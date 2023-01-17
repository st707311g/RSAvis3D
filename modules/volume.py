import json
import logging
import os
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, Union

import numpy as np
from skimage import io

from .seek import walk_to_find_directories

VOLUME_INFO_FILE_NAME: Final[str] = ".volume_info.json"


class VolumeLoader(object):
    def __init__(
        self,
        volume_path: Union[str, Path],
        minimum_file_number: int = 64,
        extensions: Tuple = (
            ".cb",
            ".png",
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
        ),
        volume_info_file_name: str = VOLUME_INFO_FILE_NAME,
    ) -> None:
        self.volume_path = Path(volume_path).resolve()
        self.minimum_file_number = minimum_file_number
        self.extensions = extensions
        self.volume_info_path = Path(self.volume_path, volume_info_file_name)

        self.__image_files: List[Path] = None

        assert self.volume_path.is_dir()

    @property
    def DEFAULT_VOLUME_INFORMATION(self) -> Dict[Any, Any]:
        return {"mm_resolution": 0.3}

    @property
    def image_files(self):
        if self.__image_files:
            return self.__image_files

        files = [
            Path(self.volume_path, f) for f in os.listdir(self.volume_path)
        ]

        ext_count = []
        for ext in self.extensions:
            ext_count.append(
                len([f for f in files if str(f).lower().endswith(ext)])
            )

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        self.__image_files = sorted(
            [f for f in files if str(f).lower().endswith(target_extension)]
        )
        return self.__image_files

    @property
    def image_file_number(self):
        return len(self.image_files)

    def is_valid_volume(self):
        return self.image_file_number >= self.minimum_file_number

    def load(self):
        return np.array([io.imread(f) for f in self.image_files])

    def load_volume_info(self):
        volume_information = self.DEFAULT_VOLUME_INFORMATION

        if self.volume_info_path.is_file():
            with open(self.volume_info_path) as f:
                volume_information.update(json.load(f))

        return volume_information


@dataclass
class VolumeSaver(object):
    def __init__(
        self,
        volume_path: Union[str, Path],
        np_volume: np.ndarray,
        volume_info: dict,
        digits: int = 4,
        extension: str = "jpg",
        volume_info_file_name: str = VOLUME_INFO_FILE_NAME,
    ) -> None:
        self.volume_path = Path(volume_path).resolve()
        self.np_volume = np_volume
        self.volume_info = volume_info
        self.digits = digits
        self.extension = extension
        self.volume_info_file_name = volume_info_file_name

        self.volume_info_path = Path(self.volume_path, volume_info_file_name)

    def save(self):
        os.makedirs(self.volume_path, exist_ok=True)

        for i, img in enumerate(self.np_volume):
            image_file_path = Path(
                self.volume_path,
                f"img{str(i).zfill(self.digits)}.{self.extension}",
            )
            io.imsave(image_file_path, img)

        with open(self.volume_info_path, "w") as f:
            json.dump(self.volume_info, f)


def volume_loading_func(
    root_src_dir: Path,
    root_dst_dir: Path,
    depth: int,
    q: Queue,
    logger: logging.Logger,
):
    for d in walk_to_find_directories(
        path=root_src_dir, depth=depth, including_source_directoriy=True
    ):
        volume_loader = VolumeLoader(d)
        if volume_loader.is_valid_volume():
            relative_path = d.relative_to(root_src_dir)
            dst_path = Path(root_dst_dir, relative_path)
            if dst_path.is_dir():
                logger.info(f"[skip] {relative_path}")
                continue
            logger.info(f"[loading start] {relative_path}")
            np_volume = volume_loader.load()
            volume_info = volume_loader.load_volume_info()
            logger.info(f"[loading end] {relative_path}")

            q.put((d, np_volume, volume_info))
        while q.qsize() == 1:
            pass

    q.put((None, None, None))


def volume_saving_func(q: Queue, logger: logging.Logger):
    while True:
        dst_path, root_dst_path, np_volume, volume_info = q.get()
        if dst_path is None:
            break

        relative_path = dst_path.relative_to(root_dst_path)
        logger.info(f"[saving start] {relative_path}")
        volume_saver = VolumeSaver(dst_path, np_volume, volume_info)
        volume_saver.save()
        logger.info(f"[saving end] {relative_path}")
