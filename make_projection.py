from __future__ import annotations

import argparse
import logging
import warnings
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import coloredlogs
import numpy as np
from skimage import io

from st_modules import config
from st_modules.config import logger
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.volume import VolumeData

warnings.simplefilter("ignore")


@dataclass
class ProjectionMaker(object):
    _src_path: Path
    _dst_path: Path
    _label: str = None

    def __post_init__(self):
        if self._src_path.is_dir():
            self._src_dir_to_load = Path(self._src_path)
        else:
            self._src_tmpdir = TemporaryDirectory()
            with zipfile.ZipFile(self._src_path, "r") as zf:
                zf.extractall(self._src_tmpdir.name)
            self._src_dir_to_load = Path(self._src_tmpdir.name)

    @cached_property
    def _image_file_names(self):
        return VolumeData(self._src_dir_to_load).image_files

    @cached_property
    def _image_file_number(self):
        return len(self._image_file_names)

    def _print_callback(self, i: int, total: int, label="processing"):
        percent = round((i + 1) * 100 / total, 1)
        print(f"\r{label}: {percent}%", end="")
        if i + 1 == total:
            print("")

    def run(self, axis: str):
        axis2shape = {
            "z": (self._shape[1], self._shape[2]),
            "y": (self._shape[0], self._shape[2]),
            "x": (self._shape[0], self._shape[1]),
        }
        projection = np.zeros(axis2shape[axis], dtype=np.uint8)

        for i, img_file in enumerate(self._image_file_names):
            self._print_callback(i, self._image_file_number, self._label)
            img = io.imread(img_file)
            if axis == "z":
                projection = np.max([projection, img], axis=0)
            if axis == "y":
                projection[i] = img.max(axis=1)
            if axis == "x":
                projection[i] = img.max(axis=0)

        io.imsave(self._dst_path, projection)

    @cached_property
    def _shape(self):
        m_img = io.imread(self._image_file_names[self._image_file_number // 2])
        return (self._image_file_number,) + m_img.shape

    def __del__(self):
        try:
            self._src_tmpdir.cleanup()
        except:  # noqa
            pass


@dataclass
class Args:
    src: Path
    axis: str
    series_src: str
    debug: bool

    @property
    def series_dst(self):
        return f"{self.series_src}_projection_{self.axis}"


def main():
    parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="source directory",
    )
    parser.add_argument(
        "--axis",
        choices=["z", "y", "x"],
        type=str,
        required=True,
        help="projection is created along the axis",
    )
    parser.add_argument(
        "--series_src",
        type=str,
        default="rsavis3d",
        help="series name in the RSA dataset, which is used to make projection.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode.",
    )

    args = Args(**parser.parse_args().__dict__)

    if args.debug:
        coloredlogs.install(level=logging.DEBUG, logger=logger)
    else:
        coloredlogs.install(level=logging.INFO, logger=logger)

    logger.debug(args.__dict__)

    if not args.src.exists():
        logger.error("indicate valid path.")
        exit()

    for dataset, skipped in walk_to_find_rsa_dataset(args.src, [args.series_src], [args.series_dst]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"{relative_path=}")
        if skipped:
            logger.info("skipped")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    dst_path = dataset.create_new_series(args.series_dst, ".jpg")
    ProjectionMaker(
        _src_path=dataset.get_series_path(args.series_src),
        _dst_path=dst_path,
        _label="making projetion",
    ).run(axis=args.axis)


if __name__ == "__main__":
    main()
