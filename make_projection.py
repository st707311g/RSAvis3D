from __future__ import annotations

import logging
import warnings
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import coloredlogs
import numpy as np
import typer
from skimage import io

from st_modules.rsa_dataset import walk_to_find_rsa_dataset
from st_modules.volume import VolumeData

warnings.simplefilter("ignore")
logger = logging.getLogger(Path(__file__).name)


@dataclass
class ProjectionMaker:
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


def main(
    search_dir: Annotated[
        Path,
        typer.Argument(help="Indicate search directory."),
    ],
    axis: Literal["x", "y", "z"] = typer.Option(
        "x",
        "--axis",
        help="Indicate format type of saving image file.",
    ),
    series_src: str = typer.Option(
        "rsavis3d",
        "--series-src",
        help="Indicate series name in the RSA dataset, which will be processed.",
    ),
    debug_mode: bool = typer.Option(
        False,
        "--debug",
        "-D",
        help="Launch in debug mode.",
    ),
):
    if debug_mode:
        coloredlogs.install(level=logging.DEBUG, logger=logger)
        logger.debug(f"{search_dir=}, {format=}, {series_src=}, {debug_mode=}")
    else:
        coloredlogs.install(level=logging.INFO, logger=logger)

    if not search_dir.exists():
        logger.error(f"Indicate valid path: {search_dir}")
        exit()

    series_dst = f"{series_src}_projection_{axis}"

    for dataset, skipped in walk_to_find_rsa_dataset(
        search_dir, [series_src], [series_dst]
    ):
        relative_path = dataset._config.path.relative_to(search_dir)
        logger.info(f"{relative_path=}")
        if skipped:
            logger.info("skipped")
            continue

        ProjectionMaker(
            _src_path=dataset.get_series_path(series_src),
            _dst_path=dataset.create_new_series(series_dst, ".jpg"),
            _label="making projetion",
        ).run(axis=axis)


if __name__ == "__main__":
    typer.run(main)
