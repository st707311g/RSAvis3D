from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import coloredlogs

from st_modules import config
from st_modules.config import logger
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.RSAvis3D import RSAvis3D

warnings.simplefilter("ignore")


@dataclass
class Args:
    src: Path
    block_size: int
    median_kernel_size: int
    cylinder_radius: int
    edge_size: int
    format: str
    intensity_factor: float
    gpu: list[int]
    series_src: str
    series_dst: str
    archive: bool
    debug: bool

    @property
    def overlapping(self):
        return max((self.median_kernel_size + 1) // 2, 1)


def main():
    parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    parser.add_argument(
        "-s",
        "--src",
        type=Path,
        required=True,
        help="source directory",
    )
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=64,
        help="divided volume size (>= 64)",
    )
    parser.add_argument(
        "-m",
        "--median_kernel_size",
        type=int,
        default=7,
        help="median kernel size (>= 1)",
    )
    parser.add_argument(
        "-c",
        "--cylinder_radius",
        type=int,
        default=None,
        help="cylinder mask radius (>= 1). If None, masking process will be skipped.",
    )
    parser.add_argument(
        "-e",
        "--edge_size",
        type=int,
        default=21,
        help="blur kernel size for edge detection (>= 1)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=("png", "tif", "jpg"),
        default="jpg",
        help="image format type",
    )
    parser.add_argument(
        "-i",
        "--intensity_factor",
        type=float,
        default=10.0,
        help="intensity factor (>0), image intensity will be multiplied by this factor",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="*",
        help="GPU device id(s) used for computing",
    )
    parser.add_argument(
        "--series_src",
        type=str,
        default="ct",
        help="series name in the RSA dataset, which will be processed",
    )
    parser.add_argument(
        "--series_dst",
        type=str,
        default="rsavis3d",
        help="series name in the RSA dataset, which will be created",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="processed images will be saved as zip archive",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
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

    if args.gpu:
        if config.is_cupy_available:
            logger.info(f"Using GPU IDs: {args.gpu}")
        else:
            logger.error("CuPy is not available")
            exit(1)

    for dataset, skipped in walk_to_find_rsa_dataset(args.src, [args.series_src], [args.series_dst]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"{relative_path=}")
        if skipped:
            logger.info(f"skipped: {relative_path}")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    dst_path = dataset.create_new_series(args.series_dst, ".zip" if args.archive else "")

    rsavis3d = RSAvis3D(
        src_path=dataset.get_series_path(args.series_src),
        dst_path=dst_path,
        dst_file_type=args.format,
        block_size=args.block_size,
        overlap=args.overlapping,
        label=str(relative_path),
        using_gpu_ids=args.gpu,
    )
    rsavis3d.run(
        median_kernel_size=args.median_kernel_size,
        edge_size=args.edge_size,
        cylinder_radius=args.cylinder_radius,
        intensity_factor=args.intensity_factor,
    )

    dataset.update_log(
        {
            dst_path.stem: {
                "median_kernel_size": args.median_kernel_size,
                "edge_size": args.edge_size,
                "cylinder_radius": args.cylinder_radius,
                "intensity_factor": args.intensity_factor,
            }
        }
    )


if __name__ == "__main__":
    main()
