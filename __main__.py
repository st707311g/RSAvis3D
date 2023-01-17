import argparse
import os
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
from skimage import io

from modules import config
from modules.config import DESCRIPTION, logger
from modules.rsavis3d import RSAvis3D
from modules.volume import volume_loading_func, volume_saving_func


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-s", "--src", type=str, help="source directory.")
    parser.add_argument("-d", "--dst", type=str, help="destination directory.")
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=64,
        help="divided volume size (>= 64)",
    )
    parser.add_argument(
        "-a",
        "--all_at_once",
        action="store_true",
        help="all-at-onec processing",
    )
    parser.add_argument(
        "-m",
        "--median_kernel_size",
        type=int,
        default=7,
        help="median kernel size (>= 1)",
    )
    parser.add_argument(
        "-e",
        "--edge_size",
        type=int,
        default=21,
        help="blur kernel size for edge detection (>= 1)",
    )
    parser.add_argument(
        "-c",
        "--cylinder_radius",
        type=int,
        default=300,
        help="cylinder mask radius (>= 64). If 0, masking process will be skipped.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=("png", "tif", "jpg"),
        default="jpg",
        help="file format type",
    )
    parser.add_argument(
        "-i",
        "--intensity_factor",
        type=int,
        default=10,
        help="intensity factor (>0)",
    )
    parser.add_argument(
        "--mm_resolution",
        type=float,
        default=0.0,
        help="spatial resolution [mm].",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=-1,
        help="depth of the maximum level to be explored. Defaults to unlimited.",
    )
    parser.add_argument(
        "--save_projection",
        action="store_true",
        help="save projection images.",
    )
    parser.add_argument(
        "--using_gpu_number",
        type=int,
        default=1,
        help="GPU number used for computing (>1)",
    )
    args = parser.parse_args()
    config.using_gpu_number = int(args.using_gpu_number)

    if args.src is None:
        parser.print_help()
        exit()

    root_src_dir: Path = Path(args.src).resolve()
    if not root_src_dir.is_dir():
        logger.error("Indicate valid virectory path.")
        exit()

    root_dst_dir = Path(args.dst or str(root_src_dir) + "_rsavis3d")

    mm_resolution = float(args.mm_resolution)
    depth = int(args.depth)

    volume_loading_queue = Queue()
    volume_loading_process = Process(
        target=volume_loading_func,
        args=(root_src_dir, root_dst_dir, depth, volume_loading_queue, logger),
    )
    volume_loading_process.start()

    volume_saving_queue = Queue()
    volume_saving_process = Process(
        target=volume_saving_func,
        args=(volume_saving_queue, logger),
    )
    volume_saving_process.start()

    while True:
        (
            volume_path,
            np_volume,
            volume_info,
        ) = volume_loading_queue.get()
        if volume_path is None:
            break

        relative_path = volume_path.relative_to(root_src_dir)

        logger.info(f"[processing start] {relative_path}")
        np_volume = perform_rsavis3d(
            np_volume=np_volume,
            block_size=args.block_size,
            median_kernel_size=args.median_kernel_size,
            edge_size=args.edge_size,
            cylinder_radius=args.cylinder_radius,
            all_at_once=args.all_at_once,
            intensity_factor=args.intensity_factor,
        )
        logger.info(f"[processing end] {relative_path}")

        if mm_resolution != 0:
            volume_info.update({"mm_resolution": mm_resolution})

        while volume_saving_queue.qsize() == 1:
            pass

        dst_path = Path(root_dst_dir, relative_path)
        if args.save_projection:
            os.makedirs(dst_path)
            for i in range(3):
                projection = np_volume.max(axis=i)
                io.imsave(
                    Path(
                        root_dst_dir, f"projection{i}_{volume_path.name}.jpg"
                    ),
                    projection,
                )

        volume_saving_queue.put(
            (dst_path, root_dst_dir, np_volume, volume_info)
        )

    volume_saving_queue.put((None, None, None, None))


def perform_rsavis3d(
    np_volume: np.ndarray,
    block_size: int,
    median_kernel_size: int,
    edge_size: int,
    cylinder_radius: int,
    all_at_once: bool,
    intensity_factor: int,
):
    rsavis3d = RSAvis3D(np_volume=np_volume)
    if cylinder_radius != 0:
        rsavis3d.make_sylinder_mask(radius=cylinder_radius)
        rsavis3d.trim_with_mask(padding=median_kernel_size // 2)
    rsavis3d.apply_median3d(
        block_size=block_size,
        median_kernel_size=median_kernel_size,
        all_at_once=all_at_once,
    )
    rsavis3d.invert()
    rsavis3d.detect_edge(
        kernel_size=edge_size,
        intensity_factor=intensity_factor,
    )
    if cylinder_radius != 0:
        rsavis3d.apply_mask()

    np_volume = rsavis3d.get_np_volume()

    return np_volume


if __name__ == "__main__":
    main()
