import argparse
import logging
import re
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
from skimage import exposure, filters

from modules.config import logger
from modules.volume import volume_loading_func, volume_saving_func


def normalize_intensity(
    np_volume: np.ndarray, relative_path: Path, logger: logging.Logger
):
    logger.info(f"[processing start] {relative_path}")
    nstack = len(np_volume)
    stack: np.ndarray = np_volume[nstack // 2 - 16 : nstack // 2 + 16]

    hist_y, hist_x = exposure.histogram(stack[stack > 0])
    thr = filters.threshold_otsu(stack[stack > 0])

    peak_air = np.argmax(hist_y[hist_x < thr]) + hist_x[0]
    peak_soil = np.argmax(hist_y[hist_x > thr]) + (thr - hist_x[0]) + hist_x[0]

    np_volume = np_volume.astype(np.int64)
    for i in range(len(np_volume)):
        np_volume[i] = (
            (np_volume[i] - peak_air).clip(0)
            / (peak_soil - peak_air)
            * 256
            / 2
        )
    logger.info(f"[processing end] {relative_path}")
    return exposure.rescale_intensity(
        np_volume, in_range=(0, 255), out_range=(0, 255)
    ).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intensity Normalizer")
    parser.add_argument("-s", "--src", type=str, help="source directory.")
    parser.add_argument("-d", "--dst", type=str, help="destination directory.")
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

    args = parser.parse_args()
    if args.src is None:
        parser.print_help()
        exit(0)

    root_src_dir: Path = Path(args.src).resolve()
    if not root_src_dir.is_dir():
        logger.error("Indicate valid virectory path.")
        exit()

    root_dst_dir = Path(
        args.dst or str(root_src_dir) + "_intensity_normalized"
    )

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
        np_volume = normalize_intensity(np_volume, relative_path, logger)

        if mm_resolution != 0:
            volume_info.update({"mm_resolution": mm_resolution})

        while volume_saving_queue.qsize() == 1:
            pass

        dst_path = Path(
            root_dst_dir, re.sub(r"_cb_\d{3}$", "", str(relative_path))
        )
        volume_saving_queue.put(
            (dst_path, root_dst_dir, np_volume, volume_info)
        )

    volume_saving_queue.put((None, None, None, None))
