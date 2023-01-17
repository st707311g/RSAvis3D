import multiprocessing
import re
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import ndimage
from skimage import util

from modules import config

median_filter_size = 7

if config.is_cupy_available:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cp_median_filter

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)


def median3d_for_multigpu(array):
    process_name = multiprocessing.current_process().name
    worker_id = int(re.sub(r"\D", "", process_name))
    dev_id = worker_id % config.using_gpu_number

    if config.is_cupy_available:
        with cp.cuda.Device(dev_id):
            piece = cp.asarray(array)
            piece = cp_median_filter(piece, size=median_filter_size)
            ret = piece.get()
            del piece
    else:
        ret = ndimage.median_filter(array, size=median_filter_size)

    return ret


class RSAvis3D(object):
    def __init__(self, np_volume: np.ndarray):
        super().__init__()
        self.__volume: np.ndarray = np_volume
        self.__mask: np.ndarray = np.empty((0))

    def get_np_volume(self):
        return self.__volume

    def block_separator(self, overlapping=0, block_size=64, all_at_once=False):
        shape = self.__volume.shape

        temp_volume = np.pad(self.__volume, overlapping, mode="symmetric")

        blocks = []
        indexes = []
        for zi in range(0, shape[0], block_size):
            for yi in range(0, shape[1], block_size):
                for xi in range(0, shape[2], block_size):
                    blocks.append(
                        temp_volume[
                            zi : zi + block_size + overlapping * 2,
                            yi : yi + block_size + overlapping * 2,
                            xi : xi + block_size + overlapping * 2,
                        ]
                    )
                    indexes.append([zi, yi, xi])

            if not all_at_once:
                yield (blocks, indexes)
                blocks = []
                indexes = []

        if blocks:
            yield (blocks, indexes)

    def __update_volume(self, blocks, indexes, overlapping=0, block_size=64):
        for block, index in zip(blocks, indexes):
            block = block[
                overlapping:-overlapping,
                overlapping:-overlapping,
                overlapping:-overlapping,
            ]
            s = [slice(index[i], index[i] + block_size) for i in range(3)]
            self.__volume[s[0], s[1], s[2]] = block

    def make_sylinder_mask(self, radius: int):
        assert radius > 0

        shape = self.__volume.shape
        x, y = np.indices((shape[1], shape[2]))
        self.__mask = (x - shape[1] / 2) ** 2 + (
            y - shape[2] / 2
        ) ** 2 < radius**2
        self.__mask: np.ndarray = np.repeat([self.__mask], shape[0], axis=0)

    def trim_with_mask(self, padding=0):
        slices = []
        for axis in range(3):
            v = np.max(
                self.__mask, axis=tuple([i for i in range(3) if i != axis])
            )
            index = np.where(v)
            slices.append(
                slice(
                    max(np.min(index) - padding, 0),
                    min(np.max(index) + padding + 1, self.__mask.shape[axis]),
                )
            )

        self.__mask = self.__mask[slices[0], slices[1], slices[2]]
        self.__volume = self.__volume[slices[0], slices[1], slices[2]]

    def apply_mask(self, padding=0):
        self.trim_with_mask(padding=padding)
        self.__volume *= self.__mask

    def apply_median3d(
        self, block_size=64, median_kernel_size=7, all_at_once=False
    ):
        global median_filter_size
        median_filter_size = median_kernel_size

        overlapping = max((median_kernel_size + 1) // 2, 1)

        i_block = self.block_separator(
            overlapping=overlapping,
            block_size=block_size,
            all_at_once=all_at_once,
        )

        for blocks, indexes in i_block:
            if config.is_cupy_available:
                if config.using_gpu_number == 1:
                    blocks = [
                        cp_median_filter(
                            cp.asarray(b), size=median_kernel_size
                        ).get()
                        for b in blocks
                    ]
                else:
                    with Pool(config.using_gpu_number) as p:
                        blocks = list(p.imap(median3d_for_multigpu, blocks))
            else:
                with Pool(multiprocessing.cpu_count()) as p:
                    blocks = list(
                        p.imap(
                            partial(
                                ndimage.median_filter, size=median_kernel_size
                            ),
                            blocks,
                        )
                    )

            self.__update_volume(
                blocks, indexes, overlapping=overlapping, block_size=block_size
            )

    def invert(self):
        self.__volume = np.array(util.invert(self.__volume))

    def detect_edge(self, kernel_size=21, intensity_factor=1.0):
        def fun(img):
            blurred_img = np.array(
                cv2.blur(
                    np.array(img, dtype=np.int64),
                    ksize=(kernel_size, kernel_size),
                ),
                dtype=np.int64,
            )
            edge_img = (
                (np.array(img, dtype=np.int64) - blurred_img)
                * intensity_factor
            ).clip(0, 255)
            return np.array(edge_img, dtype=np.uint8)

        self.__volume = np.array([fun(img) for img in self.__volume])
