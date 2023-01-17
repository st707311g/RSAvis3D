import multiprocessing
from functools import partial
from multiprocessing import Pool
from typing import Union

import cv2
import numpy as np
from scipy import ndimage
from skimage import util

try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cp_median_filter

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    is_cupy_available = True
except ModuleNotFoundError:
    is_cupy_available = False


class RSAvis3D(object):
    def __init__(self, np_volume: np.ndarray):
        super().__init__()
        self.__volume: Union[cp.ndarray, np.ndarray] = (
            cp.array(np_volume) if is_cupy_available else np_volume
        )
        self.__mask: np.ndarray = np.empty((0))

    def get_np_volume(self):
        if is_cupy_available:
            return self.__volume.get()
        else:
            return self.__volume

    def block_separator(self, overlapping=0, block_size=64, all_at_once=False):
        shape = self.__volume.shape

        pad_function = cp.pad if is_cupy_available else np.pad
        temp_volume = pad_function(
            self.__volume, overlapping, mode="symmetric"
        )

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

        mask = cp.array(self.__mask) if is_cupy_available else self.__mask
        self.__volume *= mask

    def apply_median3d(
        self, block_size=64, median_kernel_size=1, all_at_once=False
    ):
        overlapping = max((median_kernel_size + 1) // 2, 1)

        i_block = self.block_separator(
            overlapping=overlapping,
            block_size=block_size,
            all_at_once=all_at_once,
        )

        for blocks, indexes in i_block:
            if is_cupy_available:
                blocks = [
                    cp_median_filter(b, size=median_kernel_size)
                    for b in blocks
                ]
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
        if is_cupy_available:
            self.__volume = cp.array(util.invert(self.__volume.get()))
        else:
            self.__volume = np.array(util.invert(self.__volume))

    def detect_edge(self, kernel_size=21, intensity_factor=1.0):

        self.__volume = self.__volume.get()

        def fun(img):
            blurred_img = cv2.blur(img, ksize=(kernel_size, kernel_size))
            edge_img = (np.array(img, dtype=cp.float32) - blurred_img).clip(0)
            return np.array(edge_img * intensity_factor, dtype=np.uint8)

        self.__volume = np.array([fun(img) for img in self.__volume])

        if is_cupy_available:
            self.__volume = cp.array(self.__volume)
