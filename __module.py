import logging
import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Any, Final, Tuple, Union

import cv2
import numpy as np
import PIL.ExifTags as ExifTags
from PIL import Image
from scipy import ndimage
from skimage import io, util

from modules.config import (
    REGISTRATED_DESTINATION,
    SEGMENTATED_DESTINATION,
    tqdm,
)

try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cp_median_filter

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    is_cupy_available = True
except:
    is_cupy_available = False

logger = logging.getLogger("RSAvis3D")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class VolumePath(object):
    directory: str
    registrated_directory: str = REGISTRATED_DESTINATION
    segmentated_directory: str = SEGMENTATED_DESTINATION

    def __post_init__(self):
        if self.directory.endswith("/") or self.directory.endswith("\\"):
            object.__setattr__(self, "directory", self.directory[:-1])

    @property
    def SBI_pcd_file(self):
        dir_name, base_name = os.path.split(self.directory)
        destination = (
            dir_name
            + "/"
            + self.registrated_directory
            + "/"
            + base_name
            + "_SBI.pcd"
        )
        return destination

    @property
    def registrated_volume_directory(self):
        dir_name, base_name = os.path.split(self.directory)
        destination = (
            dir_name + "/" + self.registrated_directory + "/" + base_name
        )
        return destination

    @property
    def segmentated_volume_directory(self):
        dir_name, base_name = os.path.split(self.directory)
        destination = (
            dir_name + "/" + self.segmentated_directory + "/" + base_name
        )
        return destination


@dataclass(frozen=True)
class VolumeLoader(object):
    volume_directory: str
    minimum_file_number: int = 64
    extensions: Tuple = (".cb", ".png", ".tif", ".tiff", ".jpg", ".jpeg")

    def __post_init__(self):
        assert os.path.isdir(self.volume_directory)

    @property
    def image_file_list(self):
        img_files = [
            os.path.join(self.volume_directory, f)
            for f in os.listdir(self.volume_directory)
        ]

        ext_count = []
        for ext in self.extensions:
            ext_count.append(
                len([f for f in img_files if f.lower().endswith(ext)])
            )

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        return sorted(
            [f for f in img_files if f.lower().endswith(target_extension)]
        )

    @property
    def image_file_number(self):
        return len(self.image_file_list)

    def is_volume_directory(self):
        return self.image_file_number >= self.minimum_file_number

    def load(self) -> np.ndarray:
        assert os.path.isdir(self.volume_directory)

        logger.info(
            f"Loading {self.image_file_number} image files: {self.volume_directory}"
        )

        ndarray = np.array([io.imread(f) for f in tqdm(self.image_file_list)])

        return ndarray


@dataclass(frozen=True)
class VolumeInformation(object):
    volume_directory: str
    extensions: Tuple = (".cb", ".png", ".tif", ".tiff", ".jpg", ".jpeg")

    def __post_init__(self):
        assert os.path.isdir(self.volume_directory)

    @property
    def image_file_list(self):
        img_files = [
            os.path.join(self.volume_directory, f)
            for f in os.listdir(self.volume_directory)
        ]

        ext_count = []
        for ext in self.extensions:
            ext_count.append(
                len([f for f in img_files if f.lower().endswith(ext)])
            )

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        return sorted(
            [f for f in img_files if f.lower().endswith(target_extension)]
        )

    def get(self):
        image = Image.open(self.image_file_list[0])
        exif = {}

        information_dict = {}
        if image.getexif():
            for k, v in image.getexif().items():
                if k in ExifTags.TAGS:
                    exif[ExifTags.TAGS[k]] = v

        for k, v in exif.items():
            if k == "XResolution":
                information_dict.update({"resolution": float(v)})

        return information_dict


@dataclass(frozen=True)
class VolumeSaver(object):
    destination_directory: str
    np_volume: np.ndarray

    def __post_init__(self):
        assert len(self.np_volume) != 0

    def save(self, extension="jpg"):
        os.makedirs(self.destination_directory, exist_ok=True)

        logger.info(
            f"Saving {self.np_volume.shape[0]} image files: {self.destination_directory}"
        )
        for i, img in enumerate(tqdm(self.np_volume)):  # type: ignore
            image_file = os.path.join(
                self.destination_directory, f"img{str(i).zfill(4)}.{extension}"
            )
            io.imsave(image_file, img)


def tqdm_multiprocessing(fun, l):
    with Pool(multiprocessing.cpu_count()) as p:
        l = list(tqdm(p.imap(fun, l), total=len(l)))
    return l


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
        logger.info(
            f"Median3D filter: {block_size=}, {median_kernel_size=}, {all_at_once=}"
        )
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
                    for b in tqdm(blocks)
                ]
            else:
                blocks = tqdm_multiprocessing(
                    partial(ndimage.median_filter, size=median_kernel_size),
                    blocks,
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
        logger.info(
            f"Edge detection filter: {kernel_size=}, {intensity_factor=}"
        )

        self.__volume = self.__volume.get()

        def fun(img):
            blurred_img = cv2.blur(img, ksize=(kernel_size, kernel_size))
            edge_img = (np.array(img, dtype=cp.float32) - blurred_img).clip(0)
            return np.array(edge_img * intensity_factor, dtype=np.uint8)

        self.__volume = np.array([fun(img) for img in tqdm(self.__volume)])

        if is_cupy_available:
            self.__volume = cp.array(self.__volume)
