import shutil
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import cached_property, lru_cache, reduce
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ndimagetools.filter import jit8bit_3d_median_filter as jit_median_filter
from scipy.ndimage import uniform_filter
from skimage import io, util

from .volume import VolumeData

warnings.simplefilter("ignore")


class RSAvis3D:
    def __init__(
        self,
        src_path: str | Path,
        dst_path: str | Path,
        dst_file_type="jpg",
        block_size: int = 64,
        overlap: int = 8,
        label: str = None,
    ):
        self.src_path = Path(src_path)
        self.dst_path = Path(dst_path)
        self.dst_file_type = dst_file_type

        self._block_size = block_size
        self._overlap = overlap
        self._label = label

        if self.src_path.is_dir():
            self.volume_path = Path(self.src_path)
        else:
            self.tmpdir = TemporaryDirectory()
            with zipfile.ZipFile(self.src_path, "r") as zf:
                zf.extractall(self.tmpdir.name)
            self.volume_path = Path(self.tmpdir.name)

        self.image_file_names = VolumeData(self.volume_path).image_files
        self.image_file_number = len(self.image_file_names)

        self.print_callback_i = 0

        self.__get_layer_cache: dict[int, np.ndarray] = {}

    def print_callback(self, i: int, total: int, label="processing"):
        percent = round(i * 100 / self.block_count_total, 1)
        print(f"\r{label}: {percent}%", end="")
        if i + 1 == total:
            print("")

    def print_callback_(self, *args, **kwargs):
        self.print_callback(
            i=self.print_callback_i,
            total=self.block_count_total,
            label=f"{self._label or self.src_path}",
        )

        self.print_callback_i += 1

    @staticmethod
    @lru_cache(maxsize=1)
    def get_cylinder_mask(layer_shape: tuple[int], radius: int):
        assert radius > 0

        x, y = np.indices((layer_shape[1], layer_shape[2]))
        mask = (x - layer_shape[1] / 2) ** 2 + (y - layer_shape[2] / 2) ** 2 < radius**2
        mask: np.ndarray = np.repeat([mask], layer_shape[0], axis=0)

        return mask

    def run(
        self,
        median_kernel_size=7,
        edge_size=21,
        cylinder_radius: int = None,
        intensity_factor=10.0,
    ):
        temp_dst_dir = TemporaryDirectory()
        temp_dst_file_index = 0

        blocks = []
        for i in range(self.block_count_total):
            blocks.append(self.get_block(i))

            if len(blocks) != self.block_count[1] * self.block_count[2]:
                continue

            with ProcessPoolExecutor() as executor:
                futures = []
                for block in blocks:
                    futures.append(
                        executor.submit(
                            jit_median_filter,
                            block,
                            median_kernel_size,
                        )
                    )

                blocks = []
                for future in futures:
                    blocks.append(future.result())
                    self.print_callback_()

            self.__get_layer_cache.clear()
            layer = self.assemble_blocks_to_get_layer(blocks)
            blocks.clear()

            layer = np.array(util.invert(layer))
            layer = self.detect_edge(layer, edge_size, intensity_factor)
            layer = layer[:, 0 : self.shape[1], 0 : self.shape[2]]

            if cylinder_radius != 0:
                layer *= self.get_cylinder_mask(layer.shape, cylinder_radius)

            dst_file_list = []
            max_i = -1
            for i_ in range(len(layer)):
                if 0 <= temp_dst_file_index < self.shape[0]:
                    dst_file = Path(
                        temp_dst_dir.name,
                        f"img_{temp_dst_file_index:04}.{self.dst_file_type}",
                    )
                    dst_file_list.append(dst_file)
                    max_i = i_ if i_ > max_i else max_i

                temp_dst_file_index += 1

            with ThreadPoolExecutor() as executor:
                futures = []
                for f, s in zip(dst_file_list, layer[0 : max_i + 1], strict=True):
                    futures.append(executor.submit(io.imsave, f, s))

                blocks = []
                for future in futures:
                    future.result()

        if str(self.dst_path).endswith(".zip"):
            with zipfile.ZipFile(self.dst_path, "w", zipfile.ZIP_STORED) as zf:
                for p in sorted(Path(temp_dst_dir.name).glob("*")):
                    zf.write(p, p.name)
        else:
            self.dst_path.mkdir(parents=True, exist_ok=True)
            for p in sorted(Path(temp_dst_dir.name).glob("*")):
                shutil.copyfile(p, Path(self.dst_path, p.name))

        temp_dst_dir.cleanup()

    def detect_edge(self, np_ary: np.ndarray, kernel_size=21, intensity_factor=1.0):
        def fun(img: np.ndarray):
            img = img.astype(np.int16)
            blurred_img = uniform_filter(img, size=(kernel_size, kernel_size))

            blurred_img = np.asarray(blurred_img, dtype=np.int16)
            edge_img = (img - blurred_img) * intensity_factor
            edge_img = edge_img.clip(0, 255).astype(np.uint8)
            return edge_img

        with ThreadPoolExecutor() as executor:
            res_ary = np.array(list(executor.map(fun, np_ary)))

        return res_ary

    def assemble_blocks_to_get_layer(self, blocks):
        np_vol = np.zeros(
            (self._block_size - self._overlap * 2,) + self.shape_with_block_unit[1:],
            dtype=np.uint8,
        )

        i = 0
        for yi in range(self.block_count[1]):
            for xi in range(self.block_count[2]):
                np_vol[
                    :,
                    yi * self._inner_size : (yi + 1) * self._inner_size,
                    xi * self._inner_size : (xi + 1) * self._inner_size,
                ] = blocks[i][
                    self._overlap : -self._overlap,
                    self._overlap : -self._overlap,
                    self._overlap : -self._overlap,
                ]
                i += 1

        return np_vol

    @property
    def _inner_size(self):
        return self._block_size - self._overlap * 2

    def get_layer(self, layer_index: int):
        if layer_index in self.__get_layer_cache:
            return self.__get_layer_cache[layer_index]

        indexes = list(
            range(
                layer_index * self._inner_size - self._overlap,
                (layer_index + 1) * self._inner_size + self._overlap,
            )
        )

        indexes = [i if i >= 0 else -i for i in indexes]
        indexes = [
            i if i < self.image_file_number else self.image_file_number * 2 - i - 2
            for i in indexes
        ]

        self.prev_indexes = indexes.copy()

        img_files = [self.image_file_names[i] for i in indexes]

        with ThreadPoolExecutor() as executor:
            layer = np.array(list(executor.map(io.imread, img_files)))

        pad_size = [(0, 0)]
        for s in self.shape[1:]:
            pad_size.append(
                (
                    self._overlap,
                    self._inner_size * ((s + self._inner_size - 1) // self._inner_size)
                    - s
                    + self._overlap,
                )
            )

        layer = np.pad(layer, pad_size, mode="reflect")

        self.__get_layer_cache[layer_index] = layer
        return layer

    def get_block(self, block_index):
        assert block_index < self.block_count_total

        layer_index = int(
            np.floor(block_index / (self.block_count[1] * self.block_count[2]))
        )
        layer = self.get_layer(layer_index)

        block_index = (
            block_index - layer_index * self.block_count[1] * self.block_count[2]
        )
        x_index = block_index % self.block_count[1]
        y_index = block_index // self.block_count[1]

        return layer[
            :,
            y_index * self._inner_size : (y_index + 1) * self._inner_size
            + self._overlap * 2,
            x_index * self._inner_size : (x_index + 1) * self._inner_size
            + self._overlap * 2,
        ]

    @cached_property
    def shape(self):
        m_img = io.imread(self.image_file_names[self.image_file_number // 2])
        return (self.image_file_number,) + m_img.shape

    @cached_property
    def shape_with_block_unit(self):
        return tuple(
            [
                self._inner_size * ((s + self._inner_size - 1) // self._inner_size)
                for s in self.shape
            ]
        )

    @cached_property
    def block_count(self):
        return [
            int(np.ceil(it / (self._block_size - self._overlap * 2)))
            for it in self.shape
        ]

    @cached_property
    def block_count_total(self):
        return reduce(lambda x, y: x * y, self.block_count)

    def __del__(self):
        try:
            self.tmpdir.cleanup()
        except:  # noqa
            pass
