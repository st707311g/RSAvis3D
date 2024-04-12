from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass
class VolumeData(object):
    path: str | Path
    minimum_file_number: int = 64
    extensions: tuple = (
        ".cb",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
    )

    def __post_init__(self):
        self.path = Path(self.path)

    @cached_property
    def image_files(self):
        files = list(sorted(self.path.glob("*")))

        ext_count = []
        for ext in self.extensions:
            ext_count.append(len([f for f in files if str(f).lower().endswith(ext)]))

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        image_files = sorted([f for f in files if str(f).lower().endswith(target_extension)])
        return image_files

    @property
    def image_file_number(self):
        return len(self.image_files)

    def is_valid(self):
        return self.image_file_number >= self.minimum_file_number
