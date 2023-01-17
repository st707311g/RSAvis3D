import math
import operator
import os
from pathlib import Path


def walk_to_find_directories(
    path: str, depth: int = math.inf, including_source_directoriy: bool = False
):
    if including_source_directoriy:
        yield Path(path)

    depth -= 1
    with os.scandir(path) as p:
        p = list(p)
        p.sort(key=operator.attrgetter("name"))
        for entry in p:
            if entry.is_dir():
                yield Path(entry.path)
            if entry.is_dir() and depth > 0:
                yield from walk_to_find_directories(entry.path, depth)
