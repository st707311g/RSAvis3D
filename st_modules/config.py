import logging
from typing import Final

try:
    import cupy as cp  # noqa
    from cupyx.scipy.ndimage import median_filter  # noqa

    is_cupy_available = True
except:  # noqa
    is_cupy_available = False

AUTHOR: Final = "Shota Teramoto"
COPYRIGHT: Final = "2020 National Agriculture and Food Research Organization. All rights reserved."
PROGRAM_NAME: Final = "RSAvis3D"
VERSION: Final = "1.6"
DESCRIPTION: Final[str] = f"{PROGRAM_NAME} (Version {VERSION}) Author: {AUTHOR}. Copyright (C) {COPYRIGHT}"

logger = logging.getLogger(PROGRAM_NAME)
