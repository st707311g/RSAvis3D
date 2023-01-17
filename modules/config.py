import logging
import warnings
from typing import Final

warnings.filterwarnings("ignore")

AUTHOR: Final[str] = "Shota Teramoto"
COPYRIGHT: Final[
    str
] = "2020 National Agriculture and Food Research Organization. All rights reserved."
PROGRAM_NAME: Final[str] = "RSAvis3D"
VERSION: Final[str] = "1.4"
DESCRIPTION: Final[
    str
] = f"{PROGRAM_NAME} (Version {VERSION}) Author: {AUTHOR}. Copyright (C) {COPYRIGHT}"


logger = logging.getLogger(PROGRAM_NAME)
logger.setLevel(logging.INFO)

try:
    import coloredlogs

    coloredlogs.install(level=logging.INFO)
except ModuleNotFoundError:
    pass
