import logging
import warnings
from typing import Final

import tqdm as tqdm_

warnings.filterwarnings('ignore')

AUTHOR: Final[str] = 'Shota Teramoto'
COPYRIGHT: Final[str] = '2020 National Agriculture and Food Research Organization. All rights reserved.'
PROGRAM_NAME: Final[str] = 'RSAvis3D'
VERSION: Final[str] = '1.1'
DESCRIPTION: Final[str] = f'{PROGRAM_NAME} (Version {VERSION}) Author: {AUTHOR}. Copyright (C) {COPYRIGHT}'

REGISTRATED_DESTINATION: Final[str] = '.registrated'
SEGMENTATED_DESTINATION: Final[str] = '.segmentated'

logger = logging.getLogger(PROGRAM_NAME)
logger.setLevel(logging.INFO)

try:
    import coloredlogs
    coloredlogs.install(level=logging.INFO)
except:
    pass

class tqdm(tqdm_.tqdm):
    def __init__(self, *args, **kwargs):
        if not 'bar_format' in kwargs:
            kwargs.update({'bar_format': '{l_bar}{bar:10}{r_bar}{bar:-10b}'})
        super().__init__(*args, **kwargs)
