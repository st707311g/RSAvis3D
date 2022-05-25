import argparse
import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
from attr import field
from skimage import exposure, filters, io

from __common import logger, tqdm
from __module import VolumeInformation, VolumeLoader


@dataclass(frozen=True)
class IntensityNormalizer(object):
    volume: np.ndarray

    def run(self, block_size=256, all_at_once = True) -> np.ndarray:
        np_volume = self.volume

        nstack = len(np_volume)
        stack: np.ndarray = np_volume[nstack//2-16:nstack//2+16]
        
        hist_y, hist_x = exposure.histogram(stack[stack>0])
        thr = filters.threshold_otsu(stack[stack>0])
        
        peak_air = np.argmax(hist_y[hist_x<thr])+hist_x[0]
        peak_soil = np.argmax(hist_y[hist_x>thr])+(thr-hist_x[0])+hist_x[0]
        diff = peak_soil-peak_air
        
        logger.info(f'Intensity Normalization: Air: {peak_air}, Soil: {peak_soil}, diff: {diff}')

        maxid = [peak_air, peak_soil]
        maxid = [i-hist_x[0] for i in maxid]

        np_volume = np.array(np_volume, dtype=np.float32)
        for i in tqdm(range(len(np_volume))):
            img = np_volume[i]
            np_volume[i] = (img - peak_air).clip(0)/(peak_soil-peak_air)*256/2

        return np.array(exposure.rescale_intensity(np_volume, in_range=(0,255), out_range=(0,255)), dtype=np.uint8) #type: ignore

@dataclass
class CommandParameters(object):
    source: str = field(init=False)
    destination: str = field(init=False)

    def __init__(self, args):
        assert args.source is not None
        self.source = args.source

        if self.source.endswith('/') or self.source.endswith('\\'):
            self.source = self.source[:-1]

        if args.destination is not None:
            self.destination = args.destination
        else:
            self.destination = self.source+'_intensity_normalized'

        if self.destination.endswith('/') or self.destination.endswith('\\'):
            self.destination = self.destination[:-1]

    def __post_init__(self):
        assert self.source != self.destination

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intensity Normalizer') 
    parser.add_argument('-s', '--source', type=str, help='Source directory.')
    parser.add_argument('-d', '--destination', type=str, help='Destination directory.')

    args = parser.parse_args()
    if args.source is None:
        parser.print_help()
        exit(0)

    source_directory: str = args.source
    if source_directory.endswith('/') or source_directory.endswith('\\'):
        source_directory = source_directory[:-1]
    
    if not os.path.isdir(source_directory):
        logger.error(f'Indicate valid virectory path.')
        exit()


    destination_directory = args.destination if args.destination is not None else source_directory+'_intensity_normalized'
    if destination_directory.endswith('/') or destination_directory.endswith('\\'):
        destination_directory = destination_directory[:-1]

    if os.path.isdir(destination_directory):
        logger.info(f'The distination directory already exists. Do you want to overwrite it? {destination_directory}')
        while(True):
            inp=input('y/n? >> ')
            if inp == 'y':
                break
            if inp == 'n':
                exit(1)

    volume_directory_list = sorted(glob(source_directory+'/**/', recursive=True))
    volume_directory_list = [volume_directory for volume_directory in volume_directory_list if VolumeLoader(volume_directory).is_volume_directory()]

    if len(volume_directory_list) == 0:
        logger.info(f'There are no directories to be processed.')
        exit()

    for volume_directory in volume_directory_list:
        relative_path = Path(volume_directory).relative_to(source_directory)
        final_destination_directory = os.path.join(destination_directory, relative_path)
        if os.path.isdir(final_destination_directory):
            logger.info(f'[skip] {volume_directory}')
            continue

        volume = VolumeLoader(volume_directory).load()
        normalized_volume = IntensityNormalizer(volume=volume).run()

        logger.info(f'Saving {len(normalized_volume)} image files: {final_destination_directory}')
        os.makedirs(final_destination_directory, exist_ok=True)
        for i, img in enumerate(tqdm(normalized_volume)):
            out_fname = os.path.join(final_destination_directory, f'img{str(i).zfill(4)}.jpg')
            io.imsave(out_fname, img)

        information = VolumeInformation(volume_directory=volume_directory).get()
        with open(final_destination_directory+'/.volume_information', 'w') as f:
            json.dump(information, f)
