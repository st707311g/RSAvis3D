import argparse
import os
import shutil
from dataclasses import dataclass
from glob import glob
from typing import List

from PIL import Image
from skimage import io

from __common import DESCRIPTION, logger
from __module import RSAvis3D, VolumeLoader, VolumePath, VolumeSaver


@dataclass(frozen=True)
class RSAvis3D_Parameters(object):
    source: str
    block_size: int
    median_kernel_size: int
    edge_size: int
    cylinder_radius: int
    all_at_once: bool
    format:str
    intensity_factor: int

    def __post_init__(self):
        assert os.path.isdir(self.source)
        assert self.block_size >= 64
        assert self.median_kernel_size > 0
        assert self.edge_size > 0
        assert self.cylinder_radius >= 64 or self.cylinder_radius == 0
        assert self.intensity_factor > 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION) 
    parser.add_argument('-s', '--source', type=str, help='Indicate source directory.')

    parser.add_argument('-b', '--block_size', type=int, default=64, help='Indicate divided volume size (>= 64)')
    parser.add_argument('-a', '--all_at_once', action='store_true', help='Perform all-at-onec processing')
    parser.add_argument('-m', '--median_kernel_size', type=int, default=7, help='Indicate median kernel size (>= 1)')
    parser.add_argument('-e', '--edge_size', type=int, default=21, help='Indicate blur kernel size for edge detection (>= 1)')
    parser.add_argument('-c', '--cylinder_radius', type=int, default=300, help='Indicate cylinder mask radius (>= 64). If 0, masking process will be skipped.')
    parser.add_argument('-f', '--format', type=str, choices=("png", "tif", "jpg"), default = "jpg", help='Indicate file format type')
    parser.add_argument('-i', '--intensity_factor', type=int, default = 10, help='Indicate intensity factor (>0)')

    args = parser.parse_args()

    if args.source is None:
        parser.print_help()
        exit()

    source_directory: str = args.source
    if source_directory.endswith('/') or source_directory.endswith('\\'):
        source_directory = source_directory[:-1]
    
    if not os.path.isdir(source_directory):
        logger.error(f'Indicate valid directory path.')
        exit()

    rsavis3d_params = RSAvis3D_Parameters(**vars(args))

    #// listing the target volume directories
    volume_directory_list: List[str] = []
    for volume_directory in sorted(glob(rsavis3d_params.source+'/**/', recursive=True)):
        if not VolumeLoader(volume_directory=volume_directory).is_volume_directory():
            continue

        volume_directory_list.append(volume_directory)

    for volume_directory in volume_directory_list:
        destination=VolumePath(volume_directory).segmentated_volume_directory

        volume_information_source = volume_directory+'/.volume_information'
        volume_information_destination = destination+'/.volume_information'

        if os.path.isdir(rsavis3d_volume := VolumePath(volume_directory).segmentated_volume_directory):
            logger.error(f'[skip] The RSAvis3D volume already exists: {rsavis3d_volume}')
            continue

        if not os.path.isdir(target_volume_path := VolumePath(volume_directory).registrated_volume_directory):
            target_volume_path = volume_directory
        
        np_volume = VolumeLoader(target_volume_path).load()
        rsavis3d = RSAvis3D(np_volume=np_volume)
        if rsavis3d_params.cylinder_radius != 0:
            rsavis3d.make_sylinder_mask(radius=rsavis3d_params.cylinder_radius)
            rsavis3d.trim_with_mask(padding=rsavis3d_params.median_kernel_size//2)
        rsavis3d.apply_median3d(median_kernel_size=rsavis3d_params.median_kernel_size, all_at_once=rsavis3d_params.all_at_once)
        rsavis3d.invert()
        rsavis3d.detect_edge(intensity_factor=rsavis3d_params.intensity_factor)
        if rsavis3d_params.cylinder_radius != 0:
            rsavis3d.apply_mask()

        np_volume = rsavis3d.get_np_volume()
        VolumeSaver(
            destination_directory=VolumePath(volume_directory).segmentated_volume_directory, 
            np_volume=np_volume
        ).save(extension=rsavis3d_params.format)

        if os.path.isfile(volume_information_source):
            shutil.copyfile(volume_information_source, volume_information_destination)

        destination_directory=VolumePath(volume_directory).segmentated_volume_directory

        for d in range(3):
            projection_destination = os.path.dirname(destination)+f'/.projection{d}'
            os.makedirs(projection_destination, exist_ok=True)
            projection = np_volume.max(axis=d)
            io.imsave(f'{projection_destination}/{os.path.basename(destination)}.{rsavis3d_params.format}', projection)

    #// making gif animation 
    for volume_directory in volume_directory_list:
        destination=VolumePath(volume_directory).segmentated_volume_directory
        for d in range(3):
            animation_gif_file = os.path.dirname(destination)+f'/.projection{d}.gif'
            if os.path.isfile(animation_gif_file):
                continue
            img_file_list = sorted(glob(os.path.join(os.path.dirname(destination)+f'/.projection{d}', f'*.*')))
            if len(img_file_list) > 1:
                imgs = [Image.open(f) for f in img_file_list]
                imgs[0].save(animation_gif_file, save_all=True, append_images=imgs[1:], duration=300, loop=0)
