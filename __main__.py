import sys, os
from glob import glob
import argparse
from RSAvis3D import RSAvis3D

from typing import Final
version: Final[str] = '1.0'

description_text: Final[str] = f'RSAvis3D (version {version}): An robust and rapid image processing for root segmentation from X-ray CT images.'

if __name__ == '__main__':
    if sys.version_info.major!=3 or sys.version_info.minor<8:
        raise Exception('Use Python version 3.8 or later.')

    parser = argparse.ArgumentParser(description=description_text)
    parser.add_argument('-i', '--indir', type=str, help='import a directory')
    parser.add_argument('-b', '--block_size', type=int, default=64, help='determine divided volume size (>= 64)')
    parser.add_argument('-a', '--all_at_once', action='store_true', help='determine all-at-onec processing')
    parser.add_argument('-m', '--median_size', type=int, default=7, help='determine median kernel size (>= 1)')
    parser.add_argument('-e', '--edge_size', type=int, default=21, help='determine blur kernel size for edge detection (>= 1)')
    parser.add_argument('-c', '--cylinder_radius', type=int, default=300, help='determine cylinder mask radius (>= 64)')
    parser.add_argument('-f', '--format', type=str, choices=("png", "tif", "jpg"), default = "jpg", help='determine file format type')
    parser.add_argument('-w', '--overwrite', action='store_true', help='overwrite results')
    parser.add_argument('-v', '--version', action='store_true', help='show version information')
    args = parser.parse_args()

    if args.version:
        print(f'Version: {version}')
        sys.exit()

    if args.indir is None:
        parser.print_help()
        sys.exit(1)

    if (block_size:=args.block_size) < 64:
        print(f'The "block_size" should be >= 64.')
        sys.exit(1)

    if (median_size:=args.median_size) < 1:
        print(f'The "median_size" should be >= 1.')
        sys.exit(1)

    if (cylinder_radius:=args.cylinder_radius) < 64:
        print(f'The "cylinder_radius" should be >= 64.')
        sys.exit(1)

    if (edge_size:=args.edge_size) < 1:
        print(f'The "edge_size" should be >= 1.')
        sys.exit(1)


    indirs = glob(os.path.join(args.indir, '**/'), recursive=True)

    for indir in indirs:
        indir = indir[:-1] if indir.endswith("/") else indir

        if indir.endswith('_RSA'):
            print(f'Skipped: "{indir}"')
            continue

        if not os.path.isdir(indir):
            print(f'Cannnot find the directory: {indir}')
            continue

        if (outdir:=args.outdir) is None:
            outdir = indir+'_RSA'

        if not args.overwrite:
            if os.path.isdir(outdir):
                print(f'Skipped: "{indir}"')
                continue

        all_at_once = args.all_at_once
        rsavis3d = RSAvis3D()
        try:
            rsavis3d.load_from(indir=indir)
        except:
            print(f'Cannnot load images: {indir}')
            continue

        rsavis3d.normalize_intensity(block_size = block_size, all_at_once=all_at_once)
        rsavis3d.make_sylinder_mask(radius = cylinder_radius)
        rsavis3d.trim_with_mask(padding = median_size//2)
        rsavis3d.apply_median3d(median_kernel_size=median_size, all_at_once=all_at_once)
        rsavis3d.invert()
        rsavis3d.apply_edge_detection(kernel_size=edge_size)
        rsavis3d.apply_mask(padding = 0)
        rsavis3d.save_volume(outdir=outdir, format=args.format)