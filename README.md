# RSAvis3D: An robust and rapid image processing for root segmentation from X-ray CT images


![python](https://img.shields.io/badge/Python->3.8-lightgreen)
![developed_by](https://img.shields.io/badge/developed%20by-Shota_Teramoto-lightgreen)
![version](https://img.shields.io/badge/version-1.0-lightgreen)
![last_updated](https://img.shields.io/badge/last_update-July_3,_2021-lightgreen)

![top image](figures/top_image.jpg) 

RSAvis3D is the Python program for root segmentation from X-ray CT images. RSAvis3D uses a 3D median filter and edge detection algorithm to isolate root segments. 

## Installation

    $ git clone https://github.com/st707311g/RSAvis3D.git
    $ cd RSAvis3D

RSAvis3D requires Python (version > 3.8). Check it by the following command:

    $ python --version

After confirmation, install required modules:

    $ pip install --upgrade pip
    $ pip install -r requirements.txt

## How to run

    $ python . [-h] [-i INDIR] [-b BLOCK_SIZE] [-a] [-m MEDIAN_SIZE]
                [-e EDGE_SIZE] [-c CYLINDER_RADIUS] [-f {png,tif,jpg}] [-w]
                [-v]

    optional arguments:
      -h, --help            show this help message and exit
      -i INDIR, --indir INDIR
                            import a directory
      -b BLOCK_SIZE, --block_size BLOCK_SIZE
                            determine divided volume size (>= 64)
      -a, --all_at_once     determine all-at-onec processing
      -m MEDIAN_SIZE, --median_size MEDIAN_SIZE
                            determine median kernel size (>= 1)
      -e EDGE_SIZE, --edge_size EDGE_SIZE
                            determine blur kernel size for edge detection (>= 1)
      -c CYLINDER_RADIUS, --cylinder_radius CYLINDER_RADIUS
                            determine cylinder mask radius (>= 64)
      -f {png,tif,jpg}, --format {png,tif,jpg}
                            determine file format type
      -w, --overwrite       overwrite results
      -v, --version         show version information

Basic usage of RSAvis3D is

    $ python . -i INDIR

Parameters of RSAvis3D are modifiable by `-b`, `-m`, `-e`, and `-c` commands. If omitted, the parameters considered conditional on [the paper](https://doi.org/10.1186/s13007-020-00612-6) will be used.

## Demonstration data analysis

Download and unzip the demo data archive.

    $ wget https://rootomics.dna.affrc.go.jp/data/rice_root_daily_growth.zip
    $ unzip rice_root_daily_growth.zip
    $ rm rice_root_daily_growth.zip

There are 21 directories and 1 files in *rice_root_daily_growth*.

    $ ls rice_root_daily_growth
    DAS_07  DAS_10  DAS_13  DAS_16  DAS_19  DAS_22  DAS_25  description.docx
    DAS_08  DAS_11  DAS_14  DAS_17  DAS_20  DAS_23  DAS_26
    DAS_09  DAS_12  DAS_15  DAS_18  DAS_21  DAS_24  DAS_27

Let's try one CT data, *rice_root_daily_growth/DAS_27*.

    $ python . -i rice_root_daily_growth/DAS_27

If your PC has enough memory space, alternatively the following command well works.

    $ python . -i rice_root_daily_growth/DAS_27 -a

Then, one directory and three image files will appear.

    $ ls rice_root_daily_growth
    DAS_07  DAS_11  DAS_15  DAS_19  DAS_23  DAS_27            DAS_27_RSA_2.png
    DAS_08  DAS_12  DAS_16  DAS_20  DAS_24  DAS_27_RSA        description.docx
    DAS_09  DAS_13  DAS_17  DAS_21  DAS_25  DAS_27_RSA_0.png
    DAS_10  DAS_14  DAS_18  DAS_22  DAS_26  DAS_27_RSA_1.png

The processed image files are saved in *rice_root_daily_growth/DAS_27_RSA*. Parallel projections along three axis are *DAS_27_RSA_0.png*, *DAS_27_RSA_1.png*, and *DAS_27_RSA_2.png*.

<img src="figures/DAS_27_RSA_0.png" width=30% height=30% title="DAS_27_RSA_0.png"> <img src="figures/DAS_27_RSA_1.png" width=30% height=30% title="DAS_27_RSA_1.png"> <img src="figures/DAS_27_RSA_2.png" width=30% height=30% title="DAS_27_RSA_2.png">

## Citation

Please cite the following article:

Shota Teramoto et al. (2020) [High-throughput three-dimensional visualization of root system architecture of rice using X-ray computed tomography.](https://doi.org/10.1186/s13007-020-00612-6)  Plant Methods. 16, Article number: 66

## Copyright

National Agriculture and Food Research Organization (2020)

## Project homepage
https://rootomics.dna.affrc.go.jp/en/

## version policy

Version information consists of major and minor versions (major.minor). When the major version increases by one, it is no longer compatible with the original version.When the minor version invreases by one, compatibility will be maintained. Revisions that do not affect functionality, such as bug fixes and design changes, will not affect the version number.

## Update history

* version 1.0 (Dec 10th, 2020)
  * Initial version uploaded
  * README.md updated (Jan 29th, 2021)