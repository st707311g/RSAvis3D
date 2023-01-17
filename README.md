# RSAvis3D: An robust and rapid image processing for root segmentation from X-ray CT images


![python](https://img.shields.io/badge/Python-3.8.12-lightgreen)
![developed_by](https://img.shields.io/badge/developed%20by-Shota_Teramoto-lightgreen)
![version](https://img.shields.io/badge/version-1.4-lightgreen)
![last_updated](https://img.shields.io/badge/last_update-January_17,_2023-lightgreen)

![top image](figures/top_image.jpg)

RSAvis3D is the Python program for root segmentation from X-ray CT images. RSAvis3D uses a 3D median filter and edge detection algorithm to isolate root segments.

## system requirements

This software is confirmed to work with Python 3.8.12 on Ubuntu 20.04. I recommend creating a virtual environment for python 3.8.12 with `virtualenv`.

## installation

Run the following commands:

```
git clone https://github.com/st707311g/RSAvis3D.git
cd RSAvis3D
```

The following command will install the required packages.

```
pip install -U pip
pip install -r requirements.txt
```

This software can reduce processing time by using `CuPy`. Installation depends on the version of `CUDA Toolkit`. Please build the environment according to your own version of `CUDA Toolkit`. For example, if the version of `CUDA Toolkit` is 11.4, install cupy with the following command.

```
pip install cupy-cuda114
```

Please check if CuPy is available by using the following command.
```
python is_cupy_available.py
```

## how to run

```
usage: . [-h] [-s SRC] [-d DST] [-b BLOCK_SIZE] [-a] [-m MEDIAN_KERNEL_SIZE] [-e EDGE_SIZE] [-c CYLINDER_RADIUS] [-f {png,tif,jpg}]
         [-i INTENSITY_FACTOR] [--mm_resolution MM_RESOLUTION] [--depth DEPTH] [--save_projection]

optional arguments:
  -h, --help            show this help message and exit
  -s SRC, --src SRC     source directory.
  -d DST, --dst DST     destination directory.
  -b BLOCK_SIZE, --block_size BLOCK_SIZE
                        divided volume size (>= 64)
  -a, --all_at_once     all-at-onec processing
  -m MEDIAN_KERNEL_SIZE, --median_kernel_size MEDIAN_KERNEL_SIZE
                        median kernel size (>= 1)
  -e EDGE_SIZE, --edge_size EDGE_SIZE
                        blur kernel size for edge detection (>= 1)
  -c CYLINDER_RADIUS, --cylinder_radius CYLINDER_RADIUS
                        cylinder mask radius (>= 64). If 0, masking process will be skipped.
  -f {png,tif,jpg}, --format {png,tif,jpg}
                        file format type
  -i INTENSITY_FACTOR, --intensity_factor INTENSITY_FACTOR
                        intensity factor (>0)
  --mm_resolution MM_RESOLUTION
                        spatial resolution [mm].
  --depth DEPTH         depth of the maximum level to be explored. Defaults to unlimited.
  --save_projection     save projection images.

```

Basic usage of RSAvis3D is

    $ python . -s SOURCE

Parameters of RSAvis3D are modifiable by `-b`, `-m`, `-e`, and `-c` commands. If omitted, the parameters considered conditional on [the paper](https://doi.org/10.1186/s13007-020-00612-6) will be used.

The intensity factor is proportional to the signal intensity of the output images. If the image is saturated, reduce the intensity factor. The default is 10.

## demonstration

Download the demo data (1.60G), which is a time-series X-ray CT data of an upland rice cultivar from 7 to 27 days after sowing ([Teramoto et al. 2020 Plant Methods](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-020-00612-6)). The intensity of this data is normalized by `normalize_intensity_inspeXio_SMX-225CT_FPD_HR.py`. CT slice images were converted into 8 bit jpeg files, signal intensity of the air is around 0 and signal intensity of the soil is around 128.

```
wget https://rootomics.dna.affrc.go.jp/data/rice_root_daily_growth_intensity_normalized.zip
unzip rice_root_daily_growth_intensity_normalized.zip
rm rice_root_daily_growth_intensity_normalized.zip
```

There are 21 directories in *rice_root_daily_growth_intensity_normalized*.

```
ls rice_root_daily_growth_intensity_normalized
DAS07  DAS09  DAS11  DAS13  DAS15  DAS17  DAS19  DAS21  DAS23  DAS25  DAS27
DAS08  DAS10  DAS12  DAS14  DAS16  DAS18  DAS20  DAS22  DAS24  DAS26
```

If you want to obtain RSA segments, please specify the target directory.
```
python . -s rice_root_daily_growth_intensity_normalized --save_projection
```

Alternatively, if your PC has enough memory space, the following command well works.
```
python . -s rice_root_daily_growth_intensity_normalized -a --save_projection
```

Processed files are saved in the *rice_root_daily_growth_intensity_normalized_rsavis3d* directory.

<img src="figures/.projection0.gif" width=60% height=60% title=".projection0.gif"> <img src="figures/.projection1.gif" width=60% height=60% title=".projection1.gif"><img src="figures/.projection2.gif" width=60% height=60% title=".projection2.gif">

## version policy

Version information consists of major and minor versions (major.minor). When the major version increases by one, it is no longer compatible with the original version.When the minor version invreases by one, compatibility will be maintained. Revisions that do not affect functionality, such as bug fixes and design changes, will not affect the version number.

## citation

Please cite the following article:

Shota Teramoto et al. (2020) [High-throughput three-dimensional visualization of root system architecture of rice using X-ray computed tomography.](https://doi.org/10.1186/s13007-020-00612-6)  Plant Methods. 16, Article number: 66


## project homepage
https://rootomics.dna.affrc.go.jp/en/

## update history

* version 1.0 (Dec 10, 2020)
  * initial version uploaded
  * README.md updated (Jan 29th, 2021)

* version 1.1 (May 25, 2022)
  * GPU support
  * license change (NARO NON-COMMERCIAL LICENSE AGREEMENT Version 1.0)
  * rewrote the code for python 3.8.12
  * fix: typo (June 1, 2022)
  * fix: warning messages ignored (June 9, 2022)
  * add: README.md - an explanation of the intensity factor (June 9, 2022)

* version 1.2 (June 15, 2022)
  * added option of no mask processing

* version 1.3 (January 17, 2023)
  * rewrote the code of `normalize_intensity_inspeXio_SMX-225CT_FPD_HR.py`

* version 1.4 (January 17, 2023)
  * loading, image processing, and saving are parallelized.