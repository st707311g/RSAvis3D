# RSAvis3D: A robust and rapid image processing for root segmentation from X-ray CT images


![python](https://img.shields.io/badge/Python-3.10.4-lightgreen)
![developed_by](https://img.shields.io/badge/developed%20by-Shota_Teramoto-lightgreen)
![version](https://img.shields.io/badge/version-1.6-lightgreen)
![last_updated](https://img.shields.io/badge/last_update-March_6,_2024-lightgreen)

![top image](figures/top_image.jpg)

RSAvis3D is a Python program for root segmentation from X-ray computed tomography (CT) images. RSAvis3D uses a 3D median filter and an edge detection algorithm to isolate root segments.

## system requirements

This software was confirmed to work with Python 3.10.4, on Ubuntu 20.04.6 LTS. I recommend creating a virtual environment for python 3.10.4 with `virtualenv`.

## installation

The following commands were run:

```
$ git clone https://github.com/st707311g/RSAvis3D.git
$ cd RSAvis3D
```

The following commands install the required packages:

```
$ pip install -U pip
$ pip install -r requirements.txt
```

This software can reduce processing time by using `CuPy`. Installation depends on the version of `CUDA Toolkit`. Please build the environment according to your own version of `CUDA Toolkit`. 

## how to run

```
usage: run_rsavis3d.py [-h] -s SRC [-b BLOCK_SIZE] [-m MEDIAN_KERNEL_SIZE] [-c CYLINDER_RADIUS] [-e EDGE_SIZE] [-f {png,tif,jpg}] [-i INTENSITY_FACTOR] [-g [GPU ...]] [--series_src SERIES_SRC]
                       [--series_dst SERIES_DST] [--archive] [--debug]

RSAvis3D (Version 1.6) Author: Shota Teramoto. Copyright (C) 2020 National Agriculture and Food Research Organization. All rights reserved.

options:
  -h, --help            show this help message and exit
  -s SRC, --src SRC     source directory
  -b BLOCK_SIZE, --block_size BLOCK_SIZE
                        divided volume size (>= 64)
  -m MEDIAN_KERNEL_SIZE, --median_kernel_size MEDIAN_KERNEL_SIZE
                        median kernel size (>= 1)
  -c CYLINDER_RADIUS, --cylinder_radius CYLINDER_RADIUS
                        cylinder mask radius (>= 1). If None, masking process will be skipped.
  -e EDGE_SIZE, --edge_size EDGE_SIZE
                        blur kernel size for edge detection (>= 1)
  -f {png,tif,jpg}, --format {png,tif,jpg}
                        image format type
  -i INTENSITY_FACTOR, --intensity_factor INTENSITY_FACTOR
                        intensity factor (>0), image intensity will be multiplied by this factor
  -g [GPU ...], --gpu [GPU ...]
                        GPU device id(s) used for computing
  --series_src SERIES_SRC
                        series name in the RSA dataset, which will be processed
  --series_dst SERIES_DST
                        series name in the RSA dataset, which will be created
  --archive             processed images will be saved as zip archive
  --debug               debug mode
```

Basic usage of RSAvis3D is

    $ python run_rsavis3d.py --src SOURCE_DIRECTORY

SOURCE_DIRECTORY should contains RSA datasets. A description of the RSA dataset is available [here](https://github.com/st707311g/public_data/releases/tag/ct_volumes). 

Parameters of RSAvis3D are modifiable by `-b`, `-m`, and `-e` commands. If omitted, the parameters considered conditional on [the paper](https://doi.org/10.1186/s13007-020-00612-6) will be used. The option `-c` makes mask image to trim the volume 

The intensity factor is proportional to the signal intensity of the output images. Please adjust the intensity factor so that the image does not saturate.

## demonstration

Please download the demo data (1.65GB), which are the time-series X-ray CT data of an upland rice cultivar from 7 to 27 days after sowing ([Teramoto et al. 2020 Plant Methods](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-020-00612-6)). The intensities of the data were normalized. CT slice images were converted into 8 bit jpeg files, the signal intensity of the air was approximately 0, and the signal intensity of the soil was approximately 128.

```
$ wget https://github.com/st707311g/public_data/releases/download/ct_volumes/01_rice_daily_growth.zip
$ unzip 01_rice_daily_growth.zip
$ rm 01_rice_daily_growth.zip
```

There are 21 directories in `01_rice_daily_growth/`.

```
$ ls 01_rice_daily_growth/
DAS07  DAS09  DAS11  DAS13  DAS15  DAS17  DAS19  DAS21  DAS23  DAS25  DAS27
DAS08  DAS10  DAS12  DAS14  DAS16  DAS18  DAS20  DAS22  DAS24  DAS26
```

run RSAvis3D. In this case, GPU device 0 will be used.

    $ python run_rsavis3d.py --src 01_rice_daily_growth/ --gpu 0 --archive

The option `--archive` will stores the results as a zip archive.

If you want to crop the image processing results with a smaller diameter than the pot diameter, as in the paper, do the following commands:

    $ python run_rsavis3d.py --src 01_rice_daily_growth/ --gpu 0 -c 300 --archive

The value of 300 is the radius of the pixels. The resolution of these CT images is 0.3 mm/voxel, therefore the diameter is 180 mm.


Processed files are saved in each rsa_dataset directory, with series name as "rsavis3d". 

The projection images could be created by the following command:

    $ python make_projection.py --src 01_rice_daily_growth/ --axis z

This is an example of making projection along z-axis. The animations made with RSAvis3D results are shown below:

<img src="figures/.projection0.gif" width=60% height=60% title=".projection0.gif"> <img src="figures/.projection1.gif" width=60% height=60% title=".projection1.gif"><img src="figures/.projection2.gif" width=60% height=60% title=".projection2.gif">

## citation

Please cite the following article:

Shota Teramoto et al. (2020) [High-throughput three-dimensional visualization of root system architecture of rice using X-ray computed tomography.](https://doi.org/10.1186/s13007-020-00612-6)  Plant Methods. 16, Article number: 66


## update history

* version 1.6 (April 12, 2024)
  * broken links repaird
  * less memory used by not loading volumes all at once