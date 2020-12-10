# RSAvis3D: An robust and rapid image processing for root segmentation from X-ray CT images

RSAvis3D is the Python program for root segmentation from X-ray CT images. RSAvis3D uses a 3-D median filter and edge detection algorithm to isolate root segments. 

## Citation

If you use this code or modified ones, please cite our work: Shota Teramoto et al. (2020) [High-throughput three-dimensional visualization of root system architecture of rice using X-ray computed tomography.](https://doi.org/10.1186/s13007-020-00612-6) 

## Installation
    $ git clone https://github.com/st707311g/RSAvis3D.git

RSAvis3D requires Python (version > 3.6). Check it by the following command:

    $ python --version

After confirmation, install required modules:

    $ pip install --upgrade pip
    $ pip install -r RSAvis3D/requirements.txt

pyenv virtualenv 3.6.4 mypyenv