from typing import Union
from glob import glob
from tqdm import tqdm
from skimage import io, exposure, filters, util
import numpy as np
from functools import partial
from scipy import ndimage
import os
import cv2

import multiprocessing
from multiprocessing import Pool

def tqdm_multiprocessing(fun, l):
    with Pool(multiprocessing.cpu_count()) as p:
        l = list(tqdm(p.imap(fun, l), total=len(l)))
    return(l)

class RSAvis3D(object):
    def __init__(self):
        super().__init__()
        self.__indir: Union[str, None] = None
        self.__volume: Union[np.ndarray, None] = None
        self.__mask: Union[np.ndarray, None] = None

    def image_extensions(self):
        #// '.cb' is the extension of the CT iamges generated with Shimazdu X-ray CT system
        return ('.cb', '.png', '.tiff', '.tif')

    def np_volume(self) -> np.ndarray:
        assert self.__volume is not None
        return self.__volume

    def np_mask(self) -> np.ndarray:
        assert self.__mask is not None
        return self.__mask

    def has_volume(self):
        return self.__volume is not None

    def load_from(self, indir: str):
        print(f'Image loading:{indir}')
        self.__indir = indir
        files = sorted(glob(os.path.join(self.__indir, '*.*')))
        files = [f for f in files if f.lower().endswith(self.image_extensions())]
        
        if len(files) < 64:
            raise Exception(f'[error] Number of images should be >= 64: {self.__indir}')
        
        try:
            self.__volume = [io.imread(f) for f in tqdm(files)]
            self.__volume = np.asarray(self.__volume, dtype=np.uint16)
        except:
            raise Exception(f'[error] Cannot import the volume data as a NumPy array: {self.__indir}')

    def block_separator(self, overlapping=0, block_size=64, all_at_once = False):
        np_volume = self.np_volume()

        buf = np.pad(np_volume, overlapping, mode = 'symmetric')
        
        blocks = []
        indexes = []
        for zi in range(0, np_volume.shape[0], block_size):
            for yi in range(0, np_volume.shape[1], block_size):
                for xi in range(0, np_volume.shape[2], block_size):
                    blocks.append(buf[zi:zi+block_size+overlapping*2, 
                                      yi:yi+block_size+overlapping*2,
                                      xi:xi+block_size+overlapping*2])
                    indexes.append([zi, yi, xi])
                    
            if not all_at_once:
                yield (blocks, indexes)
                blocks = []
                indexes = []

        if blocks:            
            yield (blocks, indexes)
        
        return

    def __update_volume(self, blocks, indexes, overlapping=0, block_size=64):
        np_volume = self.np_volume()
        for block, index in zip(blocks, indexes):
            block = block[overlapping:-overlapping, overlapping:-overlapping, overlapping:-overlapping]
            s = [slice(index[i], index[i]+block_size) for i in range(3)]
            np_volume[s[0], s[1], s[2]] = block

        return

    def normalize_intensity(self, block_size=64, all_at_once = False):
        np_volume = self.np_volume()

        z_size = len(np_volume)
        stack = np_volume[z_size//2-16:z_size//2+16]
        
        hist_y, hist_x = exposure.histogram(stack[stack>0])
        thr = filters.threshold_otsu(stack[stack>0])
        
        peak_air = np.argmax(hist_y[hist_x<thr])+hist_x[0]
        peak_soil = np.argmax(hist_y[hist_x>thr])+(thr-hist_x[0])+hist_x[0]
        diff = peak_soil-peak_air
        
        print(f'Normalize intensity: Air: {peak_air}, Soil: {peak_soil}, diff: {diff}')
        
        maxid = [peak_air, peak_soil]
        maxid = [i-hist_x[0] for i in maxid]

        def fun(ndarry, peak_air, peak_soil):
            ret = np.asarray(ndarry, dtype=np.float32)
            ret = (ret - peak_air).clip(0)/(peak_soil-peak_air)*1024
            return np.asarray(ret, dtype=np.uint16)

        i_block = self.block_separator(overlapping = 1, block_size = block_size, all_at_once = all_at_once)
        
        for blocks, indexes in i_block:
            blocks = [fun(b, peak_air, peak_soil) for b in tqdm(blocks)]
            self.__update_volume(blocks, indexes, overlapping = 1, block_size = block_size)

    def make_sylinder_mask(self, radius):
        np_volume = self.np_volume()

        x, y = np.indices((np_volume.shape[1], np_volume.shape[2]))
        self.__mask = (x - np_volume.shape[1]/2)**2 + (y - np_volume.shape[2]/2)**2 < radius**2
        self.__mask = np.repeat([self.__mask], np_volume.shape[0], axis=0)

    def trim_with_mask(self, padding = 0):
        np_volume = self.np_volume()
        np_mask = self.np_mask()

        slices = []
        for axis in range(3):
            v = np.max(np_mask, axis=tuple([i for i in range(3) if i != axis]))
            index = np.where(v)
            slices.append(slice(max(np.min(index)-padding, 0), min(np.max(index)+padding+1, np_mask.shape[axis])))

        self.__volume = np_volume[slices[0], slices[1], slices[2]]
        self.__mask = np_mask[slices[0], slices[1], slices[2]]

    def apply_mask(self, padding = 0):
        self.trim_with_mask(padding=padding)
        self.__volume *= self.np_mask()

    def apply_median3d(self, block_size=64,  median_kernel_size = 1, all_at_once = False):
        print(f'Median3D filter: kernel size={median_kernel_size}')
        assert self.has_volume()
        
        overlapping = max((median_kernel_size+1)//2, 1)
        
        i_block = self.block_separator(overlapping=overlapping, block_size = block_size, all_at_once = all_at_once)
        
        for blocks, indexes in i_block:
            blocks = tqdm_multiprocessing(partial(ndimage.median_filter, size=median_kernel_size), blocks)
            self.__update_volume(blocks, indexes, overlapping=overlapping, block_size = block_size)

    def invert(self):
        np_volume = self.np_volume()
        self.__volume = util.invert(np_volume)

    def apply_edge_detection(self ,kernel_size = 21):
        print(f'Edge detection filter: kernel size={kernel_size}')
        np_volume = self.np_volume()

        def detect_edge(img):
            blur_ = cv2.blur(img, ksize= (kernel_size,kernel_size))
            ret = img
            ret[img<blur_] = 0
            ret[img>=blur_] = (img-blur_)[img>=blur_]

            return ret
        
        self.__volume = np.asarray([detect_edge(img) for img in tqdm(np_volume)])

    def save_volume(self, outdir: str, format="jpg"):
        np_volume = self.np_volume()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
            
        print(f'Volume saving: {outdir}')
        
        for i, img in enumerate(tqdm(np_volume)):
            img = exposure.rescale_intensity(img, in_range=(0,255), out_range=(0,255)).astype(np.uint8)
            out_fname = os.path.join(outdir, f'img{str(i).zfill(4)}.{format}')
            io.imsave(out_fname, img)

        for i in range(3):
            img = np_volume.max(axis=i)
            img = exposure.rescale_intensity(img, in_range=(0,255), out_range=(0,255)).astype(np.uint8)
            io.imsave(outdir+f'_{i}.png', img)