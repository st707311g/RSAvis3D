# -*- coding: utf-8 -*-

"""
@author: Shota Teramoto
Institute of Crop Science, National Agriculture and Food Research Organization, Tsukuba, Ibaraki 305-8518, Japan.

This source file is an implementation example of the image processing in the following article:
'High-throughput three-dimensional visualization of root system architecture of rice using X-ray computed tomography'
Shota Teramoto et al. (2020) Plant Methods

Released under the MIT license.
see https://opensource.org/licenses/MIT
"""

import os, tqdm, glob
import numpy as np
from skimage import io, exposure, util, filters
from scipy import ndimage
import cv2
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
import functools
import argparse

def tqdm_multiprocessing(fun, l):
    """
    A wrapper function for a multi-processing coding with the tqdm.
    The number of process is the number of cpu count.
    
    Basic Usage::
        result = tqdm_multiprocessing(funciton, list)
    """
    
    with Pool(multiprocessing.cpu_count()) as p:
        l = list(tqdm.tqdm(p.imap(fun, l), total=len(l)))
    return(l)
    
def normalizeIntensity(stack, peak_air, peak_soil):
    """
    A wrapper function for a normalization of CT images.
    
    @stack
    (numpy array) a Numpy array which will be normalized. 
    
    @peak_air
    (integer) The peak value of histogram coresponding to the air region.
    
    @peak_soil
    (integer) The peak value of histogram coresponding to the soil region.
    """
    stack[stack<peak_air] = 0
    stack[stack>=peak_air] = stack[stack>=peak_air]-peak_air
    diff = peak_soil-peak_air
    return (stack/diff*1024).astype(np.uint16)
    
class CT_processor:
    """
    A class used for a fully automated image processing of the CT image data.
    
    Basic Usage::
        CT = CT_processor()
        CT.loadStack(input_dir)
        CT.calculateNormalizationParam()
        CT.filterNormalization(all_at_once = True)
        CT.makeSylinderMask(300)
        CT.trim_with_mask(padding = 10)
        CT.filterMedian3d(median_kernel_size=7, all_at_once=True)
        CT.invert()
        CT.subtractBackground(kernel_size=21)
        CT.applyMask(padding = 0)
        CT.saveStack(output_dir)
        
    **Notes:** 
    If the histogram of the air or the soil region has no peaks, 
    CT.calculateNormalizationParam() and CT.filterNormalization(all_at_once = True) do not make sense.
    Please skip them or implement other normalization functions.
    
    """
    
    def __init__(self, indir=None):
        """
        Initialize the instance.
        
        @indir
        (string) The directry path containing CT iamages.
        """
        
        self.stack = None
        self.mask = None
        self.shape = None
        self.outdir = None
        self.peak_air = None
        self.peak_soil = None
        self.diff = None
        
        if indir is not None:
            self.loadStack(indir)
        else:
            self.indir = None
        
    def loadStack(self, indir):
        """
        Load the CT images.
        
        @indir
        (string) The directry path containing the CT iamages.
        """
        
        self.indir = indir
        files = glob.glob(os.path.join(self.indir, '*.*'))
        files = [f for f in files if f.endswith('.cb')]
        #// '.cb' is the extension of the CT iamges generated with Shimazdu X-ray CT system
        
        if len(files) == 0:
            raise Exception('Stack loading failed.')
            
        files.sort()
        print('Stack loading: {}'.format(self.indir))
        
        self.stack = [io.imread(f) for f in tqdm.tqdm(files)]
        self.stack = np.asarray(self.stack, dtype=np.uint16)
        #// '.cb' files is the 16-bit grayscale images
        
        self.shape = self.stack.shape
        return

    def checkStack(self):
        """
        Check whether the CT images was loaded.
        """
        
        if self.stack is None:
            raise Exception('The CT images not loaded.')
            
    def checkMask(self):
        """
        Check whether the CT mask was computed.
        """
        
        if self.mask is None:
            raise Exception('The mask not computed.')
    
    def saveStack(self, outdir):
        """
        Save the processed images.
        
        @outdir
        (string) The directry path where self.stack will be saved.
        """
        
        self.checkStack()
        self.outdir = outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
            
        print('Stack saving: {}'.format(self.outdir))
        
        for i, img in enumerate(tqdm.tqdm(self.stack)):
            img = exposure.rescale_intensity(img, in_range=(0,255), out_range=(0,255)).astype(np.uint8)
            out = os.path.join(self.outdir, 'img%s.png' % str(i).zfill(4))
            io.imsave(out, img)
            
        return
    
    def calculateNormalizationParam(self):
        """
        Analyze the CT image histogram.
        The middle 32 slices were used.
        The peak value corresponding to the air and soil regions are saved.
        """
        
        self.checkStack()
        nstack = len(self.stack)
        stack = self.stack[nstack//2-16:nstack//2+16]
        
        hist_y, hist_x = exposure.histogram(stack[stack>0])
        thr = filters.threshold_otsu(stack[stack>0])
        
        self.peak_air = np.argmax(hist_y[hist_x<thr])+hist_x[0]
        self.peak_soil = np.argmax(hist_y[hist_x>thr])+(thr-hist_x[0])+hist_x[0]
        self.diff = self.peak_soil-self.peak_air
        self.hist_y = hist_y
        self.hist_x = hist_x
        
        print('Air: {}, Soil: {}, diff: {}'.format(self.peak_air, self.peak_soil, self.diff))
    
    def makeSylinderMask(self, radius):
        """
        Make a silinder-shaped mask for triming. 
        The center of the mask is the center of the CT slices. 
        
        @radius
        (integer) Radius of the mask. The unit is pixel.
        """
        
        self.checkStack()
        self.radius = radius

        x, y = np.indices((self.shape[1], self.shape[2]))
        self.mask = (x - self.shape[1]/2)**2 + (y - self.shape[2]/2)**2 < self.radius**2
        self.mask = np.repeat([self.mask], self.shape[0], axis=0)
        
        return
        
    def trim_with_mask(self, padding = 0):     
        """
        Trim the CT images with the mask.
        
        @padding
        (integer) A padding pixel size. 
        The mask is expanded with the value and is used for trimming.
        """
        
        self.checkMask()
        def getRange(ary, axis):
            v = np.max(ary, axis=tuple([i for i in range(3) if i != axis]))
            index = np.where(v)
            return (max(np.min(index)-padding, 0), min(np.max(index)+padding+1, ary.shape[axis]))
        
        r = [getRange(self.mask, axis = i) for i in range(3)]
        
        self.stack = self.stack[r[0][0]: r[0][1], r[1][0]: r[1][1], r[2][0]: r[2][1]]
        self.mask = self.mask[r[0][0]: r[0][1], r[1][0]: r[1][1], r[2][0]: r[2][1]]
        
        self.shape = self.stack.shape
    
        return

    def applyMask(self, padding = 0):
        """
        Apply the CT images with the mask.
        
        @padding
        (integer) A padding pixel size. 
        The mask is expanded with the value and is used for masking.
        """
        
        self.checkMask()
        self.trim_with_mask(padding=padding)
        self.stack *= self.mask
        
        return
    
    def block_separator(self, overlapping=0, block_size=64, all_at_once = False):
        """
        Generator of the CT images. The CT volume is divided into smaller ones.
        
        @overlapping
        (integer) A overlapping pixel size. 
        Be sure that the value should be enough large if you apply filters later.
        
        @block_size
        (integer) A block size determining the divided volume size. 
        If the overlapping is zero, CT volume is divided into smaller ones with a side of block size. 

        @all_at_once
        (bool) A flag determining all-at-onec processing. 
        If the all_at_once is True, this function returns an iterator yielding the list containing all divided volumes.
        If False, this function returns an iterator yielding the lists containing portions of divided volumes.
        """
        
        self.checkStack()
        buf = np.pad(self.stack, overlapping, mode = 'symmetric')
        
        blocks = []
        indexes = []
        for zi in range(0, self.shape[0], block_size):
            for yi in range(0, self.shape[1], block_size):
                for xi in range(0, self.shape[2], block_size):
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
    
    def updateStack(self, blocks, indexes, overlapping=0, block_size=64):
        """
        Update the self.stack with the divided volumes.

        @blocks
        (list) The divided volumes. 
        
        @indexes
        (list) The list containing index numbers corresponding to the blocks. 

        @overlapping
        (integer) A overlapping pixel size. 
        Should be the same value used in the block_separator function.
        
        @block_size
        (integer) A block size determining the divided volume size. 
        Should be the same value used in the block_separator function.
        """
        
        self.checkStack()
        for block, index in zip(blocks, indexes):
            
            self.stack[index[0]:index[0]+block_size, 
                       index[1]:index[1]+block_size,
                       index[2]:index[2]+block_size] = block[overlapping:-overlapping, overlapping:-overlapping, overlapping:-overlapping]
            
        return
    
    def filterNormalization(self, block_size=64, all_at_once = False):
        """
        Normalize signal intensity. 
        
        @block_size
        (integer) A block size determining the divided volume size. 
        This argument is passed to the block_separator function.
        
        @all_at_once
        (bool) A flag determining all-at-onec processing. 
        This argument is passed to the block_separator function.
        """
        
        print("Intensity normalization")
        if self.peak_air == None:
            raise Exception('Call the calculateNormalizationParam in ahead.')
        
        maxid = [self.peak_air, self.peak_soil]
        maxid = [i-self.hist_x[0] for i in maxid]
        plt.figure()
        plt.plot(self.hist_x, self.hist_y)
        plt.plot(self.hist_x[maxid], self.hist_y[maxid],'ro')
        plt.xlabel('intensity')
        plt.ylabel('count')
        plt.pause(.01)
        
        i_block = self.block_separator(overlapping = 1, block_size = block_size, all_at_once = all_at_once)
        
        for blocks, indexes in i_block:
            blocks = tqdm_multiprocessing(functools.partial(normalizeIntensity, peak_air=self.peak_air, peak_soil=self.peak_soil), blocks)
            self.updateStack(blocks, indexes, overlapping = 1, block_size = block_size)
            
        return

    def filterMedian3d(self, block_size=64,  median_kernel_size = 1, all_at_once = False):
        """
        Apply Median3D filter.
        
        @block_size
        (integer) A block size determining the divided volume size. 
        The CT volume is divided into smaller ones with a side of block size, which are subjected to Median3D filter. 

        @median_kernel_size
        (integer) A kernel size used for median3D filter.
        
        @all_at_once
        (bool) A flag determining all-at-onec processing. 
        This argument is passed to the block_separator function.
        False is effective under low memory machine.
        """

        self.checkStack()
        print("Median 3D filer")
        
        overlapping = max((median_kernel_size+1)//2, 1)
        
        i_block = self.block_separator(overlapping=overlapping, block_size = block_size, all_at_once = all_at_once)
        
        for blocks, indexes in i_block:
            blocks = tqdm_multiprocessing(functools.partial(ndimage.median_filter, size=median_kernel_size), blocks)
            self.updateStack(blocks, indexes, overlapping=overlapping, block_size = block_size)
            
        return
    
    #// inverting
    def invert(self):
        """
        Invert self.stack.
        """
        
        self.checkStack()
        self.stack= util.invert(self.stack)
        
        return
    
    def subtractBackground(self ,kernel_size = 21):
        """
        Subtract background, or detect edges of the self.stack. 
        
        @kernel_size
        (integer) The CT slices is blurred with this argument.
        The difference between the original and blurred slices were calculated.
        """
        
        self.checkStack()
        def func(img):
            blur_ = cv2.blur(img, ksize= (kernel_size,kernel_size))
            ret = img
            ret[img<blur_] = 0
            ret[img>=blur_] = (img-blur_)[img>=blur_]
                
            return (ret)
        
        self.stack = [func(img) for img in self.stack]
        self.stack = np.asarray(self.stack)
        return

if __name__ == '__main__':
    """
    An implementation example.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, default=r'C:\data\test_indir')
    parser.add_argument('-o', '--outdir', type=str, default=r'C:\data\test_outdir')
    args = parser.parse_args()
    
    CT = CT_processor()
    CT.loadStack(indir=args.indir)
    CT.calculateNormalizationParam()
    CT.filterNormalization(all_at_once = True)
    CT.makeSylinderMask(300)
    CT.trim_with_mask(padding = 10)
    CT.filterMedian3d(median_kernel_size=7, all_at_once=True)
    CT.invert()
    CT.subtractBackground(kernel_size=21)
    CT.applyMask(padding = 0)
    CT.saveStack(args.outdir)