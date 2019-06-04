import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pims
import re
import json
from scipy.interpolate import interp1d
from scipy.signal import correlate2d as c2d
from scipy import interp
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import exposure,img_as_ubyte
from moviepy.editor import VideoClip
from moviepy.editor import ImageSequenceClip
from skimage import color
from skimage.measure import profile_line
import datetime
import time
import argparse
import os
from scipy import fftpack
from skimage.filters import gaussian as gauss
import matplotlib.cm as cm
import matplotlib.colors as mplcol
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from mpl_toolkits.mplot3d import axes3d 
from skimage import measure
from scipy import ndimage as ndimage
from skimage import color
from scipy import signal as sig
import glob as glob
import sys
import skimage
from scipy.interpolate import UnivariateSpline as spl
def scher(image):
    return (np.sin(2.*image)**2.)
def orderP(image):
    return np.exp(1j*image)

def norm(image,mask):
    #normalize uneven illumation by subtracting a mask
    im = image/mask
    return im-im.min()
def blowUp(image):
    #take 100x100 chunk from middle (assume it is 300x300)
    middle = image
    factor = 3
    outIm = np.zeros((factor*image.shape[0],factor*image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            outIm[i*factor:i*factor+factor,j*factor:j*factor+factor]=middle[i,j]
    return outIm

def pad(image):
    #takes square image and pads it out to 2**p for the fft
    size = image.shape[0]
    smallSize = 2*size-1 
    powSize = math.ceil(np.log(smallSize)/np.log(2))
    #print(powSize)
    finSize = 2**powSize
    #print(finSize)
    tIm = np.zeros((finSize,finSize))
    tIm[0:size,0:size] = image
    return tIm
    
     
def deBlow(image):
    factor = 3
    outIm = image[::factor,::factor]

def crop(image,startR,startC,square):
    image = color.rgb2grey(image)
    return image[startR:startR+square,startC:startC+square]

    
    #because of fucking banding noise, we can't just take the regular azimuthal average, because there are zero frequency lines that are drowning out the signal. So, my first try 
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    #import ipdb; ipdb.set_trace()
    # Calculate the indices from the image
    image = color.rgb2gray(image)
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = np.append(csim[0],tbin / nr) #readd first value in

    return radial_prof

def sim_acor_gen(imshape):
    #generate the maps for the simple autocorrelation function
    M = [[i,j] for i in np.arange(imshape[0]) for j in np.arange(imshape[1])]

    # Calculate the indices from the image
    y, x = np.indices(imshape)
    for i,s in enumerate(M):
        r = np.hypot(x- s[0], y - s[1])

        # Get sorted radii
        ind = np.argsort(r.flat)[:6]
        r_sorted = r.flat[ind]

        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)

        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
        rind = np.where(deltar)[0]       # location of changed radius
        nr = rind[1:] - rind[:-1]        # number of radius bin
        M[i] = [ind,rind,nr]
        

    return M


    
def psd(image):
    l,h = image.shape
    win =  np.outer(sig.windows.hann(l),sig.windows.hann(h))
    xp = (image-np.average(image))/np.std(image)
    return abs(fftpack.fftshift(abs(fftpack.fft2(fftpack.ifftshift(xp)))**2))/(image.shape[0]*image.shape[1])


def proLine(image,theta,length,center=None):

    cr, cc = np.indices(image.shape)
    if not center:
        center = np.array([(cr.max()-cr.min())/2.0, (cc.max()-cc.min())/2.0])
    (endrow,endcol) = (length*np.sin(theta)+center[0],length*np.cos(theta)+center[1])
    line = profile_line(image, center,(endrow,endcol))
    return line



def autocorr(image):
    #we cannot pad by zeros, as this causes a linear decrease in the
    #autocorrelation function that will swamp the signal, and be proportional
    #to the total width of the 'box'. So, we can just going to do the
    #regular circular convolution (no fancy boundaries, just fft's), and
    #we will have to hope for the best.
    #image = image.astype(float) #convert to float to avoid overflows
    oLength = image.shape[0] #assume cropped to be square
    #image = pad(image)
    #nLength = image.shape[0]
    l,h = image.shape
    win =  np.outer(sig.windows.hann(l),sig.windows.hann(h))
    winIm = win*image
    xp = (winIm-winIm.mean())
    xp = (image-image.mean())
    xxp =xp/np.sqrt((xp**2).sum())
    #return sig.correlate2d(win*image,win*image)
    return fftpack.fftshift((fftpack.ifft2(abs(fftpack.fft2(xp))**2)))/( (xp**2).sum()) #normalize to one
    #return np.real(fftpack.fftshift(fftpack.ifft2(np.absolute(fftpack.fft2(xp))**2)))

def gFfft(image):
    return fftpack.ifftshift(fftpack.fft2(image))

font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",25)
def make_frame(time):
    i = int(time*fps)
    autoim =autocorrImages[i]
    norm = mplcol.Normalize(vmin=minA,vmax=maxA)

    #ts = img.metadata['t_s']
    ts = i/24.
    frame = i
    autoim= Image.fromarray(img_as_ubyte(cm.viridis(norm(autoim)))) #get rid of alpha channel, it confuses moviepy, and multiply by 255 as that is how moviepy likes its colors for some reason
    draw= ImageDraw.Draw(autoim)
    draw.text((0,0),"time: "+str(datetime.timedelta(seconds=float(ts))),font=font,fill=(255,255,255,255))
    draw.text((0,400),"frame: "+str(frame),font=font,fill=(255,255,255,255))
    return np.asarray(autoim)[:,:,:3]
def make_frame_old(time):
    i = int(time*fps)
    im =images[i]

    #ts = img.metadata['t_s']
    ts = i/24.
    frame = i
    autoim= Image.fromarray(img_as_ubyte(exposure.rescale_intensity(im,in_range=(im.min(),im.max() ) ))) #get rid of alpha channel, it confuses moviepy, and multiply by 255 as that is how moviepy likes its colors for some reason
    draw= ImageDraw.Draw(autoim)
    draw.text((0,0),"time: "+str(datetime.timedelta(seconds=float(ts))),font=font,fill=(255))
    draw.text((0,400),"frame: "+str(frame),font=font,fill=(255))
    return color.grey2rgb(np.asarray(autoim))[:,:,:3]

def boxZero(image,sideLength):
    center = [i//2 for i in image.shape]
    image[center[0]-sideLength:center[0]+sideLength,center[1]-sideLength:center[1]+sideLength]=0
    return image
    

#Now do the radial averaging and the gaussian fits
def gausFit(x,a,sigma):
    return a*exp(-x**2/(2*sigma**2))
def lorentzFit(x,a,sigma):
    return a*(1/(1+(2*x/sigma)**2))
def expFit(x,a,xi):
    return a*exp(-x/xi)

dataSize=300
cropSize = 600
imageShape = [800,1280]
if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videofilenamepath", help='name of the file to be encoded')
    args= parser.parse_args()

    (vfilepath,vfilename)=os.path.split(args.videofilenamepath)

    image = skimage.img_as_float(plt.imread(args.videofilenamepath))
    #create noise template

    noise = fftpack.ifft2(fftpack.ifftshift(boxZero(fftpack.fftshift(fftpack.fft2(image)),100)))
    plt.imshow(np.real(noise))


