
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import argparse
import os
import skimage
from PIL import Image
from skimage.transform import AffineTransform, warp
import cv2
import glob as glob
import matplotlib.patches as patches
import yaml


def standardize(image):
    image = image.astype(np.float64)
    imgMean = np.mean(image)
    imgSTD = np.std(image)
    image= (image - imgMean)/(6*imgSTD)
    image = image+0.5
    #image = image*255
    image = np.clip(image,0,1)
    return image

class getter:
    def __init__(self,ax,fig):

        self.numClicks = 0
        self.x = [0,0]
        self.y = [0,0]
        self.ax = ax
        self.fig = fig
        self.figsize = fig.get_size_inches()*fig.dpi
        self.rect = patches.Rectangle((1,1),0,0,linewidth=1,edgecolor='r',facecolor='none')
        self.ax.add_patch(self.rect)
        self.fig.canvas.mpl_connect('button_press_event',self.click)
        self.fig.canvas.mpl_connect('motion_notify_event',self.move)
        #print(self.figsize)

    def click(self,event):
        #print(event.xdata,event.ydata)
        self.x[self.numClicks] = event.xdata
        self.y[self.numClicks] = event.ydata
        #print(self.x)
        if self.numClicks > 0:
            plt.close()
        self.numClicks = self.numClicks+1
    def move(self,event):
        if self.numClicks >0:
            mouseX = event.xdata
            mouseY = event.ydata
            if mouseX is not None:
                clickX = self.x[0]
                clickY = self.y[0]

                xsort = [mouseX,clickX]
                ysort = [mouseY,clickY]
                xsort.sort()
                ysort.sort()

                dif = min(xsort[1]-xsort[0],ysort[1]-ysort[0])

                #print(xsort)
                #print(ysort)
                self.rect.remove()
                self.rect = patches.Rectangle((xsort[0],ysort[0]),dif,dif,linewidth=1,edgecolor='r',facecolor='none')
                self.ax.add_patch(self.rect)
                self.fig.canvas.draw()

def boxZero(image,sideLength):
    center = [i//2 for i in image.shape]
    image[center[0]-sideLength:center[0]+sideLength,center[1]-sideLength:center[1]+sideLength]=0
    return image

def extractNoise(image,boxSize):
    imShape = image.shape
    r = imShape[0]
    c = imShape[1]

    if len(imShape) >2:
        image = image[:,:,1]

    imgF = fftpack.fft2(image)
    imgFS = fftpack.fftshift(imgF)
    noiseSF = boxZero(imgFS,boxSize)
    noiseF = fftpack.ifftshift(noiseSF)
    noise = fftpack.ifft2(noiseF)
    noiseR = np.real(noise)
    return noiseR

def skewImage(image,shift):

    transform = AffineTransform(translation=shift)
    shifted = warp(image,transform, mode ='wrap', preserve_range =True)
    shifted = shifted.astype(image.dtype)

    #rows,cols = shifted.shape
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    #rotated = cv2.warpAffine(shifted,M,(cols,rows))

    return rotated

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



def ExtractNoiseLocal():
    with open(configName,'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

def ExtractNoise(cfg):


    filepath = os.path.join(cfg['temp']['rootDir'],cfg['paths']['noiseSamples'])
    doCrop = cfg['noiseExtraction']['crop']
    paths = glob.glob(filepath+'/*.bmp')

    #(vfilepath,vfilename)=os.path.split(args.videofilenamepath)
    iterator = 1
    print(paths)
    for path in paths:

        
        image = skimage.img_as_float(plt.imread(path))
        dims = image.shape
        if cfg['smartNoise']['crop']:
            if iterator == 1:
                
                #fig.canvas.mpl_connect('button_press_event',onclick)
                fig,ax = plt.subplots()
                imgplot = ax.imshow(image)
                getter1 = getter(ax,fig)
                plt.show()

                x = getter1.x
                y = getter1.y

                x.sort()
                y.sort()



                for i in range(0,len(x)):
                    x[i] = int(x[i])

                for i in range(0,len(y)):
                    y[i] = int(y[i])


                minLength = min(abs(x[1]-x[0]),abs(y[1]-y[0]))
                
                first = 0
        else:
            
            x = [0,dims[0]]
            y = [0,dims[1]]
        if len(dims)>2:
            image = rgb2gray(image)
        croppedImg = image[y[0]:y[1],x[0]:x[1]]
        #image = standardize(croppedImg)
        image = croppedImg
        #plt.imshow(image)
        #plt.show()



        #print(image)
        noise = extractNoise(image,100)
        noise = standardize(noise)
        #create noise template
        h,w = noise.shape
        im = Image.fromarray(noise*255)
        #plt.imshow(im)
        #plt.show()
        outFolder = filepath+'/noiseFiles/'
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
        
        print(outFolder+'noise'+str(iterator)+'.jpg')

        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(outFolder+'noise'+str(iterator)+'.jpg')
        iterator = iterator+1
        #plt.imshow(im)
        #plt.show()



if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath",nargs='?',type = str, default='noiseSamples', help='name of the folder to extract noise from')
    parser.add_argument("setbox",nargs='?',type =str, default="y", help="y to manually set crop size")
    args= parser.parse_args()

    filepath = args.filepath
    doCrop = args.setbox
    paths = glob.glob(filepath+'/*.bmp')

    #(vfilepath,vfilename)=os.path.split(args.videofilenamepath)
    iterator = 1
    print(paths)
    for path in paths:

        
        image = skimage.img_as_float(plt.imread(path))
        dims = image.shape
        if doCrop == "y":
            if iterator == 1:
                
                #fig.canvas.mpl_connect('button_press_event',onclick)
                fig,ax = plt.subplots()
                imgplot = ax.imshow(image)
                getter1 = getter(ax,fig)
                plt.show()

                x = getter1.x
                y = getter1.y

                x.sort()
                y.sort()



                for i in range(0,len(x)):
                    x[i] = int(x[i])

                for i in range(0,len(y)):
                    y[i] = int(y[i])
                first = 0
        else:
            
            x = [0,dims[0]]
            y = [0,dims[1]]
        if len(dims)>2:
            image = rgb2gray(image)
        croppedImg = image[y[0]:y[1],x[0]:x[1]]
        #image = standardize(croppedImg)
        image = croppedImg
        #plt.imshow(image)
        #plt.show()



        #print(image)
        noise = extractNoise(image,100)
        noise = standardize(noise)
        #create noise template
        h,w = noise.shape
        im = Image.fromarray(noise*255)
        #plt.imshow(im)
        #plt.show()
        outFolder = filepath+'/noiseFiles/'
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
        
        print(outFolder+'noise'+str(iterator)+'.jpg')

        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(outFolder+'noise'+str(iterator)+'.jpg')
        iterator = iterator+1
        #plt.imshow(im)
        #plt.show()

