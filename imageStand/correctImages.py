import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pprint as pp
import numpy as np
from PIL import Image
import glob
import os
from scipy import fftpack

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm

    
    plt.imshow(np.abs(im_fft), norm = LogNorm (vmin =5))
    plt.colorbar()
    plt.title('Fourier transform')
    
def standardize(image):
    image = image.astype(np.float64)
    imgMean = np.mean(image)
    imgSTD = np.std(image)
    image= (image - imgMean)/(6*imgSTD)
    image = image+0.5
    #image = image*255
    image = np.clip(image,0,1)
    return image




class fixer:
    def __init__(self,ax,fig):

        self.numClicks = 0
        self.x = [0,0]
        self.y = [0,0]
        self.ax = ax
        self.fig = fig
        self.fig.canvas.mpl_connect('button_press_event',self)


    def __call__(self,event):
        #print(event.xdata,event.ydata)
        self.x[self.numClicks] = event.xdata
        self.y[self.numClicks] = event.ydata
        #print(self.x)
        if self.numClicks > 0:
            plt.close()
        self.numClicks = self.numClicks+1



targetDir = os.getcwd()
ext = 'bmp'
outDir = targetDir+"\\outIMG\\"

if not os.path.exists(outDir):
    os.makedirs(outDir)

filePattern = 	targetDir+"\\*." + ext
first = 1  
for filename in glob.glob(filePattern):
    
    imgcv = cv2.imread(filename)
    print(imgcv.dtype)
    fig,ax = plt.subplots()
    imgplot = ax.imshow(imgcv)
    sections = filename.split("\\")
    imName = sections[-1]
    
    #im.save(outDir+imName)
    
    #print(result)
    prePost = imName.split(".")
    noEnd = prePost[0]
    if first == 1:
        fixer1 = fixer(ax,fig)
        #fig.canvas.mpl_connect('button_press_event',onclick)
        plt.show()

        x = fixer1.x
        y = fixer1.y

        x.sort()
        y.sort()



        for i in range(0,len(x)):
            x[i] = int(x[i])

        for i in range(0,len(y)):
            y[i] = int(y[i])
        first = 0


    croppedImg = imgcv[y[0]:y[1],x[0]:x[1],:]
    croppedImg = standardize(croppedImg)

    plot_spectrum(fImg)
    plt.savefig(outDir+noEnd+'Fourier.'+'jpg')

    im = Image.fromarray(croppedImg)
    im.save(outDir+imName)
    #fig,ax = plt.subplots()
    #imgplot = ax.imshow(croppedImg)
    #plt.show()
