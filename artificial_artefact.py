# File with Functions for Artificial Inpainting
# Charles E Hill
# 21/10/2020

# In-built libraries
import os, csv, sys, copy, random
from matplotlib import image

# site-packages
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fft import ifft2, fft2, fftshift
from skimage.filters import sobel, gaussian
from PyQt5 import QtCore, QtGui, QtWidgets
import mahotas
import elasticdeform as ed

def dc_phase(phaseImg,magImg):
    """
    Function that takes a phase and a magnitude image, and creates a pixel deletion where there is a 2pi flip

    Args:
        phaseImg(np.array(x by y by 1)) - phase image for corresponding magnitude image
        magImg(np.array(x by y by 1)) - magnitude image for corresponding phase image
    """
    tempPhase = copy.deepcopy(phaseImg)
    tempMag = copy.deepcopy(magImg)
    tempPhase = sobel(gaussian(tempPhase,sigma=2)) # Finds the lines

    mask = list(map(lambda  x: x < 3*np.max(tempPhase)/4,tempPhase)) # Creates a mask

    tempPhase[mask] = 0
    tempPhase[tempMag < 3] = 0
    tempMag[tempPhase > 0] = 0
    return tempMag

def add_noise(image,mean=5,sigma=5):
    """
    Takes in an image and adds gausian noise of mean and sigma to the image (non k-space)

    Args:
        image(np.array(x by y by 1)) - image to add noise to
        mean(int, default = 5) - mean of gaussian noise
        sigma(int, default = 5) - sigma of gaussian noise
    """
    img = copy.deepcopy(image)
    row, col = img.shape[:2]
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = img + gauss
    return noisy

def add_spike(image,random = True, scaling = False, region = None):
    """
    Function for adding a spike to k space data

    Args:
        image(np.array(x by y by 1)) - input magnitude image
        rand(bool/int, default = False) - determines whether you apply random scaling (True), constant (int), no scaling (False)
        region(int, optional) - restricts the image to a region around the center of k space
    """

    img = copy.deepcopy(image)
    row, col = img.shape[:]
    fImg = fftshift(fft2(img))

    if random == True:
        if region:
            randRow = np.random.randint(row//2 - region, row//2 + region)
            randCol = np.random.randint(col//2 - region, row//2 + region)
        else:
            randRow = np.random.randint(row)
            randCol = np.random.randint(col)
    else:
        if type(random) == list:
            randRow = random[0]
            randCol = random[1]
        else:
            randRow = int(input("Row: "))
            randCol = int(input("Column: "))

    if scaling == True:
        scaling = np.random.rand()
    else:
        if type(scaling) in [int,float]:
            scaling = scaling
        else:
            scaling = 1

    fImg[randRow,randCol] = np.max(fImg)*scaling

    spikeIntensity = np.max(fImg)*scaling
    
    img = np.abs(ifft2(fImg))

    return img, spikeIntensity

def add_k_space_noise(image,mean=1000,sigma=50):
    """
    Function adds gaussian noise to k space

    Args:
        image(np.array[x,y]) - magnitude image
        mean(int, default = 1000) - mean for gaussian noise
        sigma(int, default = 50) - sgima for gaussian noise

    """
    img = copy.deepcopy(image)
    row, col = img.shape[:2]
    fImg = fftshift(fft2(img))

    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = fImg + gauss

    img = np.abs(ifft2(noisy))
    return img

def random_deformation(image, sigma=20, region=None, blur=(10,10)):
    """
    Function which takes in an image, and deforms all/a region of that image

    Args:
        image(np.array(x by y by 1)) - image to deform
        sigma(int, default=20) - sigma for the deformation, higher = more deformed
        region(list of ints, default=None) - region to deform
        blur(tuple of ints, default = (10,10)) - blurs deformed region into actual image
    """

    if region:
        tempImage = copy.deepcopy(image[region[0]:region[1],region[2]:region[3]])
    else:
        tempImage = copy.deepcopy(image)

    newImg = ed.deform_random_grid(tempImage,sigma=sigma)

    if region:
        tempImage = copy.deepcopy(image)
        for i in range(np.shape(newImg)[0]):
            for j in range(np.shape(newImg)[1]):
                if newImg[i,j] == 0:
                    newImg[i,j] = tempImage[region[0]+i,region[2]+j]
        tempImage[region[0]:region[1],region[2]:region[3]] = newImg

        if blur:
            x = tempImage[region[0]-blur[0]:region[1]+blur[0],region[2]-blur[1]:region[3]+blur[1]]
            x = gaussian(x,preserve_range=True)
            tempImage[region[0]:region[1],region[2]:region[3]] = newImg
        newImg = tempImage
        
    return newImg

class QCropLabel_Segmentation (QtWidgets.QWidget):
    """
    PyQt5 window for cropping out a drawn section of the image
    """

    def __init__(self,fols,path,random=True,parentQWidget = None):
        super(QCropLabel_Segmentation, self).__init__(parentQWidget)
        
        self.imageName = "Temp_Image.png"
        self.random = random
        self.path = path
        self.fols = fols
        self.dIdx = 0
        self.fN = self.fols[self.dIdx]
        while self.fN+"_pnts.npy" in os.listdir("./polygon/"):
            if self.random:
                self.dIdx = np.random.randint(len(self.fols))
            else:
                self.dIdx += 1
            self.fN = self.fols[self.dIdx]

        self.dicomList = os.listdir(os.path.join(self.path,self.fN))

        for dcm in self.dicomList:
            self.ds = pydicom.dcmread(os.path.join(self.path,self.fN,dcm))
            if "P" in self.ds.ImageType:
                mpimg.imsave(self.imageName,self.ds.pixel_array)
                break

        self.scaling = 800
        self.imgDims = (384,288)
        self.setWindowTitle('Select Crop Location')

        self.myPenWidth = 5
        self.myPenColor = QtGui.QColor(255,0,255,255)
        self.myBrushColor = QtGui.QColor(255,0,255,255//10)
        self.myBrush = QtGui.QBrush(self.myBrushColor)

        self.points = []
        self.firstPoint = True
        self.clicked = False
        self.initUI()

    def initUI (self):
        self.mainLabel = QtWidgets.QLabel()

        self.mainLabel.setPixmap(QtGui.QPixmap(self.imageName).scaled(self.scaling,self.scaling,QtCore.Qt.KeepAspectRatio))
        self.closeShortcut = QtWidgets.QShortcut(QtGui.QKeySequence("ESC"),self)
        self.closeShortcut.activated.connect(self.close_app)

        self.drawPolygon = QtWidgets.QShortcut(QtGui.QKeySequence("P"),self)
        self.drawPolygon.activated.connect(self.draw_function)

        self.deletePoints = QtWidgets.QShortcut(QtGui.QKeySequence("Del"),self)
        self.deletePoints.activated.connect(self.delete_points)

        self.deleteRegion = QtWidgets.QShortcut(QtGui.QKeySequence("D"),self)
        self.deleteRegion.activated.connect(self.delete_region)

        self.nextSubject = QtWidgets.QShortcut(QtCore.Qt.Key_Right,self)
        self.nextSubject.activated.connect(self.next_subject)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.mainLabel)
        self.setLayout(hbox)
        self.show()

    def mousePressEvent (self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            self.clicked = True

    def mouseMoveEvent(self, eventQMouseEvent):
        if self.clicked:
            if self.firstPoint:
                self.points.append(eventQMouseEvent.pos())
                self.firstPoint = False
            else:
                painter = QtGui.QPainter(self.mainLabel.pixmap())
                painter.setPen(self.myPenColor)
                self.points.append(eventQMouseEvent.pos())
                for ii,pnt in enumerate(self.points):
                    if ii != 0:
                        painter.drawLine(self.points[ii-1].x(),self.points[ii-1].y(),pnt.x(),pnt.y())
                painter.end()
                self.mainLabel.update()

    def mouseReleaseEvent(self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            self.clicked = False

    def draw_function(self):
        painter = QtGui.QPainter(self.mainLabel.pixmap())
        painter.setPen(self.myPenColor)
        painter.setBrush(self.myBrush)
        polygon = QtGui.QPolygon()
        for pnt in self.points:
            polygon.append(pnt)
        painter.drawPolygon(polygon,fillRule=QtCore.Qt.OddEvenFill)
        painter.end()
        self.mainLabel.update()

    def delete_points(self):
        self.points = []
        imgPaint = QtGui.QPixmap(self.imageName).scaled(self.scaling,self.scaling,QtCore.Qt.KeepAspectRatio)
        self.mainLabel.setPixmap(imgPaint)
        self.update()

    def delete_region(self):
        img = mpimg.imread("Temp_Image.png")
        self.pixelArray = copy.deepcopy(self.ds.pixel_array)
        pnts = []
        for p in self.points:
            pnt = (p.x(),p.y())
            pnts.append((pnt[1]*self.imgDims[1]//(self.scaling*self.imgDims[1]//self.imgDims[0]),pnt[0]*self.imgDims[0]//self.scaling))
        mahotas.polygon.fill_polygon(pnts,img[:,:,:],color=0)
        mahotas.polygon.fill_polygon(pnts,self.pixelArray,color=0)
        mpimg.imsave("Temp_Image_0.png",img[:,:,:])
        # np.save("./polygon/"+self.fN+".npy",self.pixelArray)
        np.save("./polygon/"+self.fN+"_pnts.npy",pnts)
        self.mainLabel.setPixmap(QtGui.QPixmap("Temp_Image_0.png").scaled(self.scaling,self.scaling,QtCore.Qt.KeepAspectRatio))
        self.update()

    def next_subject(self):
        while self.fN+"_pnts.npy" in os.listdir("./polygon/"):
            if self.random:
                self.dIdx = np.random.randint(len(self.fols))
            else:
                self.dIdx += 1
            self.fN = self.fols[self.dIdx]

        self.dicomList = os.listdir(os.path.join(self.path,self.fN))

        for dcm in self.dicomList:
            self.ds = pydicom.dcmread(os.path.join(self.path,self.fN,dcm))
            if "M" in self.ds.ImageType:
                mpimg.imsave("Temp_Image.png",self.ds.pixel_array)
                break

        self.mainLabel.setPixmap(QtGui.QPixmap(self.imageName).scaled(self.scaling,self.scaling,QtCore.Qt.KeepAspectRatio))
        self.mainLabel.update()


    def close_app(self):
        plt.imshow(self.pixelArray)
        print(np.load(self.fN+"_pnts.npy"))
        plt.show()
        self.close()   
