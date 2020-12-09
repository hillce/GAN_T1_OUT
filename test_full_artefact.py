import os, sys, json, itertools, copy
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication

from models import Discriminator, Generator
from datasets import T1_Train_Dataset, T1_Val_Dataset, T1_Test_Dataset, Random_Affine, ToTensor, Normalise, collate_fn
from param_gui import Param_GUI
from artificial_artefact import add_spike

# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--dir",help="File directory for numpy images",type=str,default="C:/fully_split_data/",dest="fileDir")
parser.add_argument("--t1dir",help="File directory for T1 matfiles",type=str,default="C:/T1_Maps/",dest="t1MapDir")
parser.add_argument("--model_name",help="Name for saving the model",type=str,dest="modelName",required=True)
# parser.add_argument("--load",help="Load the preset trainSets, or redistribute (Bool)",default=False,action='store_true',dest="load")
# parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
# parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.5,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
# parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
# parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--gui",help="Use GUI to pick out parameters (WIP)",default=False,action='store_true',dest="gui")
parser.add_argument("--artLoc","-artefact_location",help="Which slice to add a simulated artefact to [0-6]",type=list,default=[6],dest="artLoc")
args = parser.parse_args()

if args.gui:
    # Overrides the argparse parameters
    models = os.listdir("./TrainingLogs/")

    app = QApplication(sys.argv)
    main_win = Param_GUI()
    main_win.show()
    app.exec_()

    newModels = os.listdir("./TrainingLogs/")

    for mod in models:
        newModels.remove(mod)
    
    modelDir = "./TrainingLogs/{}/".format(newModels[0])
    dirHParam = "{}hparams.json".format(modelDir)

    with open(dirHParam,'r') as f:
        hParamDict = json.load(f)

    fileDir = hParamDict["fileDir"]
    t1MapDir = hParamDict["t1MapDir"]
    modelName = hParamDict["modelName"]
    load = hParamDict["load"]
    lr = hParamDict["lr"]
    b1 = hParamDict["b1"]
    bSize = hParamDict["batchSize"]
    numEpochs = hParamDict["numEpochs"]
    stepSize = hParamDict["stepSize"]
    artLoc = 6

else:

    fileDir = args.fileDir
    t1MapDir = args.t1MapDir
    modelName = args.modelName
    bSize = args.batchSize
    artLoc = args.artLoc
    modelDir = "./TrainingLogs/{}/".format(modelName)
    # modelDir = "./TrainingLogs/{}/".format(modelName)
    assert os.path.isdir(modelDir), "Model Directory is not found, please check your model name!"



figDir = "{}Test_Figures/".format(modelDir)
try:
    os.makedirs(figDir)
except FileExistsError as e:
    print(e, "This means you will be overwriting previous results!")

meanT1 = 362.66540459
stdT1 = 501.85027392

toT = ToTensor()
# norm = Normalise()

trnsInVal = transforms.Compose([toT])

datasetTest = T1_Test_Dataset(modelName,fileDir=fileDir,t1MapDir=t1MapDir,transform=trnsInVal,load=True,loadDir="{}testSet.npy".format(modelDir))
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
netG = Generator(7,288,384,outC=7)
# netD = Discriminator(1,288,384)
netG = netG.to(device)
# netD = netD.to(device)
netGT1 = Generator(7,288,384,outC=1)
netGT1.to(device)


modelDict = torch.load("{}model.pt".format(modelDir),map_location=device)
netG.load_state_dict(modelDict["Generator_state_dict"])

modelDictT1 = torch.load("./TrainingLogs/arcus_test/model.pt",map_location=device)
netGT1.load_state_dict(modelDictT1["Generator_state_dict"])

# knNums = list(itertools.combinations_with_replacement([0,1,2,3,4,5,6],2))
knNums = list(itertools.combinations([6,6],2))

loss1 = nn.SmoothL1Loss()
loss2 = nn.MSELoss()

testLen = datasetTest.__len__()

lossArr = np.zeros((testLen,2))
with torch.no_grad():
    print("\nTesting:")
    runningLoss = 0.0
    for i,data in enumerate(loaderTest):

        inpData = data["Images"]
        outGT = data["T1Map"]

        imgs = copy.deepcopy(inpData.numpy())
        for i in range(imgs.shape[0]):
            for aL in artLoc:
                imgs[i,aL,:,:],_ = add_spike(imgs[i,aL,:,:],random=[120,120])
        for kn in knNums:
            knImgs = copy.deepcopy(imgs)
            knImgs[:,kn,:,:] = 0.0
            knImgs = torch.from_numpy(knImgs)

            reconImgs = netG(knImgs)

            reconImgs = reconImgs.numpy()
            for i in range(reconImgs.shape[1]):
                if i != kn[0]:
                    reconImgs[:,i,:,:] = knImgs[:,i,:,:]
            reconImgs = torch.from_numpy(reconImgs)
            
            reconT1 = netGT1(reconImgs)

            reconNumpy = reconImgs.numpy()

            reconNumpyT1 = reconT1.numpy()
            reconNumpyT1 = np.transpose(reconNumpyT1,axes=[0,2,3,1])

            gtT1Numpy = outGT.numpy()
            gtT1Numpy = np.transpose(gtT1Numpy,axes=[0,2,3,1])

            perErrMap = np.zeros(reconNumpy.shape)
            perErrMapT1 = np.zeros(reconNumpyT1.shape)

            fig,ax = plt.subplots(4,7)

            for i in range(reconNumpy.shape[1]):
                ax[0,i].imshow(imgs[0,i,:,:])
                ax[0,i].set_xticks([])
                ax[0,i].set_yticks([])
                if i == kn[0]:
                    rect = patches.Rectangle((0+2,0+2),reconNumpy.shape[3]-2,reconNumpy.shape[2]-2,edgecolor="r",fill=False,lw=3)
                    ax[0,i].add_patch(rect)
                    
                ax[1,i].imshow(reconImgs[0,i,:,:])
                ax[1,i].set_xticks([])
                ax[1,i].set_yticks([])

                # ax[2,i].imshow(knImgs[0,i,:,:])
                # ax[2,i].set_xticks([])
                # ax[2,i].set_yticks([])

                # Percentage Error
                perErrMap[0,i,:,:] = (abs(imgs[0,i,:,:]-reconNumpy[0,i,:,:])+imgs[0,i,:,:])/imgs[0,i,:,:]
                perErrMap[0,i,:,:] = (perErrMap[0,i,:,:] - 1)*100

                for m in range(perErrMap.shape[2]):
                    for k in range(perErrMap.shape[3]):
                        if perErrMap[0,i,m,k] == np.inf:
                            perErrMap[0,i,m,k] = 0



                # ax[3,i].imshow(perErrMap[0,i,:,:],cmap="jet",vmax=50)
                im = ax[2,i].imshow(abs(imgs[0,i,:,:]-reconNumpy[0,i,:,:]),cmap="jet",vmax=50)
                fig.colorbar(im,ax=ax[2,i])
                ax[2,i].set_xticks([])
                ax[2,i].set_yticks([])

                im = ax[3,i].imshow(perErrMap[0,i,:,:],cmap="jet",vmax=50)
                fig.colorbar(im,ax=ax[3,i])
                ax[3,i].set_xticks([])
                ax[3,i].set_yticks([])

            rows = ["Original","Reconstructed","Absolute Error","Percentage Error"]
            cols = ["1","2","3","4","5","6","7"]
            for a, col in zip(ax[0], cols):
                a.set_title(col)

            for a, row in zip(ax[:,0], rows):
                a.set_ylabel(row,size="large")#, rotation=0, size='large')


            fig,ax = plt.subplots(2,2)
            im = ax[0,0].imshow(gtT1Numpy[0,:,:,:],vmax=1200)
            fig.colorbar(im,ax=ax[0,0])
            ax[0,0].set_title("Original T1")
            ax[0,0].set_xticks([])
            ax[0,0].set_yticks([])

            im = ax[1,0].imshow(reconNumpyT1[0,:,:,:],vmax=1200)
            fig.colorbar(im,ax=ax[1,0])
            ax[1,0].set_title("Recon T1")
            ax[1,0].set_xticks([])
            ax[1,0].set_yticks([])

            perErrMapT1[0,:,:,:] = (abs(gtT1Numpy[0,:,:,:]-reconNumpyT1[0,:,:,:])+gtT1Numpy[0,:,:,:])/gtT1Numpy[0,:,:,:]
            perErrMapT1[0,:,:,:] = (perErrMapT1[0,:,:,:] - 1)*100

            for m in range(perErrMapT1.shape[1]):
                for k in range(perErrMapT1.shape[2]):
                    if perErrMapT1[0,m,k,0] == np.inf:
                        perErrMapT1[0,m,k,0] = 0

            im = ax[0,1].imshow(abs(gtT1Numpy[0,:,:,:]-reconNumpyT1[0,:,:,:]),cmap="jet",vmax=50)
            fig.colorbar(im,ax=ax[0,1])
            ax[0,1].set_title("Absolute Error")
            ax[0,1].set_xticks([])
            ax[0,1].set_yticks([])


            im = ax[1,1].imshow(perErrMapT1[0,:,:,:],cmap="jet",vmax=50)
            fig.colorbar(im,ax=ax[1,1])
            ax[1,1].set_title("Percentage Error")
            ax[1,1].set_xticks([])
            ax[1,1].set_yticks([])

            plt.show()