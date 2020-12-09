import os, sys, json
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication

from models import Discriminator, Generator
from datasets import T1_Train_Dataset, T1_Val_Dataset, T1_Test_Dataset, Random_Affine, ToTensor, Normalise, collate_fn
from param_gui import Param_GUI

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

else:

    fileDir = args.fileDir
    t1MapDir = args.t1MapDir
    modelName = args.modelName
    bSize = args.batchSize

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

datasetTest = T1_Test_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,transform=trnsInVal,load=True,loadDir="{}testSet.npy".format(modelDir))
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(7,288,384)
# netD = Discriminator(1,288,384)
netG = netG.to(device)
# netD = netD.to(device)

modelDict = torch.load("{}model.pt".format(modelDir))
netG.load_state_dict(modelDict["Generator_state_dict"])

loss1 = nn.SmoothL1Loss()
loss2 = nn.MSELoss()


testLen = datasetTest.__len__()

lossArr = np.zeros((testLen,2))
with torch.no_grad():
    print("\nTesting:")
    runningLoss = 0.0
    for i,data in enumerate(loaderTest):

        inpData = data["Images"].to(device)
        outGT = data["T1Map"].to(device)

        outT1 = netG(inpData)
        err1 = loss1(outT1,outGT)
        err2 = loss2(outT1,outGT)

        lossArr[i,0] = i*testLen
        lossArr[i,1] = err2.item() + err1.item()
        runningLoss += err2.item() + err1.item()

        err = err1 + err2
        
        if i % 10 == 0:
            outT1 = outT1.detach().cpu().numpy()[0,0,:,:]#*stdT1+meanT1
            outGT = outGT.cpu().numpy()[0,0,:,:]#*stdT1+meanT1

            np.save("{}i_{}_T1.npy".format(figDir,i+1),outT1)
            np.save("{}i_{}_GT.npy".format(figDir,i+1),outGT)

            plt.figure()
            plt.imshow(outT1,cmap="plasma",vmax=1300)
            plt.axis('off')
            plt.colorbar()
            plt.savefig("{}i_{}_outT1.png".format(figDir,i+1))

            plt.figure()
            plt.imshow(outGT,cmap="plasma",vmax=1300)
            plt.axis('off')
            plt.colorbar()
            plt.savefig("{}i_{}_gtT1.png".format(figDir,i+1))

            plt.figure()
            plt.imshow(abs(outT1-outGT),vmax=100,cmap="jet")
            plt.axis('off')
            plt.colorbar()
            plt.savefig("{}i_{}_diffT1.png".format(figDir,i+1))
            
            plt.figure()
            # Percentage Error
            perErrMap = (abs(outT1-outGT)+outGT)/outGT
            perErrMap = (perErrMap - 1)*100

            for m in range(perErrMap.shape[0]):
                for k in range(perErrMap.shape[1]):
                    if perErrMap[m,k] == np.inf:
                        perErrMap[m,k] = 0

            plt.imshow(perErrMap,vmax=20,cmap="jet")
            plt.colorbar()
            plt.axis("off")
            plt.savefig("{}i_{}_percentageT1.png".format(figDir,i+1))
            plt.close("all")


    plt.figure()
    plt.plot(lossArr[:,0],lossArr[:,1],".")
    plt.savefig("{}testLoss.png".format(figDir))
    plt.close("all")
    print("-"*50)


