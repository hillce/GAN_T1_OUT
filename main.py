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
parser.add_argument("--load",help="Load the preset trainSets, or redistribute (Bool)",type=bool,default=True,dest="load")
parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.5,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--gui",help="Use GUI to pick out parameters (WIP)",type=bool,default=False,dest="gui")

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
    load = args.load
    lr = args.lr
    b1 = args.b1
    bSize = args.batchSize
    numEpochs = args.numEpochs
    stepSize = args.stepSize

    modelDir = "./TrainingLogs/{}/".format(modelName)
    os.makedirs(modelDir)

    hParamDict = {}
    hParamDict["fileDir"] = fileDir
    hParamDict["t1MapDir"] = t1MapDir
    hParamDict["modelName"] = modelName
    hParamDict["load"] = load
    hParamDict["lr"] = lr
    hParamDict["b1"] = b1
    hParamDict["batchSize"] = bSize
    hParamDict["numEpochs"] = numEpochs
    hParamDict["stepSize"] = stepSize

    with open("{}hparams.json".format(modelDir),"w") as f:
        json.dump(hParamDict,f)


figDir = "{}Training_Figures/".format(modelDir)
os.makedirs(figDir)
print(load,type(load))

meanT1 = 362.66540459
stdT1 = 501.85027392

rA = Random_Affine(degreesRot=5,trans=(0.01,0.01),shear=5)
toT = ToTensor()
norm = Normalise()

trnsInTrain = transforms.Compose([toT,norm,rA])
trnsInVal = transforms.Compose([toT,norm])

datasetTrain = T1_Train_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=200,transform=trnsInTrain,load=load)
datasetVal = T1_Val_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=50,transform=trnsInVal,load=load)
datasetTest = T1_Test_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=50,transform=trnsInVal,load=load)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(7,288,384)
# netD = Discriminator(1,288,384)
netG = netG.to(device)
# netD = netD.to(device)

loss1 = nn.SmoothL1Loss()
loss2 = nn.MSELoss()

optimG = optim.Adam(netG.parameters(),lr=lr,betas=(b1,0.999))
# optimD = optim.Adam(netD.parameters(),lr=lr,betas=(b1,0.999))

lrSchedulerG = torch.optim.lr_scheduler.StepLR(optimG,step_size=stepSize,gamma=0.1,verbose=True)
# lrSchedulerD = torch.optim.lr_scheduler.StepLR(optimD,step_size=3,gamma=0.1,verbose=True)

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000.0
lossArr = np.zeros((numEpochs,trainLen,2))
for nE in range(numEpochs):
    print("\nEpoch [{}/{}]".format(nE+1,numEpochs))
    print("\nTraining:")
    runningLoss = 0.0
    for i,data in enumerate(loaderTrain):

        inpData = data["Images"].to(device)
        outGT = data["T1Map"].to(device)

        optimG.zero_grad()
        netG.zero_grad()

        outT1 = netG(inpData)
        err1 = loss1(outT1,outGT)
        err2 = loss2(outT1,outGT)

        lossArr[nE,i,0] = i+nE*trainLen
        lossArr[nE,i,1] = err2.item() + err1.item()
        runningLoss += err2.item() + err1.item()

        err = err1 + err2
        err.backward()
        optimG.step()
        
        if i % 500 == 0:
            outT1 = outT1.detach().cpu().numpy()[0,0,:,:]*stdT1+meanT1
            outGT = outGT.cpu().numpy()[0,0,:,:]*stdT1+meanT1


            fig, ax = plt.subplots(1,3)
            ax[0].imshow(outT1,vmax=900)
            ax[0].axis('off')
            ax[1].imshow(outGT,vmax=900)
            ax[1].axis('off')
            im = ax[2].imshow(abs(outT1-outGT),vmax=100,vmin=0,cmap="jet")
            fig.colorbar(im,ax=ax[2])
            ax[2].axis('off')
            plt.savefig("{}Epoch_{}_i_{}_img.png".format(figDir,nE+1,i+1))

            plt.figure()
            plt.plot(lossArr[nE,:i,0],lossArr[nE,:i,1])
            plt.savefig("{}Epoch_{}_i_{}_loss.png".format(figDir,nE+1,i+1))
            plt.close("all")

        sys.stdout.write("\r\tSubj {}/{}: Loss = {:.4f}".format(i*bSize,trainLen,runningLoss/((i+1)*4)))

    valLoss = []
    with torch.no_grad():
        print("\nValidation:")
        for i,data in enumerate(loaderVal):
            sys.stdout.write("\r\tSubj {}/{}".format(i*bSize,valLen))

            inpData = data["Images"].to(device)
            outGT = data["T1Map"].to(device)

            optimG.zero_grad()
            netG.zero_grad()

            outT1 = netG(inpData)
            err1 = loss1(outT1,outGT)
            err2 = loss2(outT1,outGT)

           
            valLoss.append(err2.item() + err1.item())

            if i % 50 == 0:
                outT1 = outT1.detach().cpu().numpy()[0,0,:,:]*stdT1+meanT1
                outGT = outGT.cpu().numpy()[0,0,:,:]*stdT1+meanT1


                fig, ax = plt.subplots(1,3)
                ax[0].imshow(outT1,vmax=900)
                ax[0].axis('off')
                ax[1].imshow(outGT,vmax=900)
                ax[1].axis('off')
                im = ax[2].imshow(abs(outT1-outGT),vmax=100,vmin=0,cmap="jet")
                fig.colorbar(im,ax=ax[2])
                plt.savefig("{}Epoch_{}_i_{}_img_val.png".format(figDir,nE+1,i+1))
                plt.close("all")

        valLoss = sum(valLoss)/valLen
        print("\n\tVal Loss: {}".format(valLoss))

        if valLoss < lowestLoss:
            torch.save({"Epoch":nE+1,
            "Generator_state_dict":netG.state_dict(),
            "Generator_loss_function1":loss1.state_dict(),
            "Generator_loss_function2":loss2.state_dict(),
            "Generator_optimizer":optimG.state_dict(),
            "Generator_lr_scheduler":lrSchedulerG.state_dict()
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss

    lrSchedulerG.step()
    print("-"*50)


