import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import (Normalise, Random_Affine, T1_Test_Dataset,
                      T1_Train_Dataset, T1_Val_Dataset, ToTensor, collate_fn)
from models import Generator

############################ Training Preamble ###########################################################################
# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--dir",help="File directory for numpy images",type=str,default="C:/fully_split_data/",dest="fileDir")
parser.add_argument("--t1dir",help="File directory for T1 matfiles",type=str,default="C:/T1_Maps/",dest="t1MapDir")
parser.add_argument("--model_name",help="Name for saving the model",type=str,dest="modelName",required=True)
parser.add_argument("--load",help="Load the preset trainSets, or redistribute (Bool)",default=False,action='store_true',dest="load")
parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.5,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--norm",help="Normalise the input images and T1 map",default=False,action='store_true',dest="norm")
parser.add_argument("-trainS","--train_size",help="Size of the training dataset, if load, will be ignored",type=int,default=10000,dest="trainSize")
parser.add_argument("-valS","--val_size",help="Size of the validation dataset, if load, will be ignored",type=int,default=1000,dest="valSize")
parser.add_argument("-testS","--test_size",help="Size of the test dataset, if load, will be ignored",type=int,default=1000,dest="testSize")

args = parser.parse_args()

fileDir = args.fileDir
t1MapDir = args.t1MapDir
modelName = args.modelName
load = args.load
lr = args.lr
b1 = args.b1
bSize = args.batchSize
numEpochs = args.numEpochs
stepSize = args.stepSize
norm = args.norm
trainSize = args.trainSize
valSize = args.valSize
testSize = args.testSize

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
hParamDict["norm"] = norm

with open("{}hparams.json".format(modelDir),"w") as f:
    json.dump(hParamDict,f)


figDir = "{}Training_Figures/".format(modelDir)
os.makedirs(figDir)

########################################################################################################################
############################## Model Setup #############################################################################

writer = SummaryWriter("{}tensorboard".format(modelDir))

meanT1 = 362.66540459
stdT1 = 501.85027392

rA = Random_Affine(degreesRot=5,trans=(0.01,0.01),shear=5)
toT = ToTensor()
normTrns = Normalise()

if norm:
    trnsInTrain = transforms.Compose([toT,normTrns,rA])
    trnsInVal = transforms.Compose([toT,normTrns])
else:
    trnsInTrain = transforms.Compose([toT,rA])
    trnsInVal = transforms.Compose([toT])

datasetTrain = T1_Train_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=trainSize,transform=trnsInTrain,load=load)
datasetVal = T1_Val_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=valSize,transform=trnsInVal,load=load)
datasetTest = T1_Test_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=testSize,transform=trnsInVal,load=load)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(7,288,384)
netG = netG.to(device)

loss1 = nn.SmoothL1Loss()

optimG = optim.Adam(netG.parameters(),lr=lr,betas=(b1,0.999))
lrSchedulerG = torch.optim.lr_scheduler.StepLR(optimG,step_size=stepSize,gamma=0.1,verbose=False)

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000000000.0
trainLossCnt = 0
valLossCnt = 0
#####################################################################################################
#################### Start Training #################################################################

for nE in range(numEpochs):
    for i,data in enumerate(loaderTrain):
        inpData = data["Images"].to(device)
        outGT = data["T1Map"].to(device)

        optimG.zero_grad()
        netG.zero_grad()

        outT1 = netG(inpData)
        err1 = loss1(outT1,outGT)
        err1.backward()
        optimG.step()
        
        writer.add_scalar('Loss/train',err1.item(),trainLossCnt)

        trainLossCnt += 1

    valLoss = 0.0
    with torch.no_grad():
        for i,data in enumerate(loaderVal):

            inpData = data["Images"].to(device)
            outGT = data["T1Map"].to(device)

            outT1 = netG(inpData)
            err1 = loss1(outT1,outGT)

            valLoss += err1.item()

            writer.add_scalar("Loss/val",err1.item(),valLossCnt)

            valLossCnt += 1
        valLoss /= valLen

        if valLoss < lowestLoss:
            torch.save({"Epoch":nE+1,
            "Generator_state_dict":netG.state_dict(),
            "Generator_loss_function1":loss1.state_dict(),
            "Generator_optimizer":optimG.state_dict(),
            "Generator_lr_scheduler":lrSchedulerG.state_dict()
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss

    lrSchedulerG.step()
