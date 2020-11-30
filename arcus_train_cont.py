import os,json, sys
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import Generator
from datasets import T1_Train_Dataset, T1_Val_Dataset, T1_Test_Dataset, Random_Affine, ToTensor, Normalise, collate_fn

############################ Training Preamble ###########################################################################
# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--model_name",help="Name for saving the model",type=str,dest="modelName",required=True)
args = parser.parse_args()

modelName = args.modelName

modelDir = "./TrainingLogs/{}/".format(modelName)
with open("{}hparams.json".format(modelDir),"r") as f:
    hParamDict = json.load(f)

fileDir = hParamDict["fileDir"]
t1MapDir = hParamDict["t1MapDir"]
# load = hParamDict["load"]
load = True
lr = hParamDict["lr"]
b1 = hParamDict["b1"]
bSize = hParamDict["batchSize"]
numEpochs = hParamDict["numEpochs"]
stepSize = hParamDict["stepSize"]
norm = hParamDict["norm"]

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

datasetTrain = T1_Train_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,transform=trnsInTrain,load=load)
datasetVal = T1_Val_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,transform=trnsInVal,load=load)
datasetTest = T1_Test_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,transform=trnsInVal,load=load)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(7,288,384)
netG = netG.to(device)

modelDict = torch.load("{}model.pt".format(modelDir))

netG.load_state_dict(modelDict["Generator_state_dict"])

loss1 = nn.SmoothL1Loss()
loss1.load_state_dict(modelDict["Generator_loss_function1"])

optimG = optim.Adam(netG.parameters(),lr=lr,betas=(b1,0.999))
optimG.load_state_dict(modelDict["Generator_optimizer"])

lrSchedulerG = torch.optim.lr_scheduler.StepLR(optimG,step_size=stepSize,gamma=0.1,verbose=False)

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

try:
    lowestLoss = modelDict["Val_loss"]
except KeyError:
    lowestLoss = 10000000000000.0

trainLossCnt = 0
valLossCnt = 0
#####################################################################################################
#################### Start Training #################################################################

for nE in range(modelDict["Epoch"],numEpochs):
    runningLoss = 0.0
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
            "Val_loss":valLoss,
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss

    lrSchedulerG.step()
