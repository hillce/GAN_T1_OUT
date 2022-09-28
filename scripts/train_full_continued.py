import os, sys, time, copy, json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datasets import T1_Train_Dataset, T1_Val_Dataset, T1_Test_Dataset, Random_Affine, ToTensor, Normalise, collate_fn
from models import Generator, Discriminator

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
bSize = 8
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
norm = Normalise()

trnsInTrain = transforms.Compose([toT,rA])
trnsInVal = transforms.Compose([toT])

datasetTrain = T1_Train_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=10000,transform=trnsInTrain,load=load)
datasetVal = T1_Val_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInVal,load=load)
datasetTest = T1_Test_Dataset(fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInVal,load=load)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(7,288,384,outC=7)
netD = Discriminator(7,288,384)
netLoss = Generator(7,288,384)

lossDir = "./TrainingLogs/arcus_test/model.pt"
netLossDict = torch.load(lossDir)
netLoss.load_state_dict(netLossDict["Generator_state_dict"])

for param in netLoss.parameters():
    param.requires_grad = False
netLoss.eval()

modelDict = torch.load("{}model.pt".format(modelDir))
netG.load_state_dict(modelDict["Generator_state_dict"])
netD.load_state_dict(modelDict["Discriminator_state_dict"])

netG = netG.to(device)
netD = netD.to(device)
netLoss = netLoss.to(device)

real_label = 1
fake_label = 0

gen_loss = nn.MSELoss()
disc_loss = nn.BCELoss()
T1_loss = nn.SmoothL1Loss()

optim_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(b1,0.999))
optim_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(b1,0.999))

optim_G.load_state_dict(modelDict["Generator_optimizer"])
optim_D.load_state_dict(modelDict["Discriminator_state_dict"])

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000000000.0
trainLossCnt = 0
valLossCnt = 0
#####################################################################################################
#################### Start Training #################################################################

for nE in range(modelDict["Epoch"],numEpochs):
    for ii, data in enumerate(loaderTrain):
        inpData = data["Images"]
        t1GT = data["T1Map"].to(device)

        optim_G.zero_grad()
        optim_D.zero_grad()

        knImgs = copy.deepcopy(inpData).numpy()
        for j in range(knImgs.shape[0]):
            knNum = np.random.randint(knImgs.shape[1],size=np.random.randint(3)+1)
            knImgs[j,knNum,:,:] = 0.0
        knImgs = torch.from_numpy(knImgs)

        inpData = inpData.to(device)
        knImgs = knImgs.to(device)

        netD.zero_grad()

        label = torch.full((inpData.size()[0],),real_label,device=device,dtype=torch.float)
        real_out = netD(inpData).view(-1)

        errD_real = disc_loss(real_out,label)
        errD_real.backward()

        netG.zero_grad()

        fake = netG(knImgs)
        errG_recon = gen_loss(fake,inpData)
        errG_recon.backward(retain_graph=True)

        # T1 Loss
        t1Out = netLoss(fake)
        errG_T1 = T1_loss(t1Out,t1GT)
        errG_T1.backward(retain_graph=True)


        label.fill_(fake_label)
        fake_out = netD(fake.detach()).view(-1)
        errD_fake = disc_loss(fake_out,label)
        errD_fake.backward()

        optim_D.step()

        label.fill_(real_label)
        fake_out_G = netD(fake).view(-1)
        errG_fake = disc_loss(fake_out_G, label)
        errG_fake.backward()

        loss_batch = errD_real.item() + errG_recon.item() + errD_fake.item() + errG_fake.item() + errG_T1.item()

        writer.add_scalar('Loss/train',loss_batch,trainLossCnt)

        trainLossCnt += 1

        optim_G.step()

    with torch.no_grad():
        valLoss = 0.0
        for ii, data in enumerate(loaderVal):
            inpData = data["Images"]
            t1GT = data["T1Map"].to(device)

            knImgs = copy.deepcopy(inpData).numpy()
            for j in range(knImgs.shape[0]):
                knNum = np.random.randint(knImgs.shape[1],size=np.random.randint(3)+1)
                knImgs[j,knNum,:,:] = 0.0
            knImgs = torch.from_numpy(knImgs)

            inpData = inpData.to(device)
            knImgs = knImgs.to(device)

            label = torch.full((inpData.size()[0],),real_label,device=device,dtype=torch.float)
            real_out = netD(inpData).view(-1)

            errD_real = disc_loss(real_out,label)

            fake = netG(knImgs)
            errG_recon = gen_loss(fake,inpData)

            # T1 Loss
            t1Out = netLoss(fake)
            errG_T1 = T1_loss(t1Out,t1GT)

            label.fill_(fake_label)
            fake_out = netD(fake).view(-1)
            errD_fake = disc_loss(fake_out,label)

            label.fill_(real_label)
            fake_out_G = netD(fake).view(-1)
            errG_fake = disc_loss(fake_out_G, label)

            loss_batch = errD_real.item() + errG_recon.item() + errD_fake.item() + errG_fake.item() + errG_T1.item()
            valLoss += loss_batch

            writer.add_scalar("Loss/val",loss_batch,valLossCnt)

            valLossCnt += 1

        valLoss /= valLen

        if nE == 0:
            torch.save({"Epoch":nE+1,
            "Generator_state_dict":netG.state_dict(),
            "Discriminator_state_dict":netD.state_dict(),
            "Generator_optimizer":optim_G.state_dict(),
            "Discriminator_optimizer":optim_D.state_dict(),
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss
        else:
            if valLoss < lowestLoss:
                torch.save({"Epoch":nE+1,
                "Generator_state_dict":netG.state_dict(),
                "Discriminator_state_dict":netD.state_dict(),
                "Generator_optimizer":optim_G.state_dict(),
                "Discriminator_optimizer":optim_D.state_dict(),
                },"{}model.pt".format(modelDir))
                lowestLoss = valLoss
