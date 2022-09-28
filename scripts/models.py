# Generator and Discriminator models
# Charles E Hill
# 21/10/2020

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
# Classic Generator Models

class Generator(nn.Module):
    """
    Generator network for recreating the missing image in the sequence of T1,T2,PDFF maps etc.
    """
    def __init__(self,nC,xDim=128,yDim=128,outC=1):
        super(Generator,self).__init__()
        self.nC = nC

        self.inConv = DoubleConv(self.nC,64)
        self.down1 = DownConv(64,128)
        self.down2 = DownConv(128,256)
        self.down3 = DownConv(256,512)
        self.down4 = DownConv(512,512)
        self.up1 = UpConv(1024,256)
        self.up2 = UpConv(512,128)
        self.up3 = UpConv(256,64)
        self.up4 = UpConv(128,64)
        self.outConv1 = nn.Conv2d(64,32,kernel_size=1)
        self.outConv2 = nn.Conv2d(32,16,kernel_size=1)
        self.outConv3 = nn.Conv2d(16,outC,kernel_size=1)

    def forward(self,x):
        x = self.inConv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(x4,x3)
        x6 = self.up2(x5,x2)
        x7 = self.up3(x6,x1)
        x8 = self.up4(x7,x)
        out = self.outConv1(x8)
        out = self.outConv2(out)
        out = self.outConv3(out)

        return out

class Discriminator(nn.Module):
    """
    Discriminator to above Generator
    """
    def __init__(self,nC,xDim=128,yDim=128):
        super(Discriminator,self).__init__()
        self.nC = nC

        self.inConv = DoubleConv(self.nC,64)
        self.down1 = DownConv(64,128)
        self.down2 = DownConv(128,256)
        self.down3 = DownConv(256,512)
        self.down4 = DownConv(512,1024)

        self.linear1 = nn.Linear(1024*(xDim//(2**4))*(yDim//(2**4)),1024) # 2**4 determined by the number of downsamples
        self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512,64)
        self.linear4 = nn.Linear(64,1)

    def forward(self,x):
        x = self.inConv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = x.view(-1,self.flat_features(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        out = torch.sigmoid(x)
        return out

    def flat_features(self,x):
        flatFeatures = 1
        for s in x.size()[1:]:
            flatFeatures *= s
        return flatFeatures

class DoubleConv(nn.Module):
    """
    Conv-Bn-ReLu *2
    """
    def __init__(self,inC,outC):
        super(DoubleConv,self).__init__()

        self.double_conv = nn.Sequential(
                                    nn.Conv2d(inC,outC,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(outC),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(outC,outC,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(outC),
                                    nn.ReLU(inplace=True)
                                    )

    def forward(self, x):
        return self.double_conv(x)

class DownConv(nn.Module):
    """
    Downsample and Double Conv
    """
    def __init__(self,inC,outC):
        super(DownConv,self).__init__()

        self.down_conv = nn.Sequential(nn.MaxPool2d(2),DoubleConv(inC,outC))

    def forward(self, x):
        return self.down_conv(x)

class UpConv(nn.Module):
    """
    Upsample, concatenation and Double Conv
    """
    def __init__(self,inC,outC):
        super(UpConv,self).__init__()

        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.conv = DoubleConv(inC,outC)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
