import os, copy, json, sys, re, warnings

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

class T1_Train_Dataset(Dataset):
    """
    T1 Dataset
    """
    def __init__(self,size=2000,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",load=True,transform=None):
        self.fileDir = fileDir
        self.t1MapDir = t1MapDir
        self.transform = transform

        if not load:
            subjList = [x[:7] for x in os.listdir(fileDir) if x.endswith("0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(t1MapDir,x[:7]))]
            self.trainSet = np.random.choice(subjList,size)
            np.save("trainSet",self.trainSet)
        else:
            self.trainSet = np.load("trainSet.npy")

    def __getitem__(self, index):

        inpData = np.load("{}{}_20204_2_0.npy".format(self.fileDir,self.trainSet[index]))
        outGT = loadmat("{}{}_20204_2_0.mat".format(self.t1MapDir,self.trainSet[index]))['results']

        sample = {"Images":inpData,"T1Map":outGT}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.trainSet)

class T1_Val_Dataset(Dataset):
    """
    T1 Dataset
    """
    def __init__(self,size=500,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",load=True,transform=None):
        self.fileDir = fileDir
        self.t1MapDir = t1MapDir
        self.transform = transform

        if not load:
            subjList = [x[:7] for x in os.listdir(fileDir) if x.endswith("0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(t1MapDir,x[:7]))]
            self.trainSet = np.load("trainSet.npy")
            subjList = [x for x in subjList if x not in self.trainSet]
            self.valSet = np.random.choice(subjList,size)
            np.save("valSet",self.valSet)
        else:
            self.valSet = np.load("valSet.npy")

    def __getitem__(self, index):

        inpData = np.load("{}{}_20204_2_0.npy".format(self.fileDir,self.valSet[index]))
        outGT = loadmat("{}{}_20204_2_0.mat".format(self.t1MapDir,self.valSet[index]))['results']

        sample = {"Images":inpData,"T1Map":outGT}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.valSet)

class T1_Test_Dataset(Dataset):
    """
    T1 Dataset
    """
    def __init__(self,size=500,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",load=True,transform=None):
        self.fileDir = fileDir
        self.t1MapDir = t1MapDir
        self.transform = transform

        if not load:
            subjList = [x[:7] for x in os.listdir(fileDir) if x.endswith("0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(t1MapDir,x[:7]))]
            self.trainSet = np.load("trainSet.npy")
            self.valSet = np.load("valSet.npy")
            subjList = [x for x in subjList if x not in self.trainSet and x not in self.valSet]
            self.testSet = np.random.choice(subjList,size)
            np.save("testSet",self.testSet)
        else:
            self.testSet = np.load("testSet.npy")

    def __getitem__(self, index):

        inpData = np.load("{}{}_20204_2_0.npy".format(self.fileDir,self.testSet[index]))
        outGT = loadmat("{}{}_20204_2_0.mat".format(self.t1MapDir,self.testSet[index]))['results']

        sample = {"Images":inpData,"T1Map":outGT}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.testSet)

class Random_Affine(object):

    def __init__(self,degreesRot=10,trans=(0.1,0.1),shear=10):

        self.rA = transforms.RandomAffine(degreesRot,translate=trans,shear=shear)

    def __call__(self,sample):
        inpData = sample["Images"]
        outGT = sample["T1Map"]

        allData = torch.cat((inpData,outGT))

        images = self.rA(allData)

        sample = {"Images":images[:7,:,:],"T1Map":images[7,:,:].unsqueeze_(0)}
        return sample

class ToTensor(object):
    """ convert ndarrays in sample to Tensors"""

    def __call__(self,sample):
        inpData = sample["Images"]
        outGT = sample["T1Map"]

        inpData = np.transpose(inpData,axes=(2,0,1))
        inpData = torch.from_numpy(inpData).float() 
        
        outGT = outGT[:,:,0]
        outGT = torch.from_numpy(outGT).float()
        outGT.unsqueeze_(0)

        sample = {"Images":inpData,"T1Map":outGT}
        return sample

class Normalise(object):

    def __init__(self):

        self.normImg = transforms.Normalize([23.15050654, 44.1956957,  49.89248841, 51.43708248, 51.98784791, 18.21313955, 17.58093937],[33.52329497, 79.89539557, 83.93680164, 84.62633451, 84.78080839, 23.92935112, 29.45570738],inplace=True)
        self.normT1 = transforms.Normalize([362.66540459],[501.85027392])

    def __call__(self,sample):
        inpData = sample["Images"]
        outGT = sample["T1Map"]

        sample = {"Images":self.normImg(inpData),"T1Map":self.normT1(outGT)}
        return sample

def collate_fn(sampleBatch):
    inpData = [item['Images'].unsqueeze_(0) for item in sampleBatch]
    inpData = torch.cat(inpData)

    outGT = [item['T1Map'].unsqueeze_(0) for item in sampleBatch]
    outGT = torch.cat(outGT)

    sample = {"Images":inpData,"T1Map":outGT}
    return sample

if __name__ == "__main__":

    rA = Random_Affine()
    toT = ToTensor()
    trnsIn = transforms.Compose([toT,rA])

    dataset = T1_Train_Dataset(transform=trnsIn)
    loader = DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn,pin_memory=False)

    batch = next(iter(loader))
    
    batchImg = batch["Images"].numpy()
    batchT1 = batch["T1Map"].numpy()
    for i in range(batchImg.shape[0]):
        fig,ax = plt.subplots(1,batchImg.shape[1])
        for j in range(batchImg.shape[1]):
            ax[j].imshow(batchImg[i,j,:,:])

        plt.figure()
        plt.imshow(batchT1[i,0,:,:])
        plt.show()


