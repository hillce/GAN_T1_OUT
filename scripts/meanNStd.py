import os, sys

import numpy as np
from scipy.io import loadmat


fileDir = "C:/fully_split_data/"
t1MapDir = "C:/T1_Maps/"

subjList = [x[:7] for x in os.listdir(fileDir) if x.endswith("0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(t1MapDir,x[:7]))]
trainLen = len(subjList)

# inpDataMetricList = np.zeros((7))
# outGTDataMetricList = np.zeros((1))

# for i,subj in enumerate(subjList):
#     sys.stdout.write("\r Subj {}/{}".format(i,trainLen))
#     inpData = np.load("{}{}_20204_2_0.npy".format(fileDir,subj))
#     outGT = loadmat("{}{}_20204_2_0.mat".format(t1MapDir,subj))['results']

#     for j in range(7):
#         inpDataMetricList[j] += np.sum(inpData[:,:,j])
#     outGTDataMetricList[0] += np.sum(outGT[:,:,0])

numPixels = 288*384*len(subjList)

# meanValsInp = inpDataMetricList/numPixels
# print(meanValsInp)
# meanValsGT = outGTDataMetricList/numPixels
# print(meanValsGT)

meanValsInp = [23.15050654, 44.1956957,  49.89248841, 51.43708248, 51.98784791, 18.21313955, 17.58093937]
meanValsGT = [362.66540459]

inpDataMetricList = np.zeros((7))
outGTDataMetricList = np.zeros((1))

for i,subj in enumerate(subjList):
    sys.stdout.write("\r Subj {}/{}".format(i,trainLen))
    inpData = np.load("{}{}_20204_2_0.npy".format(fileDir,subj))
    outGT = loadmat("{}{}_20204_2_0.mat".format(t1MapDir,subj))['results']

    for j in range(7):
        inpDataMetricList[j] += np.sum((inpData[:,:,j]-meanValsInp[j])**2)
    outGTDataMetricList[0] += np.sum((outGT[:,:,0]-meanValsGT[0])**2)

stdValsInp = (inpDataMetricList/numPixels)**0.5
print(stdValsInp)
stdValsGT = (outGTDataMetricList/numPixels)**0.5
print(stdValsGT) 

stdValsInp = [33.52329497, 79.89539557, 83.93680164, 84.62633451, 84.78080839, 23.92935112, 29.45570738]
stdValsGT = [501.85027392]