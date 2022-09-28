# GAN_T1_OUT

This repository contains the scripts necessary to train and test the cGAN (GAN_T1_OUT) on T1-mapping MRI data from UK Biobank (available through application to [UKBiobank](https://www.ukbiobank.ac.uk/))

## train_T1.py and train_T1_continued.py

The train_T1 and train_T1_continued script allow for commencement and continuation of training of a UNet for output of T1 Maps from MOLLI T1-mapping datasets. The following command line arguments can be used for train_T1:

```txt
usage: train_T1.py [-h] [--dir FILEDIR] [--t1dir T1MAPDIR] --model_name MODELNAME [--load] [-lr LR] [-b1 B1] [-bSize BATCHSIZE] [-nE NUMEPOCHS] [--step_size STEPSIZE] [--norm] [-trainS TRAINSIZE] [-valS VALSIZE] [-testS TESTSIZE]

Training program for T1 map generation

options:
  -h, --help            show this help message and exit
  --dir FILEDIR         File directory for numpy images
  --t1dir T1MAPDIR      File directory for T1 matfiles
  --model_name MODELNAME
                        Name for saving the model
  --load                Load the preset trainSets, or redistribute (Bool)
  -lr LR                Learning rate for the optimizer
  -b1 B1                Beta 1 for the Adam optimizer
  -bSize BATCHSIZE, --batch_size BATCHSIZE
                        Batch size for dataloader
  -nE NUMEPOCHS, --num_epochs NUMEPOCHS
                        Number of Epochs to train for
  --step_size STEPSIZE  Step size for learning rate decay
  --norm                Normalise the input images and T1 map
  -trainS TRAINSIZE, --train_size TRAINSIZE
                        Size of the training dataset, if load, will be ignored
  -valS VALSIZE, --val_size VALSIZE
                        Size of the validation dataset, if load, will be ignored
  -testS TESTSIZE, --test_size TESTSIZE
                        Size of the test dataset, if load, will be ignored
```

## train_full.py and train_full_continued.py

The train_full and train_full_continued script allow for commencement and continuation of training of a Generator for full data reconstruction from partial MOLLI datasets, given a model for minimisation of T1 map error (trained in train_T1.py). The following command line arguments can be used for train_full:

```txt
usage: train_full.py [-h] [--dir FILEDIR] [--t1dir T1MAPDIR] --model_name MODELNAME [--load] [-lr LR] [-b1 B1] [-bSize BATCHSIZE] [-nE NUMEPOCHS] [--step_size STEPSIZE] [--gui GUI]

Training program for T1 map generation

options:
  -h, --help            show this help message and exit
  --dir FILEDIR         File directory for numpy images
  --t1dir T1MAPDIR      File directory for T1 matfiles
  --model_name MODELNAME
                        Name for saving the model
  --load                Load the preset trainSets, or redistribute (Bool)
  -lr LR                Learning rate for the optimizer
  -b1 B1                Beta 1 for the Adam optimizer
  -bSize BATCHSIZE, --batch_size BATCHSIZE
                        Batch size for dataloader
  -nE NUMEPOCHS, --num_epochs NUMEPOCHS
                        Number of Epochs to train for
  --step_size STEPSIZE  Step size for learning rate decay
  --gui GUI             Use GUI to pick out parameters (WIP)
```

## test_T1.py and test_full.py

The test_T1 and test_full python scripts output loss curves and figures pulled from the test set provided


## datasets.py

The datasets script contains the dataset class for loading shMOLLI images (numpy arr) and the corresponding T1 maps (.mat) files for these images, to allow training of the GAN_T1_OUT model. Adaptation of this dataset file should allow the user to perform similar transformations between one dataset and a target dataset as performed in [GAN_PDFF_OUT](https://github.com/hillce/GAN_PDFF_OUT)

## models.py

Contained within models.py are the Generator and Discriminator classes used for both T1 map generation (train.py) and Dataset regeneration (train_full.py)
