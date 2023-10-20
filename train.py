import glob, os
from natsort import natsorted
import attention_unet # in same directory
import numpy as np
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
# make sure 'models' folder is in same directory for next import to run
from data import datasets, trans # in same directory, could rewrite these .py files to be cleaner too
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import losses as lf2 # in same directory
import EdgeLoss3D # in same directory
from models import unet3d


def main():

    data_dir = '' # Directory containing for each subject a folder of nifti files
    training_regions = 'overlapping' # or 'disjoint'
    lr = pass
    batch_size = pass
    loss = pass
    model = unet3d.U_Net3d()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)

    print(f"Training on {training_regions} regions")

    train_composed = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32))
                                              ])
    train_set = datasets.BratsDataset(data_dir, transforms=train_composed, mode='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    

    for epoch in range(epoch_start, max_epoch+1):


        for imgs, seg, info in train_loader:

            model.train()
