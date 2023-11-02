import os
from natsort import natsorted
import torch
from torchvision import transforms
from data import datasets, trans # in same directory, could rewrite these .py files to be cleaner too
import numpy as np
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import argparse
from utils import *


import matplotlib.pyplot as plt
import numpy as np

def plot_slices(images=None, seg=None, pred=None, nslice=None):
    n_rows = 0
    if images is not None:
        n_rows += 1
    if seg is not None:
        n_rows += 1
    if pred is not None:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 5*n_rows))

    row = 0
    if images is not None:
        for i, suffix in enumerate(['t1c', 't1n', 't2f', 't2w']):
            axes[row, i].imshow(images[i][:, :, nslice], cmap='gray')
            axes[row, i].set_title(f'{suffix} modality')
            axes[row, i].axis('off')
        row += 1

    if seg is not None:
        for i in range(3):
            axes[row, i].imshow(seg[i, :, :, nslice], vmin=0, vmax=1)
            axes[row, i].set_title(f'Segmentation Channel {i+1}')
            axes[row, i].axis('off')
        seg_combined = seg[0] + 2*seg[1] + 3*seg[2]
        axes[row, 3].imshow(seg_combined[:, :, nslice], vmin=0, vmax=1)
        axes[row, 3].set_title('All Segmentation Channels')
        axes[row, 3].axis('off')
        row += 1

    if pred is not None:
        for i in range(3):
            axes[row, i].imshow(pred[i, :, :, nslice], vmin=0, vmax=1)
            axes[row, i].set_title(f'Prediction Channel {i+1}')
            axes[row, i].axis('off')
        pred_combined = pred[0] + 2*pred[1] + 3*pred[2]
        axes[row, 3].imshow(pred_combined[:, :, nslice], vmin=0, vmax=1)
        axes[row, 3].set_title('All Prediction Channels')
        axes[row, 3].axis('off')

    plt.tight_layout()
    return fig