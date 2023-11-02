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

def median_slice(seg):

    one_hot_encoded = np.where(seg != 0, 1, 0)

    slice_sums = np.sum(one_hot_encoded, axis=(0,1))
    median_value = np.median(slice_sums)

    median_indices = np.argwhere(slice_sums == median_value)
    median_index = median_indices[len(median_indices) // 2][0]

    return median_index

def plot_slices(images=None, seg=None, pred=None, nslice=None):
    n_rows = 0
    if images is not None:
        n_rows += 1
    if seg is not None:
        n_rows += 1
    if pred is not None:
        n_rows += 1

    if nslice is None:
        if seg is not None:
            n_slice = median_slice(seg)
        elif pred is not None:
            n_slice = median_slice(pred)
        else:
            n_slice=64

    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 5*n_rows))

    row = 0
    if images is not None:
        for i, suffix in enumerate(['t1c', 't1n', 't2f', 't2w']):
            axes[row, i].imshow(images[i][:, :, nslice], cmap='gray')
            axes[row, i].set_title(f'{suffix}')
            axes[row, i].axis('off')
        row += 1

    if seg is not None:
        seg_slice = seg[:, :, nslice]
        for i, region in enumerate(['NCR', 'ED', 'ET']):
            axes[row, i].imshow(np.where(seg_slice == i+1, i+1, 0), cmap='viridis', vmin=0, vmax=3)
            axes[row, i].set_title(f'{region} ground truth')
            axes[row, i].axis('off')
        axes[row, 3].imshow(seg_slice, cmap='viridis', vmin=0, vmax=3)
        axes[row, 3].set_title('All ground truth channels')
        axes[row, 3].axis('off')
        row += 1

    if pred is not None:
        pred_slice = pred[:, :, nslice]
        for i, region in enumerate(['NCR', 'ED', 'ET']):
            axes[row, i].imshow(np.where(pred_slice == i+1, i+1, 0), cmap='viridis', vmin=0, vmax=3)
            axes[row, i].set_title(f'{region} prediction')
            axes[row, i].axis('off')
        axes[row, 3].imshow(pred_slice, cmap='viridis', vmin=0, vmax=3)
        axes[row, 3].set_title('All prediction channels')
        axes[row, 3].axis('off')

    plt.tight_layout()
    return fig