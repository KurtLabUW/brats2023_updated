import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from models import unet3d
from data import datasets, trans
import numpy as np
import nibabel as nib
import os

LOSS_STR_TO_FUNC = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.CrossEntropyLoss(),
    # 'edge-loss': EdgeLoss3D.GMELoss3D(),
    # 'dice': lf.DiceLoss(),
    # 'focal': lf.FocalLoss()
    # 'hd'
}

MODEL_STR_TO_FUNC = {
    'unet3d': unet3d.U_Net3d()
}

def make_dataloader(data_dir, shuffle, mode, batch_size):
    transforms_composed = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32))
                                              ])
    dataset = datasets.BratsDataset(data_dir, transforms=transforms_composed, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)
    return dataloader

def exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate):
    """Exponentially decays learning rate of optimizer at given epoch."""
    lr = init_lr * (decay_rate ** (epoch-1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def seg_to_one_hot_channels(seg):
    """Converts segmentation to 3 channels, each a one-hot encoding of a tumour region label."""
    B,_,H,W,D = seg.shape
    seg3 = torch.zeros((B,3,H,W,D))
    for channel_value in [1,2,3]:
        seg3[:, channel_value-1, :, :, :] = (seg == channel_value).type(torch.float)
    return seg3

def disjoint_to_overlapping(seg):
    """Converts tensor channels from representing disjoint regions to overlapping ones."""
    mask = torch.zeros_like(seg)
    mask[:,0] = seg[:, 0] + seg[:, 1] + seg[:, 2] #WHOLE TUMOR
    mask[:,1] = seg[:, 0] + seg[:, 2] #TUMOR CORE
    mask[:,2] = seg[:, 2] #ENHANCING TUMOR
    return mask

def reshape_input(input):
    out = np.zeros((240, 240, 155))
    out[56:184,24:216,14:142] = input 
    return out

def overlapping_probs_to_preds(output, t1=0.45, t2=0.4, t3=0.45):
    output_ = np.squeeze(output.cpu().detach().numpy())
    c1, c2, c3 = output_[0] > t1, output_[1] > t2, output_[2] > t3
    pred = (c1 > 0).astype(np.uint8) # NCR
    pred[(c2 == False) * (c1 == True)] = 2 # ED
    pred[(c3 == True) * (c1 == True)] = 3 # ET
    output_plot = np.zeros_like(output_)
    output_plot[0] = (pred == 1) #NCR
    output_plot[1] = (pred == 2) #ED
    output_plot[2] = (pred == 3) #ET
    output_plot = output_plot.astype(np.uint8)
    return output_plot

def disjoint_probs_to_preds(output, t=0.5):
    output_ = np.squeeze(output.cpu().detach().numpy())
    c1, c2, c3 = output_[0], output_[1], output_[2]
    max_label = np.maximum(np.maximum(c1, c2), c3)
    pred = np.zeros_like(output_)
    pred[0] = np.where(c1 < max_label, 0, max_label)
    pred[1] = np.where(c2 < max_label, 0, max_label)
    pred[2] = np.where(c3 < max_label, 0, max_label)
    output_plot = np.zeros_like(output_)
    for c in range(0, 3):
        output_plot[c] = np.where(pred[c] > t, 1., 0.)
    output_plot = output_plot.astype(np.uint8)
    return output_plot

def probs_to_preds(output, training_regions):

    if training_regions == 'overlapping':
        pred = overlapping_probs_to_preds(output)
    elif training_regions == 'disjoint':
        pred = disjoint_probs_to_preds(output)

    return pred

def fetch_affine_header(subject_name, data_dir):

    modality_nifti_filename = f'{subject_name}-t1c.nii.gz'
    modality_nifti_path = os.path.join(data_dir, subject_name, modality_nifti_filename)
    nifti = nib.load(modality_nifti_path)
    
    return nifti.affine, nifti.header

def save_pred_as_nifti(pred, save_dir, data_dir, subject_name):

    # Convert back from 3 one-hot encoded channels to 1 channel with 3 tumour region labels
    pred_for_nifti = np.zeros_like(pred[0])
    pred_for_nifti = pred[0] + pred[1]*2 + pred[2]*3
    pred_for_nifti = np.squeeze(pred_for_nifti)
    pred_for_nifti = reshape_input(pred_for_nifti)

    affine, header = fetch_affine_header(subject_name, data_dir)
    pred_nifti = nib.nifti1.Nifti1Image(pred_for_nifti, affine=affine, header=header)
    filename = f'{subject_name}.nii.gz'
    nib.nifti1.save(pred_nifti, os.path.join(save_dir, filename))