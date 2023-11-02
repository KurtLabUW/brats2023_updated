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

def make_dataloader(data_dir, shuffle, mode, batch_size=1):
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

def disjoint_to_overlapping(seg_disjoint):
    """Converts tensor representing one-hot encoding of disjoint regions to that of overlapping ones."""
    seg_overlapping = torch.zeros_like(seg_disjoint)
    seg_overlapping[:,0] = seg_disjoint[:, 0] + seg_disjoint[:, 1] + seg_disjoint[:, 2] #WHOLE TUMOR
    seg_overlapping[:,1] = seg_disjoint[:, 0] + seg_disjoint[:, 2] #TUMOR CORE
    seg_overlapping[:,2] = seg_disjoint[:, 2] #ENHANCING TUMOR
    return seg_overlapping

def reshape_input(input):
    out = np.zeros((240, 240, 155))
    out[56:184,24:216,14:142] = input 
    return out

def overlapping_probs_to_preds(output, t1=0.45, t2=0.4, t3=0.45):
    output = output.cpu().detach()
    c1, c2, c3 = output[:, 0] > t1, output[:, 1] > t2, output[:, 2] > t3
    preds = (c1 > 0).to(torch.uint8) # NCR
    preds[(c2 == False) * (c1 == True)] = 2 # ED
    preds[(c3 == True) * (c1 == True)] = 3 # ET
    output_plot = torch.zeros_like(output)
    output_plot[:, 0] = (preds == 1).to(torch.uint8) #NCR
    output_plot[:, 1] = (preds == 2).to(torch.uint8) #ED
    output_plot[:, 2] = (preds == 3).to(torch.uint8) #ET
    output_plot = output_plot.to(torch.uint8)
    return output_plot

def disjoint_probs_to_preds(output, t=0.5):
    output = output.cpu().detach()
    c1, c2, c3 = output[:, 0], output[:, 1], output[:, 2]
    max_label = torch.max(torch.max(c1, c2), c3)
    preds = torch.zeros_like(output)
    preds[:, 0] = torch.where(c1 < max_label, torch.tensor(0), max_label)
    preds[:, 1] = torch.where(c2 < max_label, torch.tensor(0), max_label)
    preds[:, 2] = torch.where(c3 < max_label, torch.tensor(0), max_label)
    output_plot = torch.zeros_like(output)
    for c in range(0, 3):
        output_plot[:, c] = torch.where(preds[:, c] > t, torch.tensor(1.), torch.tensor(0.))
    output_plot = output_plot.to(torch.uint8)
    return output_plot

def probs_to_preds(output, training_regions):

    if training_regions == 'overlapping':
        preds = overlapping_probs_to_preds(output)
    elif training_regions == 'disjoint':
        preds = disjoint_probs_to_preds(output)

    return preds

def fetch_affine_header(subject_name, data_dir):

    modality_nifti_filename = f'{subject_name}-t1c.nii.gz'
    modality_nifti_path = os.path.join(data_dir, subject_name, modality_nifti_filename)
    nifti = nib.load(modality_nifti_path)
    
    return nifti.affine, nifti.header

def one_hot_channels_to_three_labels(pred):
    return pred[0] + pred[1]*2 + pred[2]*3

def save_pred_as_nifti(pred, save_dir, data_dir, subject_name, postprocess_function=None):

    # Convert back from 3 one-hot encoded channels to 1 channel with 3 tumour region labels
    pred = np.array(pred)
    pred_for_nifti = one_hot_channels_to_three_labels(pred)
    pred_for_nifti = np.squeeze(pred_for_nifti)
    pred_for_nifti = reshape_input(pred_for_nifti)
    pred_for_nifti = pred_for_nifti.astype(np.uint8)

    if postprocess_function:
        pred_for_nifti = postprocess_function(pred_for_nifti)

    affine, header = fetch_affine_header(subject_name, data_dir)
    pred_nifti = nib.nifti1.Nifti1Image(pred_for_nifti, affine=affine, header=header)
    filename = f'{subject_name}.nii.gz'
    nib.nifti1.save(pred_nifti, os.path.join(save_dir, filename))

def compute_loss(output, seg, loss_functs, loss_weights):
    # Compute weighted loss, summed across each region.
    loss = 0.
    for n, loss_function in enumerate(loss_functs):      
        temp = 0
        for i in range(3):
            temp += loss_function(output[:,i:i+1].cuda(), seg[:,i:i+1].cuda())

        loss += temp * loss_weights[n]
    return loss