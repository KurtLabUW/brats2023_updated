import torch.nn as nn
import os
from natsort import natsorted
import torch
from torchvision import transforms
from data import datasets, trans # in same directory, could rewrite these .py files to be cleaner too
import numpy as np
import matplotlib.pyplot as plt
from monai.metrics import HausdorffDistanceMetric

from utils import *

def validate(data_dir, ckpt_path, eval_regions='overlapping', out_dir=None, batch_size=1):

    # Set up directory.
    if out_dir is None:
        out_dir = os.getcwd()

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path)

    epoch = checkpoint['epoch']
    training_regions = checkpoint['training_regions']
    model_str = checkpoint['model_str']
    model = MODEL_STR_TO_FUNC[model_str]
    model.load_state_dict(checkpoint['model_sd'])
    loss_functs_str = checkpoint['loss_functs_str']
    loss_functs = [LOSS_STR_TO_FUNC[l] for l in loss_functs_str]
    loss_weights = checkpoint['loss_weights']

    # print(f"Loaded {model_str} model trained on {training_regions} regions for {epoch} epochs.")

    # print(f"Evaluating model on {eval_regions} regions.")

    print("---------------------------------------------------")
    print(f"TRAINING SUMMARY")
    print(f"Model: {model_str}")
    print(f"Loss functions: {loss_functs_str}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Training regions: {training_regions}")
    print(f"Epochs trained: {epoch}")
    print("---------------------------------------------------")
    print("VALIDATION SUMMARY")
    print(f"Evaluation regions: {eval_regions}")
    print(f"Data directory: {data_dir}")
    print(f"Out directory: {out_dir}")
    print("---------------------------------------------------")

    val_loader = make_dataloader(data_dir, shuffle=False, mode='train')

    val_loss_vals = []

    print('Validation starts.')
    with torch.no_grad():
        for _, imgs, seg in val_loader:

            model.eval()

            # Move data to GPU.
            imgs = [img.cuda() for img in imgs]
            seg = seg.cuda()

            # Split segmentation into 3 channels.
            seg = seg_to_one_hot_channels(seg)

            if training_regions == 'overlapping':
                seg_train = disjoint_to_overlapping(seg)
            elif training_regions == 'disjoint':
                seg_train = seg

            x_in = torch.cat(imgs, dim=1)
            output = model(x_in)
            output = output.float()

            val_loss = compute_loss(output, seg_train, loss_functs, loss_weights)
            val_loss_vals.append(val_loss.detach().cpu())

            preds = probs_to_preds(output, training_regions)

            print(seg.shape, output.shape, preds.shape)

            if eval_regions == 'overlapping':
                # Convert seg and pred to 3 channels corresponding to overlapping regions
                seg_eval = disjoint_to_overlapping(seg)
                preds_eval = disjoint_to_overlapping(preds)
                
            elif eval_regions == 'disjoint':
                # Convert seg and pred to 3 channels corresponding to disjoint regions
                seg_eval = seg
                preds_eval = preds

            print(seg_eval.shape, preds_eval.shape)

            # Compute metrics between seg_eval and preds_eval.

if __name__ == '__main__':

    data_dir = '/mmfs1/home/ehoney22/debug_data/train'
    ckpt_path = '/mmfs1/home/ehoney22/debug/saved_ckpts/epoch20.pth.tar'
    out_dir = '/mmfs1/home/ehoney22/debug'

    validate(data_dir, ckpt_path, out_dir=out_dir)