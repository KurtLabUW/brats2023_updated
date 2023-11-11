import os
from natsort import natsorted
import torch
from torchvision import transforms
from data import datasets, trans # in same directory, could rewrite these .py files to be cleaner too
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

from utils import *
from postprocess import *

def infer(data_dir, ckpt_path, out_dir=None, batch_size=1, postprocess_function=None):

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    preds_dir = os.path.join(out_dir, 'preds')
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
        os.system(f'chmod a+rwx {preds_dir}')

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path)

    epoch = checkpoint['epoch']
    training_regions = checkpoint['training_regions']
    model_str = checkpoint['model_str']
    model = MODEL_STR_TO_FUNC[model_str]
    model.load_state_dict(checkpoint['model_sd'])

    print(f"Loaded {model_str} model trained on {training_regions} regions for {epoch} epochs.")

    test_loader = make_dataloader(data_dir, shuffle=False, mode='test', batch_size=batch_size)

    print('Inference starts.')
    with torch.no_grad():
        for subject_names, imgs in test_loader:

            model.eval()

            # Move data to GPU.
            imgs = [img.cuda() for img in imgs]

            x_in = torch.cat(imgs, dim=1)
            output = model(x_in)
            output = output.float()

            preds = probs_to_preds(output, training_regions)
            # preds is B3HWD

            # Iterate over batch and save each prediction.
            for i, subject_name in enumerate(subject_names):
                save_pred_as_nifti(preds[i], preds_dir, data_dir, subject_name, postprocess_function)

    print(f'Inference completed. Predictions saved in {preds_dir}.')

if __name__ == '__main__':

    data_dir = '/mmfs1/home/ehoney22/debug_data/test'
    ckpt_path = '/mmfs1/home/ehoney22/debug/saved_ckpts/epoch20.pth.tar'
    out_dir = '/mmfs1/home/ehoney22/debug'
    postprocess_function = rm_dust_fh

    infer(data_dir, ckpt_path, out_dir=out_dir, postprocess_function=postprocess_function)