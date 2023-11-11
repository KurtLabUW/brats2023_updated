from torch.utils.data import Dataset
import os
import nibabel as nib
from brats_clean.processing.preprocess import znorm_rescale, center_crop
import numpy as np
import torch

class BratsDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.subject_list = os.listdir(data_dir)
        self.mode = mode

    def __len__(self):
        return len(self.subject_list)
    
    def load_nifti(self, subject_name, suffix):
        """Loads nifti file for given subject and suffix."""

        nifti_filename = f'{subject_name}-{suffix}.nii.gz'
        nifti_path = os.path.join(self.data_dir, subject_name, nifti_filename)
        nifti = nib.load(nifti_path)
        return nifti
    
    def load_subject_data(self, subject_name):
        """Loads images, segmentation (if training) and extra info for a subject."""

        modalities_data = []
        for suffix in ['t1c', 't1n', 't2f', 't2w']:
            modality_nifti = self.load_nifti(subject_name, suffix)
            modality_data = modality_nifti.get_fdata()
            modalities_data.append(modality_data)

        if self.mode == 'train':
            seg_nifti = self.load_nifti(subject_name, 'seg')
            seg_data = seg_nifti.get_fdata()
            return modalities_data, seg_data
        elif self.mode == 'test':
            return modalities_data
    
    def __getitem__(self, idx):
        subject_name = self.subject_list[idx]

        # Load the data and extra info.
        if self.mode == 'train':
            imgs, seg = self.load_subject_data(subject_name)
        elif self.mode == 'test':
            imgs = self.load_subject_data(subject_name)

        # Do Z-score norm and rescaling preprocessing.
        imgs = [znorm_rescale(img) for img in imgs]

        # Perform center crop.
        imgs = [center_crop(img) for img in imgs]

        imgs = [x[None, ...] for x in imgs]
        imgs = [np.ascontiguousarray(x, dtype=np.float32) for x in imgs]

        # Convert to torch tensors.
        imgs = [torch.from_numpy(x) for x in imgs]

        # If train mode, process segmentation similarly.
        if self.mode == 'train':
            seg = center_crop(seg)
            seg = seg[None, ...]
            seg = np.ascontiguousarray(seg)
            seg = torch.from_numpy(seg)

            return subject_name, imgs, seg
        
        elif self.mode == 'test':
            return subject_name, imgs