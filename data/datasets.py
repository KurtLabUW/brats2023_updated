from torch.utils.data import Dataset
import os
import nibabel as nib
from preprocess import znorm_rescale
import numpy as np
import torch

class BratsDataset(Dataset):
    def __init__(self, data_dir, transforms, mode):
        self.data_dir = data_dir
        self.subject_list = os.listdir(data_dir)
        self.transforms = transforms
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

        case_info = subject_name.split('-', maxsplit=2)[-1]
        affine, header = modality_nifti.affine, modality_nifti.header
        extra_info = (case_info, affine, header)

        if self.mode == 'train':
            seg_nifti = self.load_nifti(subject_name, 'seg')
            seg_data = seg_nifti.get_fdata()
            return modalities_data, seg_data, extra_info
        elif self.mode == 'test':
            return modalities_data, extra_info
    
    def __getitem__(self, idx):
        subject_name = self.subject_list[idx]

        # Load the data and extra info.
        if self.mode == 'train':
            imgs, seg, extra_info = self.load_subject_data(subject_name)
        elif self.mode == 'test':
            imgs, extra_info = self.load_subject_data(subject_name)

        # Do Z-score norm and rescaling preprocessing.
        imgs = [znorm_rescale(img) for img in imgs]

        # Do transformations of data.
        imgs = [x[None, ...] for x in imgs]
        imgs = self.transforms(imgs)

        # Convert to torch tensors.
        imgs = [np.ascontiguousarray(x) for x in imgs]
        imgs = [torch.from_numpy(x) for x in imgs]

        # If train, process segmentation similarly.
        if self.mode == 'train':
            seg = seg[None, ...]
            seg = self.transforms(seg)
            seg = np.ascontiguousarray(seg)
            seg = torch.from_numpy(seg)

            return imgs, seg, extra_info
        
        elif self.mode == 'test':
            return imgs, extra_info