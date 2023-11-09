import nibabel as nib
import torch
import os
import numpy as np

def find_indicies_of_nonzeros(x):
    start_index = None
    end_index = None

    for i in range(len(x)):
        if x[i] != 0:
            if start_index is None:
                start_index = i
            end_index = i

    return start_index, end_index

if __name__ == '__main__':
    
    preds_dir = ''

    X, Y, Z = [], [], []

    for subject_name in os.listdir(preds_dir):
        modality_nifti = f'{subject_name}-t1n.nii.gz'
        modality_nifti_path = os.path.join(preds_dir, subject_name, modality_nifti)
        nifti = nib.load(modality_nifti_path)
        modality_arr = nifti.get_fdata()

        x = np.sum(modality_arr, axis=(1,2))
        y = np.sum(modality_arr, axis=(0,2))
        z = np.sum(modality_arr, axis=(0,1))

        x_indices = find_indicies_of_nonzeros(x)
        y_indices = find_indicies_of_nonzeros(y)
        z_indices = find_indicies_of_nonzeros(z)

        X.append(x_indices)
        Y.append(y_indices)
        Z.append(z_indices)

        print(x_indices, y_indices, z_indices)