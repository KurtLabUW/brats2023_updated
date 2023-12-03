"""This module contains functions for postprocessing predicted segmentations, to minimize false positives."""

import numpy as np
import cc3d

def simple_rm_dust(pred_mat):
    """Removes connected components (of each disjoint label) that are smaller than 50 voxels.

    Args:
        pred_mat: Array of shape HWD for a predicted segmentation.

    Returns:
        The postprocessed array.
    """
    pred_mat_rm_dust = cc3d.dust(pred_mat, threshold=50, connectivity=26)

    return pred_mat_rm_dust

def get_tissue_wise_seg(pred_mat, tissue_type):
    """Returns one-hot encoding of the specified overlapping region for a given prediction.

    Args:
        pred_mat: Array of shape HWD for a predicted segmentation.
        tissue_type: Either 'WT', 'TC' or 'ET' - the overlapping region to encode.

    Returns:
        The one-hot encoded array.
    """
    pred_mat_tissue_wise = np.zeros_like(pred_mat[0])

    if tissue_type == 'WT':
        pred_mat_tissue_wise = pred_mat > 0
    elif tissue_type == 'TC':
        pred_mat_tissue_wise = np.logical_or(pred_mat == 1, pred_mat == 3)
    elif tissue_type == 'ET':
        pred_mat_tissue_wise = pred_mat == 3

    return pred_mat_tissue_wise.astype(np.uint16)

def rm_tt_dust(pred_mat, tt):
    """Removes dust of an overlapping region from a predicted segmentation array.

    Args:
        pred_mat: Array of shape HWD for a predicted segmentation.
        tissue_type: Either 'WT', 'TC' or 'ET' - the overlapping region for which to remove dust.

    Returns:
        The mask corresponding to the voxels removed as dust.
    """
    pred_mat_tt = get_tissue_wise_seg(pred_mat, tt)
    pred_mat_tt_rm_dust = cc3d.dust(pred_mat_tt, threshold=50, connectivity=26)
    rm_dust_mask = np.logical_and(pred_mat_tt==1, pred_mat_tt_rm_dust==0)
    pred_mat[rm_dust_mask] = 0
    return rm_dust_mask

def fill_holes(pred_mat, tt, label, rm_dust_mask):
    """Fills holes created by removal of dust of an overlapping region, by relabelling voxels.

    Args:
        pred_mat: Array of shape HWD for a predicted segmentation.
        tissue_type: Either 'WT', 'TC' or 'ET' - the overlapping region to detect holes in.
        label: The value that holes should be relabelled to.
        rm_dust_mask: The mask corresponding to the voxels removed as dust.
    """
    pred_mat_tt = get_tissue_wise_seg(pred_mat, tt)
    tt_holes = 1 - pred_mat_tt
    tt_holes_rm = cc3d.dust(tt_holes, threshold=50, connectivity=26)
    tt_filled = 1 - tt_holes_rm
    holes_mask = np.logical_and(tt_filled==1, pred_mat==0) * rm_dust_mask
    pred_mat[holes_mask == 1] = label

def rm_dust_fh(pred_mat):
    """Iterates through each overlapping region (from ET to TC to WT), removing dust and filling any holes created in the overlapping region above it.

    Args:
        pred_mat: Array of shape HWD for a predicted segmentation.

    Returns:
        The postprocessed array.
    """

    rm_et_mask = rm_tt_dust(pred_mat, 'ET')
    fill_holes(pred_mat, 'TC', 1, rm_et_mask)

    rm_tc_mask = rm_tt_dust(pred_mat, 'TC')
    fill_holes(pred_mat, 'WT', 2, rm_tc_mask)

    _ = rm_tt_dust(pred_mat, 'WT')

    rm_tc_mask = rm_tt_dust(pred_mat, 'WT')

    return pred_mat