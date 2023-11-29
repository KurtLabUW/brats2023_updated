import numpy as np
import cc3d

def quick_rm_dust(pred_mat):

    pred_mat_rm_dust = cc3d.dust(pred_mat, threshold=50, connectivity=26)

    return pred_mat_rm_dust

def get_tissue_wise_seg(pred_mat, tissue_type):

    pred_mat_tissue_wise = np.zeros_like(pred_mat[0])

    if tissue_type == 'WT':
        pred_mat_tissue_wise = pred_mat > 0
    elif tissue_type == 'TC':
        pred_mat_tissue_wise = np.logical_or(pred_mat == 1, pred_mat == 3)
    elif tissue_type == 'ET':
        pred_mat_tissue_wise = pred_mat == 3

    return pred_mat_tissue_wise.astype(np.uint16)

def rm_dust(pred_mat, tt):
    pred_mat_tt = get_tissue_wise_seg(pred_mat, tt)
    pred_mat_tt_rm_dust = cc3d.dust(pred_mat_tt, threshold=50, connectivity=26)
    rm_dust_mask = np.logical_and(pred_mat_tt==1, pred_mat_tt_rm_dust==0)
    pred_mat[rm_dust_mask] = 0
    return rm_dust_mask

def fill_holes(pred_mat, tt, label, rm_dust_mask):
    pred_mat_tt = get_tissue_wise_seg(pred_mat, tt)
    tt_holes = 1 - pred_mat_tt
    tt_holes_rm = cc3d.dust(tt_holes, threshold=50, connectivity=26)
    tt_filled = 1 - tt_holes_rm
    holes_mask = np.logical_and(tt_filled==1, pred_mat==0) * rm_dust_mask
    pred_mat[holes_mask == 1] = label

def rm_dust_fh(pred_mat):
    # Receives prediction as HWD with labels for NCR, ED, ET

    rm_et_mask = rm_dust(pred_mat, 'ET')
    fill_holes(pred_mat, 'TC', 1, rm_et_mask)

    rm_tc_mask = rm_dust(pred_mat, 'TC')
    fill_holes(pred_mat, 'WT', 2, rm_tc_mask)

    _ = rm_dust(pred_mat, 'WT')

    rm_tc_mask = rm_dust(pred_mat)

    return pred_mat

def OLD_rm_dust_fh(pred_mat):

    pred_mat_new = pred_mat.copy()

    pred_mat_et = get_tissue_wise_seg(pred_mat_new, 'ET')
    pred_mat_et_rm_dust = cc3d.dust(pred_mat_et, threshold=50, connectivity=26)
    rm_et_mask = np.logical_and(pred_mat_et==1, pred_mat_et_rm_dust==0)
    pred_mat_new[rm_et_mask] = 0

    pred_mat_tc = get_tissue_wise_seg(pred_mat_new, 'TC')
    tc_holes = 1 - pred_mat_tc
    tc_holes_rm = cc3d.dust(tc_holes, threshold=50, connectivity=26)
    tc_filled = 1 - tc_holes_rm
    fill_ncr_mask = np.logical_and(tc_filled==1, pred_mat_new==0) * rm_et_mask
    pred_mat_new[fill_ncr_mask] = 1 #Fill holes with NCR

    pred_mat_tc = get_tissue_wise_seg(pred_mat_new, 'TC')
    pred_mat_tc_rm_dust = cc3d.dust(pred_mat_tc, threshold=50, connectivity=26)
    rm_tc_mask = np.logical_and(pred_mat_tc==1, pred_mat_tc_rm_dust==0)
    pred_mat_new[rm_tc_mask] = 0

    pred_mat_wt = get_tissue_wise_seg(pred_mat_new, 'WT')
    wt_holes = 1 - pred_mat_wt
    wt_holes_rm = cc3d.dust(wt_holes, threshold=50, connectivity=26)
    wt_filled = 1- wt_holes_rm
    fill_ed_mask = np.logical_and(wt_filled==1, pred_mat_new==0) * rm_tc_mask
    pred_mat_new[fill_ed_mask] = 2 #Fill holes with ED

    pred_mat_wt = get_tissue_wise_seg(pred_mat_new, 'WT')
    pred_mat_wt_rm_dust = cc3d.dust(pred_mat_wt, threshold=50, connectivity=26)
    rm_wt_mask = np.logical_and(pred_mat_wt==1, pred_mat_wt_rm_dust==0)
    pred_mat_new[rm_wt_mask] = 0

    return pred_mat_new