
import numpy as np
import nibabel as nib
from nilearn.image import load_img, smooth_img
from numpy import std, mean, sqrt

def get_T2_bg_img_controls(mask_size, nifti_masker):
    """
    returns the mean image of the masked T2 images for a set of healthy controls.
    """
    with open(f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/brain_age_disease_diff/T2_models/seed_2/control/SmoothGrad/eids.npy", "rb") as file:
        control_eids = np.load(file)

    imgs_masked = np.zeros((len(control_eids), mask_size))

    for i, eid in enumerate(control_eids):
        eid = int(eid)
        img = np.load(f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/lin_reg_imgs/brain_age_dis_diff/T2/controls/{eid}.npy")
        imgs_masked[i, :] = img

    mean_img = np.mean(imgs_masked, axis = 0)

    T2_bg_img = nifti_masker.inverse_transform(mean_img)

    return T2_bg_img

#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    
    return (mean(x, axis=0) - mean(y, axis=0)) / sqrt(((nx-1)*std(x, ddof=1, axis=0) ** 2 + (ny-1)*std(y, ddof=1, axis=0) ** 2) / dof)

def get_regressors_and_masked_expls_faster(n_dis, mask_size, current_dis_eids, current_control_eids, ukbb_df, dis_eids, dis_expls, control_eids, control_expls, nifti_masker,
                                           fwhm, scale_percentile, clip_percentile, sign):
    """
    Prepares the regressors and masked explanations for investigating disease based differences in brain age explanations using mass univariate analysis (nilearn.mass_univariate.permuted_ols).
    """
                                        
    dis_xpl_indices = []
    for eid in current_dis_eids:
        index = list(dis_eids).index(eid)
        dis_xpl_indices.append(index)

    control_xpl_indices = []
    for eid in current_control_eids:
        index = list(control_eids).index(eid)
        control_xpl_indices.append(index)

    dis_xpls = dis_expls[dis_xpl_indices, :]
    control_xpls = control_expls[control_xpl_indices, :]

    masked_expls = np.vstack((dis_xpls, control_xpls))

    # sign
    if sign == "abs":
        masked_expls = np.abs(masked_expls)
    elif sign == "pos":
        masked_expls[masked_expls < 0] = 0


    # smooth
    expl_imgs = nifti_masker.inverse_transform(np.float32(masked_expls))
    smoothed_expl_imgs = smooth_img(expl_imgs, fwhm)
    masked_expls = nifti_masker.transform(smoothed_expl_imgs)

    # scale
    masked_expls = masked_expls/np.percentile(masked_expls, scale_percentile, axis=1)[:,None]

    # clip
    clip_val = np.percentile(masked_expls, clip_percentile, axis=1)[:,None]
    masked_expls = np.clip(masked_expls, a_max=clip_val, a_min=(-clip_val))

    regressors = np.zeros((len(current_dis_eids) + len(current_control_eids), 3))

    bags_dis = []
    bags_cn = []
    for i, eid in enumerate(current_dis_eids + current_control_eids):
        age = ukbb_df[ukbb_df["f.eid"]==eid]["f.21003.2.0"].values[0]
        sex = int(ukbb_df[ukbb_df["f.eid"]==eid]["f.31.0.0"].values[0])
        pred = int(ukbb_df[ukbb_df["f.eid"]==eid]["predictions"].values[0])
        bag = pred - age
        
        regressors[i, 1] = age
        regressors[i, 2] = sex
        # regressors[i, 3] = bag
        # regressors[i, 4] = np.abs(bag)

        if i < n_dis:
            regressors[i, 0] = 1
            bags_dis.append(bag)

        else:
            regressors[i, 0] = 0
            bags_cn.append(bag)


    d = cohen_d(np.array(bags_dis), np.array(bags_cn))
    # print(f"effect size: {d}")
        
    return masked_expls, regressors

def get_mni_bg_img():
    img_bg_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI152_T1_2mm_brain.nii.gz"
    bg_img = load_img(img_bg_path)
    bg_img = nib.Nifti1Image(bg_img.get_fdata(), np.eye(4))
    return bg_img