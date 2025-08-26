import numpy as np
import nibabel as nib
from nilearn.image import smooth_img
import scipy.sparse as sp
from nilearn.image import load_img
import skimage
import nilearn
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from nilearn.plotting import plot_stat_map, show
from nilearn.plotting.displays import MosaicSlicer
import pandas as pd
import json
import random
import os
from pathlib import Path

def get_ground_truth_regions(path_to_metanalysis_csv):

    # mapping 
    mapping = {
    "Cerebral Cor":"Cortex",
    "Cerebral WM":"CerebralWhiteMatter",
    "Lat Vent":"Lateral-Ventricle",
    "Inf Lat Vent":"Inf-Lat-Vent",
    "Cerebel WM":"Cerebellum-White-Matter",
    "Cerebel Cor":"Cerebellum-Cortex",
    "Thalamus":"Thalamus-Proper",
    "Caudate":"Caudate",
    "Putamen":"Putamen",
    "Pallidum":"Pallidum",
    "Hippocampus":"Hippocampus",
    "Amygdala":"Amygdala",
    "Accumbens":"Accumbens-area",
    "3rd Vent":"3rd-Ventricle",
    "4th Vent":"4th-Ventricle",
    "Brain Stem":"Brain-Stem",
    "CSF":"CSF",
    }

    ### get ground truth aging table
    ground_truth_df = pd.read_csv(path_to_metanalysis_csv, index_col=0, dtype= {'18–29\nto\n30–39': np.float64, '30–39\nto\n40–49': np.float64, '40–49\nto\n50–59': np.float64,'50–59\nto\n60–69': np.float64, '60–69\nto\n70–79': np.float64, '70–79\nto\n80–95': np.float64, '18–29\nto\n80–95': np.float64})

    for column in ['18–29\nto\n30–39','30–39\nto\n40–49','40–49\nto\n50–59','50–59\nto\n60–69','60–69\nto\n70–79','70–79\nto\n80–95','18–29\nto\n80–95']:
        ground_truth_df[column]=ground_truth_df[column] * 0.01

    ground_truth_df["ukbb age range"] = (1*(1 + ground_truth_df["30–39\nto\n40–49"].values)*(1 + ground_truth_df["40–49\nto\n50–59"].values)*(1 + ground_truth_df["50–59\nto\n60–69"].values)*(1 + ground_truth_df["60–69\nto\n70–79"].values)*(1 + ground_truth_df["70–79\nto\n80–95"].values)) - 1

    ground_truth_df["ukbb age range"] = np.abs(ground_truth_df["ukbb age range"])

    ground_truth_df = ground_truth_df.sort_values(by=["ukbb age range"], ascending = False)

    gt_list = ground_truth_df.index.tolist()

    for key in mapping:
        idx = gt_list.index(key)
        gt_list[idx] = mapping[key]

    return gt_list

def process_xpls_for_scoring(expls, nifti_masker, fwhm_expls, cutoff_mode = "no_cutoff", scale_perc = 99, scale = True):
    # abs oder square
    expls = np.nan_to_num(expls)

    if cutoff_mode == "squared":
        expls = np.square(expls)
    else:
        expls = np.abs(expls)

    # to img space
    expl_imgs = nifti_masker.inverse_transform(np.float32(expls))
    
    # smooth
    expl_imgs = smooth_img(expl_imgs, fwhm_expls)
    
    expl_imgs_data = expl_imgs.get_fdata()

    # scale in participant space
    if scale:
        scale_percentiles = np.percentile(expl_imgs_data, scale_perc, axis = [0,1,2])
        expl_imgs_data = expl_imgs_data/scale_percentiles

    # cutoffs
    if cutoff_mode in ["80th", "90th", "95th", "99th"]:
        if cutoff_mode == "80th":
            cutoff_values = np.percentile(expl_imgs_data, 80, axis = [0,1,2])

        if cutoff_mode == "90th":
            cutoff_values = np.percentile(expl_imgs_data, 90, axis = [0,1,2])

        if cutoff_mode == "95th":
            cutoff_values = np.percentile(expl_imgs_data, 95, axis = [0,1,2])

        if cutoff_mode == "99th":
            cutoff_values = np.percentile(expl_imgs_data, 99, axis = [0,1,2])

        mask = expl_imgs_data < cutoff_values
        expl_imgs_data[mask] = 0

    expl_imgs = nib.Nifti1Image(expl_imgs_data, np.eye(4))
    expls = nifti_masker.transform(expl_imgs)
    
    return expls


def get_roi_atlas_from_atlas_keys(atlas_keys, warped_rois_path, eid, masked_shape):
    # get_atlas:
    roi_atas = np.zeros(masked_shape)
    for i, atlas_key in enumerate(atlas_keys):

        roi_eid_atlas_path = f"{warped_rois_path}/{eid}/{atlas_key}.npz"
        loaded_sparse = sp.load_npz(roi_eid_atlas_path)

        masked_roi_atlas = loaded_sparse.toarray()[0,:]
        roi_atas += masked_roi_atlas

    roi_atlas = np.clip(roi_atas, 0, 1)

    return roi_atlas


def get_mni2mm_bg():
    img_bg_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI152_T1_2mm_brain.nii.gz"
    bg_img = load_img(img_bg_path)
    bg_img = nib.Nifti1Image(bg_img.get_fdata(), np.eye(4))
    return bg_img

def discard_worse_lateral_roi(roi_names):
    kept_rois = []
    for roi in roi_names:
        roi = roi.replace(" (left hemisphere)", "")
        roi = roi.replace(" (right hemisphere)", "")
        roi = roi.replace(" (whole brain)", "")

        if roi not in kept_rois:
            kept_rois.append(roi)

    return kept_rois

def get_lesion_mask_img(lesion_mask, dilation_iter):
    lesion_mask[lesion_mask>0]=1
    if dilation_iter > 0:
        lesion_mask = ndimage.binary_dilation(lesion_mask, iterations=dilation_iter)
    rescaled_lesion_mask = skimage.transform.downscale_local_mean(lesion_mask, (2,2,2), cval=0)
    rescaled_lesion_mask[rescaled_lesion_mask > 0] = 1
    lesion_img = nib.Nifti1Image(rescaled_lesion_mask, np.eye(4))
    return lesion_img

def get_eid_bg_img(eid):
    bg_img_path = f"/sc-resources/ukb/data/bulk/imaging/brain/20253/20253_{eid}_2/T2_FLAIR/T2_FLAIR_brain.nii.gz"
    bg_img = nilearn.image.load_img(bg_img_path)
    bg_img_data = bg_img.get_fdata()
    rescaled_bg_img_data = skimage.transform.downscale_local_mean(bg_img_data, (2,2,2), cval=0)
    rescaled_bg_img = nib.Nifti1Image(rescaled_bg_img_data, affine=np.eye(4))
    return rescaled_bg_img

def get_expl(eid, expls, eids):
    idx = list(eids).index(eid)
    xpl_masked = expls[idx,:]    
    return xpl_masked

def process_expl_for_plotting(expl, nifti_masker):
    """
    input: masked explanations
    processing steps:
    1. abs
    2. scale 99th percentile
    3. clip (0, 10)
    4. inverse masking
    5. smooth mean expl fwhm=1.5
    retrun: mean expl rdy to plot
    """
    expl = np.abs(expl)
    expl = expl/np.percentile(expl, 99)
    expl = np.clip(expl, a_min=0, a_max=10)

    expl = nifti_masker.inverse_transform(np.float32(expl))
    expl = smooth_img(expl, 1.5)
    return expl

def create_result_dir(res_path):
    # create result directory
    isExist = os.path.exists(res_path)
    if not isExist:
        os.makedirs(res_path)

def plot_heatmap_on_bg(expl, bg_img, display_mode, cut_coords, label_img="", save_path="", vmax=10, cbar=True, neg_atlas_label_img=None, ax=None, supress_subplots = False):
    
    if display_mode != "mosaic":

        if vmax != "":
            display = plot_stat_map(
            expl,
            bg_img=bg_img,
            display_mode=display_mode,
            cut_coords = cut_coords,
            vmax=10,
            threshold=1,
            colorbar=cbar,
            draw_cross=False,
            black_bg=False,
            cmap="black_red",
            axes=ax,
            figure=ax.figure if ax is not None else None,
            )
        else:
            display = plot_stat_map(
            expl,
            bg_img=bg_img,
            display_mode=display_mode,
            cut_coords = cut_coords,
            threshold=1,
            colorbar=cbar,
            draw_cross=False,
            black_bg=False,
            axes=ax,
            cmap="black_red",
            figure=ax.figure if ax is not None else None,
            )
    else:
        
        display = MosaicSlicer.init_with_figure(
                            img=bg_img,
                            cut_coords=cut_coords,
                            figure=plt.figure(figsize=(12, 8)),
                            threshold=1,
                            colorbar=True,
                            )

        display.add_overlay(expl)

    if label_img != "":
        display.add_contours(label_img, colors="lime")

    if neg_atlas_label_img is not None:
        display.add_contours(neg_atlas_label_img, colors="fuchsia")
    
    if cbar == True:
        display._colorbar_margin["left"] = 0.15
        display._cbar.set_label('Feature Relevance', labelpad=5, color='black', fontsize=9)

    # remove orentation labels
    for cut_ax in display.axes.values():
        for txt in cut_ax.ax.texts:
            if txt.get_text() in ['L', 'R']:
                txt.set_visible(False)

    # Add vertical solid grey line
    if display_mode == "z": 
        
        fig = ax.figure
        ax_pos = ax.get_position() 
        y_fig = ax_pos.y0 

        # Optional: match the x-range of the axes in figure coords
        x_start = ax_pos.x0
        x_end   = ax_pos.x1

        # Create the line
        from matplotlib.lines import Line2D
        line = Line2D([x_start, x_end], [y_fig, y_fig],
                    transform=fig.transFigure,
                    color='grey', linestyle='dotted', linewidth=3)

        # Add the line to the figure
        fig.add_artist(line)


    if save_path != "":
        fig = plt.gcf()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    if not supress_subplots:
        show()
        display.close()
        if save_path != "":
            plt.close()


def filter_split(split_path, filtered_split_path, warp_table_path):
    with open(split_path, 'r') as file:
        # Load JSON data into a Python dictionary
        split = json.load(file)

    table_df = pd.read_csv(warp_table_path)

    eids_lesions = table_df["eid"].values

    for key in list(split.keys()):
        if key != "auxiliary":
            split_eids = split[key]
            lesion_eids = [eid for eid in split_eids if eid in eids_lesions]
            split[key] = lesion_eids

    with open(filtered_split_path, "w") as f:
        json.dump(split, f)

def get_mean_of_scaled_abs_expls(expls, nifti_masker, scale_perc=99):
    expls = np.abs(expls)

    # to img space
    expl_imgs = nifti_masker.inverse_transform(np.float32(expls))
    expl_imgs_data = expl_imgs.get_fdata()

    # scale in participant space
    scale_percentiles = np.percentile(expl_imgs_data, 99, axis = [0,1,2])
    expl_imgs_data = expl_imgs_data/scale_percentiles

    expl_imgs = nib.Nifti1Image(expl_imgs_data, np.eye(4))
    expls = nifti_masker.transform(expl_imgs)

    mean_expl = np.mean(expls, axis=0)
    return mean_expl

def plot_mean_expl(expls, nifti_masker, atlas_label_img, display_modes=["x", "z"], cut_coords=None, cbar=True, save_path = "", neg_atlas_label_img=None, axes=None, supress_subplots=False, cutoff_mode = "99th"):
    print("Mean expls")
    
    mean_expl = get_mean_of_scaled_abs_expls(expls, nifti_masker)

    mean_expl = process_xpls_for_scoring(mean_expl, nifti_masker, fwhm_expls=2, cutoff_mode = cutoff_mode, scale_perc = 99, scale = True)
    mean_expl = nifti_masker.inverse_transform(np.float32(mean_expl))

    bg_img = get_mni2mm_bg()

    if cut_coords is None:
        cut_coords = np.linspace(10, 80, num=10, dtype="int16")

    for i, display_mode in enumerate(display_modes):
        if axes is not None:
            plot_heatmap_on_bg(mean_expl, bg_img, display_mode, cut_coords, atlas_label_img, cbar=cbar, save_path=save_path, neg_atlas_label_img=neg_atlas_label_img, ax=axes[i], supress_subplots=supress_subplots)
        else:
            plot_heatmap_on_bg(mean_expl, bg_img, display_mode, cut_coords, atlas_label_img, cbar=cbar, save_path=save_path, neg_atlas_label_img=neg_atlas_label_img, supress_subplots=supress_subplots)

def plot_random_single_subjects(expls, eids, nifti_masker,atlas_keys, warped_rois_path, n=3, display_modes=["x", "z"], cut_coords=None, vmax=10,save_folder="", cbar=True, neg_atlas_keys=None, axes=None, supress_subplots=False, random_subject_eids=None):
    if cut_coords is None:
        cut_coords = np.linspace(10, 80, num=10, dtype="int16")

    for i in range(0, n):
        if random_subject_eids is None:
            expl_idx = random.sample(range(0, expls.shape[0]), 1)[0]
            expl = expls[expl_idx, :]
            eid = int(eids[expl_idx])

        roi_atlas = get_roi_atlas_from_atlas_keys(atlas_keys, warped_rois_path, eid, nifti_masker.n_elements_)
        roi_atlas_img = nifti_masker.inverse_transform(roi_atlas.astype(np.float32))

        if neg_atlas_keys is not None:
            roi_atlas_neg = get_roi_atlas_from_atlas_keys(neg_atlas_keys, warped_rois_path, eid, nifti_masker.n_elements_)
            neg_roi_atlas_img = nifti_masker.inverse_transform(roi_atlas_neg.astype(np.float32))
        else:
            neg_roi_atlas_img = None

        expl = process_xpls_for_scoring(expl, nifti_masker, fwhm_expls=2, cutoff_mode = "99th", scale_perc = 99, scale = True)
        expl = nifti_masker.inverse_transform(np.float32(expl))
        bg_img = load_eid_img(eid, nifti_masker)

        for i, display_mode in enumerate(display_modes):
            if axes is not None:
                ax = axes[i]
            else:
                ax = None

            if save_folder != "":
                plot_heatmap_on_bg(expl, bg_img, display_mode, cut_coords, roi_atlas_img, save_path=f"{save_folder}/{i}_{display_mode}_{cbar}.png", vmax=vmax, cbar=cbar, neg_atlas_label_img=neg_roi_atlas_img, ax=ax, supress_subplots=supress_subplots)
            
            else:
                plot_heatmap_on_bg(expl, bg_img, display_mode, cut_coords, roi_atlas_img, save_path="", vmax=vmax, cbar=cbar, neg_atlas_label_img=neg_roi_atlas_img, ax=ax, supress_subplots=supress_subplots)


def load_eid_img(eid, masker):
    masked_img = np.load(f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/lin_reg_imgs/all_used_test_eids/{eid}.npy")
    img = masker.inverse_transform(masked_img.astype(np.float32))
    return img


def get_all_used_test_eids():
    """
    Get all used eids for which explanations have been computed in hardcoded fashion.
    """

    eids = []


    ### brain age eids:
    seeds = [3]
    for cutoff_mode in ["99th"]:
        
        for xai_m in ["LRP_EpsilonPlusFlat"]:
            for seed in seeds:
                with open(f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/brain_age_disease_diff/T1_models/seed_{seed}/control/{xai_m}/eids.npy", "rb") as file:
                    eids_brain_age = np.load(file)
                eids += list(eids_brain_age)

    res_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/4cort_4subcort_2dummydis"

    for cutoff_mode in ["95th"]:

        p = Path(res_dir)

        score_dict = {}
        for sub_dir in p.iterdir():
            idp_type = str(sub_dir).split("/")[-1]
            print(idp_type)
            if idp_type in ["high_contrast", "cort_areas", "cort_thicknesses","subcort_intensities", "subcort_volumes", "dummy_dis"]:
                p2 = Path(str(sub_dir))
                for sub_dir_field in p2.iterdir():
                    
                    if idp_type != "dummy_dis":
                        raw_field = sub_dir_field.name
                    
                    elif idp_type == "dummy_dis":
                        raw_field = sub_dir_field.name


                    for xai_m in ["LRP_EpsilonPlusFlat"]:
                        path_to_expls = f"{res_dir}/{idp_type}/{raw_field}/{xai_m}"

                        with open(f"{path_to_expls}/eids.npy", "rb") as file:
                            eids_idps = np.load(file)

                        eids += list(eids_idps)

    eids = list(set(eids))

    return eids