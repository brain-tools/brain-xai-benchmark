import pandas as pd
import numpy as np
import pandas as pd
import skimage
from pathlib import Path
import nibabel as nib
from sklearn import linear_model
import nibabel as nib
from nilearn.image import load_img, resample_to_img
from nilearn.mass_univariate import permuted_ols
import os
from nilearn.plotting import plot_stat_map, show


def correct_target(confounds, targets_scaled):
    """
    Correct target from confounds using linear regression.
    """
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(confounds, targets_scaled)
    corrected_target = targets_scaled - regr.predict(confounds)
    return corrected_target

def remove_unwanted_fields(idp_df, target_field, array_csv_df):
    """"
    Preare idp_df to only contain idps, and fields not related to target_field.
    """

    # remove non idp columns:
    only_idps = idp_df.drop((['reg_img_path', 'eid']), axis=1)
    
    # delete all information related to field:
    field_nr = int(target_field.split(".")[1])
    field_name = array_csv_df[array_csv_df["field"] == field_nr]["name"].values[0]

    field_name_stripped = strip_field_name(field_name)

    # related_fields = 
    relevant_fields = list(array_csv_df[array_csv_df['name'].str.contains(field_name_stripped)]["field"])

    for field in relevant_fields:
        only_idps = only_idps.drop(([f'f.{field}.2.0']), axis=1)

    only_idps = only_idps.to_numpy()
    return only_idps

def img_paths_to_idp_df_eid_in_fname(path_to_images, idp_df, n_imgs):
    # First, read filenames of lin. registered images (see save_linearly_registered_images.py)
    img_paths = []

    p = Path(path_to_images)
    for sub_folder in p.iterdir():
        if "eid" not in str(sub_folder):
            img_paths.append(path_to_images + "/" + sub_folder.name)

    reg_img_df = pd.DataFrame(columns = ["eid", "reg_img_path"])

    for i, img_path in enumerate(img_paths):
        if i < n_imgs:
            eid = int(img_path.split("/")[-1].split(".")[0])
            reg_img_df.loc[len(reg_img_df)] = {"eid":eid, "reg_img_path":img_path}

    # keep non image idps:
    idp_df = pd.merge(idp_df, reg_img_df, on='eid', how="left")

    return idp_df

def strip_field_name(field_name):
    field_name_stripped = field_name.replace("Volume of ", '')
    field_name_stripped = field_name_stripped.replace(" (whole brain)", '')
    field_name_stripped = field_name_stripped.replace("(left hemisphere)", '')
    field_name_stripped = field_name_stripped.replace("(right hemisphere)", '')
    field_name_stripped = field_name_stripped.replace("Mean intensity of ", '')
    field_name_stripped = field_name_stripped.replace("Area of ", '')
    field_name_stripped = field_name_stripped.replace("Mean thickness of ", '')
    return field_name_stripped


def get_voxel_target_correlation(target, imgs_transformed, nifti_masker,  bias, n_perm=10):
    neg_log_pvals, t_scores, _ = permuted_ols(
        target,
        imgs_transformed,
        model_intercept=False,
        confounding_vars = bias,
        n_perm=n_perm,  # 1,000 in the interest of time; 10000 would be better
        verbose=1,  # display progress bar
        n_jobs=1)

    print(neg_log_pvals.max())
    signed_neg_log_pvals = neg_log_pvals*np.sign(t_scores)
    neg_log_pvals_image_space = nifti_masker.inverse_transform(signed_neg_log_pvals)
    return neg_log_pvals_image_space.get_fdata()


def create_result_dir(path_to_res_folder, field):
    isExist = os.path.exists(path_to_res_folder)
    if not isExist:
        os.makedirs(path_to_res_folder)

   # create directory for specific target
    res_dir = f"{path_to_res_folder}/{field}"
    isExist = os.path.exists(res_dir)
    if not isExist:
        os.makedirs(res_dir)
    return res_dir


def plot_corr_target_voxel_corr(target, idp_df, imgs_transformed, atlas, nifti_masker,  bias, field, n_perm=10, n=0):
    neg_log_pvals, t_scores, _ = permuted_ols(
        target,
        imgs_transformed,
        model_intercept=False,
        confounding_vars = bias,
        n_perm=n_perm,  # 1,000 in the interest of time; 10000 would be better
        verbose=1,  # display progress bar
        n_jobs=1)

    neg_log_pvals_image_space = nifti_masker.inverse_transform(neg_log_pvals)

    print(t_scores.max())
    print(t_scores.min())


    cut_coords = np.linspace(50, 70, num=8, dtype="int16")

    # plot results
    display = plot_stat_map(
        neg_log_pvals_image_space,
        bg_img=idp_df[idp_df["reg_img_path"].notna()]["reg_img_path"].values[0],
        display_mode="x",
        cut_coords=cut_coords,
        threshold=1.3,
    )

    display.add_contours(atlas, colors="lime")
    
    img_filename = f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/results/debug_vis/{field}_pc{n}.png"
    display.savefig(img_filename)
    show()

def extract_integer(filename):
    return int(str(filename).split('.')[-2].split('_')[2])