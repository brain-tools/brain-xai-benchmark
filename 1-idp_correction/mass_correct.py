"""
This script performs mass-correction of IDP targets.
"""

import numpy as np
import pandas as pd
import pickle
from cidp_methods import correct_target, remove_unwanted_fields, img_paths_to_idp_df_eid_in_fname, get_voxel_target_correlation, create_result_dir, plot_corr_target_voxel_corr
from atlas_methods import get_atlas_in_img_space
import datetime
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Variables and paths:
idp_type = "subcortical" # "subcortical" or "cortical"
debug = True

step = 5
n_perm = 200
n_imgs = 5000

if idp_type == "cortical":
    n_pcs = 440 
elif idp_type == "subcortical":
    n_pcs = 100

# trial name for results
trial_name = f"{idp_type}_refactoring_test"

# where to save results
path_to_res_folder = f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/results/{trial_name}"

# .nii images in 2mm MNI space
# path_to_images = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/nii"
path_to_images = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/lin_reg_imgs/int_downsampling/"

# universal 10k masker w smoothing
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/10k_brain_masker.pkl"

if idp_type == "cortical":
   # idp df created in prep/prep_mass_decorrelate.py
   idp_save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/idp_dfs/idp_df_cortical.pkl"
   # previously masked images as 2D array:
   masked_imgs_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/masked_array/masked_imgs_aparc2009.npy"
   # config csv relating atlas keys, ukbb field ids, and field names
   array_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/config_csvs/config_cortical.csv"
   # which atlas to use for (debug) visualization
   which_atlas = "aparc2009"

elif idp_type == "subcortical":
   idp_save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/idp_dfs/idp_df_subcortical.pkl"
   masked_imgs_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/masked_array/masked_imgs_aseg.npy"
   array_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/config_csvs/config_subcortical.csv"
   which_atlas = "aseg"


if debug:
   # debug input
   field = 26562
   atlas_key = 17
   region_name = "Volume of Hippocampus (left hemisphere)"
else:
   # read input
   field = int(sys.argv[1])
   atlas_key = int(sys.argv[2])
   region_name = str(sys.argv[3])


# correct field format
field = f"f.{field}.2.0"

# for logging:
print(f"array job: {field}, {atlas_key}, {region_name}")

# load config csv
array_csv_df = pd.read_csv(array_csv_path, names=['idx', 'field', 'atlas_key', 'name'])

# create result directory if not existing
res_dir = create_result_dir(path_to_res_folder, field)

# read prepared idp df
idp_df = pd.read_pickle(idp_save_path)

# drop nan rows
idp_df = idp_df.dropna()

# match .nii img paths to idp data
idp_df = img_paths_to_idp_df_eid_in_fname(path_to_images, idp_df, n_imgs)

# load masked images and masker:
with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

imgs_transformed = np.load(masked_imgs_path)

# prepare idp df for PCA. Here, we remove all fields related to the target field, as well as non-idp columns (e.g. eid).
idp_df_pca = remove_unwanted_fields(idp_df, field, array_csv_df)

# standardize data
idp_df_pca = StandardScaler().fit_transform(idp_df_pca)

# perform PCA
pca = PCA()
transformed_idp_data = pca.fit_transform(idp_df_pca)

# add bias:
transformed_idp_data = np.pad(transformed_idp_data, ((0,0),(1,0)), mode='constant', constant_values=1)

# get and scale target
target = idp_df[field].to_numpy()
target_scaled = StandardScaler().fit_transform(target.reshape(-1, 1))

# get binary atlas target for (debug) visualization
atlas = get_atlas_in_img_space(atlas_key, which=which_atlas)

target_df = {}

for i in range(0, n_pcs+1, step):
    """
    Correct target using progressively more PCs.
    """
    now = datetime.datetime.now()
    print(f"time: {now.time()}")
    current_max_pc = i+1
    print(f"max pc: {current_max_pc}")
    
    # correct for the first i PCs and bias
    confounds = transformed_idp_data[:, :i+1]
    corrected_target = correct_target(confounds, target_scaled)

   # now, compute voxel-wise correlation with the corrected target for visualisatiuon later
   # only look at targets where we have images
    img_indices = list(idp_df[idp_df["reg_img_path"].notnull()].index)
    corrected_target_vis = corrected_target[img_indices]
    bias = np.ones(corrected_target_vis.shape)
    
    # compute voxel-wise between image and corrected target
    neg_log_pvals = get_voxel_target_correlation(corrected_target_vis, imgs_transformed, nifti_masker,  bias, n_perm=n_perm)

    if debug:
        plot_corr_target_voxel_corr(corrected_target_vis, idp_df, imgs_transformed, atlas, nifti_masker,  bias, field, n_perm=50, n=i)

    # save pvals and targets
    np.save(f'{res_dir}/p_vals_{current_max_pc}.npy', neg_log_pvals, allow_pickle=True)
    np.save(f'{res_dir}/decorr_target_{current_max_pc}.npy', corrected_target, allow_pickle=True)
