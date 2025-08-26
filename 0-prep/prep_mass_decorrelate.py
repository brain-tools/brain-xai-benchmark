"""
reads IDP data from UKBB, matches it to images, masks images with brain mask, saves masked images as .npy array
"""

from prep_methods import creat_idp_df_pkl, img_paths_to_idp_df_eid_in_fname
import numpy as np
from nilearn.maskers import NiftiMasker
import pickle

idp_type = "subcortical" # "subcortical" or "cortical"

# these are .nii images in 2mm MNI space
path_to_images = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/lin_reg_imgs/int_downsampling/"

# number of images later used for IDP correction
n_imgs = 5000

# prev fitted masker
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/brain_masker.pkl"
with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

if idp_type == "cortical":
    # where to save this dataframe (input for mass decorrelation)
    idp_df_save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/idp_dfs/idp_df_cortical.pkl"
    # UKBB identifier for cortical IDPs
    subset_of_idp_identifier_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/prep/setup_files/idp_fields_aparc2009.pkl"
    # where to save masked images
    save_masked_imgs_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/masked_array/masked_imgs_aparc2009.npy"

elif idp_type == "subcortical":
    idp_df_save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/idp_dfs/idp_df_subcortical.pkl" 
    subset_of_idp_identifier_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/prep/setup_files/idp_fields_aseg.pkl"
    save_masked_imgs_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/masked_array/masked_imgs_aseg.npy"


# UKBB identifier for all IDPs
idp_fields_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/0-prep/setup_files/idp_ids.pkl"

# reads required IDP fields and respective eids from UKBB table, saves df to disk
idp_df = creat_idp_df_pkl(save_path = idp_df_save_path, idp_identifier_path=idp_fields_path, subset_of_idp_identifier_path=subset_of_idp_identifier_path)

# drop nan rows
idp_df = idp_df.dropna() 

# match .nii img paths to idp data
idp_df = img_paths_to_idp_df_eid_in_fname(path_to_images, idp_df, n_imgs)

imgs_transformed = nifti_masker.transform(idp_df["reg_img_path"][idp_df["reg_img_path"].notna()])

# save masked images
np.save(save_masked_imgs_path, imgs_transformed)
