import numpy as np
import json
import pickle
import os
import json
import nibabel as nib
from nilearn.image import load_img, resample_to_img
import pandas as pd
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import create_atlas_target, add_lat_target

def load_masker(masker_path):
    with open(masker_path, 'rb') as file:
        nifti_masker = pickle.load(file)
    return nifti_masker

def create_result_dir(res_path):
    # create result directory
    isExist = os.path.exists(res_path)
    if not isExist:
        os.makedirs(res_path)

def init_xai_dict(split_path, xai_methods, mask_size):
    xai_dict = {}
    with open(split_path, 'r') as file:
        split = json.load(file)

    n_test = len(split["test"])

    for xai_m in xai_methods:
        mean_expl = np.zeros((182, 218, 182))
        # for aseg:
        masked_expl = np.zeros((n_test, mask_size), dtype="float16")
        eids = np.zeros((n_test))

        m_dict = {"mean_expl" : mean_expl, "cont_dice" : [], "masked_expl":masked_expl, "eids":eids}
        xai_dict[xai_m] = m_dict

    return xai_dict, n_test

def get_full_atlas(res = "2mm"):

    atlas = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc.a2009s+aseg.mgz")

    atlas_data = atlas.get_fdata()

    field_atlas = nib.Nifti1Image(atlas_data, atlas._affine)
    if res == "2mm":
        atlas_rs = resample_to_img(field_atlas, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz", interpolation="nearest")

    elif res == "1mm":
        atlas_rs = resample_to_img(field_atlas, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_1mm_brain_mask.nii.gz", interpolation="nearest")
    
    atlas_rs_data = atlas_rs.get_fdata()
    atlas_rs_data = np.rint(atlas_rs_data)

    final_atlas = nib.Nifti1Image(atlas_rs_data, np.eye(4))

    return final_atlas

def init_faith_dict(split_path, xai_methods):
    xai_dict = {}
    with open(split_path, 'r') as file:
        split = json.load(file)

    n_test = len(split["test"])

    for xai_m in xai_methods:
        m_dict = {"applied_mask_sizes":[], "flipped_mask_overlap_percentage":[] ,"predictions_original" : [], "predictions_fp_replace" : [], "predictions_flip_replace" : [], "labels" :[], "eids":[], "FPs":[], "relative_relevance_mass_replaced":[], "relative_relevance_mass_replaced_th":[]}
        xai_dict[xai_m] = m_dict

    return xai_dict, n_test
    
        
def add_atlas_to_model(field, model, full_atlas):
    if field < 27000:
        atlas_id = "aseg"
                
    if field >= 27000: 
        atlas_id = "aparc2009"

    array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv"
    config_df_aseg = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
    
    array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys_extended.csv"
    config_df_aparc = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
    config_df = pd.concat([config_df_aseg, config_df_aparc])

    model.config_df = config_df
    atlas_key = config_df[config_df["field"] == field]["atlas_key"].values[0]

    model.atlas_key = atlas_key
    
    # dilation should be zero here!
    atlas = create_atlas_target(atlas_key, atlas_id, dilation_iter=0, one_mm = True)
    atlas_target = add_lat_target(config_df, field, atlas_id, atlas, dilation_iter = 0, one_mm = True)

    model.atlas_target = atlas_target

    model.full_atlas = full_atlas