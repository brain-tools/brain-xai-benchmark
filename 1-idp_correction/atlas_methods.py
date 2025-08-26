import numpy as np
import skimage
import nibabel as  nib
from nilearn.image import load_img, resample_to_img
from scipy import ndimage
import pandas as pd
import scipy.sparse as sp
import torch
from brain_deform.cuda.deform import inverse_warp, coefs_to_field, Interpolation, Addressing, resample
from brain_deform.registration import premat_coords

def get_atlas_in_img_space(atlas_field, which="HO", one_mm = False):
    if which == "HO":
        # for visualisation of expected target area
        atlas = load_img("/home/nysi10/paper_xai_benchmark/generate_clean_results/files/MRIcron_Workshop_Materials/Research_Atlases/Harvard-Oxford cortical and subcortical structural atlases/HarvardOxford-cort_and_sub-maxprob-thr25-1mm.nii")

        atlas_data = atlas.get_fdata()
        
        if atlas_field != "full_atlas":
            atlas_data[atlas_data != atlas_field] = 0
            atlas_data[atlas_data == atlas_field] = 1

        resample_shape = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz").shape
        rescaled_atlas_data = skimage.transform.resize(atlas_data, resample_shape, preserve_range=True)
        rescaled_atlas_data = np.rint(rescaled_atlas_data)

        atlas_correct_affine = nib.Nifti1Image(rescaled_atlas_data, np.eye(4))
        return atlas_correct_affine
    
    if which == "aseg":
        aseg = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/raw_atlases/aseg.mgz")

        aseg_data = aseg.get_fdata()

        if atlas_field != "full_atlas":
            aseg_data[aseg_data != atlas_field] = 0
            aseg_data[aseg_data == atlas_field] = 1

        field_aseg = nib.Nifti1Image(aseg_data, aseg._affine)

        if one_mm:
            aseg_rs = resample_to_img(field_aseg, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_1mm_brain_mask.nii.gz", interpolation="nearest")
        else:
            aseg_rs = resample_to_img(field_aseg, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz", interpolation="nearest")

        aseg_rs_data = aseg_rs.get_fdata()
        aseg_rs_data = np.rint(aseg_rs_data)

        # final_atlas = nib.Nifti1Image(aseg_rs_data, aseg_rs._affine)
        final_atlas = nib.Nifti1Image(aseg_rs_data, np.eye(4))

        # 26553

        return final_atlas
    
    if which == "aparc2009":
        atlas = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc.a2009s+aseg.mgz")

        atlas_data = atlas.get_fdata()
        if atlas_field != "full_atlas":
            atlas_data[atlas_data != atlas_field] = 0
            atlas_data[atlas_data == atlas_field] = 1

        field_atlas = nib.Nifti1Image(atlas_data, atlas._affine)
        if one_mm:
            atlas_rs = resample_to_img(field_atlas, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_1mm_brain_mask.nii.gz", interpolation="nearest")
        else:
            atlas_rs = resample_to_img(field_atlas, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz", interpolation="nearest")

        atlas_rs_data = atlas_rs.get_fdata()
        atlas_rs_data = np.rint(atlas_rs_data)

        # final_atlas = nib.Nifti1Image(aseg_rs_data, aseg_rs._affine)
        final_atlas = nib.Nifti1Image(atlas_rs_data, np.eye(4))

        return final_atlas
    
def create_atlas_target(atlas_key, atlas_id, score_mode=None, smooth_sigma=None, clip_th=None, dilation_iter=None, one_mm = False):
        
    atlas_correct_affine = get_atlas_in_img_space(atlas_key, which=atlas_id, one_mm = one_mm)

    atlas_data = atlas_correct_affine.get_fdata()
            
    if dilation_iter > 0:
        atlas_data = ndimage.binary_dilation(atlas_data, iterations=dilation_iter)
    
    return atlas_data

def add_lat_target(config_df, field, atlas_id, atlas_target, dilation_iter = None, score_mode=None, smooth_sigma=None, clip_th=None, one_mm = False):
    # get atlas target also for lateral field
    field_name = config_df[config_df["field"] == field]["name"].iloc[0]
    
    lat_field_name =  None
    if "(left hemisphere)" in field_name:
        lat_field_name = field_name.replace("(left hemisphere)", "(right hemisphere)")
    
    if "(right hemisphere)" in field_name:
        lat_field_name = field_name.replace("(right hemisphere)", "(left hemisphere)")
    
    if lat_field_name != None:
        
        lat_atlas_key = int(config_df[config_df["name"] == lat_field_name]["atlas_key"].iloc[0])

        if dilation_iter == None:
            lat_atlas_target = create_atlas_target(lat_atlas_key, atlas_id, score_mode=score_mode, smooth_sigma=smooth_sigma, clip_th=clip_th, one_mm=one_mm)
        else:
            lat_atlas_target = create_atlas_target(lat_atlas_key, atlas_id, dilation_iter=dilation_iter, one_mm=one_mm)

        atlas_target += lat_atlas_target
        atlas_target = np.clip(atlas_target, 0, 1)
    return atlas_target

def get_lat_key(config_df, field):
    field_name = config_df[config_df["field"] == field]["name"].iloc[0]

    lat_field_name =  None
    if "(left hemisphere)" in field_name:
        lat_field_name = field_name.replace("(left hemisphere)", "(right hemisphere)")

    if "(right hemisphere)" in field_name:
        lat_field_name = field_name.replace("(right hemisphere)", "(left hemisphere)")

    if lat_field_name != None:
        
        lat_atlas_key = int(config_df[config_df["name"] == lat_field_name]["atlas_key"].iloc[0])
        return lat_atlas_key
    else:
        return None

def get_full_atlas(which="HO", one_mm = False):
    if which == "HO":
        # for visualisation of expected target area
        atlas = load_img("/home/nysi10/paper_xai_benchmark/generate_clean_results/files/MRIcron_Workshop_Materials/Research_Atlases/Harvard-Oxford cortical and subcortical structural atlases/HarvardOxford-cort_and_sub-maxprob-thr25-1mm.nii")

        atlas_data = atlas.get_fdata()

        resample_shape = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz").shape
        rescaled_atlas_data = skimage.transform.resize(atlas_data, resample_shape, preserve_range=True)
        rescaled_atlas_data = np.rint(rescaled_atlas_data)

        atlas_correct_affine = nib.Nifti1Image(rescaled_atlas_data, np.eye(4))
        return atlas_correct_affine
    
    if which == "aseg":
        aseg = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/raw_atlases/aseg.mgz")

        aseg_data = aseg.get_fdata()

        field_aseg = nib.Nifti1Image(aseg_data, aseg._affine)

        if one_mm:
            aseg_rs = resample_to_img(field_aseg, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_1mm_brain_mask.nii.gz", interpolation="nearest")    
        else:
            aseg_rs = resample_to_img(field_aseg, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz", interpolation="nearest")

        aseg_rs_data = aseg_rs.get_fdata()
        aseg_rs_data = np.rint(aseg_rs_data)

        final_atlas = nib.Nifti1Image(aseg_rs_data, np.eye(4))

        # 26553

        return final_atlas
    
    if which == "aparc2009":
        atlas = load_img("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc.a2009s+aseg.mgz")

        atlas_data = atlas.get_fdata()

        field_atlas = nib.Nifti1Image(atlas_data, atlas._affine)
        if one_mm:
            atlas_rs = resample_to_img(field_atlas, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_1mm_brain_mask.nii.gz", interpolation="nearest")
        else:
            atlas_rs = resample_to_img(field_atlas, "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/MNI/MNI_T1_2mm_brain_mask.nii.gz", interpolation="nearest")

        atlas_rs_data = atlas_rs.get_fdata()
        atlas_rs_data = np.rint(atlas_rs_data)

        final_atlas = nib.Nifti1Image(atlas_rs_data, np.eye(4))

        return final_atlas

def get_atlas_info():
    config_df_1 = pd.read_csv("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv", header=None, names=["id", "field", "atlas_key", "name"])
    config_df_2 = pd.read_csv("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys.csv", header=None, names=["id", "field", "atlas_key", "name"])
    atlas_keys = list(set(list(set(config_df_2["atlas_key"].values)) + list(set(config_df_1["atlas_key"].values))))
    roi_names = [get_roi_name(key) for key in atlas_keys]
    return atlas_keys, roi_names

def get_roi_name(key):
    config_df_1 = pd.read_csv("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv", header=None, names=["id", "field", "atlas_key", "name"])
    config_df_2 = pd.read_csv("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys.csv", header=None, names=["id", "field", "atlas_key", "name"])

    if len(config_df_1[config_df_1["atlas_key"] == key]["name"].values) > 0:
        field_string = config_df_1[config_df_1["atlas_key"] == key]["name"].values[0]
        field_string = field_string.replace("Volume of ", "")
        field_string = field_string.replace("Mean intensity of ", "")
        field_string = field_string.replace("Mean thickness of ", "")
        field_string = field_string.replace("Area of ", "")

    else:
        field_string = config_df_2[config_df_2["atlas_key"] == key]["name"].values[0]
        field_string = field_string.replace("Volume of ", "")
        field_string = field_string.replace("Mean intensity of ", "")
        field_string = field_string.replace("Mean thickness of ", "")
        field_string = field_string.replace("Area of ", "")

    return field_string

def get_atlas_keys_and_field_string(idp_type, sub_dir_field, lat_key=True):
    atlas_keys = []
    if idp_type != "dummy_dis":
        raw_field = sub_dir_field.name
        field = int(raw_field.split("_")[0])
        if field < 27000:
            array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv"
        
        if field >= 27000: 
            array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys.csv"

        config_df = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
        atlas_key = config_df[config_df["field"] == field]["atlas_key"].values[0]
        
        atlas_keys.append(atlas_key)

        if lat_key:
            lat_atlas_key = get_lat_key(config_df, field)
            if lat_atlas_key is not None:
                atlas_keys.append(lat_atlas_key)

        print(raw_field)
        field_string = config_df[config_df["field"] == field]["name"].values[0]
        print(field_string)
        field_strings = [field_string]
    
    elif idp_type == "dummy_dis":
        raw_field = sub_dir_field.name
        field_1 = int(raw_field.split("_")[1])
        field_2 = int(raw_field.split("_")[4])

        field_strings = []
        for field in [field_1, field_2]:
        
            if field < 27000:
                array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv"

            if field >= 27000: 
                array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys.csv"

            config_df = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
            atlas_key = config_df[config_df["field"] == field]["atlas_key"].values[0]
            atlas_keys.append(atlas_key)

            if lat_key:
                lat_atlas_key = get_lat_key(config_df, field)
                if lat_atlas_key is not None:
                    atlas_keys.append(lat_atlas_key)

            field_strings.append(config_df[config_df["field"] == field]["name"].values[0])

        field_string = f"high {field_strings[0]} and\nlow {field_strings[1]}"
        print(field_string)

    return atlas_keys, field_string, raw_field, field_strings


def get_roi_ranking_with_warped_atlas(expl, eid, warped_rois_path, roi_keys, roi_names, nifti_masker, score_mode="density"):
    df = pd.DataFrame(columns=['roi', 'roi_sig'])

    for i, key in enumerate(roi_keys):
        roi_atlas = get_roi_atlas_from_atlas_keys([key], warped_rois_path, eid, nifti_masker.n_elements_)
        roi_size = np.sum(roi_atlas)
        if roi_size > 0:

            if score_mode == "density":
                roi_sig = np.nansum(roi_atlas*expl)/roi_size

            if score_mode == "percentile":

                voxel_in_mask = expl[roi_atlas.astype("bool")]
                
                roi_sig = np.percentile(voxel_in_mask, 99)
                # roi_sig = voxel_in_mask.max()

            new_row = pd.DataFrame([{'roi':roi_names[i], 'roi_sig':roi_sig}])
            df = pd.concat([df, new_row], ignore_index=True)

    df = df.sort_values(by=['roi_sig'], ascending = False)

    return df

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

def warp_atlas(coefs, device, atlas, i=0):

    premat = torch.eye(4).cuda()

    # coefs_to_field_batched
    field = coefs_to_field(182, 218, 182, 10, 10, 10, 1.0, premat, coefs[i, :, :, :, :])
    field = field - premat_coords().cuda()

    # Invert the warp field, produce absolute coordinates
    field = field.float().cuda().contiguous()
    
    # inverse_warp_batched
    inv = inverse_warp(field, 5000, False)

    atlas_tensor = torch.FloatTensor(atlas.astype(np.float32)).cuda().contiguous()

    # Resample using the inverse warp field
    postmat = torch.eye(4).cuda()
    out_image = resample(atlas_tensor, inv, postmat, Interpolation.Nearest, Addressing.Clamp, 0.0)
    out_image = out_image.cpu().numpy()

    # get indices:
    indices = np.unique(out_image)
    indices = np.delete(indices, 0)

    return out_image, indices