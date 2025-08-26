"""
This script inverse-warpes target regions from nonlinear MNI space to each participant’s linear MNI space using deformation fields provided by the UKBB.
"""
import numpy as np
from pathlib import Path
import pandas as pd
import os
import skimage
from nilearn.image import load_img
import numpy as np
import torch
import pickle
import nibabel as nib
import scipy.sparse as sp
from datetime import datetime
from xai_validation_helper_methods import create_result_dir
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import warp_atlas, create_atlas_target
from xai_validation_helper_methods import get_all_used_test_eids

debug_input = True

# prev. fitted brain masker
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/10k_brain_masker.pkl"

# config csv relating atlas keys, ukbb field ids, and field names
array_csv_path_cort = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/config_csvs/config_cortical.csv"

# subcortical target fields, target strings, and atlas keys
array_csv_path_subcort = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/idp_correction/setup_files/config_csvs/config_subcortical.csv"

# where to save prewarped target regions
save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/prewarped_rois"

# read files
config_df_aseg = pd.read_csv(array_csv_path_cort, header=None, names=["id", "field", "atlas_key", "name"])
config_df_aparc = pd.read_csv(array_csv_path_subcort, header=None, names=["id", "field", "atlas_key", "name"])

# atlas keys for all possible target regions
unique_keys = list(set(list(set(config_df_aseg["atlas_key"].values)) + list(set(config_df_aparc["atlas_key"].values))))

with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

# get all used eids
eids = get_all_used_test_eids()

# split eids because for parallel processing
eid_chunks = np.array_split(eids, 4)

if debug_input:
    dilation_iter = 2
    chunk_idx = 3
else:
    dilation_iter = str(sys.argv[1])
    chunk_idx = str(sys.argv[2])

for dilation_iter in [dilation_iter]:
    create_result_dir(f"{save_path}/dil_iter_{dilation_iter}")
    for key in unique_keys:
        
        # get atlas region in MNI space
        if key in config_df_aseg["atlas_key"]:
            roi_atlas = create_atlas_target(key, "aseg", dilation_iter=dilation_iter, one_mm = True)
        else:
            roi_atlas = create_atlas_target(key, "aparc2009", dilation_iter=dilation_iter, one_mm = True)

        # iterate subjects
        for eid in eid_chunks[chunk_idx]:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("Current Time:", current_time)

            eid = int(eid)
            print(eid)

            if not os.path.isfile(f"{save_path}/dil_iter_{dilation_iter}/{eid}/{key}.npz"):
                
                create_result_dir(f"{save_path}/dil_iter_{dilation_iter}/{eid}")
                file_count = len([f for f in os.listdir(f"{save_path}/dil_iter_{dilation_iter}/{eid}") if os.path.isfile(os.path.join(f"{save_path}/dil_iter_{dilation_iter}/{eid}", f))])

                # load coefficients
                coefs = load_img(f"/sc-resources/ukb/data/bulk/imaging/brain/20252/20252_{eid}_2/T1/transforms/T1_to_MNI_warp_coef.nii.gz")
                coefs = torch.FloatTensor(coefs.get_fdata().astype(np.float32)).cuda().contiguous()[None, :, :, :, :]
                
                # atlas 3D numpy
                # coefs 5D cuda tensor
                warped_atlas, indices = warp_atlas(coefs, "cuda", roi_atlas)
                
                # downsample to 2mm resolution
                warped_atlas = skimage.transform.downscale_local_mean(warped_atlas, (2,2,2), cval=0)

                atlas_img = nib.Nifti1Image(warped_atlas.astype(np.float32), np.eye(4))
                
                # mask atlas
                atlas_label = nifti_masker.transform(atlas_img)

                atlas_label[atlas_label < 0.5] = 0
                atlas_label[atlas_label >= 0.5] = 1

                # save boolean atlas
                atlas_label = atlas_label.astype(bool)
                atlas_label_sparse = sp.csr_matrix(atlas_label)
                sp.save_npz(f"{save_path}/dil_iter_{dilation_iter}/{eid}/{key}", atlas_label_sparse)