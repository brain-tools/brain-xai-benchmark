"""
Computes RMA scores given precomputed explanations and atlas targets.
"""

import numpy as np
import pickle
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from metrics import get_relevance_mass_accuracy_batch
from xai_validation_helper_methods import process_xpls_for_scoring, get_roi_atlas_from_atlas_keys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import get_lat_key, get_atlas_in_img_space, get_atlas_keys_and_field_string
import pandas as pd
from pathlib import Path
from datetime import datetime

# if no, pass inputs via the command line
debug_input = True

# how much smoothing on explanations
fwhm_expls_score = 2

# how much to dilate atlas-based ground-truth targets
dilation_iter = 2

# target types
target_types = ["high_contrast", "cort_areas", "cort_thicknesses","subcort_intensities", "subcort_volumes", "dummy_dis"]

# Path to explanations that should be evaluated
expl_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/4cort_4subcort_2dummydis"

# XAI methods to evaluate
xai_methods = ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop", "DeepLift", "DeepLift_mean_img", "GuidedBackprop", "GuidedGradCam", "GradCAM", "GradCAM_l1", "GradCAM_l2", "GradCAM_l3", "InputXGradient"]

# where to save evalutation
df_save_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation/result_dfs"

# prev. fitted brain masker
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/10k_brain_masker.pkl"

# which threshold to apply to explanations
if debug_input:
    cutoff_mode = "99th"
else:
    cutoff_mode = str(sys.argv[1])

# We previously inverse-warped target regions from nonlinear MNI space to each participant’s linear MNI space using deformation fields provided by the UKBB. These are stored in the directory below:
warped_rois_path = f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/prewarped_rois/dil_iter_{dilation_iter}"

# init score df
df = pd.DataFrame()

# load masker
with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

p = Path(expl_dir)

full_atlas = get_atlas_in_img_space("full_atlas", which="aparc2009", one_mm = True)

# iterate over directories with explanations for each target and method
for sub_dir in p.iterdir():
    idp_type = str(sub_dir).split("/")[-1]
    print(idp_type)
    if idp_type in target_types:
        p2 = Path(str(sub_dir))
        for sub_dir_field in p2.iterdir():            
            # get the atlas keys for constructing the ground-truth targets from binary atlas brain atlas.
            # in case of a single cidp target, this includes the target-related region, as well as its lateral coutnerpart.
            # for artificial diseases (here: dummy dis), this includes all target-related regions and their lateral counterparts.
            atlas_keys, field_string, raw_field, _ = get_atlas_keys_and_field_string(idp_type, sub_dir_field, lat_key=True)

            for xai_m in xai_methods:
    
                print(xai_m)
                path_to_expls = f"{expl_dir}/{idp_type}/{raw_field}/{xai_m}"

                with open(f"{path_to_expls}/eids.npy", "rb") as file:
                    eids = np.load(file)

                with open(f"{path_to_expls}/masked_expls.npy", "rb") as file:
                    expls = np.load(file)

                # preprocess for computing scores:
                expls = process_xpls_for_scoring(expls, nifti_masker, fwhm_expls_score, cutoff_mode = cutoff_mode, scale_perc = 99, scale = False)
                
                # get rma scores for each subject
                relevance_mass_accuracies = []
                for i, eid in enumerate(eids):
                    eid = int(eid)

                    if i % 100 == 0:
                        current_time = datetime.now()
                        print("Current Time:", current_time.strftime("%H:%M:%S"))
                        print(i)
                    
                    # get subject specific groud-truth atlas targets.
                    roi_atlas = get_roi_atlas_from_atlas_keys(atlas_keys, warped_rois_path, eid, nifti_masker.n_elements_)

                    expl = expls[i, :][None, :]

                    # compute score:
                    relevance_mass_accuracy = get_relevance_mass_accuracy_batch(expl, roi_atlas)[0]
                    relevance_mass_accuracies.append(relevance_mass_accuracy)
                
                mean_rma = np.mean(relevance_mass_accuracies)
                std_rma = np.std(relevance_mass_accuracies)
                print(f"mean rma score: {mean_rma}")

                # save results
                new_row = pd.DataFrame([{'task':field_string, 'xai_m':xai_m, 'score':mean_rma, 'score_std':std_rma}])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_pickle(f"{df_save_dir}/rma_df_th_{cutoff_mode}.pkl")