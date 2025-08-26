import time
import numpy as np
import pickle
import sys
from datetime import datetime
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from metrics import check_true_positive
from xai_validation_helper_methods import process_xpls_for_scoring, discard_worse_lateral_roi
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import get_atlas_info, get_atlas_keys_and_field_string
import pandas as pd
from pathlib import Path    

# if no, pass inputs via the command line. If yes, FPs are plotted.
debug_input = True

# how much smoothing on explanations
fwhm_expls_score = 2

# how much to dilate atlas-based ground-truth targets
dilation_iter = 0

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
    xai_m = "SmoothGrad"
else:
    cutoff_mode = str(sys.argv[1])
    xai_m = str(sys.argv[2])

# We previously inverse-warped target regions from nonlinear MNI space to each participant’s linear MNI space using deformation fields provided by the UKBB. These are stored in the directory below:
warped_rois_path = f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/prewarped_rois/dil_iter_{dilation_iter}"

roi_keys, roi_names = get_atlas_info()

with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

df = pd.DataFrame(columns=['task', 'xai_m', 'score'])

p = Path(expl_dir)

# iterate over directories with explanations for each target and given method
score_dict = {}
for sub_dir in p.iterdir():
    idp_type = str(sub_dir).split("/")[-1]
    print(idp_type)

    # Get and print the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Time:", current_time)
    
    if idp_type in target_types:

        p2 = Path(str(sub_dir))
        for sub_dir_field in p2.iterdir():
            # get the atlas keys for constructing the ground-truth targets from binary atlas brain atlas.
            # in case of a single cidp target, this includes the target-related region, as well as its lateral coutnerpart.
            # for artificial diseases (here: dummy dis), this includes all target-related regions and their lateral counterparts.
            atlas_keys, field_string, raw_field, field_strings_list = get_atlas_keys_and_field_string(idp_type, sub_dir_field)
            
            # log time
            start_time = time.time()
            print(f"Processing for {xai_m} started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

            # load explanations and eids
            path_to_expls = f"{expl_dir}/{idp_type}/{raw_field}/{xai_m}"

            with open(f"{path_to_expls}/eids.npy", "rb") as file:
                eids = np.load(file)

            with open(f"{path_to_expls}/masked_expls.npy", "rb") as file:
                expls = np.load(file)

            # standard setting was preprocessed:
            if fwhm_expls_score == 2 and cutoff_mode == "99th":

                expls = np.load(f"{path_to_expls}/processed_masked_expls.npy")                    
            else:
                # preprocess for computing score:
                expls = process_xpls_for_scoring(expls, nifti_masker, fwhm_expls_score, cutoff_mode = cutoff_mode, scale_perc = 99, scale = True)               

            tps = []
            tps_weak = []
            
            for i in range(expls.shape[0]):
                if i % 200 == 0:
                    print(f"{i} subjects processed.")
                    
                    current_time = time.time()
                    print(f"Processing subject {i} at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")

                expl = expls[i, :]
                eid = int(eids[i])

                strong_match, weak_match = check_true_positive(expl, eid, warped_rois_path, roi_keys, roi_names, nifti_masker, field_string, field_strings_list, idp_type, rank_by="percentile")
                
                tps.append(strong_match)
                tps_weak.append(weak_match)

            tpr = np.mean(tps)
            tpr_weak = np.mean(tps_weak)

            # save results
            new_row = pd.DataFrame([{'task':field_string, 'xai_m':xai_m, 'score':tpr, 'score_weak':tpr_weak}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_pickle(f"{df_save_dir}/tpr_percentile_{cutoff_mode}_{xai_m}.pkl")

            # log time
            end_time = time.time()
            print(f"Processing ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")