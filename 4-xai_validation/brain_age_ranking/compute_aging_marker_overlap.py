import numpy as np
import pickle
import sys
import pandas as pd
import rbo
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from xai_validation_helper_methods import discard_worse_lateral_roi, get_ground_truth_regions, process_xpls_for_scoring
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import get_roi_ranking_with_warped_atlas, get_atlas_info

debug_input = True

# path to meta analysis table with aging markers from Walhovd et al. (2011)
path_to_metanalysis_csv = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation/brain_age_ranking/aging_volume_table.csv"

# how much smoothing on explanations
fwhm_expls_score = 2

# how much to dilate atlas-based ground-truth targets
dilation_iter = 0

# rank brain regions by expl percentile within region's mask or by density
score_mode = "percentile" # "density"

# Path to explanations that should be evaluated
expl_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/brain_age_disease_diff/T1_models"

# XAI methods to evaluate
xai_methods = ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop", "DeepLift", "DeepLift_mean_img", "GuidedBackprop", "GuidedGradCam", "GradCAM", "GradCAM_l1", "GradCAM_l2", "GradCAM_l3", "InputXGradient"]

# identifier for different model seeds
seed_ids = [3, 4, 5]

# where to save evalutation
df_save_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation/result_dfs"

# prev. fitted brain masker
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/10k_brain_masker.pkl"

# We previously inverse-warped target regions from nonlinear MNI space to each participant’s linear MNI space using deformation fields provided by the UKBB. These are stored in the directory below:
warped_rois_path = f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/prewarped_rois/dil_iter_{dilation_iter}"

# which threshold to apply to explanations
if debug_input:
    cutoff_mode = "99th"
else:
    cutoff_mode = str(sys.argv[1])

# read meta-analyisis table, select relevant age range, and map region names
gt_list = get_ground_truth_regions(path_to_metanalysis_csv)

with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

roi_keys, roi_names = get_atlas_info()

score_df = pd.DataFrame()

save_top_rois = {}

xai_m_mean_expls = {}

for xai_m in xai_methods:
    print(xai_m)
    
    # rbo score takes order into account, overlap does not
    xai_m_rbos = []
    xai_m_overlaps = []

    for seed in seed_ids:
        print(seed)

        with open(f"{expl_dir}/seed_{seed}/control/{xai_m}/eids.npy", "rb") as file:
            eids = np.load(file)

        with open(f"{expl_dir}/seed_{seed}/control/{xai_m}/masked_expls.npy", "rb") as file:
            expls = np.load(file)
            expls = process_xpls_for_scoring(expls, nifti_masker, fwhm_expls_score, cutoff_mode = cutoff_mode, scale_perc = 99, scale = False)

        for i in range(expls.shape[0]):
            roi_ranking = get_roi_ranking_with_warped_atlas(expls[i, :], int(eids[i]), warped_rois_path, roi_keys, roi_names, nifti_masker, score_mode=score_mode)

            top_rois = list(roi_ranking["roi"].values)
            top_rois = discard_worse_lateral_roi(top_rois)
            top_rois = top_rois[:17]
            rbo_sim = rbo.RankingSimilarity(top_rois, gt_list).rbo()
            overlap = len(list(set(top_rois) & set(gt_list)))/len(top_rois)

            xai_m_rbos.append(rbo_sim)
            xai_m_overlaps.append(overlap)
    
    mean_rbo = np.mean(xai_m_rbos)
    std_rbo = np.std(xai_m_rbos)

    mean_overlap = np.mean(xai_m_overlaps)
    std_overlap = np.std(xai_m_overlaps)

    new_row = pd.DataFrame([{'xai_m':xai_m, 'rbo':mean_rbo, 'rbo_std':std_rbo, 'overlap':mean_overlap, 'overlap_std':std_overlap}])
    score_df = pd.concat([score_df, new_row], ignore_index=True)

score_df.to_pickle(f'{df_save_dir}/brain_age_overlap_scores_{score_mode}_{cutoff_mode}.pkl')
