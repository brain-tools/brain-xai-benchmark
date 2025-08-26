import numpy as np
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import get_roi_ranking_with_warped_atlas
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from xai_validation_helper_methods import discard_worse_lateral_roi

def get_relevance_mass_accuracy_batch(xpls, atlas_label):
    AB = atlas_label * xpls
    r_within = np.sum(AB, axis=1)
    r_total = np.sum(xpls, axis=1)
    rma_accuracies = r_within/r_total
    
    # for some edge cases, r_total is 0. set that cases to rma=0:
    rma_accuracies = np.nan_to_num(rma_accuracies)
    
    return rma_accuracies

def check_false_positive(expl, roi_atlas):
    voxel_inside_mask = expl[0,:][roi_atlas.astype("bool")]
    voxel_outside_mask = expl[0,:][~roi_atlas.astype("bool")]
    
    # get X*99th percentile of XPL
    false_positive_th = np.percentile(voxel_inside_mask, 99)

    # if value outside dilated atlas -> thats a false positive
    if voxel_outside_mask.max() > false_positive_th:
        return 1

    else:
        return 0

def check_true_positive(expl, eid, warped_rois_path, roi_keys, roi_names, nifti_masker, field_string, field_strings_list, idp_type, rank_by="percentile"):
    roi_ranking = get_roi_ranking_with_warped_atlas(expl, eid, warped_rois_path, roi_keys, roi_names, nifti_masker, score_mode=rank_by)
    top_rois = list(roi_ranking["roi"].values)
    top_rois = discard_worse_lateral_roi(top_rois)

    if idp_type != "dummy_dis":

        strong_match = 0
        # perfect match
        top_roi = list(roi_ranking.head(1)["roi"].values)[0]
        roi_match = top_roi in field_string
        if roi_match:
            strong_match = 1

        # top 3 match
        weak_match = 0
        top_3_rois = list(roi_ranking.head(3)["roi"].values)

        for top_roi in top_3_rois:
            if top_roi in field_string:
                weak_match = 1

        return strong_match, weak_match

    else:
        # perfect match
        strong_match = 0
        top_rois = list(roi_ranking.head(2)["roi"].values)
        roi_match_1 = any(top_rois[0] in full_string for full_string in field_strings_list)
        roi_match_2 = any(top_rois[1] in full_string for full_string in field_strings_list)
        
        if roi_match_1 and roi_match_2:
            strong_match = 1

        # top 6 match
        weak_match = 0
        top_rois = list(roi_ranking.head(6)["roi"].values)
        
        found_first_field = False
        found_second_field = False
        
        for top_roi in top_rois:
            if top_roi in field_strings_list[0]:
                found_first_field = True
            
            if top_roi in field_strings_list[1]:
                found_second_field = True
            
        if found_first_field and found_second_field:
            weak_match = 1
        
        return strong_match, weak_match