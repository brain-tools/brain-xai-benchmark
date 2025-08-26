"""
Evaluates precomputed explanations for models predicting WMH lesion load using lesion masks provided by the UKBB and RMA.
Lesions masked need to be aligned to the explanation space by registering them linearly to MNI.
"""

import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from xai_validation_helper_methods import get_lesion_mask_img, get_eid_bg_img, get_expl, process_xpls_for_scoring, process_expl_for_plotting, plot_heatmap_on_bg, filter_split
from metrics import get_relevance_mass_accuracy_batch
from brain_deform.lightning import BrainDataModule
from brain_deform.augmentation import process_image
import numpy as np
import pickle
import pandas as pd
import sys

debug_input = True

# how much smoothing on explanations
fwhm_expls_score = 2

# how much to dilate atlas-based ground-truth targets
dilation_iter = 0

# table with paths to lesion masks and warp coefficients
warp_table_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/lesion_mask_warp_table.csv"

# json with test eids related to explanations that should be evaluated.
split_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/pv_lesion_load.json"

# not all subject that have a T2 image available have a lesion mask as well. hence, filter eids in split for available lesion masks:
filtered_split_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/dummy_lesion_eids_split.json"
filter_split(split_path, filtered_split_path, warp_table_path)

# path to explanations
expl_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/pv_lesion_load"

# where to save evaluation results
df_save_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation/result_dfs"

# XAI methods to evaluate
xai_methods = ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop", "DeepLift", "DeepLift_mean_img", "GuidedBackprop", "GuidedGradCam", "GradCAM", "GradCAM_l1", "GradCAM_l2", "GradCAM_l3", "InputXGradient"]

# prev. fitted brain masker
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/10k_brain_masker.pkl"

# prediciton target name
prediciton_target = "pv_lesion_load"

# table with targets
prediction_target_table = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/pv_lesion_load.csv"

# how many explanations to plot
n_plot = 5

# which xai methods to plot
xai_methods_to_plot = []

if debug_input:
    cutoff_mode = "99th"
else:
    cutoff_mode = str(sys.argv[1])

# init score df
df = pd.DataFrame(columns=['task', 'xai_m', 'score', 'score_std'])

# load masker
with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

for xai_m in xai_methods:

    print(xai_m)
    relevance_mass_accuracies = []
 
    # load explanations and eids
    with open(f"{expl_dir}/{prediciton_target}/{xai_m}/masked_expls.npy", "rb") as file:
        expls = np.load(file)

    with open(f"{expl_dir}/{prediciton_target}/{xai_m}/eids.npy", "rb") as file:
        eids = np.load(file)

    # setup datamodule for registering lesion masks
    data_module = BrainDataModule(data_table_path=warp_table_path,
                    split_path=filtered_split_path,
                    batch_size=1,
                    registration="linear",
                    augmentation_probability = 0,
                    index_column= "eid",
                    t1_column= "raw",
                    coefs_to_mni_column= "coefs_to_mni",
                    target_column= "target")

    data_module.setup()
    data_loader = data_module.test_dataloader()

    # pipe lesion masks
    for i, x in enumerate(data_loader):
        plot_counter = 0
        data_main, data_aux, data_aug = x
        img_main, coefs_main, premat_main, postmat_main, label_main, eid = data_main
        img_aux, coefs_aux, premat_aux, postmat_aux, label_aux, _ = data_aux
        coefs_aug, premat_aug = data_aug
        
        mask_main_registered, _ = process_image(
                img_main.to("cuda:0"),
                coefs_main.to("cuda:0"),
                premat_main.to("cuda:0"),
                postmat_main.to("cuda:0"),
                img_aux.to("cuda:0"),
                coefs_aux.to("cuda:0"),
                premat_aux.to("cuda:0"),
                postmat_aux.to("cuda:0"),
                coefs_aug.to("cuda:0"),
                premat_aug.to("cuda:0"),
                data_module.hparams,
            )
        
        lesion_mask = mask_main_registered.cpu().numpy()[0,0,:,:,:]
        # get lesion masks as nii image, dilated if desired
        lesion_mask_img = get_lesion_mask_img(lesion_mask, dilation_iter)        

        eid = int(eid)
        label = label_main.cpu().numpy()[0]
        print(i)

        # get score:
        # first, get explanation related to mask
        xpl_masked = get_expl(eid, expls, eids)
        xpl_masked = process_xpls_for_scoring(xpl_masked, nifti_masker, fwhm_expls_score, cutoff_mode = cutoff_mode, scale_perc = 99, scale = False)

        # mask lesion mask to match masked explanation's shape
        masked_lesion_mask = nifti_masker.transform(lesion_mask_img)
        
        # compute RMA
        relevance_mass_accuracy = get_relevance_mass_accuracy_batch(xpl_masked, masked_lesion_mask)
        relevance_mass_accuracies.append(relevance_mass_accuracy)

        if plot_counter < n_plot and xai_m in xai_methods_to_plot:
            # plot some examples if desired
            plot_counter += 1
            xpl_masked = get_expl(eid, expls, eids)
            xpl_img = process_expl_for_plotting(xpl_masked, nifti_masker)
            bg_img = get_eid_bg_img(eid)
            cut_coords = np.linspace(10, 80, num=12, dtype="int16")
            for display_mode in ["x", "z"]:
                plot_heatmap_on_bg(xpl_img, bg_img, display_mode, cut_coords, lesion_mask_img)

    mean_rma = np.mean(relevance_mass_accuracies)
    std_rma = np.std(relevance_mass_accuracies)
    print(f"mean rma: {mean_rma}")

    new_row = pd.DataFrame([{'task':prediciton_target, 'xai_m':xai_m, 'score':mean_rma, 'score_std':std_rma}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_pickle(f"{df_save_dir}/rma_df_lesions_{cutoff_mode}.pkl")

