import os
import numpy as np
import pickle
import nibabel as nib
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from xai_validation_helper_methods import plot_mean_expl, plot_random_single_subjects
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import create_atlas_target, add_lat_target
import pandas as pd
from pathlib import Path
import random
from matplotlib import pyplot as plt
from pypdf import PdfWriter
import random

# how much smoothing on explanations
fwhm_expls_score = 2

# how much to dilate atlas-based ground-truth targets
dilation_iter = 2

# equivalent to 2 dilation iterations in 1mm space
dilation_iter_2mm = 1

# which threshold to apply to explanations
cutoff_mode = "99th"

# tasks to plot. coding: ukbb-field-idp_n-pc-corrected; high*low* identify artificial diseases
tasks_to_plot = ["27352_146", "27420_326", "27359_426", "27652_101", "26544_21", "26576_76", "26559_26", "26562_46", "26526_16", "26554_56", "high_27359_426_low_26559_26_0.64ds", "high_27652_101_low_26562_46_0.64ds"]

# Path to explanations that should be evaluated
expl_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/results/explanations/norm_bug_fixed/4cort_4subcort_2dummydis"

# XAI methods to plot
xai_methods_to_plot = ["LRP_EpsilonPlusFlat", "SmoothGrad", "GradCAM"]

# whether to plot the lateral region of the ground truth target as well
plot_lat_target = False

# where to save evalutation
plot_save_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation/result_plots"

# prev. fitted brain masker
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/10k_brain_masker.pkl"

# We previously inverse-warped target regions from nonlinear MNI space to each participant’s linear MNI space using deformation fields provided by the UKBB. These are stored in the directory below:
warped_rois_path = f"/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/prewarped_rois/dil_iter_{dilation_iter}"

with open(masker_path, 'rb') as file:
    nifti_masker = pickle.load(file)

p = Path(expl_dir)

for xai_m in xai_methods_to_plot:
    # for plotting the same subjects for each xai method
    random.seed(42)
    np.random.seed(42)

    single_pdf_paths = []

    score_dict = {}
    # iterate folder with explanations for different categories and tasks, until the desired task is found
    for task in tasks_to_plot:
        for sub_dir in sorted(p.iterdir()):
            idp_type = str(sub_dir).split("/")[-1]
            p2 = Path(str(sub_dir))
            for sub_dir_field in p2.iterdir():
                raw_field = sub_dir_field.name
                if task == raw_field:
                    # get atlas target in mni space, atlas keys and field names for grount-truth region
                    # note that we use the standard mni space atlas as ground truth target when plotting mean explanations,
                    # while we use the atlas targets inverse warped to each participant’s linear MNI space when plotting single subject explanations
                    atlas_keys = []
                    if idp_type != "dummy_dis":
                        field = int(raw_field.split("_")[0])
                        if field < 27000:
                            atlas_id = "aseg"
                            array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction/setup_files/config_csvs/config_subcortical.csv"
                        
                        if field >= 27000: 
                            atlas_id = "aparc2009"
                            array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction/setup_files/config_csvs/config_cortical.csv"

                        config_df = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
                        atlas_key = config_df[config_df["field"] == field]["atlas_key"].values[0]
                        atlas_keys.append(atlas_key)

                        atlas_target = create_atlas_target(atlas_key, atlas_id, dilation_iter=dilation_iter_2mm)
                        
                        if plot_lat_target == True:
                            atlas_target = add_lat_target(config_df, field, atlas_id, atlas_target, dilation_iter = dilation_iter_2mm)

                        label_size = np.sum(atlas_target)
                        print(f"label_size = {label_size}")

                        atlas_label_img = nib.Nifti1Image(np.float32(atlas_target), np.eye(4))

                        print(raw_field)
                        field_string = config_df[config_df["field"] == field]["name"].values[0]
                        print(field_string)
                    
                    elif idp_type == "dummy_dis":
                        field_1 = int(raw_field.split("_")[1])
                        field_2 = int(raw_field.split("_")[4])

                        atlas_labels = []
                        field_strings = []
                        for field in [field_1, field_2]:
                        
                            if field < 27000:
                                atlas_id = "aseg"
                                array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv"

                            if field >= 27000: 
                                atlas_id = "aparc2009"
                                array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys.csv"

                            config_df = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
                            atlas_key = config_df[config_df["field"] == field]["atlas_key"].values[0]
                            atlas_keys.append(atlas_key)

                            atlas_target = create_atlas_target(atlas_key, atlas_id, dilation_iter=dilation_iter_2mm)
                            if plot_lat_target == True:
                                atlas_target = add_lat_target(config_df, field, atlas_id, atlas_target, dilation_iter=dilation_iter_2mm)

                            label_size = np.sum(atlas_target)
                            print(f"label_size = {label_size}")
                            
                            atlas_labels.append(atlas_target)

                            field_strings.append(config_df[config_df["field"] == field]["name"].values[0])


                        atlas_data_1 = atlas_labels[0]
                        atlas_data_2 = atlas_labels[1]
                        atlas_label = np.clip(atlas_data_1 + atlas_data_2, 0, 1)

                        atlas_label_img = nib.Nifti1Image(np.float32(atlas_label), np.eye(4))

                        field_string = f"high {field_strings[0]} and\nlow {field_strings[1]}"
                        print(field_string)

                    if field_string == "high Mean thickness of G-postcentral (right hemisphere) andlow Volume of Hippocampus (left hemisphere)":
                        field_string = "Artificial Disease 2"
                    elif field_string == "high Area of G-rectus (left hemisphere) andlow Volume of Caudate (left hemisphere)":
                        field_string = "Artificial Disease 2"


                    # plotting      
                    fig, axes = plt.subplots(12, 1, figsize=(16, 22))  # adjust figsize as needed

                    if idp_type == "high_contrast":
                        fig.suptitle(f"{xai_m} - {field_string}\n[expected atypicall behaviour because target is potentially too big/hard to correct]", fontsize=16, y=0.925)

                    else:
                        fig.suptitle(f"{xai_m} - {field_string}", fontsize=16, y=0.925)


                    print(xai_m)
                    path_to_expls = f"{expl_dir}/{idp_type}/{raw_field}/{xai_m}"

                    with open(f"{path_to_expls}/eids.npy", "rb") as file:
                        eids = np.load(file)

                    with open(f"{path_to_expls}/masked_expls.npy", "rb") as file:
                        expls = np.load(file)

                    cut_coords = np.linspace(10, 80, num=12, dtype="int16")
                    display_modes = ["x", "z"]
                    
                    axes[0].set_title("Mean Explanation")
                    plot_mean_expl(
                        expls,
                        nifti_masker,
                        atlas_label_img,
                        cut_coords=cut_coords,
                        display_modes=display_modes,
                        axes=[axes[0], axes[1]],
                        supress_subplots=True,
                    )
                    
                    for i in range(2, 11, 2):
                        
                        if i % 2 == 0:
                            current_subject = int(i/2)

                            axes[i].text(0.5, 1.02, f"Random Subject {current_subject}", transform=axes[i].transAxes, ha='center', va='bottom', fontsize=12)
                        else:
                            axes[i].text(0.5, 1.02, f" ", transform=axes[i].transAxes, ha='center', va='bottom', fontsize=12)
                        
                        plot_random_single_subjects(
                            expls,
                            eids,
                            nifti_masker,
                            atlas_keys,
                            warped_rois_path,
                            n=1,
                            cut_coords=cut_coords,
                            display_modes=display_modes,
                            axes=[axes[i], axes[i+1]],
                            supress_subplots=True,
                        )

                    # save plots          
                    save_dir = f"{plot_save_dir}/{xai_m}"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = f"/{save_dir}/{field_string}.pdf"
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
                    single_pdf_paths.append(save_path)

    # merge single task pdfs to large pdf
    merger = PdfWriter()

    for pdf in single_pdf_paths:
        merger.append(pdf)

    merger.write(f"{save_dir}/all_tasks.pdf")
    merger.close()