"""
This scripts computes explanations for a pretrained ResNet and ImageNet test and validation images.
RMA scores are computed for each explanation using object segmentation masks
"""
from datetime import datetime
from torch.nn.functional import softmax
from torchvision import models
from xai_methods import get_explanation
from natural_image_helper_methods import get_imgnet_idx, get_cat_name, sample_shap_bg, Preprocessor, create_dir, get_label_from_category, prepare_segmentation_mask, get_img, save_xpl_image_w_neg
from pathlib import Path
import numpy as np
import json
import pandas as pd
import sys
from pathlib import Path
import torch
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation')
from metrics import get_relevance_mass_accuracy_batch    

# which xai methods to evaluate
xai_methods = ["DeepLiftShap", "LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop", "DeepLift", "GuidedBackprop", "GuidedGradCam", "GradCAM", "GradCAM_smooth", "GradCAM_l1", "GradCAM_l2", "GradCAM_l3", "InputXGradient"]

# compute explanations on gpu?
gpu = False

# how much to dilate segmentation masks (to include boundary voxels in ground truth masks)
dilation_iters = 3

# path to imagenet test images
test_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/XAI-imagenet-benchmark/Imagenet-S-prepared/ImageNetS919/test"

# path to imagenet val images
val_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/XAI-imagenet-benchmark/Imagenet-S-prepared/ImageNetS919/validation"

# path to imagenet segmentation masks (from https://github.com/LUSSeg/ImageNet-S)
seg_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/XAI-imagenet-benchmark/Imagenet-S-prepared/validation-segmentation"

# where to save explanations
expl_base_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/5-natural_image_comparison/results/expls"

# where to save validation results
val_result_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/5-natural_image_comparison/results/val_dfs"

# file matching imagenet indices to class ids
imagenetS_indices_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/5-natural_image_comparison/imagenet_files/ImageNetS_categories_im919.txt"

# file matching imagenet indices to class-strings
imagenet_class_index_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/5-natural_image_comparison/imagenet_files/imagenet_class_index.json"

# load index-class_id-class_string mappings
with open(imagenetS_indices_path, "r") as file:
    imagenetS_indices = file.readlines()
    imagenetS_indices = [x.strip("\n") for x in imagenetS_indices]

with open(imagenet_class_index_path) as handle:
    imagenet_class_index = json.loads(handle.read())      

# Load the pre-trained ResNet50 model
if gpu:
    model = models.resnet50(pretrained=True).to("cuda")
else:
    model = models.resnet50(pretrained=True)

# Set to evaluation mode
model.eval()

# data frame to save evaluation 
large_df = pd.DataFrame(columns=['img_filename', 'expl_path', 'category', 'xai_m', 'score', 'probability_correct_class', 'correct_prediction', 'th'])

preprocessor = Preprocessor(gpu)

# sample background distribution for deeplift shap
shap_baselines = sample_shap_bg(preprocessor, gpu, n_bg = 150)

# iterate over segmentation masks
p = Path(seg_path)

# first iteration level: imagenet categories
for sub_dir in p.iterdir():
    # get category id from folder name
    category = str(sub_dir).split("/")[-1]
    p2 = Path(str(sub_dir))

    # get category_string from category id
    category_name = get_cat_name(category, imagenet_class_index)
    print(category_name)

    xpl_save_path = f"{expl_base_path}/{category}"
    create_dir(xpl_save_path)

    # second iteration level: segmentation masks
    for sub_dir_seg in p2.iterdir():
        # Get the current time
        current_time = datetime.now().time()

        # Print the current time
        print("Current time:", current_time)
        
        # load segmentation mask from file and extract mask related to class object (sometimes, there are multiple classe's objects)
        binary_segmentation_mask = prepare_segmentation_mask(sub_dir_seg, imagenetS_indices, category, dilation_iters)
        flat_binary_segmentation_mask = binary_segmentation_mask.flatten()[None, :]
        
        # returns image related to segmentation mask
        img = get_img(sub_dir_seg, test_path, val_path, category)
        input_tensor = preprocessor.preprocess_img(img)

        # get integer label/class index for current category
        imgnet_label = get_imgnet_idx(category, imagenet_class_index)

        # pass image to model
        with torch.no_grad():
            outputs = model(input_tensor)
            # get metrics for xai evaluation later
            logit_true_class = float(outputs[0, imgnet_label])
            probability_true_class = float(softmax(outputs)[0, imgnet_label])
            prediction = int(torch.argmax(outputs))
            prediction_correct = prediction == imgnet_label

        for xai_m in xai_methods:
            # xai method specific path for saving explanations
            xpl_xai_m_path = f"{xpl_save_path}/{xai_m}"
            create_dir(xpl_xai_m_path)

            # get explanations
            xpl = get_explanation(model, input_tensor, xai_m, imgnet_label, shap_baselines)
            
            shape = xpl.shape

            # concatenate over channel dimension, if expls is no single channel image by default
            if xai_m not in ["GradCAM", "GradCAM_l1", "GradCAM_l2", "GradCAM_l3", "HiResCAM", "GradCAMPlusPlus", "GradCAM_smooth", "HiResCAM_smooth", "GradCAMPlusPlus_smooth"]:
                xpl =  xpl.squeeze().transpose(1, 2, 0)
                xpl = np.sum(xpl, axis = -1)

            else:
                xpl = xpl.squeeze()
                            
            # save expl
            seg_mask_path = str(sub_dir_seg).split("/")[-1]
            xpl_save_path_full = f"{xpl_xai_m_path}/{seg_mask_path}"
            xpl_save_path_full = xpl_save_path_full.replace(".png", ".npy")
            np.save(xpl_save_path_full, xpl.astype(np.float16))
                
            # get RMA scores for different cutoffs
            for th in [0, 70, 80, 90, 95, 99]:
                xpl_flat = xpl.flatten()[None, :]

                xpl = post_proc_expl(xpl, th)

                # get RMA score
                score = get_relevance_mass_accuracy_batch(xpl_flat, flat_binary_segmentation_mask)[0]
                
                # save explanation images
                seg_mask_path = str(sub_dir_seg).split("/")[-1]
                xpl_img_path = f"{xpl_xai_m_path}/{seg_mask_path}"
                img_ =  input_tensor.cpu().numpy().squeeze().transpose(1, 2, 0)
                img_ = np.mean(img_, axis = 2)
                save_xpl_image_w_neg(xpl, img_, binary_segmentation_mask, xpl_img_path)
                
                # add validation results to results df
                img_filename = str(sub_dir_seg).split("/")[-1]
                new_row = pd.DataFrame([{'img_filename':img_filename, 'expl_path':xpl_save_path,'category':category, 'xai_m':xai_m, 'score':score, 'probability_correct_class':probability_true_class, 'correct_prediction':prediction_correct, 'th':th}])
                large_df = pd.concat([large_df, new_row], ignore_index=True)

    # save results
    large_df.to_pickle(val_result_path)
