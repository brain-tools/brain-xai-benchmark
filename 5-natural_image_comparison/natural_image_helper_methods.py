import torch
from PIL import Image
import random
import os
from scipy import ndimage
from matplotlib import pyplot as plt
import os
import torchvision.transforms as transforms
import numpy as np

def post_proc_expl(expl, th, scale_perc=None):
    # only look at features that increase current class' logit
    expl[expl < 0] = 0
    
    if scale_perc is not None:
        scale_percentile = np.percentile(expl, scale_perc)
        expl = expl/scale_percentile

    cutoff_values = np.percentile(expl, th)

    mask = expl < cutoff_values
    expl[mask] = 0
    
    return expl

def get_img_path(test_path, val_path, filename):
    
    image_path_test = os.path.join(test_path, filename.replace(".png", ".JPEG")) 
    image_path_val = os.path.join(val_path, filename.replace(".png", ".JPEG"))

    try:
        return image_path_test
    except:
        z=1
    try:
        return image_path_val
    except:
        print(f"image relaetd to mask ({filename}) not found.")

def get_imgnet_idx(cat, imagenet_class_index):
    for key, value in imagenet_class_index.items():
        if value[0] == cat:
            return int(key)
        
def get_cat_name(cat, imagenet_class_index):
    for key, value in imagenet_class_index.items():
        if value[0] == cat:
            return value[1]
        
def sample_deep_lift_background(root_folder, sample_size):
    all_files = []
    
    # Walk through the two-level folder structure
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if os.path.relpath(dirpath, root_folder).count(os.sep) <= 1:  # Limit to two levels
            for filename in filenames:
                all_files.append(os.path.join(dirpath, filename))
    
    # If there are fewer files than the sample size, adjust the sample size
    sample_size = min(sample_size, len(all_files))
    
    # Randomly sample files
    random.seed(42)
    sampled_files = random.sample(all_files, sample_size)
    
    return sampled_files

def post_proc_expl(expl, th, scale_perc=None):
    # only look at features that increase current class' logit
    expl[expl < 0] = 0
    
    if scale_perc is not None:
        scale_percentile = np.percentile(expl, scale_perc)
        expl = expl/scale_percentile

    cutoff_values = np.percentile(expl, th)

    mask = expl < cutoff_values
    expl[mask] = 0
    
    return expl



def get_label_from_category(category,imagenet_class_index):
    for key, value in imagenet_class_index.items():
        if value[1] == category:
            return int(key)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_segmentation_mask(sub_dir_seg, imagenetS_indices, category, dilation_iters):
    transform_mask = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ])

    seg = np.array(Image.open(sub_dir_seg))
    s_idx = imagenetS_indices.index(category) + 1

    segmentation_id = seg[:, :, 1] * 256 + seg[:, :, 0]
    binary_segmentation_mask = segmentation_id == s_idx
    
    binary_segmentation_mask = torch.tensor(binary_segmentation_mask*1)

    binary_segmentation_mask = binary_segmentation_mask[None, :, :]

    binary_segmentation_mask = transform_mask(binary_segmentation_mask)

    binary_segmentation_mask = np.array(binary_segmentation_mask)

    binary_segmentation_mask = ndimage.binary_dilation(binary_segmentation_mask, iterations=dilation_iters)

    return binary_segmentation_mask

def get_img(sub_dir_seg, test_path, val_path, category):

    filename = str(sub_dir_seg).split("/")[-1]

    image_path_test = os.path.join(test_path, category, filename.replace(".png", ".JPEG")) 
    image_path_val = os.path.join(val_path, category, filename.replace(".png", ".JPEG"))

    try:
        img = Image.open(image_path_test)
    except:
        print("not a test image")
    try:
        img = Image.open(image_path_val)
    except:
        print("not a val image")

    return img

def save_xpl_image_w_neg(xpl, img_, binary_segmentation_mask, path):
    fig, ax = plt.subplots()
    ax.imshow(img_, cmap="Greys")
    
    xpl = xpl/np.percentile(np.abs(xpl), 80)
    
    cutoff = 0.05
    xpl[(xpl < cutoff) & (xpl > -cutoff)] = 0
    xpl[xpl > 5] = 5
    xpl[xpl < -5] = -5
        
    alpha_mask = np.where(xpl == 0, 0, 0.5)
    cax = ax.imshow(xpl, cmap='seismic', alpha=alpha_mask, vmin=-5, vmax=5)
    
    fig.colorbar(cax)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

def save_xpl_image(xpl, img_, binary_segmentation_mask, path):
    fig, ax = plt.subplots()
    ax.imshow(img_, cmap="Greys")
    
    xpl = xpl/np.percentile(xpl, 95)
    
    cutoff = 0.3
    xpl[xpl < cutoff] = 0
    xpl[xpl > 5] = 5
    
    alpha_mask = np.where(xpl == 0, 0, 0.5)

    cax = ax.imshow(xpl, cmap='jet', alpha=alpha_mask, vmin=0, vmax=5)
    fig.colorbar(cax)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def get_channel_dim(img):
    return np.array(img).shape[-1]


class Preprocessor:
    def __init__(self, gpu):
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_gs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.gpu = gpu
    
    def preprocess_img(self, img):
        
        channel_dim = get_channel_dim(img)
            
        if channel_dim == 3:
            input_tensor = self.transform(img).unsqueeze(0)
            
        elif channel_dim > 3:
            img_array = np.array(img, dtype="float")
            img_array = np.stack((img_array, img_array, img_array), axis=0)
            img_tensor = torch.FloatTensor(img_array)
            input_tensor = self.transform_gs(img_tensor).unsqueeze(0)

        if self.gpu:
            return input_tensor.to("cuda")
        else:
            return input_tensor
        
def sample_shap_bg(preprocessor, gpu, n_bg = 200):

    bg_paths = sample_deep_lift_background("/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/XAI-imagenet-benchmark/Imagenet-S-prepared/ImageNetS919", n_bg)

    baselines = torch.randn(n_bg, 3, 224, 224)

    for i, img_path in enumerate(bg_paths):
        img = Image.open(img_path)
        img = preprocessor.preprocess_img(img)
        baselines[i, :, :, :] = img    

    if gpu:
        return baselines.to("cuda")
    else:
        return baselines
    
def get_image_paths_from_score_df(score_df):

    image_paths = []

    for index, row in score_df.iterrows():

        category = row["category"]
        filename = row["img_filename"]

        image_paths.append(f"{category}/{filename}")
        

    print(set(image_paths))

    image_paths = list(set(image_paths))

    return image_paths


def get_segmentation_mask(sub_dir_seg, imagenetS_indices, category, dilation_iters):
    transform_mask = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ])

    seg = np.array(Image.open(sub_dir_seg))
    s_idx = imagenetS_indices.index(category) + 1

    segmentation_id = seg[:, :, 1] * 256 + seg[:, :, 0]

    n_objects = len(np.unique(segmentation_id))

    binary_segmentation_mask = segmentation_id == s_idx

    
    binary_segmentation_mask = torch.tensor(binary_segmentation_mask*1)

    binary_segmentation_mask = binary_segmentation_mask[None, :, :]

    binary_segmentation_mask = transform_mask(binary_segmentation_mask)

    binary_segmentation_mask = np.array(binary_segmentation_mask)

    # plt.imshow(np.squeeze(binary_segmentation_mask))
    # plt.show()

    # dilate
    if dilation_iters > 0:
        binary_segmentation_mask = ndimage.binary_dilation(binary_segmentation_mask, iterations=dilation_iters)

    return binary_segmentation_mask, n_objects - 1, seg