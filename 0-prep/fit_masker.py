"""
script to fit brain masker
"""

from nilearn.maskers import NiftiMasker
from os import listdir
import pickle

# these are 10k .nii images in 2mm MNI space
path_to_imgs = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/lin_reg_imgs/int_downsampling"
masker_save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/brain_masker.pkl"

smoothing_fwhm = 2

img_paths = []
for f in listdir(path_to_imgs):
    if f != "eids.pkl":
        img_paths.append(f"{path_to_imgs}/{f}")


nifti_masker_fitted = NiftiMasker(smoothing_fwhm=smoothing_fwhm)
nifti_masker_fitted.fit(img_paths)

with open(masker_save_path, 'wb') as file:
    pickle.dump(nifti_masker_fitted, file)