"""
This script linearly registers 3D MRI brain images to MNI space, downscales them by factor 2, and saves them as .nii or .npy (after masking).
"""

from brain_deform.lightning import BrainDataModule
from brain_deform.augmentation import process_image
import nibabel as nib
import numpy as np
import pickle
import skimage



# save as .nii or .npy (masked)
save_type = "nii" 

if save_type == "npy":
    save_dir = '/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/masked'
    
    masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/brain_masker/brain_masker.pkl"
    with open(masker_path, 'rb') as file:
        nifti_masker = pickle.load(file)


elif save_type == "nii":
    save_dir = '/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/files/lin_reg_imgs/nii'



data_module = BrainDataModule(
                # table with eids and image paths
                data_table_path="/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/table_single_visit.csv",
                # which images to register
                split_path="/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/all_test_eids.json",
                batch_size=1,
                registration="linear",
                augmentation_probability = 0,
                index_column= "eid",
                t1_column= "raw",
                coefs_to_mni_column= "coefs_to_mni",
                target_column= "target",
                drop_last=False)


data_module.setup()

data_loader = data_module.test_dataloader()

eids = []

for i, x in enumerate(data_loader):

    data_main, data_aux, data_aug = x
    img_main, coefs_main, premat_main, postmat_main, label_main, eids_current = data_main
    img_aux, coefs_aux, premat_aux, postmat_aux, label_aux, _ = data_aux
    coefs_aug, premat_aug = data_aug
    
    img_main_registered, img_main_registered_augmented = process_image(
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

    print(f"processing image {i}")
    
    for j in range(0, img_main_registered.shape[0]):

        img_data = img_main_registered.cpu().numpy()[j,0,:,:,:]
        
        rescaled_img_data = skimage.transform.downscale_local_mean(img_data, (2,2,2), cval=0)
        
        rescaled_image = nib.Nifti1Image(rescaled_img_data, affine=np.eye(4))

        eid = int(eids_current[j])


        if save_type == "npy":
            masked_rescaled_image = nifti_masker.transform(rescaled_image)        
            np.save(f'{save_dir}/{eid}.npy', masked_rescaled_image)

        elif save_type == "nii":    
            nib.save(rescaled_image, f'{save_dir}/{eid}.nii')
        
        eids.append(int(eid))

        with open(f'{save_dir}/eids.pkl', 'wb') as file:
            pickle.dump(eids, file)
