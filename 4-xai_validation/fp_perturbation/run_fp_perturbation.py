"""
This scripts triggers the Lighning Models' test_step() to compute explanations, perturb false positive regions and hemisphere-flipped baseline regions, and save predictions for each of these conditions for a given set of XAI methods and task.
"""

from pytorch_lightning import Trainer
import torch
import json
from brain_deform.lightning import BrainDataModule
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/3-compute_explanations')
from xai_helper_methods import load_masker, create_result_dir, init_faith_dict, add_atlas_to_model, get_full_atlas

model_type = "ResNet"
batch_size = 1
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/masker/10k_imgs_masker.pkl"
xai_methods = ["LRP_EpsilonGamma_0.1", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1"]
result_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/4-xai_validation/fp_perturbation/results"

if model_type == "ResNet": 
    import sys
    sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/models/ResNet')
    from main_fp_perturb import Model

full_atlas = get_full_atlas(res = "1mm")

cidp_dict = {"cort_thicknesses/27652_101":{"split_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/simple_split_aparc.json", 
                                          "seed_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/model_checkpoints/4cort_4subcort_2dummydis/cort_thicknesses/27652_101/epoch=31-step=150000.ckpt",
                                          "table_csv": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/aparc2009/table_27652_101.csv",
                                          "target_mean": 0.0007857433849485547,
                                          "target_std": 0.639308312084868},
                                          }


for idp_id in cidp_dict:
    print(f"Performing FP perturbation for {idp_id}", flush=True)
    res_path = f"{result_path}/{idp_id}"
    
    split_path = cidp_dict[idp_id]["split_path"]
    seed_path = cidp_dict[idp_id]["seed_path"]
    table_csv = cidp_dict[idp_id]["table_csv"]
    target_mean = cidp_dict[idp_id]["target_mean"]
    target_std = cidp_dict[idp_id]["target_std"]
    
    create_result_dir(res_path)

    with open(f'{res_path}/xai_setup_dict.json', 'w') as f:
        json.dump(cidp_dict[idp_id], f)

    # load masker
    nifti_masker = load_masker(masker_path)
    mask_size = nifti_masker.n_elements_

    # for storing fp-perturbation results
    faith_dict, n_test = init_faith_dict(split_path, xai_methods)

    # load trained model
    model = Model.load_from_checkpoint(seed_path, map_location=torch.device('cuda:0')).to("cuda:0")
    
    # add atlas target
    field = int(idp_id.split("/")[-1].split("_")[0])
    add_atlas_to_model(field, model, full_atlas)

    # pass attributes needed for fp replacement
    model.test_mode = "fp-perturb"
    model.target_mean = target_mean
    model.target_std = target_std
    model.faith_dict = faith_dict
    model.expl_methods = xai_methods
    model.res_path = res_path
    model.batch_size = batch_size
    model.nifti_masker = nifti_masker
    model.idp_id = idp_id
    model.n_test = n_test
    model.eval()

    # create data module
    data_module = BrainDataModule(data_table_path=table_csv,
                            split_path=split_path,
                            batch_size=batch_size,
                            registration="linear",
                            augmentation_probability=0,
                            index_column="eid",
                            t1_column="raw",
                            coefs_to_mni_column="coefs_to_mni",
                            target_column="target",
                            drop_last=False)


    data_module.setup()
    data_loader = data_module.test_dataloader()
    
    # pipe test set
    trainer = Trainer(devices=[0], accelerator="gpu", inference_mode=False)
    trainer.test(model=model, datamodule=data_module, verbose=True)