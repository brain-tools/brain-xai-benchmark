"""
This scripts triggers the Lighning Models' test_step() to compute explanations for a given set of XAI methods and task.
"""

from pytorch_lightning import Trainer
import torch
import json
from brain_deform.lightning import BrainDataModule
import sys
from xai_helper_methods import load_masker, create_result_dir, init_xai_dict

# which model architecture to use
model_type = "DenseNet" # "ResNet" or "DenseNet"

batch_size = 1
masker_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/masker/10k_imgs_masker.pkl"
xai_methods = ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop", "DeepLift", "DeepLift_mean_img", "GuidedBackprop", "GuidedGradCam", "GradCAM", "InputXGradient", "GradCAM_l3"]
# just for diagnostic/debugging plots
smooth_sigma_expl = 1.5

expl_save_dir = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/3-compute_explanations/results/exps"

if model_type == "ResNet": 
    import sys
    sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/models/ResNet')
    from main import Model

elif model_type == "DenseNet":
    sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/models/DenseNet')
    from main_densenet import Model


cidp_dict = {"seed_2/disease":{"split_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/brain_age_disease_diff/used_disease_as_test_split_T2.json",
                          # "seed_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/model_checkpoints/T2_brain_age/seed_2/epoch=43-step=150000.ckpt",
                          "seed_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/model_checkpoints/densenet/brain_age/seed_3/lightning_logs/version_5025487/checkpoints/epoch=42-step=150000.ckpt",
                          "table_csv": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/T2_age_table.csv",
                          "target_mean":64.5549726448325,
                          "target_std":7.73845669467534,
                          },
            "seed_2/control":{"split_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/split_matched_controls_T2.json",
                          "seed_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/model_checkpoints/T2_brain_age/seed_2/epoch=43-step=150000.ckpt",
                          "table_csv": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/T2_age_table.csv",
                          "target_mean":64.5549726448325,
                          "target_std":7.73845669467534,
                          },
            "seed_3/disease":{"split_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/brain_age_disease_diff/used_disease_as_test_split_T2.json",
                          "seed_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/model_checkpoints/T2_brain_age/seed_3/epoch=43-step=150000.ckpt",
                          "table_csv": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/T2_age_table.csv",
                          "target_mean":64.5549726448325,
                          "target_std":7.73845669467534,
                          },
            "seed_3/control":{"split_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/split_matched_controls_T2.json",
                          "seed_path": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/model_checkpoints/T2_brain_age/seed_3/epoch=43-step=150000.ckpt",
                          "table_csv": "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/training_files/T2_flair/T2_age_table.csv",
                          "target_mean":64.5549726448325,
                          "target_std":7.73845669467534,
                          },

                        }

for idp_id in cidp_dict:
    print(f"Computing explanations for {idp_id}", flush=True)
    res_path = f"{expl_save_dir}/{model_type}/brain_age_disease_diff/T2_models/{idp_id}"
    
    split_path = cidp_dict[idp_id]["split_path"]
    seed_path = cidp_dict[idp_id]["seed_path"]
    table_csv = cidp_dict[idp_id]["table_csv"]
    target_mean = cidp_dict[idp_id]["target_mean"]
    target_std = cidp_dict[idp_id]["target_std"]
    
    create_result_dir(res_path)

    with open(f'{res_path}/xai_setup_dict.json', 'w') as f:
        json.dump(cidp_dict[idp_id], f)

    nifti_masker = load_masker(masker_path)
    mask_size = nifti_masker.n_elements_

    # for storing the explanations across test subjects
    xai_dict, n_test = init_xai_dict(split_path, xai_methods, mask_size)

    # load trained model
    model = Model.load_from_checkpoint(seed_path, map_location=torch.device('cuda:0')).to("cuda:0")
    
    # pass attributes needed for computing explanations
    model.target_mean = target_mean
    model.target_std = target_std
    model.xai_dict = xai_dict
    model.expl_methods = xai_methods
    model.res_path = res_path
    model.smooth_sigma = smooth_sigma_expl
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
    print(trainer.logger.save_dir)
    trainer.test(model=model, datamodule=data_module, verbose=True)