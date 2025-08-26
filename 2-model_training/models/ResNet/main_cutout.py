from multiprocessing import freeze_support
import numpy as np
import torch
from pytorch_lightning.cli import LightningCLI, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import torch.nn as nn
import sys
from resnet import generate_model
import nibabel as nib
from brain_deform.lightning import BrainDataModule
import wandb
import os
import yaml
from zennit.composites import EpsilonPlusFlat, GuidedBackprop, ExcitationBackprop, EpsilonPlus, EpsilonGammaBox, EpsilonAlpha2Beta1Flat, EpsilonAlpha2Beta1
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient, SmoothGrad
from nilearn.plotting import plot_stat_map, show
import skimage
import pandas as pd
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction/')
from atlas_methods import create_atlas_target, add_lat_target

freeze_support()

class Model(LightningModule):
    def __init__(self, model_depth=50,
                 one_cycle=False,
                 max_lr=0,
                 reduce_on_plat=True,
                 reduce_on_plat_start_lr=0.0001,
                 reduce_on_plat_red= 0.5,
                 wandb_run_name=None,
                 slurm_run=1,
                 target_mean = 64.27008,
                 target_std = 7.7529,
                 wandb_proj = "swin-transformer-brain-age",
                 field = 27352,
                 dilation_iter = 2,
                 ):
        
        super().__init__()

        # generate model
        self.model = generate_model(model_depth=model_depth, n_input_channels=1, n_classes=1)

        if wandb_run_name is not None:
            # save config to wandb
            with open(f'/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/run_models/{slurm_run}/config.yaml', 'r') as stream:
                config_dictionary=yaml.safe_load(stream)
                
            wandb_id = os.environ["SLURM_JOB_ID"]
            self.rank = os.environ.get('LOCAL_RANK')
            self.task_id = os.environ.get('SLURM_PROCID')
            print(self.task_id)

            run = wandb.init(
                id = wandb_id + "_" + str(self.task_id),
                resume="allow",
                name = wandb_run_name,
                # Set the project where this run will be logged
                project=wandb_proj,
                # Track hyperparameters and run metadata
                config=config_dictionary,
                entity="students")

        # setup loss
        self.mse = nn.MSELoss()
        
        # mean and std for target normalization
        self.target_mean = target_mean
        self.target_std = target_std
        
        # mean and std for input normalization
        self.train_mean = 232.5522
        self.train_std = 414.4120

        # to save predictions, eids, and explanations during testing
        self.test_predictions = []
        self.test_eids = []
        self.xai_dict = None
        self.test_mode = "xai"
        
        # initialization of optimizer and scheduler
        self.one_cycle = one_cycle
        self.max_lr = max_lr
        self.reduce_on_plat = reduce_on_plat
        self.reduce_on_plat_start_lr = reduce_on_plat_start_lr
        self.reduce_on_plat_red = reduce_on_plat_red
        self.val_epoch = -1
        self.test_counter = 0

        # setup atlas for cutout of ground-truth region
        if field < 27000:
            atlas_id = "aseg"
            array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/xai_benchmark_array_job_all_aseg_fields.csv"
                    
        if field >= 27000: 
            atlas_id = "aparc2009"
            array_job_csv_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/atlas_files/aparc2009_all_fields_w_matched_atlas_keys.csv"

        config_df = pd.read_csv(array_job_csv_path, header=None, names=["id", "field", "atlas_key", "name"])
        atlas_key = config_df[config_df["field"] == field]["atlas_key"].values[0]
                    
        atlas = create_atlas_target(atlas_key, atlas_id, dilation_iter=dilation_iter, one_mm = True)
        atlas_target = add_lat_target(config_df, field, atlas_id, atlas, dilation_iter = dilation_iter, one_mm = True)

        self.atlas_target = atlas_target

    def forward(self, x):
        # normalize
        x_norm = (x - self.train_mean)/self.train_std

        return self.model(x_norm).squeeze()

    def training_step(self, batch, batch_idx):
        # use augmented version for training
        (_, x, y, eid), _ = batch
        
        # cutout ground-truth region
        batch_size = x.shape[0]
        atlas_target = torch.tensor(np.repeat(np.expand_dims(self.atlas_target[None, :, :, :], axis=0), batch_size, axis=0)).to("cuda")
        mask = (atlas_target == 1)
        x[mask] = 0
        
        # get the loss
        y_hat = self.forward(x)

        # Normalize loss with precomputed values
        y_norm = (y - self.target_mean)/self.target_std
        loss = self.mse(y_hat, y_norm)

        self.log("train_loss", loss)
        
        if self.one_cycle == True:
            wandb.log({"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]}, step=self.trainer.global_step)
        else:
            lightning_optimizer = self.optimizers()  # self = your model
            for param_group in lightning_optimizer.optimizer.param_groups:
                self.log(f"leanring rate", param_group['lr'])
                wandb.log({"train_loss": loss, "lr": param_group['lr']}, step=self.trainer.global_step)
                        
        return loss

    def validation_step(self, batch, batch_idx):
        # use unaugmented version for validation
        (x, _, y, eid ), _ = batch

        # cutout ground-truth region
        batch_size = x.shape[0]
        atlas_target = torch.tensor(np.repeat(np.expand_dims(self.atlas_target[None, :, :, :], axis=0), batch_size, axis=0)).to("cuda")
        mask = (atlas_target == 1)
        x[mask] = 0
        
        # get the loss
        y_hat = self.forward(x)
        
        # denormalize
        y_hat_scalar = y_hat*self.target_std + self.target_mean

        # Normalize loss with precomputed values
        y_norm = (y - self.target_mean)/self.target_std
        loss = self.mse(y_hat, y_norm)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        
        return {"val_loss": loss, "diff": (y - y_hat_scalar), "target": y}

    def validation_epoch_end(self, outputs):
        target_mean = torch.cat([x["target"] for x in outputs]).mean(dim=0)
        ss_tot = torch.cat([(x["target"] - target_mean) ** 2 for x in outputs]).mean(
            dim=0
        )
        mse = torch.cat([x["diff"] ** 2 for x in outputs]).mean(dim=0)
        mae = torch.cat([torch.abs(x["diff"]) for x in outputs]).mean(dim=0)
        r2 = 1.0 - mse / ss_tot

        self.log(f"val_mse", mse, sync_dist=True)
        self.log(f"val_mae", mae, sync_dist=True)
        self.log(f"val_r2", r2, sync_dist=True)
        self.log("epoch", self.trainer.current_epoch, sync_dist=True)

        wandb.log({"val_mse": mse, "val_mae": mae, "val_r2": r2, "epoch": self.trainer.current_epoch})
            
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.reduce_on_plat_start_lr, weight_decay=0)

        if self.reduce_on_plat:
            self.scheduler = ReduceLROnPlateau(optimizer, monitor="val_mae", mode='min', factor=self.reduce_on_plat_red, patience=50)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val_mae",
                    "frequency": 1
                },
            }
        
        if self.one_cycle:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.trainer.max_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step"
                },
            }

    def remember_explanation(self, expl, batch_counter, eid, xai_method):
        rescaled_expl = skimage.transform.downscale_local_mean(expl, (2,2,2), cval=0)
        
        rescaled_expl_img = nib.Nifti1Image(rescaled_expl, affine=np.eye(4))
        masked_expl = self.nifti_masker.transform(rescaled_expl_img)

        self.xai_dict[xai_method]["masked_expl"][self.test_counter + batch_counter, :] = masked_expl[0,:]
        self.xai_dict[xai_method]["eids"][self.test_counter + batch_counter] = eid[batch_counter].detach().cpu().numpy()


    def test_step(self, batch, batch_idx):
        (x, _, y, eid ), _ = batch
        
        # mask out ROI
        batch_size = x.shape[0]
        atlas_target = torch.tensor(np.repeat(np.expand_dims(self.atlas_target[None, :, :, :], axis=0), batch_size, axis=0)).to("cuda")
        mask = (atlas_target == 1)
        x[mask] = 0

        # get test predictions and save
        self.save_predictions(x, eid)
        self.save_labels(y)

    def save_labels(self, y):
        y_scalar = y.detach().cpu().numpy()

        if y_scalar.size == 1:
            self.test_labels.append(float(y_scalar))

        else:

            for i, y_scalar_ in enumerate(y_scalar):
                self.test_labels.append(y_scalar_)

        with open(f'{self.res_path}/labels.npy', 'wb') as f:
            np.save(f, self.test_labels)


    def save_predictions(self, x, eid):
        y_hat = self.forward(x)
        y_hat_scalar = y_hat*self.target_std + self.target_mean
        y_hat_scalar = y_hat_scalar.detach().cpu().numpy()
        eid = eid.detach().cpu().numpy()

        if y_hat_scalar.size == 1:
            self.test_predictions.append(float(y_hat_scalar))
            self.test_eids.append(int(eid))

        else:

            for i, y_hat_ in enumerate(y_hat_scalar):
                self.test_predictions.append(y_hat_)
                self.test_eids.append(eid[i])

        with open(f'{self.res_path}/predictions.npy', 'wb') as f:
            np.save(f, self.test_predictions)
    
        with open(f'{self.res_path}/prediction_eids.npy', 'wb') as f:
            np.save(f, self.test_eids)

if __name__ == '__main__':

    cli = LightningCLI(Model, BrainDataModule, save_config_callback=None)

# python main.py fit -c config.yaml
