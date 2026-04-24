from multiprocessing import freeze_support
import numpy as np
import torch
from pytorch_lightning.cli import LightningCLI, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import torch.nn as nn
from resnet import generate_model
import nibabel as nib
from brain_deform.lightning import BrainDataModule
import wandb
import os
import yaml
from nilearn.plotting import plot_stat_map, show
import skimage
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/3-compute_explanations')
from xai_methods import ExplComputer
from xai_helper_methods import get_full_atlas

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
                 wandb_proj = "paper-xai-benchmark",
                 ):
        
        super().__init__()

        # generate model
        self.model = generate_model(model_depth=model_depth, n_input_channels=1, n_classes=1)

        if wandb_run_name is not None:
            # save config to wandb
            with open(f'/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/run_models/{slurm_run}/config.yaml', 'r') as stream:
                config_dictionary=yaml.safe_load(stream)
            
            # setup wandb logging:
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

        # setup mean image for DeepLift
        self.mean_img = np.load('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/avg_brains/mean_img.npy')
        self.mean_img = (self.mean_img - self.train_mean)/self.train_std
        self.mean_img = np.expand_dims(self.mean_img, axis=0)
        self.mean_img = np.expand_dims(self.mean_img, axis=0)
        self.mean_img = torch.from_numpy(self.mean_img)

        # get brain atlas for AtlasOcclusion
        self.full_atlas = get_full_atlas(res = "1mm")
        self.expl_computer = ExplComputer(model_type="ResNet", train_mean=self.train_mean, train_std=self.train_std, deep_lift_mean_img= self.mean_img, full_atlas=self.full_atlas)

    def forward(self, x):
        # normalize
        x_norm = (x - self.train_mean)/self.train_std

        return self.model(x_norm).squeeze()

    def training_step(self, batch, batch_idx):
        # use augmented version for training
        (_, x, y, eid), _ = batch

        y_hat = self.forward(x)

        # Normalize label with precomputed values
        y_norm = (y - self.target_mean)/self.target_std
        loss = self.mse(y_hat, y_norm)
        
        self.log("train_loss", loss)

        if self.one_cycle == True:
            wandb.log({"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]}, step=self.trainer.global_step)
        
        else:
            lightning_optimizer = self.optimizers()
            for param_group in lightning_optimizer.optimizer.param_groups:
                self.log(f"leanring rate", param_group['lr'])
                wandb.log({"train_loss": loss, "lr": param_group['lr']}, step=self.trainer.global_step)
                        
        return loss

    def validation_step(self, batch, batch_idx):
        # use unaugmented version for validation
        (x, _, y, eid ), _ = batch
        y_hat = self.forward(x)
        
        # normalize label
        y_norm = (y - self.target_mean)/self.target_std
        loss = self.mse(y_hat, y_norm)

        # denormalize
        y_hat_scalar = y_hat*self.target_std + self.target_mean

        # note: val loss and target are in normalized space, diff is not.
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
        """
        Downscale and mask explanation, then save it to the xai_dict.
        """
        rescaled_expl = skimage.transform.downscale_local_mean(expl, (2,2,2), cval=0)
        
        rescaled_expl_img = nib.Nifti1Image(rescaled_expl, affine=np.eye(4))
        masked_expl = self.nifti_masker.transform(rescaled_expl_img)

        self.xai_dict[xai_method]["masked_expl"][self.test_counter + batch_counter, :] = masked_expl[0,:]
        self.xai_dict[xai_method]["eids"][self.test_counter + batch_counter] = eid[batch_counter].detach().cpu().numpy()


    def test_step(self, batch, batch_idx):
        """
        Test step to compute explanations and infer test-set performance.
        """
        if self.test_mode == "xai":
            (x, _, y, eid ), _ = batch
            # coefs for atlas warping in AtlasOcclusion
            coefs = _[3]

            for xai_method in self.expl_methods:
                batch_expl = batch_expl = self.expl_computer.get_explanation(self.model, x, xai_method, coefs=coefs)
                
                for i in range(0, batch_expl.shape[0]):
                    
                    expl = batch_expl[i, :, :, :]

                    # save raw explanation (will be written to disk later)
                    self.remember_explanation(expl, i, eid, xai_method)

                    # just for debugging/diagnostic plots:
                    # abs contributions 
                    expl = np.abs(expl)

                    # smooth
                    # expl = gaussian_filter(expl, sigma=self.smooth_sigma)

                    # normalize
                    expl = expl/np.percentile(expl, 95)
                    
                    if self.test_counter < 30:
                        
                        current_eid = int(eid[i].detach().cpu().numpy())
                        self.plot_xai_instance(expl, x, xai_method, i, eid=current_eid)
                        
                    self.xai_dict[xai_method]["mean_expl"] += expl

            self.save_predictions(x, eid)

            # last batch can be smaller than batch size        
            self.test_counter += x.shape[0]
            print(f"test counter {self.test_counter}", flush=True)

            # save explanations to disk
            if len(self.expl_methods) > 0:
                if self.test_counter == self.n_test:
                    print(f"saving explanations", flush=True)
                    
                    # save explanations to disk
                    self.save_expl_eval()
                    
                    # plot mean expl image
                    for xai_method in self.expl_methods:
                        mean_expl = self.xai_dict[xai_method]["mean_expl"]/(self.test_counter)
                        # random bg image
                        self.plot_xai_instance(mean_expl, x, xai_method, 0, mean_expl=True)

        
        if self.test_mode == "predictions":
            (x, _, y, eid ), _ = batch
            self.save_predictions(x, eid)
            self.save_labels(y)



    def save_predictions(self, x, eid):
        """
        Save predicitons and eids to disk.
        """
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

    def save_labels(self, y):
        y_scalar = y.detach().cpu().numpy()

        if y_scalar.size == 1:
            self.test_labels.append(float(y_scalar))

        else:

            for i, y_scalar_ in enumerate(y_scalar):
                self.test_labels.append(y_scalar_)

        with open(f'{self.res_path}/labels.npy', 'wb') as f:
            np.save(f, self.test_labels)


    def save_expl_eval(self):
        for xai_method in self.expl_methods:
            mean_expl = self.xai_dict[xai_method]["mean_expl"]/(self.test_counter)
            
            with open(f'{self.res_path}/{xai_method}/mean_expl.npy', 'wb') as f:
                np.save(f, mean_expl)

            with open(f'{self.res_path}/{xai_method}/masked_expls.npy', 'wb') as f:
                np.save(f, self.xai_dict[xai_method]["masked_expl"])
            
            with open(f'{self.res_path}/{xai_method}/eids.npy', 'wb') as f:
                np.save(f, self.xai_dict[xai_method]["eids"])
    
    def plot_xai_instance(self, expl, x, xai_method, batch_counter, eid=None, mean_expl=False):
        """
        Plot explanation for a single instance. Just for debugging/diagnostic purposes.
        """
        expl_img = nib.Nifti1Image(expl, np.eye(4))

        bg_img = x.detach().cpu().numpy()[batch_counter, 0, :, :, :]

        bg_img = nib.Nifti1Image(bg_img, np.eye(4))

        display = plot_stat_map(
            expl_img,
            bg_img=bg_img,
            display_mode="mosaic",
            cut_coords=12,
            threshold=0.5,
            vmax=10,
        )

        res_path = f"{self.res_path}/{xai_method}"
        isExist = os.path.exists(res_path)
        if not isExist:
            os.makedirs(res_path)

        if not mean_expl:
            img_filename = f"{res_path}/expl_{eid}.png"
        else:
            img_filename = f"{res_path}/mean_expl_{xai_method}.png"
        
        display.savefig(img_filename)
        show()
        display.close()

if __name__ == '__main__':
    cli = LightningCLI(Model, BrainDataModule, save_config_callback=None)
