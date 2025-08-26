from multiprocessing import freeze_support
import numpy as np
import torch
from pytorch_lightning.cli import LightningCLI, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import torch.nn as nn
from densenet import generate_model
import nibabel as nib
from brain_deform.lightning import BrainDataModule
import wandb
import os
import yaml
from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall

freeze_support()


class Model(LightningModule):
    def __init__(self, model_depth=121,
                 one_cycle=False,
                 max_lr=0,
                 reduce_on_plat=True,
                 reduce_on_plat_start_lr=0.0001,
                 reduce_on_plat_red= 0.5,
                 wandb_run_name=None,
                 slurm_run=1,
                 wandb_proj = "paper-xai-benchmark",
                 ):
        
        super().__init__()

        # generate model
        self.model = generate_model(model_depth=model_depth, n_input_channels=1)
        
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

        # mean and std for input normalization
        self.train_mean = 232.5522
        self.train_std = 414.4120

        # setup loss
        self.bce = nn.BCELoss()
        # setup final activation function
        self.sigmoid = nn.Sigmoid()

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

        # for logging:
        self.val_step_outputs = []
        self.val_step_targets = []


    def forward(self, x):
        # normalize input with precomputed values
        x_norm = (x - self.train_mean)/self.train_std

        out = self.model(x_norm).squeeze()
        return self.sigmoid(out)


    def training_step(self, batch, batch_idx):
        # use augmented version for training
        (_, x, y, eid), _ = batch

        y_hat = self.forward(x)

        loss = self.bce(y_hat, y)
        
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
        loss = self.bce(y_hat, y)

        y_hat_binary = []
        for prediction in y_hat:
             y_hat_binary.append(int(round(prediction.item())))

        y_binary = []
        for target in y:
            y_binary.append(int(target))

        self.val_step_outputs.extend(y_hat_binary)
        self.val_step_targets.extend(y_binary)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        tensor_outputs = torch.IntTensor(self.val_step_outputs)
        tensor_targets = torch.IntTensor(self.val_step_targets)
        
        precision = binary_precision(tensor_outputs, tensor_targets)
        recall = binary_recall(tensor_outputs, tensor_targets)
        accuracy = binary_accuracy(tensor_outputs, tensor_targets)
        
        self.log(f"precision", precision, sync_dist=True)
        self.log(f"recall", recall, sync_dist=True)
        self.log(f"accuracy", accuracy, sync_dist=True)
        self.log("epoch", self.trainer.current_epoch, sync_dist=True)

        wandb.log({"precision": precision, "recall": recall, "accuracy": accuracy, "epoch": self.trainer.current_epoch})

        self.val_step_outputs.clear()
        self.val_step_targets.clear()
            
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
        
    def test_step(self, batch, batch_idx):
        (x, _, y, eid ), _ = batch
        coefs = _[3]

        y_hat = self.forward(x)

        y_hat_binary = []
        for prediction in y_hat:
             y_hat_binary.append(int(round(prediction.item())))

        y_binary = []
        for target in y:
            y_binary.append(int(target))
        
        self.save_predictions(y_hat_binary, eid)
        self.save_labels(y_binary)

    def save_labels(self, y_binary):
        
        if len(y_binary) == 1:
            self.test_labels.append(float(y_binary))

        else:

            for i, y_scalar_ in enumerate(y_binary):
                self.test_labels.append(y_scalar_)

        with open(f'{self.res_path}/labels.npy', 'wb') as f:
            np.save(f, self.test_labels)

    def save_predictions(self, y_binary, eid):
        
        eid = eid.detach().cpu().numpy()

        if len(y_binary) == 1:
            self.test_predictions.append(float(y_binary))
            self.test_eids.append(int(eid))

        else:

            for i, y_hat_ in enumerate(y_binary):
                self.test_predictions.append(y_hat_)
                self.test_eids.append(eid[i])


        with open(f'{self.res_path}/predictions.npy', 'wb') as f:
            np.save(f, self.test_predictions)
    
        with open(f'{self.res_path}/prediction_eids.npy', 'wb') as f:
            np.save(f, self.test_eids)

if __name__ == '__main__':

    cli = LightningCLI(Model, BrainDataModule, save_config_callback=None)