#!/bin/bash

#SBATCH --job-name=art_disease_1_426
#SBATCH --output=/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/model_training/slurm_runs/slurm_logs/%x-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --signal=SIGUSR1@90

export LD_LIBRARY_PATH=/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/envs/brain-xai-benchmark-env/lib

wandb online

cd /sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/run_models/train_densenet_bcl
srun /sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/envs/brain-xai-benchmark-env/bin/python /sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/models/DenseNet/main_densenet_bcl.py fit -c /sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/run_models/train_densenet_bcl/config.yaml