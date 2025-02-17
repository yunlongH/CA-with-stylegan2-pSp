#!/bin/bash

#SBATCH --job-name=train_cls
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_cls_out.txt
#SBATCH --error=./reports/train_cls_err.txt

source ./set_cuda_IDS.sh # Set cuda dependencies for Jeanzay

### source environments ###
source ~/anaconda3/bin/activate styleGANenv  # if using IDS cluster

python training_scripts/train.py \
--dataset_type=ffhq_glasses \
--stylegan_weights=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--exp_dir=results/baseline/


