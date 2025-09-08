#!/bin/bash

#SBATCH --job-name=diffAE
#SBATCH --nodes=1
#SBATCH --partition=V100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80GB
#SBATCH --time=20:00:00
#SBATCH --output=./train_scripts/slurm_reports/out.txt
#SBATCH --error=./train_scripts/slurm_reports/err.txt


source ~/anaconda3/bin/activate styleGAN2env

source ./set_cuda_env.sh

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/diffusion-AE

echo "Changed directory to $(pwd)"

# python encode_latents_ffhq.py


