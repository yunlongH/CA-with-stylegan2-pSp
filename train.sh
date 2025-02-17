#!/bin/bash
#SBATCH --job-name=cmlp_adv          # Job name
#SBATCH --output=reports/a100.out      # Output file (%j = job ID)
#SBATCH --error=reports/a100.err       # Error file (%j = job ID)
#SBATCH --constraint=a100              # Reserve A100 GPUs (80 GB memory)
#SBATCH --nodes=1                      # Number of nodes to reserve
#SBATCH --ntasks=1                     # Number of tasks (or processes)
#SBATCH --gres=gpu:1                   # GPUs per node (total: 2 nodes * 4 GPUs = 8 GPUs)
#SBATCH --cpus-per-task=12            # Number of CPUs per task
#SBATCH --time=20:00:00                # Maximum allocation time (HH:MM:SS)
#SBATCH --hint=nomultithread           # Disable hyperthreading
#SBATCH --account=ggs@a100             # Use A100 account for job allocation


source ./set_cuda_Jeanzay.sh # Set cuda dependencies for Jeanzay

### source environments ###
# source ~/anaconda3/bin/activate styleGANenv  # if using IDS cluster
module load miniforge/24.9.0
conda activate styleGANenv
module purge
module load cuda/12.4.1


python training_scripts/train.py \
--dataset_type=ffhq_glasses \
--stylegan_weights=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--exp_dir=results/baseline/


