#!/bin/bash
#SBATCH --job-name=train_cls          # Job name
#SBATCH --output=reports/v100.out      # Output file (%j = job ID)
#SBATCH --error=reports/v100.err       # Error file (%j = job ID)
#SBATCH --constraint=v100-32g              # Reserve V100 GPUs (32 or 16 GB memory)
#SBATCH --nodes=1                      # Number of nodes to reserve
#SBATCH --ntasks=1                     # Number of tasks (or processes)
#SBATCH --gres=gpu:1                   # GPUs per node (total: 2 nodes * 4 GPUs = 8 GPUs)
#SBATCH --cpus-per-task=10            # Number of CPUs per task
#SBATCH --time=20:00:00                # Maximum allocation time (HH:MM:SS)
#SBATCH --hint=nomultithread           # Disable hyperthreading
#SBATCH --account=ggs@v100             # Use V100 account for job allocation

# Environment setup
# module purge                           # Purge all inherited modules
# conda deactivate                       # Deactivate any inherited Conda environments
# module load arch/h100                  # Load H100 architecture-specific modules
# module load pytorch-gpu/py3/2.4.0      # Load PyTorch GPU module for Python 3

# # Echo all commands for debugging
# set -x

# Source the script to set TORCH_CUDA_ARCH_LIST
source $HOME/set_cuda_arch.sh

module load miniforge/24.9.0
conda activate styleGANenv
module purge
module load cuda/12.4.1

# Export environment variables
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

cd $SCRATCH/3_code/styleGAN/pSp_encoder_constructive

rm -rf $WORK/.cache/torch_extensions
# rm -rf ~/.cache/torch_extensions

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

#export TORCH_HOME=/lustre/fswork/projects/rech/ggs/uri15na/.cache/torch
# export PYTHONPATH=$SCRATCH/3_code/styelGAN/pSp_encoder_constructive:$PYTHONPATH
# CUDA_LAUNCH_BLOCKING=1

# python -c "import torch; print(torch.__version__, torch.version.cuda)"


python training_scripts/train.py \
--exp_scheme=train_cls_common \
--exp_dir=results/train_cls_common_cbg/lr2e-4_layer1_global \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--cmlp_checkpoint_path=results/cmlp_ffhq_glasses/iteration_130000.pt \
--disc_checkpoint_path=None \
--max_steps=800000 \
--log_interval=250 \
--val_interval=500 \
--save_interval=500 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--n_layer_disc=1 \
--D_learning_rate=2e-4 \
--ref_latent_dist=cbg \
--disc_type=global \
# --batch_size=4 \
# --test_batch_size=4 \
# --workers=4 \
# --test_workers=4 \

# mv ./results/alternative_training/classifiers/latent_130k ./results/alternative_training/cmlp130k_latent

