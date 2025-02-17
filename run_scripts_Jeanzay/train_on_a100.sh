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
rm -rf ~/.cache/torch_extensions

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

#export TORCH_HOME=/lustre/fswork/projects/rech/ggs/uri15na/.cache/torch
# export PYTHONPATH=$SCRATCH/3_code/styelGAN/pSp_encoder_constructive:$PYTHONPATH
# CUDA_LAUNCH_BLOCKING=1

# python -c "import torch; print(torch.__version__, torch.version.cuda)"

python training_scripts/train.py \
--exp_scheme=cmlp_adv_common \
--exp_dir=results/cmlp_adv_common_wbg/preted_CMLP_preted_D/lr0.01_layer1_w0.01 \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--cmlp_checkpoint_path=./results/cmlp_ffhq_glasses/iteration_130000.pt \
--disc_checkpoint_path=./results/train_cls_common_wbg/lr1e-4_layer1_bs8/checkpoints/iteration_161500.pt \
--max_steps=200000 \
--log_interval=250 \
--val_interval=500 \
--save_interval=500 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--disc_type=global \
--n_layer_disc=1 \
--learning_rate=0.01 \
--D_learning_rate=0.01 \
--ref_latent_dist=wbg \
--disc_type=global \
--w_adv_lambda=0.01 \
# --train_D_common \


