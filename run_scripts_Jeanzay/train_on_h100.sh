#!/bin/bash
#SBATCH --job-name=adv_common          # Job name
#SBATCH --output=reports/h100.out      # Output file (%j = job ID)
#SBATCH --error=reports/h100.err       # Error file (%j = job ID)
#SBATCH --constraint=h100              # Reserve H100 GPUs (80 GB memory)
#SBATCH --nodes=1                      # Number of nodes to reserve
#SBATCH --ntasks=1                     # Number of tasks (or processes)
#SBATCH --gres=gpu:1                   # GPUs per node (total: 2 nodes * 4 GPUs = 8 GPUs)
#SBATCH --cpus-per-task=24             # Number of CPUs per task
#SBATCH --time=20:00:00                # Maximum allocation time (HH:MM:SS)
#SBATCH --hint=nomultithread           # Disable hyperthreading
#SBATCH --account=ggs@h100             # Use H100 account for job allocation

# Source the script to set TORCH_CUDA_ARCH_LIST
source $HOME/set_cuda_arch.sh

# Environment setup
module load miniforge/24.9.0

# module purge                           # Purge all inherited modules
# conda deactivate                       # Deactivate any inherited Conda environments
# module load arch/h100                  # Load H100 architecture-specific modules
# module load pytorch-gpu/py3/2.4.0      # Load PyTorch GPU module for Python 3
conda activate styleGANenv
module purge
module load cuda/12.4.1

cd $SCRATCH/3_code/styleGAN/pSp_encoder_constructive
# rm -rf $WORK/.cache/torch_extensions
# rm -rf ~/.cache/torch_extensions

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

#export TORCH_HOME=/lustre/fswork/projects/rech/ggs/uri15na/.cache/torch
export PYTHONPATH=$SCRATCH/3_code/styelGAN/pSp_encoder_constructive:$PYTHONPATH
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1


# python training_scripts/train.py \
# --exp_scheme=train_cls_common \
# --exp_dir=results/train_cls_common/on_sbg \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --cmlp_checkpoint_path=results/cmlp_ffhq_glasses/iteration_130000.pt \
# --disc_checkpoint_path=None \
# --log_interval=200 \
# --val_interval=1000 \
# --save_interval=500 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses \
# --disc_type=global \
# --n_layer_disc=2 \
# --D_learning_rate=0.0002 \
# --learning_rate=0.0002 \
# --ref_latent_dist=sbg \
# --disc_type=global \


python training_scripts/train.py \
--exp_scheme=cmlp_adv_common \
--exp_dir=results/cmlp_adv_common_cbg/lr1e-4_layer2_global/lr0.01_w0.05 \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--cmlp_checkpoint_path=./results/cmlp_ffhq_glasses/iteration_130000.pt \
--disc_checkpoint_path=./results/train_cls_common_cbg/lr1e-4_layer1_global/checkpoints/iteration_132500.pt \
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
--ref_latent_dist=cbg \
--disc_type=global \
--w_adv_lambda=0.05 \
# --train_D_common \