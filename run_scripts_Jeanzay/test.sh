#!/bin/bash
#SBATCH --job-name=eval          # Job name
#SBATCH --output=reports/eval_v100.out      # Output file (%j = job ID)
#SBATCH --error=reports/eval_v100.err       # Error file (%j = job ID)
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


########## latent glasses classification/separately (Logistic regression score) ############
python evaluation/latent_classification.py \
    --model_ckpt_path ./results/cmlp_adv_common_wbg/ckpt2/preted_CMLP_preted_D/lr0.01_layer1_w0.01/checkpoints/iteration_133000.pt \
    --label_csv_path ./evaluation/results_age-gender-smile/latent_classification/glasses/labeled_glasses.csv \
    --results_dir ./results/EVALUATION \
    --cls_type bg-glasses \
    --reduced_dim None
