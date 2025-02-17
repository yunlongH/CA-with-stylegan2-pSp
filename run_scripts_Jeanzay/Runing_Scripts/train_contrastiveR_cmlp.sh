#!/bin/bash
#SBATCH --job-name=h100          # Job name
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

source ./set_cuda_env.sh

python training_scripts/train.py \
--dataset_type=ffhq_glasses \
--exp_scheme=contrastive_R \
--cmlp_checkpoint_path=./results/others/cmlp_baseline_h100/3Dmlp/checkpoints/iteration_130000.pt \
--c2s_checkpoint_path=./results/contrastiveR/c2smlp/checkpoints/model_epoch_80.pth \
--exp_dir=./results/contrastiveR_cmlp_adjust/w1.0_lr0.001_80ep \
--log_interval=200 \
--val_interval=200 \
--save_interval=200 \
--max_steps=140000 \
--n_layers_mlp=12 \
--n_c2s_layers=12 \
--mlp_norm_type=nodim \
--learning_rate=0.001 \
--w_c2s_lambda=1.0 \
--c2s_loss_type=mse \
--c2s_net_type=c2smlp \

# --lr=1e-3 \
# --n_c2s_layers=8 \
# --loss_type=mse \
# --network_type=c2smlp \


