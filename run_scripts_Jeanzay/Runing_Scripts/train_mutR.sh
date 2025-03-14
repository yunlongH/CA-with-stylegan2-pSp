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

python training_R/train_orn.py \
--model_path=./results/baseline/iteration_130000.pt \
--latent_hdf5_path=./results/baseline/cmlp130k_latent \
--results_dir=./results/PCA_recon/c2smlp/lr1e-4 \
--max_epochs=1000 \
--save_interval=20 \
--log_interval=1 \
--val_interval=2 \
--lr=1e-4 \
--network_type=c2smlp \
--no_reproduce_latent \
--n_c2s_layers=12 \
# --pca_latent_dim=512

