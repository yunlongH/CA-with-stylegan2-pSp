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

python training_classifier/train.py \
--model_path=./results/baseline/iteration_130000.pt \
--save_path=./results/PCA_classifiers \
--latent_hdf5_path=./results/baseline/cmlp130k_latent \
--max_epochs=100 \
--save_interval=2 \
--lr=0.0001 \
--k=9216 \
--no_reproduce_latent \


# --pca_load_path=./PCA \



