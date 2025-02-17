#!/bin/bash
#SBATCH --job-name=h100          # Job name
#SBATCH --output=reports/h100pca.out      # Output file (%j = job ID)
#SBATCH --error=reports/h100pca.err       # Error file (%j = job ID)
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
--exp_scheme=adverserial_common \
--cmlp_checkpoint_path=./results/baseline/iteration_130000.pt \
--cls_checkpoint_path=./results/PCA_classifiers/U_pca_k512/checkpoints/model_epoch_50.pth \
--exp_dir=./results/PCA_classifiers_mlp_lr1e-2/cmlp130_with_pca512/ \
--k=512 \
--log_interval=100 \
--val_interval=100 \
--save_interval=200 \
--max_steps=135000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--learning_rate=1e-2 \
--w_cls_lambda=1.0 \
# --pca_load_path=./PCA \

# --pca_load_path=./PCA \



