#!/bin/bash
#SBATCH --job-name=h100          # Job name
#SBATCH --output=reports/h1002.out      # Output file (%j = job ID)
#SBATCH --error=reports/h1002.err       # Error file (%j = job ID)
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
--exp_scheme=adverserial_mutR \
--cmlp_checkpoint_path=./results/baseline/iteration_130000.pt \
--cls_checkpoint_path=./results/adverserial_mutualR/1st_alternate/recontor_1e-5_4layers/checkpoints/model_epoch_500.pth \
--exp_dir=./results/adverserial_mutualR/1st_alternate/cmlp130000_mutRcmlp/lr1e-2 \
--log_interval=1 \
--val_interval=20 \
--save_interval=20 \
--max_steps=130200 \
--n_layers_mlp=12 \
--n_c2s_layers=4 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--learning_rate=0.01 \
--w_mi_lambda=1.0 \


