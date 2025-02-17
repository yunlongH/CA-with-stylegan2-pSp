#!/bin/bash
#SBATCH --job-name=h100          # Job name
#SBATCH --output=reports/h100v2.out      # Output file (%j = job ID)
#SBATCH --error=reports/h100v2.err       # Error file (%j = job ID)
#SBATCH --constraint=h100              # Reserve H100 GPUs (80 GB memory)
#SBATCH --nodes=1                      # Number of nodes to reserve
#SBATCH --ntasks=1                     # Number of tasks (or processes)
#SBATCH --gres=gpu:1                   # GPUs per node (total: 2 nodes * 4 GPUs = 8 GPUs)
#SBATCH --cpus-per-task=24             # Number of CPUs per task
#SBATCH --time=20:00:00                # Maximum allocation time (HH:MM:SS)
#SBATCH --hint=nomultithread           # Disable hyperthreading
#SBATCH --account=ggs@h100             # Use H100 account for job allocation


source ./set_cuda_env.sh

rm -rf $WORK/.cache/torch_extensions
rm -rf ~/.cache/torch_extensions

# # train discriminator 
# python main/train_D.py --config=main/configs_D.json --experiment=CA_with_RD

# ## train reconstructor 
# python main/train_R_Wpsp.py --config=main/configs_R.json --experiment=CA_with_RD

# python training_scripts/train.py \
# --dataset_type=ffhq_glasses \
# --exp_scheme=mult_optims \
# --cmlp_checkpoint_path=./results/CA_with_R/CA_from_130k/config1/checkpoints/iteration_140000.pt \
# --disc_checkpoint_path=./results/baseline/discriminator/checkpoints/model_epoch_200.pth \
# --c2s_checkpoint_path=./results/baseline/reconstructor/strong/lr1e-5/checkpoints/model_epoch_500.pth \
# --exp_dir=./results/CA_with_R/CA_from_140k/config5_strong_lrCA0.001_lrR0.002 \
# --log_interval=100 \
# --val_interval=200 \
# --save_interval=200 \
# --max_steps=150000 \
# --mlp_norm_type=nodim \
# --n_c2s_layers=12 \
# --c2s_net_type=strong \
# --learning_rate=0.001 \
# --w_cls_lambda=0.0 \
# --w_recon_lambda=1.0 \
# --lr_CA=0.001 \
# --lr_R=0.002

python training_scripts/train.py \
--dataset_type=ffhq_glasses \
--exp_scheme=improved_loss \
--cmlp_checkpoint_path=./results/CA_with_RD/CA_from_145600/config1_lambda_s0.1_lr5e-4/checkpoints/iteration_147200.pt \
--disc_checkpoint_path=./results/CA_with_RD/discriminators/for_iter145600/checkpoints/model_epoch_500.pth \
--c2s_checkpoint_path=./results/CA_with_RD/reconstructors/for_iter147200_wpsp_1e-3_8layers/checkpoints/model_epoch_100.pth \
--exp_dir=./results/CA_with_RD/CA_from_147200/stronglr1e-3_100ep_w1.0_Wpsp  \
--log_interval=100 \
--val_interval=100 \
--save_interval=100 \
--max_steps=150000 \
--mlp_norm_type=nodim \
--n_c2s_layers=8 \
--c2s_net_type=c2smlp \
--learning_rate=0.001 \
--w_cls_lambda=0.0 \
--w_recon_lambda=1.0 \
--lambda_s=0.1 \
--num_D_layers=2 \

# # for discriminator
# python training_scripts/train.py \
# --dataset_type=ffhq_glasses \
# --exp_scheme=improved_loss \
# --cmlp_checkpoint_path=./results/CA_with_D/CA_from_145k/config1/checkpoints/iteration_145600.pt \
# --disc_checkpoint_path=./results/CA_with_D/discriminators/for_iter145600/checkpoints/model_epoch_500.pth \
# --c2s_checkpoint_path=./results/baseline/reconstructor/c2smlp/checkpoints/model_epoch_140.pth \
# --exp_dir=./results/CA_with_D/CA_from_145600/config1 \
# --log_interval=200 \
# --val_interval=200 \
# --save_interval=200 \
# --max_steps=160000 \
# --mlp_norm_type=nodim \
# --n_c2s_layers=12 \
# --c2s_net_type=c2smlp \
# --learning_rate=0.001 \
# --w_cls_lambda=1.0 \
# --w_recon_lambda=0.0 \
# --lambda_s=0.2 \