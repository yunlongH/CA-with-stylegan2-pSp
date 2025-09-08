#!/bin/bash

#SBATCH --job-name=diffAE
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./train_cs/slurm_reports/out.txt
#SBATCH --error=./train_cs/slurm_reports/err.txt


source ~/anaconda3/bin/activate styleGAN2env

source ./set_cuda_env.sh

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/diffusion-AE

echo "Changed directory to $(pwd)"

# python encode_latents_ffhq.py

# train with only latent loss (baseline)
python train_cs/train_common_salient_baseline.py \
    --results_dir=./results/layers2/lr=0.0001 \
    --n_layers=2 \
    --learning_rate=0.0001 \
    --w_bg=10.0 \
    --w_t=10.0 \
    --w_sbg=10.0 \
    --w_mmd=0.0
#     --w_adv=0.1 \
#     --disc_iter_interval=1 \
#     --disc_n_layers=1 \
#     --lr_disc=0.001

# python train_cs/train_common_salient_adv_bg.py \
#     --results_dir=./results/adv_bg/Diter2_layer1_w0.01_lr1e-3_lrmain1e-3 \
#     --disc_iter_interval=2 \
#     --disc_n_layers=1 \
#     --w_adv=0.01 \
#     --lr_disc=0.001 \
#     --learning_rate=0.001

    # --w_id=0.0 \
    # --w_pix=0.0 \
    # --w_lpips=0.0 \
    # --T_noise=20

## train with reconstruction + Adv loss
# python train_cs/train_common_salient_baseline.py \
#     --results_dir=./results/baseline/latent_recon \
#     --max_epochs=1000  \
#     --adv_target=diffAE_zbg \
#     --image_interval=100 \
#     --save_interval=100 \
#     --w_adv=0.2 \

# ## train discriminator
# python resume_training/train_discriminator_stable.py \
#     --cs_model_ckpt=./results/baseline/lr0.01/checkpoints/model_epoch_800.pth \
#     --results_dir=./results/resume_training/cxsy_diffAE_y/lr1e-4_layer1 \
#     --max_epochs=500  \
#     --lr_cls=1e-4 \
#     --cls_n_layers=1 \
#     --classifier_type=cxsy_diffAE_y

# ## train discriminator
# python resume_training/train_discriminator_stable.py \
#     --cs_model_ckpt=./results/baseline/lr0.01/checkpoints/model_epoch_800.pth \
#     --results_dir=./results/resume_training/cxsy_diffAE_y/lr1e-4_layer1 \
#     --max_epochs=500  \
#     --lr_cls=1e-4 \
#     --cls_n_layers=1 \
#     --classifier_type=cxsy_diffAE_y

# ## resume training with discriminator
# python resume_training/resume_training_with_Disc.py \
#     --cs_model_ckpt=./results/baseline/lr0.01/checkpoints/model_epoch_800.pth \
#     --results_dir=./results/resume_training/hyparam_v2 \
#     --latent_type=Zx_Cy \
#     --alt_max_phases=10  \
#     --disc_max_epochs=10  \
#     --main_max_epochs=10  \
#     --disc_n_layers=2 \
#     --w_adv=0.1 \
#     --lr_main=1e-3 \
#     --lr_disc=1e-4

# python ./train_cs/preprocess_lmdb.py
