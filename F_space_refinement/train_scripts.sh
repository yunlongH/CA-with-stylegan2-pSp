#!/bin/bash

#SBATCH --job-name=s1s2CAT
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
# SBATCH -w node01,node02,node03,node04,node05,node06,node07
#SBATCH --output=./reports/s1s2CAT.out
#SBATCH --error=./reports/s1s2CAT.err

# Optional: Manually configure CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set Compute Capability for A100

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS

source ~/anaconda3/bin/activate styleGANenv

source set_cuda_env.sh


#rm -rf /home/ids/yuhe/.cache/torch_extensions

# ##### training inverter
# python scripts/train.py \
#     exp.exp_dir=./experiments/inverter/ \
#     exp.config_dir=configs \
#     exp.config=fse_inverter_train.yaml \
#     exp.name=fse_inverter_train \
#     data.transform=face_1024 \
#     data.dataset=ffhq \
#     train.log_step=20 \
#     train.val_step=5000 \
#     train.checkpoint_step=5000 \
    
    # data.input_train_dir=path/to/train/images \
    # data.input_val_dir=path/to/validation/images \
    # data.special_dir=path/to/several/special/images


# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=ffhq_glasses \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/ablation/ffhq_glasses_cs1s2 \
#     methods_args.fse_full.inverter_pth=./pretrained_models/sfe_inverter_light.pt \
#     train.train_runner=fse_editor_cs1s2 \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=0 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=1024 \
#     #model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_100k_000/iteration_350000.pt \


# ############## CelebAHQ smile ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=celebahq_smile \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/celebaHQ/celebahq_smile \
#     methods_args.fse_full.inverter_pth=./pretrained_models/sfe_inverter_light.pt \
#     train.start_step=330000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=2 \
#     model.w_space_encoder=pSp \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/celebaHQ/celebahq_smile_000/iteration_330000.pt \
    #SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_gender_001/iteration_340000.pt

#     ############## church ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=lsun_church \
#     data.transform=face_256 \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/lsun_church_9k/ \
#     methods_args.fse_full.inverter_pth=../../pretrained_models/sfe/sfe_inverter_church_165k.pt \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=3 \
#     model.w_space_encoder=e4e \
#     model.stylegan_size=256 \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_gender_001/iteration_340000.pt \


#     ############## Brats ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=brats_edit \
#     data.transform=face_256 \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/brats/edit/ \
#     methods_args.fse_full.inverter_pth=../../pretrained_models/sfe/brats_inverter.pt \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=20 \
#     train.val_step=20 \
#     train.checkpoint_step=20 \
#     data.special_idx=2 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=256 \
#     model.checkpoint_path=../../pretrained_models/sfe/refined_sfe_brats_170k.pt \

#     ############## ffhq_glassesvssmile editor ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=ffhq_glassesvssmile \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_40k \
#     methods_args.fse_full.inverter_pth=./pretrained_models/sfe_inverter_light.pt \
#     train.train_runner=fse_editor_cs1s2 \
#     train.start_step=360000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=0 \
#     model.w_space_encoder=pSp \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_40k_001/iteration_360000.pt \


#     ############## AFHQ editor ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=afhq_cat_dog \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/AFHQ/S1S2 \
#     methods_args.fse_full.inverter_pth=../../pretrained_models/sfe/afhq_inverter.pt \
#     train.train_runner=fse_editor_cs1s2 \
#     train.start_step=320000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=0 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=512 \
#     data.transform=face_512 \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_40k_000/iteration_320000.pt \