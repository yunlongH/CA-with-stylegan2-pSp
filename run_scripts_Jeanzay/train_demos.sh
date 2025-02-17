
python training_scripts/train.py \
--exp_scheme=baseline \
--exp_dir=results/cmlp_baseline/3Dmlp \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--image_interval=10000 \
--print_interval=10000 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses

# python test_cuda.py
python training_scripts/train.py \
--exp_scheme=swap_loss \
--exp_dir=results/cmlp_swap_loss/swap1.0_rec1.0 \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--image_interval=10000 \
--print_interval=10000 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--sbg_lambda=1.0 \
--w_dist_lambda=1.0 \
--w_dist_swap_lambda=1.0 \
--w_dist_rec_lambda=1.0




python training_scripts/train.py \
--exp_scheme=Swap_Lat \
--exp_dir=results/cmlp_Swap_Lat/3Dmlp \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--image_interval=10000 \
--print_interval=10000 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glass \
--w_lat_swap_lambda=1.0



python training_scripts/train.py \
--exp_dir=results/train_c2smlp/ \
--exp_scheme=train_c2smlp \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=results/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim


scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Projects/CA_with_GAN/2_data/ uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/3_code
nohup scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/3_code

nohup scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Projects/CA_with_GAN/3_code/diffusion-AE uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/3_code

nohup scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Shared/ uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/2_data/MRI_dataset



python training_scripts/train.py \
--exp_dir=results/train_c2smlp/12layers_l1_lr0.01 \
--exp_scheme=train_c2smlp \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=results/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--learning_rate=0.01 \
--c2s_loss_type=l1

python training_scripts/train.py \
--exp_dir=results/train_c2smlp/12layers_l1_lr0.01 \
--exp_scheme=train_c2smlp \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=results/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--learning_rate=0.01 \
--c2s_loss_type=l1 \
--n_layer_disc=2


python training_scripts/train.py \
--exp_dir=results/train_mi_disc/indept_lr0.01 \
--exp_scheme=train_mi_disc \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=./results/cmlp_ffhq_glasses/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--optim_name=admn \
--seed=99 \
--disc_type=indept \
--learning_rate=0.01 \
--batch_size=8 \
--test_batch_size=8 \
--workers=8 \
--test_workers=8

python training_scripts/train.py \
--exp_dir=results/train_c2s_mlp/12layers_l1_lr0.01 \
--exp_scheme=train_c2s_mlp \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=./results/cmlp_ffhq_glasses/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--n_layer_c2s=12 \
--c2s_loss_type=l1 \
--learning_rate=0.01
