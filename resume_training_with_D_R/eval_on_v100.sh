#!/bin/bash
#SBATCH --job-name=eval          # Job name
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


# # # #############  reproduce_eval_images ##############  
# python evaluation/reproduce_eval_images.py \
#     --model_ckpt_path ./results/cmlp_ffhq_glasses/iteration_130000.pt \
#     --recon_swap_path ./eval_images/glasses/3Dmlp/recon_swap \
#     --references_path ./eval_images/glasses/references
    #--eval_ref

# # ############## FID score ############# 
# python evaluation/image_fid_score.py \
#     --eval_images_dir ./eval_images/smile/2Dmlp/recon_swap \
#     --real_images_dir ./eval_images/smile/references \
#     --save_results_dir ./evaluation/results_age-gender-smile/fid_scores/fid_age-gender-smile.txt \
#     --eval_pSp 

# # ############  Image identity ############# 
# python evaluation/image_identity.py \
#     --eval_images_dir ./eval_images/age/3Dmlp/recon_swap \
#     --real_images_dir ./eval_images/age/references \
#     --save_results_dir ./evaluation/results_fid_age-gender-smile/identity \
#     --max_images -1 --dist_metric cosine

    #--eval_ref
    
############# calculate glasses detection #############
# python evaluation/image_glasses_detection.py \
#     --recon_swap_path ./eval_images/sparsity_3e-3_sbg1.5 \
#     --save_results_path ./evaluation/eval_results/glasses_detection/SwinB_sparsity_3e-3_sbg1.5 \
#     --max_images -1 --box_threshold 0.0 --model_type SwinB \

    #--eval_ref --references_path ./eval_images/references
    #--reproduce_images

### results/CA_with_RD/CA_from_145600/config1_lambda_s0.0_lr5e-4/checkpoints/iteration_147200.pt
########## latent glasses classification/separately (Logistic regression score) ############
python evaluation/latent_classification.py \
    --model_ckpt_path ./results/CA_with_RD/CA_from_147200/stronglr1e-3_500ep_w1.0_onlyCt/checkpoints/iteration_148000.pt \
    --label_csv_path ./evaluation/results_age-gender-smile/latent_classification/glasses/labeled_glasses.csv \
    --results_dir ./results/EVALUATION \
    --cls_type bg-glasses \
    --reduced_dim None
# python get_labels.py
# results/cmlp_c2s_adv/with_up_c2snet_/lambda0.01_lr0.0001/checkpoints/iteration_30000.pt
    # parser.add_argument('--model_ckpt_path', type=str, required=True, help='Path to the real images directory')
    # parser.add_argument('--reduced_dim', type=str, default="50", help='Directory where output reconstruction images will be saved')
    # parser.add_argument('--results_dir', type=str, default='./evaluation/separately.txt', help='Directory where fid txt results will be saved')
    # parser.add_argument('--random_seed', type=int, default=42, help='set a random seed for different results')
    # parser.add_argument('--label_csv_path', type=str, default='/labeled_gender_age.csv', help='path to label attributes')
    # parser.add_argument('--cls_type', type=str, default='male-female', help='gender or age')

# ########### calculate gender separately (Logistic regression score) ############
# python evaluation/latent_gender_classification.py \
#     --model_ckpt_path ./results/network_architectures/cnn/checkpoints/iteration_100000.pt \
#     --save_results_dir ./evaluation/eval_results/latent_gender_cls.txt \
#     --reduced_dim 50


# ########### calculate ages separately (Logistic regression score) ############
# python evaluation/latent_age_regression.py \
#     --model_ckpt_path ./results/csmlp_sparsity/mlp3D/effect_l1reg_types/thresholded/element_zeroOut1e-4_lasso3.6e-3_threshold1e-3/checkpoints/iteration_100000.pt \
#     --save_results_dir ./evaluation/eval_results/latent_age_reg_lasso.txt \
#     --reduced_dim None --l1_ratio=0.5