# Step 1:  set the dataset path in "./configs/paths_config.py" 
Set the 4 paths :

dataset_paths = {
	'ffhq_bg_train': 'path/to/your background train images',
	'ffhq_glass_train': 'path/to/your target train images',
	'ffhq_bg_test': 'path/to/your background test images',
	'ffhq_glass_test': 'path/to/your target test images',
  }



# Step 2:  set paths for pretrained models' in train.sh

python training_scripts/train.py \
--dataset_type=ffhq_glasses \
--stylegan_weights=./psp_ffhq_encode.pt \
--pSp_checkpoint_path=./stylegan2-ffhq1024.pt \
--exp_dir=results/baseline/


# Step 3: source environment, run script:  ./train.sh

