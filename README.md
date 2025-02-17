# Step 1:  set the dataset path in "./configs/paths_config.py" 

Before running, set the four dataset paths: the background and target folders for training data, and the background and target folders for test data. We support .png images as input data.

dataset_paths = {
	'ffhq_bg_train': 'path/to/your background train images', \
	'ffhq_glass_train': 'path/to/your target train images', \
	'ffhq_bg_test': 'path/to/your background test images', \
	'ffhq_glass_test': 'path/to/your target test images', \
  }

# Step 2: source environment, run script  

Depends on which cluster you want to use:
./train_Jeanzay.sh or 
./train_IDS.sh

# Example for runing:

python training_scripts/train.py --dataset_type=ffhq_glasses --stylegan_weights=./psp_ffhq_encode.pt --pSp_checkpoint_path=./stylegan2-ffhq1024.pt --exp_dir=results/baseline/


