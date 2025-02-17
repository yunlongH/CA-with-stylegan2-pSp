from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	# def load_previous(self):

		
	def initialize(self):

		# basic settings
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--exp_scheme', default='baseline', type=str, help='Type of experiment to run, we have "csmlp" or "cskd"')
		self.parser.add_argument('--seed', default=99, type=int, help='Set the seed for reproducibility')

		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--latent_dim', default=512, type=int, help='Output size of generator')
		self.parser.add_argument('--style_dim', default=18, type=int, help='Output size of generator')

		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--train_pSp_encoder', default=False, type=bool, help='Whether to train the decoder model')		
		self.parser.add_argument('--train_stylegan_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', default=True, type=bool, help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', default=False, type=bool, help='Whether to learn in w space instead of w+')
		
		# arguments for cs_mlp net
		#self.parser.add_argument('--mlp_model', default='normal', type=str, help='Which network to use, choices: normal, individal')
		self.parser.add_argument('--mlp_norm_type', default='nodim', type=str, help='Which type of weight to use, choices: 2D or 3D')		
		self.parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use, adam or ranger')
		self.parser.add_argument('--learning_rate', default=0.01, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--net_type', default='independent', type=str, help='net_type, independent, shared, fn')
		self.parser.add_argument('--n_layers_mlp', default=12, type=int, help='number of layers in MLP')
		self.parser.add_argument('--n_shared_layers', default=4, type=int, help='number of shared layers in FN')
		self.parser.add_argument('--n_branch_layers', default=8, type=int, help='number of branch layers in FN')
		self.parser.add_argument('--spatial_encoding', default='none', type=str, choices=['spatial-aware', 'inter-row-attention', 'none'],  help='Spatial encoding type')

		# # arguments for mi discriminator
		self.parser.add_argument('--n_mut_layers', default=2, type=int, help='number of branch layers in FN')
		self.parser.add_argument('--disc_type', default='global', type=str, help='global or indenpendent')
		self.parser.add_argument('--mi_loss_method', default='bce', type=str)
		self.parser.add_argument('--train_D_common', action='store_true')
		self.parser.add_argument('--D_learning_rate', default=0.0002, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--ref_latent_dist', default='cbg', type=str, help='reference latent distribution, c_bg or w_bg')
		
		# for pca
		self.parser.add_argument('--pca_load_path', type=str, default="./PCA")
		self.parser.add_argument('--k', type=int, default=512)
		self.parser.add_argument('--lr_CA', default=0.001, type=float)
		self.parser.add_argument('--lr_R', default=0.005, type=float)
		# # arguments for classifier cbg ct
		# self.parser.add_argument('--cls_num_epochs', default=50, type=int)
		# self.parser.add_argument('--cls_val_interval', default=2, type=int)
		self.parser.add_argument('--num_D_layers', default=2, type=int)
		self.parser.add_argument('--cls_lr', default=0.0001, type=float)
		# self.parser.add_argument('--train_cls_interval', default=5000, type=int)
		# self.parser.add_argument('--max_cls_steps', default=1000, type=int)


		# arguments for reconstruction from c to s
		self.parser.add_argument('--recon_loss_type', default='mse', type=str)
		self.parser.add_argument('--c2s_net_type',  default='c2smlp', type=str)  
		self.parser.add_argument('--n_c2s_layers',  default=12, type=int)
		self.parser.add_argument('--w_recon_lambda', default=1.0, type=float) 
		self.parser.add_argument('--w_cls_lambda', default=1.0, type=float)
		self.parser.add_argument('--lambda_s', default=0.1, type=float)

		# arguments for adjust losses 
		self.parser.add_argument('--id_lambda', default=0.4, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--pix_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--w_dist_lambda', default=1.0, type=float, help='similarity loss for latent w')
		self.parser.add_argument('--sbg_lambda', default=1.0, type=float, help='l1 loss for silent factor of target')
		self.parser.add_argument('--w_dist_swap_lambda', default=0.0, type=float, help='similarity loss for latent w')
		self.parser.add_argument('--w_dist_rec_lambda', default=0.0, type=float, help='similarity loss for latent w')

		self.parser.add_argument('--st_lambda', default=1.0, type=float, help='l1 loss for silent factor of target')
		self.parser.add_argument('--s_scale_lambda', default=0.3, type=float, help='similarity loss for latent w')
		self.parser.add_argument('--sbg_type', default='l2', type=str, help='type of norm forcing sbg toward zeros')

		# arguments for zero out sparsity
		self.parser.add_argument('--zero_out_silent_bg', action="store_true", help='zero out silent factor for background')
		self.parser.add_argument('--zero_out_silent_t', action="store_true", help='zero out silent factor for target')
		self.parser.add_argument('--zero_out_type', default='hard', type=str, help='Output size of generator')
		self.parser.add_argument('--zero_out_threshold', default=0.0, type=float, help='threshold for zeroout')

		# arguments for lasso
		self.parser.add_argument('--lasso_weight_type', default='all', type=str, help='regularization type')
		self.parser.add_argument('--elastic_alpha', default=0.0, type=float, help='to balance between l1 and l2 regularization')
		self.parser.add_argument('--lasso_sbg_lambda', default=0.0, type=float, help='lasso loss for latent sbg')
		self.parser.add_argument('--lasso_st_lambda', default=0.0, type=float, help='lasso loss for latent st')
		self.parser.add_argument('--lasso_weight_lambda', default=0.0, type=float, help='similarity loss for latent w')
		self.parser.add_argument('--lasso_direction', default='element', type=str, help='Define the regularization direction (row, column, or element)')
		self.parser.add_argument('--lasso_threshold', default=1e-4, type=float, help='Direction of L1 regularization')
		self.parser.add_argument('--lasso_reg_type', default='traditional', type=str, help='Define the regularization type (traditional, logarithmic, adaptive, thresholded)')
		
		# arguments for weights of pretained models
		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--pSp_checkpoint_path', default='../pretrained_models/pSp_models/psp_ffhq_encode.pt', type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--cmlp_checkpoint_path', default=None, type=str, help='Path to csmlp model checkpoint')
		self.parser.add_argument('--disc_checkpoint_path', default=None, type=str, help='Path to discriminator checkpoint')
		self.parser.add_argument('--c2s_checkpoint_path', default=None, type=str, help='Path to reconstructor checkpoint')
		self.parser.add_argument('--previous_train_ckpt_path', default=model_paths['previous_train_ckpt_path'], type=str, help='Path to previous model weights')		

		# self.parser.add_argument('--mi_disc_checkpoint_path', default=None, type=str, help='Path to mi_disc model checkpoint')
		# self.parser.add_argument('--disc_checkpoint_path', default=None, type=str, help='Path to discriminator checkpoint')
		# self.parser.add_argument('--w_mut_lambda', default=1.0, type=float)
		

		# arguments for training steps & logging interval
		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=10000, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--print_interval', default=10000, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--log_interval', default=10000, type=int, help='Interval for logging metrics to text file')
		self.parser.add_argument('--val_interval', default=10000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')
		self.parser.add_argument('--save_training_data',  default=True, type=bool, help='Save intermediate training data to resume training from the checkpoint')
		self.parser.add_argument('--keep_optimizer',  default=True, type=bool, help='Whether to continue from the checkpoint\'s optimizer')

		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
