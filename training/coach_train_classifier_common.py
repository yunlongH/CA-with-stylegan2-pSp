import os
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
from argparse import Namespace
matplotlib.use('Agg')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import tqdm
from utils import common, train_utils
from criteria import id_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from models.mlp3D import MappingNetwork_cs_independent, EqualizedLinear
from models.discriminator import Discriminator, DiscriminatorGlobal

class Coach_csmlp:
	def __init__(self, opts, previous_train_ckpt=None):
		self.opts = opts
		self.global_step = 0
		self.n_print = 0
		self.n_adv_print=0
		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		self.seed_experiments(opts.seed)

		# Initialize network
		self.init_pSp_cmlp_networks(ckpt_path = self.opts.cmlp_checkpoint_path)
		self.init_discriminator(disc_ckpt_path = self.opts.disc_checkpoint_path)
		self.criterion = nn.BCELoss()

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.pSp_net.latent_avg is None:
			self.pSp_net.latent_avg = self.pSp_net.decoder.mean_latent(int(1e5))[0].detach()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:	
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizerD = self.configure_optimizers()

		# Initialize dataset
		self.train_bg_dataset, self.train_t_dataset, self.test_bg_dataset, self.test_t_dataset = self.configure_datasets()

		self.train_bg_dataloader = DataLoader(self.train_bg_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		
		self.train_t_dataloader = DataLoader(self.train_t_dataset,
									batch_size=self.opts.batch_size,
									shuffle=True,
									num_workers=int(self.opts.workers),
									drop_last=True)
		
		self.test_bg_dataloader = DataLoader(self.test_bg_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)
		
		self.test_t_dataloader = DataLoader(self.test_t_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		print(f"Number of training batches per epoch: {len(self.train_bg_dataloader)}")						
		print(f"Number of testing batches per epoch: {len(self.test_bg_dataloader)}")		

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		#self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		self.init_additional_params(previous_train_ckpt)
		
	def seed_experiments(self, seed):
		# Set the random seed for reproducibility
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.

		# Ensures deterministic behavior for some PyTorch operations
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def init_pSp_cmlp_networks(self, ckpt_path):

		if ckpt_path is not None and os.path.exists(ckpt_path):
			ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)		
			self.global_step = ckpt['global_step']
			model_opts = ckpt['opts']
			model_opts = Namespace(**model_opts)

			self.pSp_net = pSp(model_opts).to(self.device).eval()

			self.cs_mlp_net = MappingNetwork_cs_independent(model_opts).to(self.device)	
			print(f'Resuming training cmlp from path: {ckpt_path} \n Step-{self.global_step}')
			self.cs_mlp_net.load_state_dict(ckpt['state_dict_cs_enc']).eval()	
		else:
			self.pSp_net = pSp(self.opts).to(self.device).eval()
			self.cs_mlp_net = MappingNetwork_cs_independent(self.opts).to(self.device).eval()	

	def init_discriminator(self, disc_ckpt_path):

		if disc_ckpt_path is not None and os.path.exists(disc_ckpt_path):
			print(f'Resume training discriminator from path: {disc_ckpt_path}')
			disc_ckpt = torch.load(disc_ckpt_path, map_location='cpu', weights_only=True)
			disc_opts = disc_ckpt['opts']
			disc_opts = Namespace(**disc_opts)

			if disc_opts.disc_type == 'global':
				self.discriminator = DiscriminatorGlobal(disc_opts).to(self.opts.device)
			elif disc_opts.disc_type == 'indept':
				self.discriminator = Discriminator(disc_opts).to(self.opts.device)
			self.discriminator.load_state_dict(disc_ckpt['discriminator'])

		else:

			if self.opts.disc_type == 'global':
				self.discriminator = DiscriminatorGlobal(self.opts).to(self.opts.device)
			elif self.opts.disc_type == 'indept':
				self.discriminator = Discriminator(self.opts).to(self.opts.device)


	def	init_additional_params(self, prev_train_ckpt):
		if self.opts.save_training_data and prev_train_ckpt is not None:
			self.best_val_loss = prev_train_ckpt['best_val_loss']
			if self.opts.keep_optimizer :
				self.optimizer.load_state_dict(prev_train_ckpt['optimizer'])

	def update_discriminator(self, x_bg, x_t):
		self.discriminator.train()
		self.optimizerD.zero_grad()  # Zero out gradients

		# Obtain latents from pSp network
		_, w_bg_pSp = self.pSp_net.forward(x_bg, return_latents=True)
		_, w_t_pSp = self.pSp_net.forward(x_t, return_latents=True) 

		latent_t_c, _ = self.cs_mlp_net(w_t_pSp, zero_out_silent=self.opts.zero_out_silent_t) 

		if self.opts.ref_latent_dist == 'cbg': 
			latent_bg_c, _ = self.cs_mlp_net(w_bg_pSp, zero_out_silent=self.opts.zero_out_silent_bg)
			real_latent = latent_bg_c
		elif self.opts.ref_latent_dist == 'wbg':
			real_latent = w_bg_pSp
		else:
			raise ValueError(f"Invalid ref_latent_dist: {self.opts.ref_latent_dist}. Expected 'cbg' or 'wbg'.")

		# get labels
		b_size = real_latent.size(0)
		real_label = torch.ones(b_size, device=real_latent.device)
		fake_label = torch.zeros(b_size, device=latent_t_c.device)

		# Forward pass real batch through D
		output_real = self.discriminator(real_latent).view(-1)
		errD_real = self.criterion(output_real, real_label)

		# Forward pass fake batch through D
		output_fake = self.discriminator(latent_t_c.detach()).view(-1)
		errD_fake = self.criterion(output_fake, fake_label)

		# Compute total loss and update Discriminator
		errD = errD_real + errD_fake
		errD.backward()
		self.optimizerD.step()

		return errD_real.item(), errD_fake.item(), errD.item()


	def val_discriminator(self, x_bg, x_t):

		self.discriminator.eval()  # Set discriminator to evaluation mode

		with torch.no_grad():
			# Obtain latents from pSp network
			_, w_bg_pSp = self.pSp_net.forward(x_bg, return_latents=True)
			_, w_t_pSp = self.pSp_net.forward(x_t, return_latents=True) 

			latent_t_c, _ = self.cs_mlp_net(w_t_pSp, zero_out_silent=self.opts.zero_out_silent_t) 

			if self.opts.ref_latent_dist == 'cbg': 
				latent_bg_c, _ = self.cs_mlp_net(w_bg_pSp, zero_out_silent=self.opts.zero_out_silent_bg)
				real_latent = latent_bg_c
			elif self.opts.ref_latent_dist == 'wbg':
				real_latent = w_bg_pSp
			else:
				raise ValueError(f"Invalid ref_latent_dist: {self.opts.ref_latent_dist}. Expected 'cbg' or 'wbg'.")

			# get labels
			b_size = real_latent.size(0)
			real_label = torch.ones(b_size, device=real_latent.device)
			fake_label = torch.zeros(b_size, device=latent_t_c.device)

			# Forward pass real batch through D
			output_real = self.discriminator(real_latent).view(-1)
			errD_real = self.criterion(output_real, real_label)

			# Forward pass fake batch through D
			output_fake = self.discriminator(latent_t_c.detach()).view(-1)
			errD_fake = self.criterion(output_fake, fake_label)
			errD = errD_fake + errD_real

		return errD_real, errD_fake, errD

	def train(self):

		while self.global_step < self.opts.max_steps:
			print(f"Training step {self.global_step}")
			for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.train_bg_dataloader, self.train_t_dataloader)):
				
				x_bg, _ = batch_bg
				x_t, _ = batch_t

				x_bg, x_t = x_bg.to(self.device).float(), x_t.to(self.device).float()
				
				errD_real, errD_fake, errD = self.update_discriminator(x_bg, x_t)

				if self.global_step % self.opts.log_interval == 0 or (self.n_adv_print < 100 and self.global_step % 25 == 0):
					log_msg = ('[%d/%d]\tLoss_D real: %.4f\tLoss_D fake: %.4f\tLoss: %.4f \n'
								% (self.global_step, self.opts.max_steps, errD_real, errD_fake, errD))
					self.write_log_to_txt(log_msg, filename='train_loss.txt')
					self.n_adv_print += 1

				# Validation
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0:
					val_loss_dict, val_log_msg = self.validate(max_batches=100)
					self.write_log_to_txt(val_log_msg, filename='val_loss.txt')
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict < self.best_val_loss):
						self.best_val_loss = val_loss_dict
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(train_loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self, max_batches=None):
		"""
		Validate the MI discriminator on a subset of the validation set.
		Args:
			max_batches: Maximum number of validation batches to process (optional).
		"""
		total_loss_r = 0.0
		total_loss_f = 0.0
		total_loss = 0.0
		num_batches = 0

		for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.test_bg_dataloader, self.test_t_dataloader)):
			x_bg, _ = batch_bg
			x_t, _ = batch_t

			x_bg, x_t = x_bg.to(self.device).float(), x_t.to(self.device).float()

			# Compute validation loss for the current batch
			errD_real, errD_fake, errD = self.val_discriminator(x_bg, x_t)

			total_loss_r += errD_real
			total_loss_f += errD_fake
			total_loss += errD
			num_batches += 1

			# Stop after max_batches
			if max_batches is not None and num_batches >= max_batches:
				break

		# Compute average validation loss
		avg_val_loss_r = total_loss_r / num_batches
		avg_val_loss_f = total_loss_f / num_batches
		avg_val_loss = total_loss / num_batches
		log_msg = ('[%d/%d]\tLoss_D real: %.4f\tLoss_D fake: %.4f\tLoss: %.4f \n'
								% (self.global_step, self.opts.max_steps, avg_val_loss_r.item(), avg_val_loss_f.item(), avg_val_loss.item()))
		return avg_val_loss.item(), log_msg

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def write_log_to_txt(self, log_msg, filename):
		with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
			f.write(log_msg)

	def write_metrics_to_txt(self, metrics_dict, prefix, filename):
		with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
			f.write(f'Metrics for {prefix}, Step - {self.global_step}')
			f.write(f'\n{metrics_dict}\n')		

	# def print_metrics(self, metrics_dict, prefix):
	# 	print(f'Metrics for {prefix}, step {self.global_step}')
	# 	for key, value in metrics_dict.items():
	# 		print(f'\t{key} = ', value)

	def configure_optimizers(self):
		params = list(self.discriminator.parameters())
		
		optimizerD = torch.optim.Adam(params, lr=self.opts.D_learning_rate)

		return optimizerD

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

		train_bg_dataset = ImagesDataset(source_root=dataset_args['train_bg_source_root'],
									  target_root=dataset_args['train_bg_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		
		train_t_dataset = ImagesDataset(source_root=dataset_args['train_t_source_root'],
									  target_root=dataset_args['train_t_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
				
		test_bg_dataset = ImagesDataset(source_root=dataset_args['test_bg_source_root'],
									 target_root=dataset_args['test_bg_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		
		test_t_dataset = ImagesDataset(source_root=dataset_args['test_t_source_root'],
									 target_root=dataset_args['test_t_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
				
		# if self.opts.use_wandb:
		# 	self.wb_logger.log_dataset_wandb(train_bg_dataset, dataset_name="Train_bg")
		# 	self.wb_logger.log_dataset_wandb(train_t_dataset, dataset_name="Train_t")
		# 	self.wb_logger.log_dataset_wandb(test_bg_dataset, dataset_name="Test_bg")
		# 	self.wb_logger.log_dataset_wandb(test_t_dataset, dataset_name="Test_t")

		print(f"Number of training samples: {len(train_bg_dataset)}")
		print(f"Number of test samples: {len(test_bg_dataset)}")
		return train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset
	
	def output_regularization(self, output, direction='element', reg_type='traditional', lambda_reg=0.1, a=3.7, gamma=3, threshold=0.0, epsilon=1e-8):

		abs_output = torch.abs(output)

		if reg_type == 'traditional':
			regularization_fn = torch.abs
		elif reg_type == 'logarithmic':
			regularization_fn = lambda x: torch.log(1 + torch.abs(x))
		elif reg_type == 'adaptive':
			regularization_fn = lambda x: (1.0 / (torch.abs(x) + epsilon)) * torch.abs(x)
		elif reg_type == 'thresholded':
			regularization_fn = lambda x: torch.where(torch.abs(x) < threshold, torch.abs(x), torch.tensor(0.0, device=x.device))
		elif reg_type == 'SCAD':
			# SCAD penalty function
			def scad_penalty(x):
				penalty = torch.zeros_like(x)
				mask1 = abs_output <= lambda_reg
				penalty[mask1] = lambda_reg * abs_output[mask1]
				mask2 = (abs_output > lambda_reg) & (abs_output <= a * lambda_reg)
				penalty[mask2] = (-x[mask2]**2 + 2 * a * lambda_reg * abs_output[mask2] - lambda_reg**2) / (2 * (a - 1))
				mask3 = abs_output > a * lambda_reg
				penalty[mask3] = (a + 1) * lambda_reg**2 / 2
				return penalty
			regularization_fn = scad_penalty
		elif reg_type == 'MCP':
			# MCP penalty function
			def mcp_penalty(x):
				penalty = torch.zeros_like(x)
				mask1 = abs_output <= gamma * lambda_reg
				penalty[mask1] = lambda_reg * abs_output[mask1] - (x[mask1]**2) / (2 * gamma)
				mask2 = abs_output > gamma * lambda_reg
				penalty[mask2] = (gamma * lambda_reg**2) / 2
				return penalty
			regularization_fn = mcp_penalty
		else:
			raise ValueError(f"Invalid regularization type: '{reg_type}'.")

		# Apply the selected regularization function based on direction
		if direction == 'row':
			return torch.sum(regularization_fn(output).sum(dim=2))
		elif direction == 'column':
			return torch.sum(regularization_fn(output).sum(dim=1))
		elif direction == 'element':
			return torch.sum(regularization_fn(output))
		else:
			raise ValueError(f"Invalid direction: '{direction}'.")


	def elastic_net_regularization(self, model, network='net_s', reg_type='all', alpha=0.5):
		"""
		Apply Elastic Net regularization on the specified layers of either `net_s` or `net_c` in the model, 
		handling both 2D and 3D weight matrices.

		Args:
			model (MappingNetwork_cs_sparsity): The model with the network to regularize.
			network (str): The network to apply Elastic Net regularization ('net_s' or 'net_c').
			reg_type (str): Type of weight regularization ('all' or 'last').
				- 'all': Applies Elastic Net to all EqualizedLinear layers in the specified network.
				- 'last': Applies Elastic Net only to the last EqualizedLinear layer in the specified network.
			alpha (float): The balance between L1 and L2 regularization. 0.0 corresponds to pure L2 (Ridge),
						1.0 to pure L1 (Lasso), and values in between apply both.

		Returns:
			torch.Tensor: Computed Elastic Net regularization penalty for the selected layers.
		"""
		elastic_net_penalty = 0.0

		# Select the target network based on the network argument
		target_network = getattr(model, network, None)
		if target_network is None:
			raise ValueError(f"Invalid network '{network}'. Use 'net_s' or 'net_c'.")

		# Get layers of the selected network
		layers = list(target_network.children())

		# Apply Elastic Net regularization based on reg_type
		if reg_type == 'all':
			# Apply Elastic Net to all EqualizedLinear layers in the target network
			for layer in layers:
				if isinstance(layer[0], EqualizedLinear):
					weights = layer[0].weight.weight
					l1_penalty = torch.sum(torch.abs(weights))
					l2_penalty = torch.sum(weights ** 2)
					elastic_net_penalty += alpha * l1_penalty + (1 - alpha) * l2_penalty
		elif reg_type == 'last':
			# Apply Elastic Net only to the last EqualizedLinear layer in the target network
			last_layer = layers[-1][0]
			if isinstance(last_layer, EqualizedLinear):
				weights = last_layer.weight.weight
				l1_penalty = torch.sum(torch.abs(weights))
				l2_penalty = torch.sum(weights ** 2)
				elastic_net_penalty += alpha * l1_penalty + (1 - alpha) * l2_penalty
		else:
			raise ValueError(f"Invalid reg_type '{reg_type}'. Use 'all' or 'last'.")

		return elastic_net_penalty

	
	def calc_image_loss(self, x, x_hat):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		# Calculate pSp id, rec, lpips losses for images
		if self.opts.id_lambda > 0:			
			loss_id, sim_improvement, id_logs = self.id_loss(x_hat, x, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda

		if self.opts.pix_lambda > 0:
			loss_pix = F.mse_loss(x, x_hat)
			loss_dict['loss_pix'] = float(loss_pix)
			loss += loss_pix * self.opts.pix_lambda

		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(x_hat, x)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda

		loss_dict['loss'] = float(loss)

		return loss, loss_dict, id_logs


	def calc_latent_loss(self, latent_bg_c, latent_bg_s, latent_t_c, latent_t_s, w_bg_pSp, w_t_pSp):
		"""
		Calculate total loss, including L1 regularization for output and weights.

		Args:
			latent_bg_s: Background latent tensor.
			latent_t_s: Target latent tensor.
			latent_bg: Background tensor.
			latent_t: Target tensor.
			latent_bg_target: Target tensor for background.
			latent_t_target: Target tensor for output.
			lasso_sbg_lambda: Weight for output L1 regularization.
			lasso_st_lambda: Weight for output L1 regularization.
			lasso_weight_lambda: Weight for weight L1 regularization.
			model: The model on which to apply regularization.
			lasso_output_type: Type of L1 regularization for output ('row', 'column', 'element').
			lasso_weight_type: Type of L1 regularization for weights ('all', 'last').
			network: Specifies 'net_s' or 'net_c' for weight regularization.

		Returns:
			total_loss: Computed total loss.
			loss_dict: Dictionary with individual loss components.
		"""
		loss_dict = {}
		loss = 0.0

		if self.opts.sbg_lambda > 0:
			if self.opts.sbg_type == 'l2':
				loss_silent_bg = F.mse_loss(latent_bg_s, torch.zeros_like(latent_bg_s).to(self.device))
			elif self.opts.sbg_type == 'fro':
				loss_silent_bg = torch.norm(latent_bg_s, p='fro')  # Frobenius norm
			loss_dict['loss_silent_bg'] = float(loss_silent_bg)
			loss += loss_silent_bg * self.opts.sbg_lambda
			
		if self.opts.w_dist_lambda > 0:
			loss_distance_bg = F.mse_loss(latent_bg_c, w_bg_pSp)
			loss_distance_t = F.mse_loss(latent_t_c + latent_t_s, w_t_pSp)
			loss_dict['loss_distance_bg'] = float(loss_distance_bg)
			loss_dict['loss_distance_t'] = float(loss_distance_t)
			loss += (loss_distance_bg + loss_distance_t) * self.opts.w_dist_lambda	

		if self.opts.lasso_sbg_lambda > 0:
			lasso_sbg_loss = self.output_regularization(latent_bg_s, direction=self.opts.lasso_direction, reg_type=self.opts.lasso_reg_type, threshold=self.opts.lasso_threshold)
			loss_dict['lasso_sbg_loss'] = float(lasso_sbg_loss)  #  direction=self.opts.lasso_direction, reg_type=self.opts.lasso_reg_type, threshold=self.opts.lasso_threshold
			loss += lasso_sbg_loss * self.opts.lasso_sbg_lambda

		if self.opts.lasso_st_lambda > 0:
			lasso_st_loss = self.output_regularization(latent_t_s, direction=self.opts.lasso_direction, reg_type=self.opts.lasso_reg_type, threshold=self.opts.lasso_threshold)
			loss_dict['lasso_st_loss'] = float(lasso_st_loss)
			loss += lasso_st_loss * self.opts.lasso_st_lambda

		with torch.no_grad():	
			zero_count_bg = (latent_bg_s.abs() <= 1e-7).sum().item()	
			zero_count_t = (latent_t_s.abs() <= 1e-7).sum().item()
			total_count = latent_t_s.numel()
			loss_dict['zero_count_sbg'] = int(zero_count_bg)
			loss_dict['zero_count_st'] = int(zero_count_t)
			loss_dict['total_count_s'] = int(total_count)

		if self.opts.lasso_weight_lambda > 0:
			lasso_weight_loss = (self.elastic_net_regularization(self.cs_mlp_net, network='net_s', reg_type=self.opts.lasso_weight_type, alpha=self.opts.elastic_alpha) 
				if self.cs_mlp_net else 0.0)
			loss_dict['lasso_weight_loss'] = float(lasso_weight_loss)
			loss += lasso_weight_loss * self.opts.lasso_weight_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict		


	def merge_loss_dict(self, loss_lat_dict, loss_img_dict_bg, loss_img_dict_t):
		
		# loss_dict = loss_cs_dict | loss_dict_bg_pSp | loss_dict_t_pSp
		loss_dict = {}
		loss_dict['loss_lat'] = loss_lat_dict
		loss_dict['loss_img_bg'] = loss_img_dict_bg
		loss_dict['loss_img_t'] = loss_img_dict_t
		loss_dict['loss_sum'] = loss_lat_dict['loss'] + loss_img_dict_bg['loss'] + loss_img_dict_t['loss']
		
		return loss_dict
		

	# def log_metrics(self, metrics_dict, prefix):
	# 	for key, value in metrics_dict.items():
	# 		self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
	# 	if self.opts.use_wandb:
	# 		self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	# def parse_and_log_images(self, id_logs, x, y_pSp, y_ours, title, subscript=None, display_count=2):
	# 	im_data = []
	# 	for i in range(display_count):
	# 		cur_im_data = {
	# 			'input_face': common.log_input_image(x[i], self.opts),
	# 			'pSp_output_face': common.tensor2im(y_pSp[i]),
	# 			'Our_output_face': common.tensor2im(y_ours[i])
	# 		}
	# 		if id_logs is not None:
	# 			for key in id_logs[i]:
	# 				cur_im_data[key] = id_logs[i][key]
	# 		im_data.append(cur_im_data)
	# 	self.log_images(title, im_data=im_data, subscript=subscript)

	# def log_images(self, name, im_data, subscript=None, log_latest=False):
	# 	fig = common.vis_faces(im_data)
	# 	step = self.global_step
	# 	if log_latest:
	# 		step = 0
	# 	if subscript:
	# 		path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
	# 	else:
	# 		path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
	# 	os.makedirs(os.path.dirname(path), exist_ok=True)
	# 	fig.savefig(path)
	# 	plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'discriminator': self.discriminator.state_dict(),
			# 'state_dict_cs_enc': self.cs_mlp_net.state_dict(),
			'opts': vars(self.opts)
		}
		if self.opts.save_training_data:
			save_dict['global_step'] = self.global_step
			save_dict['optimizerD'] = self.optimizerD.state_dict()
			save_dict['best_val_loss'] = self.best_val_loss
				
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.pSp_net.latent_avg
		return save_dict
	
		
