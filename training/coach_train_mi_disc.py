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
from models.MI_discriminator import Discriminator, DiscriminatorGlobal

class Coach_csmlp:
	def __init__(self, opts, previous_train_ckpt=None):
		self.opts = opts
		self.global_step = 0
		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		self.seed_experiments(opts.seed)
		self.criterion = nn.BCELoss()

		if opts.disc_type == 'global':
			self.mi_discriminator = DiscriminatorGlobal(self.opts).to(self.opts.device)
		else:
			self.mi_discriminator = Discriminator(self.opts).to(self.opts.device)

		self.init_pretrained_models(self.opts.csmlp_checkpoint_path)

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

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

		# self.init_additional_params(previous_train_ckpt)
		
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
		
	def init_pretrained_models(self, pretrained_ckpt_path):

		model_ckpt = torch.load(pretrained_ckpt_path, map_location=self.device, weights_only=True)
		cs_ckpt  = model_ckpt['state_dict_cs_enc']
		model_opts = model_ckpt['opts']
		model_opts = Namespace(**model_opts)

		# load pSp model and pretrained weights
		self.pSp_net = pSp(model_opts).to(self.device)

		# load csmlp model and pretrained weights
		self.cs_mlp_net = MappingNetwork_cs_independent(model_opts).to(self.device)
		print('Loading csmlp from path: {}'.format(model_opts.exp_dir))     
		self.cs_mlp_net.load_state_dict(cs_ckpt)

		self.pSp_net.eval()
		self.cs_mlp_net.eval()

	def pSp_csmlp_encoding(self, images):
		
		# with torch.no_grad():
		w_pSp = self.pSp_net.forward(images, encode_only=True)
		c, s = self.cs_mlp_net(w_pSp, zero_out_silent=self.opts.zero_out_silent_t) 

		return c, s

	def update_mi_disc(self, x_t):
		"""
		Train the MI discriminator to classify real (P(c, s)) vs. fake (P(c)P(s)) pairs.
		Args:
			x_t: Full input batch of data (Tensor of shape [batch_size, ...]).
		"""
		self.mi_discriminator.train()
		self.optimizer.zero_grad()

		# Ensure batch size is even
		batch_size = x_t.size(0)
		if batch_size % 2 != 0:
			x_t = x_t[:-1]  # Remove the last sample to make it even
		mid = batch_size // 2

		# Split into two sub-batches
		batch_a, batch_b = x_t[:mid], x_t[mid:]

		# Encode batch_a and batch_b into (c, s) representations
		c_a, s_a = self.pSp_csmlp_encoding(batch_a)  # From batch_a
		c_b, s_b = self.pSp_csmlp_encoding(batch_b)  # From batch_b

		# **Real Pairs (P(c, s))**
		pos_1 = self.mi_discriminator(c_a, s_a)  # Real Pair 1
		pos_2 = self.mi_discriminator(c_b, s_b)  # Real Pair 2

		# **Fake Pairs (P(c)P(s))**
		neg_1 = self.mi_discriminator(c_a, s_b)  # Fake Pair 1
		neg_2 = self.mi_discriminator(c_b, s_a)  # Fake Pair 2

		# **Step 4: Create Labels**
		labels = torch.cat([
			torch.ones(mid, 1),  # Real (c_a, s_a)
			torch.ones(mid, 1),  # Real (c_b, s_b)
			torch.zeros(mid, 1),  # Fake (c_a, s_b)
			torch.zeros(mid, 1)   # Fake (c_b, s_a)
		]).to(self.device)

		predictions = torch.cat([pos_1, pos_2, neg_1, neg_2], dim=0)

		# **Step 5: Compute Loss and Update**
		loss = self.criterion(predictions, labels)
		loss.backward()
		self.optimizer.step()

		train_loss_dict = float(loss.item())  # Ensure it's a Python float
		return train_loss_dict


	def val_mi_disc(self, x_t):
		"""
		Validate the MI discriminator on a single validation batch.
		Args:
			x_t: Full input batch of validation data (Tensor of shape [batch_size, ...]).
		Returns:
			val_loss_dict: Validation loss as a Python float.
		"""
		self.mi_discriminator.eval()  # Set discriminator to evaluation mode

		with torch.no_grad():  # Disable gradient computation for validation
			# Ensure batch size is even
			batch_size = x_t.size(0)
			if batch_size % 2 != 0:
				x_t = x_t[:-1]  # Remove the last sample to make it even
			mid = batch_size // 2

			# Split into two sub-batches
			batch_a, batch_b = x_t[:mid], x_t[mid:]

			# Encode batch_a and batch_b into (c, s) representations
			c_a, s_a = self.pSp_csmlp_encoding(batch_a)
			c_b, s_b = self.pSp_csmlp_encoding(batch_b)

			# **Real Pairs (P(c, s))**
			pos_1 = self.mi_discriminator(c_a, s_a)  # Real Pair 1
			pos_2 = self.mi_discriminator(c_b, s_b)  # Real Pair 2

			# **Fake Pairs (P(c)P(s))**
			neg_1 = self.mi_discriminator(c_a, s_b)  # Fake Pair 1
			neg_2 = self.mi_discriminator(c_b, s_a)  # Fake Pair 2

			# **Step 4: Create Labels**
			labels = torch.cat([
				torch.ones(mid, 1),  # Real (c_a, s_a)
				torch.ones(mid, 1),  # Real (c_b, s_b)
				torch.zeros(mid, 1),  # Fake (c_a, s_b)
				torch.zeros(mid, 1)   # Fake (c_b, s_a)
			]).to(self.device)

			predictions = torch.cat([pos_1, pos_2, neg_1, neg_2], dim=0)

			# **Step 5: Compute Loss**
			val_loss = self.criterion(predictions, labels)

			# Return loss as a Python float
			val_loss_dict = float(val_loss.item())

		return val_loss_dict


	def train(self):

		while self.global_step < self.opts.max_steps:

			for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.train_bg_dataloader, self.train_t_dataloader)):
				
				x_t, _ = batch_t
				x_t = x_t.to(self.device).float()

				# Update MI discriminator with the full batch
				train_loss_dict = self.update_mi_disc(x_t)

				if self.global_step % self.opts.log_interval == 0 or (self.global_step < 1000 and self.global_step % 50 == 0):
					self.write_metrics_to_txt(train_loss_dict, prefix='train', filename='train_loss.txt')

				# Validation
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or (self.global_step < 1000 and self.global_step % 50 == 0):
					val_loss_dict = self.validate()
					self.write_metrics_to_txt(val_loss_dict, prefix='validate', filename='validate_loss.txt')
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
		total_loss = 0.0
		num_batches = 0

		for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.test_bg_dataloader, self.test_t_dataloader)):
			x_t, _ = batch_t
			x_t = x_t.to(self.device).float()

			# Compute validation loss for the current batch
			val_loss_dict = self.val_mi_disc(x_t)

			total_loss += val_loss_dict
			num_batches += 1

			# Stop after max_batches
			if max_batches is not None and num_batches >= max_batches:
				break

		# Compute average validation loss
		avg_val_loss = total_loss / num_batches
		return avg_val_loss


	def write_metrics_to_txt(self, metrics, prefix, filename):
		"""
		Write metrics to a text file in a readable format.
		Args:
			metrics: Can be a float (single value) or a dictionary of metric names and values.
			prefix: A string to identify the type of metrics (e.g., 'train', 'val').
			filename: Name of the file to write metrics to.
		"""
		with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
			if isinstance(metrics, dict):
				# Write each key-value pair from the dictionary
				f.write(f"{prefix} step-{self.global_step}, ")
				f.write(", ".join([f"{key}={value:.4f}" for key, value in metrics.items()]))
			elif isinstance(metrics, float):
				# Directly write the float value as a single loss
				f.write(f"{prefix} step-{self.global_step}, Loss={metrics:.4f}")
			else:
				raise TypeError("Metrics must be either a float or a dictionary.")
			f.write("\n")  # Add a newline for better readability


	# def write_metrics_to_txt(self, metrics_dict, prefix, filename):
	# 	with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
	# 		f.write(f'Metrics for {prefix}, Step - {self.global_step}')
	# 		f.write(f'\n{metrics_dict}\n')	

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

	def configure_optimizers(self):
		params = list(self.mi_discriminator.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

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

		print(f"Number of traing bg samples: {len(train_bg_dataset)}")
		print(f"Number of traing t samples: {len(train_t_dataset)}")
		print(f"Number of test bg samples: {len(test_bg_dataset)}")
		print(f"Number of test t samples: {len(test_t_dataset)}")
		return train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset

	def __get_save_dict(self):
		save_dict = {
			#'state_dict_pSp': self.pSp_net.state_dict(),
			'state_dict_mi_disc': self.mi_discriminator.state_dict(),
			'opts': vars(self.opts)
		}
		if self.opts.save_training_data:
			save_dict['global_step'] = self.global_step
			save_dict['optimizer'] = self.optimizer.state_dict()
			save_dict['best_val_loss'] = self.best_val_loss
				
		# # save the latent avg in state_dict for inference if truncation of w was used during training
		# if self.opts.start_from_latent_avg:
		# 	save_dict['latent_avg'] = self.pSp_net.latent_avg
		return save_dict
	
		
