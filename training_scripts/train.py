"""
This file runs the main training/val loop
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import json
import sys
import pprint
import torch
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from configs.paths_config import model_paths
from argparse import Namespace

def main():

	previous_train_path = model_paths['previous_train_ckpt_path']
	previous_train_ckpt = None

	if previous_train_path is not None:
		print('start from previous training checkpoint...')
		previous_train_ckpt = torch.load(previous_train_path, map_location='cpu', weights_only=True)
		opts = load_opts_from_checkpoint(previous_train_ckpt)
	else:
		opts = TrainOptions().parse()
		create_initial_experiment_dir(opts)

	if opts.exp_scheme=='baseline':
		print('using coach_csmlp_baseline.py ...')
		from training.coach_csmlp_baseline import Coach_csmlp
		
	# elif opts.exp_scheme=='swap_loss':
	# 	print('using coach_csmlp_swap_losses.py ...')
	# 	from training.coach_csmlp_swap_losses import Coach_csmlp

	# elif opts.exp_scheme=='train_c2s_mlp':
	# 	print('using coach_train_c2s_mlp.py ...')
	# 	from training.coach_train_c2s_mlp import Coach_csmlp

	# elif opts.exp_scheme=='train_mi_disc':
	# 	print('using coach_train_mi_disc.py ...')
	# 	from training.coach_train_mi_disc import Coach_csmlp

	elif opts.exp_scheme=='adverserial_common':
		print('using coach_adverserial_common.py ...')
		from training.coach_adverserial_common import Coach_csmlp

	elif opts.exp_scheme=='adverserial_mutR':
		print('using coach_adverserial_mutual_R.py ...')
		from training.coach_adverserial_mutual_R import Coach_csmlp

	elif opts.exp_scheme=='improved_loss':
		print('using coach_contrastive_Dist_Recon.py ...')
		from main.coach_contrastive_Disc_Recon import Coach_csmlp

	elif opts.exp_scheme=='mult_optims':
		print('using coach_alternate_optimizers.py ...')
		from main.coach_alternate_optimizers import Coach_csmlp

	# elif opts.exp_scheme=='cmlp_mi_disc':
	# 	print('using coach_resume_train_mi_disc.py ...')
	# 	from training.coach_resume_train_mi_disc import Coach_csmlp

	elif opts.exp_scheme=='2D-weights':
		print('coach_csmlp_2Dweights.py ...')
		from training.coach_csmlp_2Dweights import Coach_csmlp	

	elif opts.exp_scheme=='3D-weights':
		print('using coach_csmlp_3Dweights.py ...')
		from training.coach_csmlp_3Dweights import Coach_csmlp
	else:
		raise Exception('errors in loading coach file')
	
	coach = Coach_csmlp(opts, previous_train_ckpt)
	coach.train()

def load_opts_from_checkpoint(previous_train_ckpt):
	
	opts = previous_train_ckpt['opts']
	opts = Namespace(**opts)	
	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	
	return opts


def create_initial_experiment_dir(opts):
	if os.path.exists(opts.exp_dir):
		shutil.rmtree(opts.exp_dir)
		#raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)

	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

if __name__ == '__main__':
	main()
