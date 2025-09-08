import torch
from torch import optim
from templates import *
from model.mlp2D import MappingNetwork_cs
from model.discriminators import CustomLatentClassifier
from argparse import Namespace
import json

def load_diffusion_model(device):
    """Load Diffusion AE and Mapping Network models."""
    
    # Load Diffusion AE Model
    conf = ffhq256_autoenc()
    # print(conf.name)
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    return model, conf


def load_cs_model(device, args, is_train=True):
    """
    Load the Mapping Network model (common and salient model) for Diffusion AE.
    If training, also return the optimizer.
    Supports loading from checkpoint if provided.
    """
    # Update network params if checkpoint path provided
    if args.cs_model_ckpt is not None:
        exp_dir = args.cs_model_ckpt.split("checkpoints")[0].rstrip("/")
        hyparams_path = os.path.join(exp_dir, "hyparams.json")
        with open(hyparams_path, 'r') as f:
            net_args_dict = json.load(f)
            net_args = Namespace(**net_args_dict)

        # update only the network arguments
        if hasattr(net_args, 'features'):
            args.features = net_args.features
        if hasattr(net_args, 'n_layers'):
            args.n_layers = net_args.n_layers

    cs_model = MappingNetwork_cs(
        features=args.features,
        n_layers=args.n_layers
    ).to(device)

    # Load model weights if checkpoint path provided
    if hasattr(args, 'cs_model_ckpt') and args.cs_model_ckpt is not None:
        cs_model.load_state_dict(torch.load(args.cs_model_ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.cs_model_ckpt}")

    if is_train:
        optimizer = optim.Adam(cs_model.parameters(), lr=args.learning_rate)
        return cs_model, optimizer
    else:
        cs_model.eval()
        return cs_model


def load_disc_model(device, args, is_train=True):
    """Load the classifier for the common salient learning."""

    # Update network params if checkpoint path provided
    if args.disc_model_ckpt is not None:
        exp_dir = args.disc_model_ckpt.split("checkpoints")[0].rstrip("/")
        hyparams_path = os.path.join(exp_dir, "hyparams.json")
        with open(hyparams_path, 'r') as f:
            net_args_dict = json.load(f)
            net_args = Namespace(**net_args_dict)
        # update only the network arguments
            args.disc_input_dim = net_args.disc_input_dim
            args.disc_n_layers = net_args.disc_n_layers

    disc_model = CustomLatentClassifier(
        input_dim=args.disc_input_dim,
        num_layers=args.disc_n_layers
    ).to(device)

    # Load model weights if checkpoint path provided
    if hasattr(args, 'disc_model_ckpt') and args.disc_model_ckpt is not None:
        disc_model.load_state_dict(torch.load(args.disc_model_ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.disc_model_ckpt}")
    if is_train:
        optimizer = optim.Adam(disc_model.parameters(), lr=args.lr_disc)
        return disc_model, optimizer
    else:
        disc_model.eval()
        return disc_model
    
