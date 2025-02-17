import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import shutil
import torch
import json
import os
import sys
import torch.nn.functional as F
sys.path.append(".")
sys.path.append("..")
from base_functions.model_funcs import load_pSp_cmlp_models
from base_functions.data_funcs import reproduce_latent_hdf5, HDF5LatentDataset
from base_functions.base_funcs import seed_experiments
import random
from torch.utils.data import Dataset, DataLoader
from models.c2s_mlp import SimpleLinearModel, DeepC2SModel, StrongerC2SModel
from models.mlp3D import MappingNetwork_c2s


def train(model, train_dataloader, val_dataloader, optimizer, args, device):
    """
    Train the model with mixed precision, MSE + L1 loss, and gradient accumulation.
    """
    model.train()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"  # ✅ Convert `torch.device` to string

    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)

    scaler = torch.amp.GradScaler(device=device_type)  # ✅ Enable AMP
    gradient_accumulation_steps = 8  # ✅ Accumulate gradients over multiple steps

    for epoch in range(args.max_epochs):

        running_loss_bg, running_loss_t, running_loss = 0.0, 0.0, 0.0

        for step, (latent_bg_c, latent_bg_s, latent_t_c, latent_t_s) in enumerate(train_dataloader):

            # Move data to GPU efficiently
            latent_bg_c = latent_bg_c.to(device, non_blocking=True)
            latent_bg_s = latent_bg_s.to(device, non_blocking=True)
            latent_t_c = latent_t_c.to(device, non_blocking=True)
            latent_t_s = latent_t_s.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type):  # ✅ Enable Mixed Precision
                recon_bg_s = model(latent_bg_c)
                recon_t_s = model(latent_t_c)

                if args.loss_type == 'mse':
                    loss_bg = F.mse_loss(recon_bg_s, torch.zeros_like(latent_bg_s))
                    loss_t = F.mse_loss(recon_t_s, latent_t_s)
                elif args.loss_type == 'l1':
                    loss_bg = F.l1_loss(recon_bg_s, torch.zeros_like(latent_bg_s))
                    loss_t = F.l1_loss(recon_t_s, latent_t_s)

                loss_unscaled = loss_bg + loss_t  # ✅ Keep unscaled loss for tracking
                loss = loss_unscaled / gradient_accumulation_steps  # ✅ Only divide for backpropagation

            # ✅ Gradient accumulation for small batch sizes
            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ✅ Track losses using unscaled loss
            running_loss_bg += loss_bg.item()
            running_loss_t += loss_t.item()
            running_loss += loss_unscaled.item()  # ✅ Track the real loss, not the divided one

        # ✅ Compute the average loss for the epoch
        avg_loss_bg = running_loss_bg / len(train_dataloader)
        avg_loss_t = running_loss_t / len(train_dataloader)
        avg_loss = running_loss / len(train_dataloader)  # ✅ Now matches validation loss

        # ✅ Validation step
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.max_epochs:
            val_loss_bg, val_loss_t, val_loss = validate(model, args, val_dataloader, device)
            log_msg = (
                f"Epoch {epoch+1} | Train Loss_bg: {avg_loss_bg:.4f}, Loss_t: {avg_loss_t:.4f}, Loss: {avg_loss:.4f} | "
                f"Val Loss_bg: {val_loss_bg:.4f}, Loss_t: {val_loss_t:.4f}, Loss: {val_loss:.4f}\n"
            )

        else:
            log_msg = f"Epoch {epoch+1} | Train Loss_bg: {avg_loss_bg:.4f}, Loss_t: {avg_loss_t:.4f}, Loss: {avg_loss:.4f}\n"

        write_log_to_txt(log_msg, args.results_dir, "train_val_loss.txt")

        # ✅ Save model checkpoint every `save_interval` epochs
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.max_epochs:
            checkpoint_path = f"{args.results_dir}/checkpoints/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")



def validate(model, args, val_dataloader, device):
    """
    Evaluate the model on the validation dataset with mixed precision.
    """
    model.eval()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"  # ✅ Convert `torch.device` to string
    running_loss_bg, running_loss_t, running_loss = 0.0, 0.0, 0.0

    with torch.no_grad():
        for latent_bg_c, latent_bg_s, latent_t_c, latent_t_s in val_dataloader:
            latent_bg_c = latent_bg_c.to(device, non_blocking=True)
            latent_bg_s = latent_bg_s.to(device, non_blocking=True)
            latent_t_c = latent_t_c.to(device, non_blocking=True)
            latent_t_s = latent_t_s.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type):  # ✅ Enable AMP for validation too
                recon_bg_s = model(latent_bg_c)
                recon_t_s = model(latent_t_c)

                if args.loss_type == 'mse':
                    loss_bg = F.mse_loss(recon_bg_s, torch.zeros_like(latent_bg_s))
                    loss_t = F.mse_loss(recon_t_s, latent_t_s)
                elif args.loss_type == 'l1':
                    loss_bg = F.l1_loss(recon_bg_s, torch.zeros_like(latent_bg_s))
                    loss_t = F.l1_loss(recon_t_s, latent_t_s)

                loss = loss_bg + loss_t

            # ✅ Track losses
            running_loss_bg += loss_bg.item()
            running_loss_t += loss_t.item()
            running_loss += loss.item()

        # ✅ Compute the average loss for the epoch
        val_loss_bg = running_loss_bg / len(val_dataloader)
        val_loss_t = running_loss_t / len(val_dataloader)
        val_loss = running_loss / len(val_dataloader)

    model.train()  # ✅ Return to training mode
    return val_loss_bg, val_loss_t, val_loss
    
def write_log_to_txt(log_msg, results_dir, filename):
    with open(f"{results_dir}/logs/{filename}", 'a') as f:
        f.write(log_msg)

def load_config(config_path, experiment_name):
    """Load JSON config file and return the selected experiment's config."""
    with open(config_path, "r") as file:
        full_config = json.load(file)

    if experiment_name not in full_config:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")

    return full_config[experiment_name]

def save_hyparams(args):
    # Save hyperparameters to JSON
    args_json_path = os.path.join(args.results_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    print(f"Arguments saved to {args_json_path}")


def main():
    # Step 1: Parse `--config` and `--experiment` first
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--experiment', type=str, required=True, help="Name of the experiment to run")

    args, remaining_args = parser.parse_known_args()
    print("Parsed args before loading config:", vars(args))

    # Step 2: Load experiment settings from JSON
    config = load_config(args.config, args.experiment)
    print("Loaded config:", config)

    parser = argparse.ArgumentParser()
    # Define arguments using JSON configuration as defaults
    parser.add_argument('--model_path', type=str, default=config.get('model_path', './results/baseline/iteration_130000.pt'))
    parser.add_argument('--latent_hdf5_path', type=str, default=config.get('latent_hdf5_path', './results/baseline/cmlp130k_latent'))
    parser.add_argument('--results_dir', type=str, default=config.get('results_dir', './results/baseline/TEST'))
    parser.add_argument('--max_epochs', type=int, default=config.get('max_epochs', 1000))
    parser.add_argument('--log_interval', type=int, default=config.get('log_interval', 1))
    parser.add_argument('--val_interval', type=int, default=config.get('val_interval', 20))
    parser.add_argument('--save_interval', type=int, default=config.get('save_interval', 20))
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 4))
    parser.add_argument('--n_c2s_layers', type=int, default=config.get('n_c2s_layers', 12))
    parser.add_argument('--network_type', type=str, default=config.get('network_type', 'c2smlp'))
    parser.add_argument('--loss_type', type=str, default=config.get('loss_type', 'mse'))
    parser.add_argument('--style_dim', type=int, default=config.get('style_dim', 18))
    parser.add_argument('--latent_dim', type=int, default=config.get('latent_dim', 512))
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.0001))
    parser.add_argument('--prev_train_path', type=str, default=config.get('prev_train_path', None))

    # # Boolean flags
    # parser.add_argument('--reproduce_latent', action='store_true', help="Enable latent reproduction")
    # parser.add_argument('--no_reproduce_latent', action='store_false', dest="reproduce_latent", help="Disable latent reproduction")
    # parser.set_defaults(reproduce_latent=not config.get('no_reproduce_latent', True))

    # Parse final arguments (command-line overrides JSON)
    args = parser.parse_args(remaining_args)

    # Generate a random seed
    args.seed = random.randint(0, 2**10 - 1)
    print("Random Seed:", args.seed)

    # Print loaded settings for debugging
    print("Training arguments:", vars(args))

    # Ensure results directory is fresh
    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    # Save arguments
    save_hyparams(args)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_hdf5_path = f"{args.latent_hdf5_path}/train_latents.h5"
    val_hdf5_path = f"{args.latent_hdf5_path}/val_latents.h5"

    # Check if both files exist
    if os.path.exists(train_hdf5_path) and os.path.exists(val_hdf5_path):
        pass  # Do nothing, as the files already exist
    else:
        print("Latent files not found. Running reproduction code...")
        pSp_net, cs_mlp_net, opts = load_pSp_cmlp_models(args.model_path, device=device)
        reproduce_latent_hdf5(cs_mlp_net, pSp_net, train_hdf5_path, val_hdf5_path, opts, seed=args.seed, device=device)

    # latent_keys = ["w_pSp_bg", "w_pSp_t", "latent_bg_c", "latent_bg_s", "latent_t_c", "latent_t_s"]
    latent_keys = [ "latent_bg_c", "latent_bg_s", "latent_t_c", "latent_t_s"]
    
    # Initialize dataset
    train_dataset = HDF5LatentDataset(train_hdf5_path, latent_keys)
    train_latent_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = HDF5LatentDataset(val_hdf5_path, latent_keys)
    val_latent_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    #model = MappingNetwork_c2s(features=args.pca_latent_dim, n_layers=args.n_c2s_layers).to(device)
    if args.network_type == 'simple':
        model = SimpleLinearModel().to(device)
    elif args.network_type == 'deep':
        model = DeepC2SModel().to(device)
    elif args.network_type == 'strong': 
        model = StrongerC2SModel().to(device)
    elif args.network_type == 'c2smlp': 
        model = MappingNetwork_c2s(args).to(device)

    if args.prev_train_path:
        print(f"Loaded model from {args.prev_train_path}")
        model.load_state_dict(torch.load(args.prev_train_path, map_location='cuda'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model with periodic validation
    train(model, train_latent_dataloader, val_latent_dataloader, optimizer, args, device)

if __name__ == "__main__":
    main()
