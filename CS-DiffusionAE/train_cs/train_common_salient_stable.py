import os
import sys

import shutil
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from pathlib import Path
from PIL import Image

sys.path.append(".")
sys.path.append("..")
from templates import *
from model.mlp2D import MappingNetwork_cs
from dataset import LatentDataset  
from base_functions.data_funcs import get_dataloaders, write_log_to_txt, save_hyparams, get_fixed_for_test, show_image_results

def train(cs_model, diffAE_model, optimizer, train_dataloader, val_dataloader, test_fixed, args, device):
    print("training start.....")
    cs_model.train()

    # Create results directories
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.results_dir}/images", exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()  # Enable AMP
    gradient_accumulation_steps = args.grad_acc_steps  # Accumulate gradients over multiple steps

    for epoch in range(args.max_epochs):  # Iterate over epochs
        train_dict = {"total_loss": 0}
        
        for step, batch in enumerate(train_dataloader):
            diffAE_zbg = batch["diffAE_zbg"].to(device, non_blocking=True)
            diffAE_zt = batch["diffAE_zt"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():  # Enable Mixed Precision
                # Forward pass
                c_bg, s_bg = cs_model(diffAE_zbg)
                c_t, s_t = cs_model(diffAE_zt)
                
                # Compute losses
                loss_list = calc_latent_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args)
                total_batch_loss = sum(loss_list.values())
            
            # Gradient accumulation for small batch sizes
            scaler.scale(total_batch_loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Accumulate training loss dynamically
            for loss_name, loss_value in loss_list.items():
                train_dict.setdefault(loss_name, 0)
                train_dict[loss_name] += loss_value.item()
            train_dict["total_loss"] += total_batch_loss.item()

        # Compute average loss per epoch
        for key in train_dict.keys():
            train_dict[key] /= len(train_dataloader)

        # Save train loss
        if (epoch + 1) % args.train_interval == 0:
            train_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in train_dict.items()])
            write_log_to_txt(f"Epoch {epoch+1}/{args.max_epochs} | Train Loss {train_loss_msg}\n", args.results_dir, "train_loss.txt")

        # Run validation and log if it's time
        if (epoch + 1) % args.val_interval == 0:
            val_dict = validate(cs_model, val_dataloader, args, device)
            val_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in val_dict.items()])
            write_log_to_txt(f"Epoch {epoch+1}/{args.max_epochs} | Validation Loss {val_loss_msg}\n", args.results_dir, "val_loss.txt")

        if (epoch + 1) % args.image_interval == 0:
            save_path = f"{args.results_dir}/images/recon_{epoch + 1}.png"
            show_image_results(cs_model, diffAE_model, test_fixed, save_path)

        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.results_dir}/checkpoints/model_epoch_{epoch+1}.pth"
            torch.save(cs_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("training finish")

def validate(cs_model, val_dataloader, args, device):
    """Evaluate the model on the validation dataset."""
    cs_model.eval()
    val_dict = {"total_loss": 0}
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in val_dataloader:
            diffAE_zbg = batch["diffAE_zbg"].to(device, non_blocking=True)
            diffAE_zt = batch["diffAE_zt"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():  # Enable Mixed Precision:
                c_bg, s_bg = cs_model(diffAE_zbg)
                c_t, s_t = cs_model(diffAE_zt)
                loss_list = calc_latent_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args)
                total_batch_loss = sum(loss_list.values())

            # Accumulate validation losses
            for loss_name, loss_value in loss_list.items():
                val_dict.setdefault(loss_name, 0)
                val_dict[loss_name] += loss_value.item()
            val_dict["total_loss"] += total_batch_loss.item()

    # Compute average validation loss
    for key in val_dict.keys():
        val_dict[key] /= num_batches

    cs_model.train()  # Switch back to training mode
    return val_dict

def calc_latent_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args):
    """Compute different loss components for training."""
    loss_list = {
        "loss_bg": args.w_bg * F.mse_loss(c_bg, diffAE_zbg),
        "loss_t": args.w_t * F.mse_loss(c_t + s_t, diffAE_zt),
        "loss_sbg": args.w_sbg * F.mse_loss(s_bg, torch.zeros_like(s_bg))
    }
    return loss_list

def load_models(device, args):
    """Load Diffusion AE and Mapping Network models."""
    
    # Load Diffusion AE Model
    conf = ffhq256_autoenc()
    diffAE_model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    diffAE_model.load_state_dict(state['state_dict'], strict=False)
    diffAE_model.ema_model.eval()
    diffAE_model.ema_model.to(device)

    # Load Mapping Network
    cs_model = MappingNetwork_cs(features=args.features, n_layers=args.n_layers).to(device)

    # Define Optimizer
    optimizer = optim.Adam(cs_model.parameters(), lr=args.learning_rate)

    return diffAE_model, cs_model, optimizer


def main():
    """Main function to initialize and run training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffAE_path', type=str, default='model_path')
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--grad_acc_steps', type=int, default=8)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--image_data_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42) 

    parser.add_argument('--w_bg', type=float, default=1.0)
    parser.add_argument('--w_t', type=float, default=1.0)
    parser.add_argument('--w_sbg', type=float, default=1.0)

    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--train_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--image_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    random.seed(args.seed)  # Ensure reproducibility
    print("Training arguments:", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shutil.rmtree(args.results_dir, ignore_errors=True)
    os.makedirs(args.results_dir, exist_ok=True)
    save_hyparams(args)

    diffAE_model, cs_model, optimizer = load_models(device, args)

    image_data_dir = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs"
    train_dataloader, val_dataloader, test_batch = get_dataloaders(args)

    test_fixed = get_fixed_for_test(test_batch, image_data_dir, diffAE_model, device)

    train(cs_model, diffAE_model, optimizer, train_dataloader, val_dataloader, test_fixed, args, device)


if __name__ == "__main__":
    main()
