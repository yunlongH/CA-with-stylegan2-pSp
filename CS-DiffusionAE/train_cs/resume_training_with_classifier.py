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
from custom_funcs.data_funcs import get_dataloaders
from custom_funcs.model_funcs import load_diffusion_model, load_cs_model
from custom_funcs.eval_funcs import get_fixed_for_test, visualize_eval_recons
from custom_funcs.utils import write_log_to_txt, save_hyparams 
from custom_funcs.loss_funcs import calc_latent_loss


def train(cs_model, diffAE_model, optimizer, train_dataloader, val_dataloader, test_image_fixed, args, device):

    print("training start.....")
    cs_model.train()

    # Create results directories
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.results_dir}/images", exist_ok=True)

    for epoch in range(args.max_epochs):  # Iterate over epochs
        train_dict = {"total_loss": 0}
        
        for batch in train_dataloader:
            diffAE_zbg = batch["diffAE_zbg"].to(device, non_blocking=True)
            diffAE_zt = batch["diffAE_zt"].to(device, non_blocking=True)

            # Forward pass
            c_bg, s_bg = cs_model(diffAE_zbg)
            c_t, s_t = cs_model(diffAE_zt)

            # Compute losses
            loss_list = calc_latent_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args)
            total_batch_loss = sum(loss_list.values())

            # Backpropagation
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            # Accumulate training loss dynamically
            for loss_name, loss_value in loss_list.items():
                train_dict.setdefault(loss_name, 0)
                train_dict[loss_name] += loss_value.item()
            train_dict["total_loss"] += total_batch_loss.item()

        # Compute average loss per epoch
        for key in train_dict.keys():
            train_dict[key] /= len(train_dataloader)

        # Save **train loss** independently if it's the train interval
        if (epoch + 1) % args.train_interval == 0:
            train_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in train_dict.items()])
            #print(f"Epoch {epoch+1}/{args.max_epochs} | Train Loss | {train_loss_msg}")
            write_log_to_txt(f"Epoch {epoch+1}/{args.max_epochs} | Train Loss {train_loss_msg}\n", args.results_dir, "train_loss.txt")

        # Run validation and log if it's time
        if (epoch + 1) % args.val_interval == 0:
            val_dict = validate(cs_model, val_dataloader, args, device)

            # Save **validation loss** independently if it's the val interval
            val_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in val_dict.items()])
            #print(f"Epoch {epoch+1}/{args.max_epochs} | Validation Loss | {val_loss_msg}")
            write_log_to_txt(f"Epoch {epoch+1}/{args.max_epochs} | Validation Loss {val_loss_msg}\n", args.results_dir, "val_loss.txt")

        if (epoch + 1) % args.image_interval == 0:
            # print("showing start.....")
            save_path = f"{args.results_dir}/images/recon_{epoch + 1}.png"
            #show_image_results(cs_model, diffAE_model, test_fixed, save_path)
            visualize_eval_recons(cs_model, diffAE_model, test_image_fixed, T_render=10, save_dir=save_path, is_train=True)
            
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

            # Forward pass
            c_bg, s_bg = cs_model(diffAE_zbg)
            c_t, s_t = cs_model(diffAE_zt)

            # Compute validation losses
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


def main():
    """Main function to initialize and run training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffAE_path', type=str, default='model_path')
    parser.add_argument('--cs_model_ckpt', type=str, default=None)
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    shutil.rmtree(args.results_dir, ignore_errors=True)
    os.makedirs(args.results_dir, exist_ok=True)
    save_hyparams(args)

    diff_model = load_diffusion_model(device)
    cs_model, optimizer = load_cs_model(device, args, is_train=True)

    train_dataloader, val_dataloader = get_dataloaders(args)
    real_image_dir = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs"
    test_batch = next(iter(val_dataloader))
    test_image_fixed = get_fixed_for_test(test_batch, real_image_dir, diff_model, device, T_noise=250, idx=1)

    train(cs_model, diff_model, optimizer, train_dataloader, val_dataloader, test_image_fixed, args, device)


if __name__ == "__main__":
    main()
