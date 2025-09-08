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
from custom_funcs.data_funcs import config_dataset_path, load_lmdb_dataset       
from custom_funcs.model_funcs import load_diffusion_model, load_cs_model
from custom_funcs.eval_funcs import get_fixed_for_test, eval_recons
from custom_funcs.utils import write_log_to_txt, save_hyparams 
from custom_funcs.loss_funcs import calc_latent_loss, calc_recon_images, calc_image_loss

def train(cs_model, diff_model, optimizer, 
          trainloader_bg, trainloader_t, valloader_bg, valloader_t,
          test_image_fixed, args, device):

    print("training start.....")
    cs_model.train()
    
    # Create results directories
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.results_dir}/images", exist_ok=True)
    steps = 0

    while steps < args.max_steps:
        train_dict = {"total_loss": 0}
        
        for batch_idx, (batch_bg, batch_t) in enumerate(zip(trainloader_bg, trainloader_t)):
            imgs_bg = batch_bg['img'].to(device)
            imgs_t = batch_t['img'].to(device)
            diffAE_zbg = batch_bg['latent'].to(device)
            diffAE_zt = batch_t['latent'].to(device)
            
            # Forward pass
            c_bg, s_bg = cs_model(diffAE_zbg)
            c_t, s_t = cs_model(diffAE_zt)

            recon_bg, recon_t = calc_recon_images(diff_model, imgs_bg, imgs_t, 
                                                  latents_bg = c_bg, latents_t=c_t + s_t, 
                                                  T_noise = args.T_noise,
                                                  T_render = args.T_noise)

            # Compute losses
            latent_loss = calc_latent_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args)

            # Compute image losses and logs
            image_loss_bg, _ = calc_image_loss(imgs_bg, recon_bg, args) 
            image_loss_t, _ = calc_image_loss(imgs_t, recon_t, args)

            # Sum total image losses
            total_image_loss_bg = sum(image_loss_bg.values())
            total_image_loss_t = sum(image_loss_t.values())

            # Combine all losses
            total_batch_loss = sum(latent_loss.values()) + total_image_loss_bg + total_image_loss_t

            # Backpropagation
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            # Record latent losses
            for loss_name, loss_value in latent_loss.items():
                train_dict[loss_name] = loss_value.item()

            # Record image losses
            for loss_name, loss_value in image_loss_bg.items():
                train_dict[f"{loss_name}_bg"] = loss_value.item()
            for loss_name, loss_value in image_loss_t.items():
                train_dict[f"{loss_name}_t"] = loss_value.item()

            # Record total loss
            train_dict["total_loss"] = total_batch_loss.item()

            if (steps + 1) % args.train_interval == 0:
                train_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in train_dict.items()])
                write_log_to_txt(f"Steps {steps+1}/{args.max_steps} | Train Loss {train_loss_msg}\n", args.results_dir, "train_loss.txt")

            # Run validation and log if it's time
            if (steps + 1) % args.val_interval == 0:
                val_dict = validate(cs_model, diff_model, valloader_bg, valloader_t, args, device)
                val_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in val_dict.items()])
                write_log_to_txt(f"Steps {steps+1}/{args.max_steps} | Validation Loss {val_loss_msg}\n", args.results_dir, "val_loss.txt")

            if (steps + 1) % args.image_interval == 0:
                # print("showing start.....")
                save_path = f"{args.results_dir}/images/recon_step-{steps + 1}.png"
                eval_recons(cs_model, diff_model, test_image_fixed, T_render=50, save_dir=save_path, is_train=True)
                
            # Save model checkpoint
            if (steps + 1) % args.save_interval == 0:
                checkpoint_path = f"{args.results_dir}/checkpoints/model_epoch_{steps+1}.pth"
                torch.save(cs_model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

            steps += 1

    print("training finish")

def validate(cs_model, diff_model, valloader_bg, valloader_t, args, device):
    """Evaluate the model on the validation dataset."""
    cs_model.eval()
    val_dict = {"total_loss": 0}

    with torch.no_grad():
        for batch_idx, (batch_bg, batch_t) in enumerate(zip(valloader_bg, valloader_t)):
            # imgs_bg = batch_bg['img'].to(device)
            # imgs_t = batch_t['img'].to(device)
            diffAE_zbg = batch_bg['latent'].to(device)
            diffAE_zt = batch_t['latent'].to(device)

            # Forward pass
            c_bg, s_bg = cs_model(diffAE_zbg)
            c_t, s_t = cs_model(diffAE_zt)

            # recon_bg, recon_t = calc_recon_images(
            #     diff_model, imgs_bg, imgs_t,
            #     latents_bg=c_bg, latents_t=c_t + s_t,
            #     T_noise=args.T_noise,
            #     T_render=args.T_noise
            # )

            # Compute losses
            latent_loss = calc_latent_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args)
            
            # Total loss
            total_batch_loss = sum(latent_loss.values()) #+ total_image_loss_bg + total_image_loss_t

            # Accumulate latent losses
            for loss_name, loss_value in latent_loss.items():
                val_dict.setdefault(loss_name, 0)
                val_dict[loss_name] += loss_value.item()

            # # Accumulate image losses (bg)
            # for loss_name, loss_value in image_loss_bg.items():
            #     key = f"{loss_name}_bg"
            #     val_dict.setdefault(key, 0)
            #     val_dict[key] += loss_value.item()

            # # Accumulate image losses (t)
            # for loss_name, loss_value in image_loss_t.items():
            #     key = f"{loss_name}_t"
            #     val_dict.setdefault(key, 0)
            #     val_dict[key] += loss_value.item()

            val_dict["total_loss"] += total_batch_loss.item()

    # Compute average losses
    for key in val_dict.keys():
        val_dict[key] /= len(valloader_bg)

    cs_model.train()  # Set model back to train mode
    return val_dict


def main():
    """Main function to initialize and run training."""
    parser = argparse.ArgumentParser()
    #parser.add_argument('--diffAE_path', type=str, default='model_path')
    parser.add_argument('--cs_model_ckpt', type=str, default=None)
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_main', type=float, default=1e-3)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--T_noise', type=int, default=10)

    parser.add_argument('--w_bg', type=float, default=1000.0)
    parser.add_argument('--w_t', type=float, default=1000.0)
    parser.add_argument('--w_sbg', type=float, default=1000.0)
    parser.add_argument('--w_id', type=float, default=0.2)
    parser.add_argument('--w_pix', type=float, default=1.0)
    parser.add_argument('--w_lpips', type=float, default=0.8)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--train_interval', type=int, default=1000)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--image_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=500)

    args = parser.parse_args()
    random.seed(args.seed)  # Ensure reproducibility
    print("Training arguments:", vars(args))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    shutil.rmtree(args.results_dir, ignore_errors=True)
    os.makedirs(args.results_dir, exist_ok=True)
    save_hyparams(args)

    diff_model, conf = load_diffusion_model(device)
    cs_model, optimizer = load_cs_model(device, args, is_train=True)

    train_path_bg, train_path_t, val_path_bg, val_path_t = config_dataset_path(dataset_type="ffhq")
    trainloader_bg, trainloader_t = load_lmdb_dataset(train_path_bg, train_path_t, shuffle=True)
    valloader_bg, valloader_t = load_lmdb_dataset(val_path_bg, val_path_t, shuffle=False)
    
    test_batch_bg, test_batch_t = next(iter(valloader_bg)), next(iter(valloader_t))
    test_image_fixed = get_fixed_for_test(test_batch_bg, test_batch_t, diff_model, device, T_noise=100, T_render=100, idx=1)


    train(cs_model, diff_model, optimizer, 
          trainloader_bg, trainloader_t, valloader_bg, valloader_t,
          test_image_fixed, args, device)


if __name__ == "__main__":
    main()
