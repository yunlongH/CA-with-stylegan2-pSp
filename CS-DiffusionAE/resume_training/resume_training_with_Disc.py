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
from custom_funcs.model_funcs import load_diffusion_model, load_cs_model, load_disc_model
from custom_funcs.eval_funcs import get_fixed_for_test
from custom_funcs.utils import save_hyparams 
from custom_funcs.train_funcs import train_main, train_disc

def train_alternating(cs_model, diff_model, disc_model, 
                      optimizer_main, optimizer_disc, 
                      train_dataloader, val_dataloader, test_image_fixed, 
                      device, args):
    
    print("Starting alternating training...")

    for phase in range(args.alt_max_phases):
        print(f"\n🔁 Alternating Phase {phase + 1}/{args.alt_max_phases}")

        # === Train the Discriminator ===
        print(f"\n[Phase {phase + 1}] Training Discriminator for {args.disc_max_epochs} epochs")
        train_disc(
            disc_model=disc_model,
            cs_model=cs_model,
            optimizer_disc=optimizer_disc,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=args.disc_max_epochs,
            phase=phase,
            device=device,
            args=args
        )

        # === Train the Main Model ===
        print(f"\n[Phase {phase + 1}] Training Main Model for {args.main_max_epochs} epochs")
        train_main(
            cs_model=cs_model,
            diff_model=diff_model,
            disc_model=disc_model,
            optimizer=optimizer_main,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_image_fixed=test_image_fixed,
            max_epochs=args.main_max_epochs,
            phase=phase,
            device=device,
            args=args
        )

    print("✅ Alternating training completed.")




def main():
    """Main function to initialize and run training."""
    parser = argparse.ArgumentParser()

    # === Checkpoints ===
    #parser.add_argument('--diffAE_path', type=str, default='model_path')
    parser.add_argument('--cs_model_ckpt', type=str, default=None, help='Path to pre-trained cs_model checkpoint')
    parser.add_argument('--disc_model_ckpt', type=str, default=None, help='Path to pre-trained discriminator checkpoint')
    
    # === General Training Arguments ===
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--latent_type', type=str, default="Cx_Cy", help="Latent type to use for discriminator input")
    parser.add_argument('--alt_max_phases', type=int, default=10, help='Number of alternating training phases')
    parser.add_argument('--disc_max_epochs', type=int, default=10, help='Epochs to train discriminator per phase')
    parser.add_argument('--main_max_epochs', type=int, default=5, help='Epochs to train main model per phase')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42) 

    # === cs_model Arguments ===
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--train_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--image_interval', type=int, default=1)

    # === Discriminator Arguments ===
    parser.add_argument('--disc_input_dim', type=int, default=512)
    parser.add_argument('--disc_n_layers', type=int, default=1)
    parser.add_argument('--disc_train_interval', type=int, default=1)
    parser.add_argument('--disc_val_interval', type=int, default=1)

    # === Loss Weights ===
    parser.add_argument('--w_bg', type=float, default=1.0, help='Weight for background latent loss')
    parser.add_argument('--w_t', type=float, default=1.0, help='Weight for target latent loss')
    parser.add_argument('--w_sbg', type=float, default=1.0, help='Weight for style disentanglement loss')
    parser.add_argument('--w_adv', type=float, default=1.0, help="Weight for adversarial loss")

    # === Optimizer / Learning Rate ===
    parser.add_argument('--lr_main', type=float, default=1e-4, help='Learning rate for cs_model')
    parser.add_argument('--lr_disc', type=float, default=1e-4, help='Learning rate for discriminator')



    args = parser.parse_args()
    random.seed(args.seed)  # Ensure reproducibility
    print("Training arguments:", vars(args))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    shutil.rmtree(args.results_dir, ignore_errors=True)
    os.makedirs(args.results_dir, exist_ok=True)
    save_hyparams(args)


    cs_model, optimizer_main = load_cs_model(device, args, is_train=True)
    diff_model = load_diffusion_model(device)
    disc_model, optimizer_disc = load_disc_model(device, args, is_train=True)

    train_dataloader, val_dataloader = get_dataloaders(args)

    train_dataloader, val_dataloader = get_dataloaders(args)
    real_image_dir = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs"
    test_batch = next(iter(val_dataloader))
    test_image_fixed = get_fixed_for_test(test_batch, real_image_dir, diff_model, device, T_noise=250, idx=1)

    # Create results directories
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.results_dir}/images", exist_ok=True)

    train_alternating(cs_model, diff_model, disc_model, 
                        optimizer_main, optimizer_disc, 
                        train_dataloader, val_dataloader, test_image_fixed, 
                        device, args)


if __name__ == "__main__":
    main()
