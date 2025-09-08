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
from custom_funcs.model_funcs import load_diffusion_model, load_cs_model, load_cls_model
from custom_funcs.eval_funcs import get_fixed_for_test, visualize_eval_recons
from custom_funcs.train_funcs import get_latents_from_batch 
from custom_funcs.utils import write_log_to_txt, save_hyparams 
from custom_funcs.loss_funcs import calc_latent_loss

def train(cls_model, cs_model, optimizer_cls, criterion_cls, train_dataloader, val_dataloader, device, args):

    print("Classifier training started...")

    best_val_loss = float('inf')
    # Create results directories
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()  # Enable AMP
    gradient_accumulation_steps = 8  # Accumulate gradients over multiple steps

    for epoch in range(args.max_epochs):

        # Training phase
        cls_model.train()
        train_loss_epoch = 0
        train_correct = 0
        train_total = 0
        optimizer_cls.zero_grad()  # Clear gradients at the start of each epoch

        for step, batch in enumerate(train_dataloader):
            diffAE_zbg = batch["diffAE_zbg"].to(device)
            diffAE_zt = batch["diffAE_zt"].to(device)

            # Mixed precision context
            with torch.cuda.amp.autocast():
                latent_x, latent_y = get_latents_from_batch(diffAE_zbg, diffAE_zt, cs_model)
                loss, acc = run_classifier_step(latent_x, latent_y, cls_model, criterion_cls, is_train=True)
                scaled_loss = loss / gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            # Gradient accumulation: Step only after `gradient_accumulation_steps` batches
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer_cls)
                scaler.update()
                optimizer_cls.zero_grad()

            train_loss_epoch += loss.item()
            train_correct += acc * (diffAE_zbg.size(0) + diffAE_zt.size(0))
            train_total += (diffAE_zbg.size(0) + diffAE_zt.size(0))

        avg_train_loss = train_loss_epoch / len(train_dataloader)
        avg_train_acc = train_correct / train_total

        if (epoch + 1) % args.train_interval == 0:
            train_log_msg = f"Epoch {epoch+1}/{args.max_epochs} | Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.4f}"
            write_log_to_txt(train_log_msg + "\n", args.results_dir, "train_loss.txt")
            print(train_log_msg)

        # Validation phase
        if (epoch + 1) % args.val_interval == 0:
            avg_val_loss, avg_val_acc = validate(cls_model, cs_model, val_dataloader, criterion_cls, device)
            
            val_log_msg = f"Epoch {epoch+1}/{args.max_epochs} | Validation Loss: {avg_val_loss:.6f} | Validation Acc: {avg_val_acc:.4f}"
            write_log_to_txt(val_log_msg + "\n", args.results_dir, "val_loss.txt")
            print(val_log_msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = f"{args.results_dir}/checkpoints/best_classifier_epoch_{epoch+1}.pth"
                torch.save(cls_model.state_dict(), checkpoint_path)
                print(f"Best checkpoint saved: {checkpoint_path}")

    print("Classifier training finished")


def validate(cls_model, cs_model, val_dataloader, criterion_cls, device):
    cls_model.eval()
    val_loss_epoch = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        # Mixed precision context for validation
        with torch.cuda.amp.autocast():
            for batch in val_dataloader:
                diffAE_zbg = batch["diffAE_zbg"].to(device)
                diffAE_zt = batch["diffAE_zt"].to(device)

                latent_x, latent_y = get_latents_from_batch(diffAE_zbg, diffAE_zt, cs_model)
                loss, acc = run_classifier_step(latent_x, latent_y, cls_model, criterion_cls, is_train=False)

                val_loss_epoch += loss.item()
                val_correct += acc * (diffAE_zbg.size(0) + diffAE_zt.size(0))
                val_total += (diffAE_zbg.size(0) + diffAE_zt.size(0))

    avg_val_loss = val_loss_epoch / len(val_dataloader)
    avg_val_acc = val_correct / val_total
    return avg_val_loss, avg_val_acc



def main():
    """Main function to initialize and run training."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--diffAE_path', type=str, default='model_path')

    # Separating Network arguments
    parser.add_argument('--cs_model_ckpt', type=str, default=None)
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=10)

    # Discriminator arguments
    parser.add_argument('--cls_input_dim', type=int, default=512)
    parser.add_argument('--cls_n_layers', type=int, default=1)
    parser.add_argument('--cls_model_ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_cls', type=float, default=1e-3)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--train_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--classifier_type', type=str, default = "cxy")

    # Create results directories
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)

    args = parser.parse_args()
    random.seed(args.seed)  # Ensure reproducibility
    print("Training arguments:", vars(args))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    shutil.rmtree(args.results_dir, ignore_errors=True)
    os.makedirs(args.results_dir, exist_ok=True)
    save_hyparams(args)


    cs_model = load_cs_model(device, args, is_train=False)

    # Initialize the classifier model and optimizer
    cls_model, optimizer_cls = load_cls_model(device, args, is_train=True)

    train_dataloader, val_dataloader = get_dataloaders(args)

    criterion_cls = torch.nn.BCEWithLogitsLoss()

    # # Load pretrained models
    # real_image_dir = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs"
    # test_batch = next(iter(val_dataloader))
    # diff_model = load_diffusion_model(device)
    # test_image_fixed = get_fixed_for_test(test_batch, real_image_dir, diff_model, device, T_noise=250, idx=1)
    # save_path = f"{args.results_dir}/images/recon.png"
    # os.makedirs(f"{args.results_dir}/images", exist_ok=True)
    # visualize_eval_recons(cs_model, diff_model, test_image_fixed, T_render=50, save_dir=save_path, is_train=False)
    
    train(cls_model, cs_model, optimizer_cls, criterion_cls, train_dataloader, val_dataloader, device, args)


if __name__ == "__main__":
    main()
