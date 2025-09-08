import torch
from custom_funcs.loss_funcs import disc_step_loss, calc_latent_loss, calc_adv_loss
from custom_funcs.utils import write_log_to_txt
from custom_funcs.eval_funcs import visualize_eval_recons
from custom_funcs.loss_funcs import calc_cls_loss


citration_disc = torch.nn.BCEWithLogitsLoss()

def select_latentxy(zbg, zt, c_bg, s_bg, c_t, s_t, latent_type="Cx_Cy"):
    
    if latent_type == "Cx_Cy":
        # c_bg vs. s_bg
        latent_x = c_bg
        latent_y = c_t

    elif latent_type == "Zx_Cy":
        # zbg vs. c_t
        latent_x = zbg
        latent_y = c_t

    elif latent_type == "Zy_CxSy":
        # (c_bg + s_t) vs. zt
        latent_x = zt
        latent_y = c_bg + s_t

    elif latent_type == "CxSx_CyCx":
        # (c_bg + s_bg) vs. (c_t + s_bg)
        latent_x = c_bg + s_bg   
        latent_y = c_t + s_bg   

    elif latent_type == "CxSy_CyCy":
        # (c_bg + s_t) vs. (c_t + s_t)
        latent_x = c_bg + s_t 
        latent_y = c_t + s_t # = zt

    elif latent_type == "Zx_Zy":
        latent_x = zbg
        latent_y = zt
        
    else:
        raise ValueError("Invalid classifier type. Choose from: 'cxy', 'cxy_sx', 'cxy_sy', 'diff_xy'")
    
    return latent_x, latent_y


# def train_classifier_step(latent_x, latent_y, cls_model, optimizer_cls):
#     """
#     Run a forward pass, compute loss/accuracy for a binary classifier,
#     and explicitly update classifier weights.
#     """
#     cls_model.train()

#     loss, accuracy = calc_cls_loss(latent_x, latent_y, cls_model)

#     # Backpropagation and classifier update
#     optimizer_cls.zero_grad()
#     loss.backward()
#     optimizer_cls.step()


#     return loss.item(), accuracy



# def train_disc_step(latent_real, latent_fake, disc_model, disc_optimizer, criterion_disc):
#     disc_model.train()

#     # Labels explicitly defined
#     labels_real = torch.ones(latent_real.size(0), device=latent_real.device)
#     labels_fake = torch.zeros(latent_fake.size(0), device=latent_fake.device)
    
#     # Detach explicitly the fake latent vectors
#     latent_fake = latent_fake.detach()

#     # Forward pass explicitly
#     logits_real = disc_model(latent_real)
#     logits_fake = disc_model(latent_fake)
    
#     # Compute losses explicitly
#     loss_real = criterion_disc(logits_real, labels_real)
#     loss_fake = criterion_disc(logits_fake, labels_fake)
    
#     loss = 0.5 * (loss_fake + loss_real)

#     # Backpropagation explicitly
#     disc_optimizer.zero_grad()
#     loss.backward()
#     disc_optimizer.step()

#     # Return explicitly loss values for logging clearly
#     loss_D_list = {
#         "loss_D_fake": loss_fake.item(),
#         "loss_D_real": loss_real.item(),
#         "loss_D_total": loss.item()
#     }

#     return loss_D_list



# def train_adv_step(latent_real, latent_fake, disc_model, cs_model):

#     disc_model.eval()
#     # Prepare labels: latent_x -> class 0, latent_y -> class 1
#     labels_x = torch.zeros(latent_fake.size(0), device=latent_fake.device)
#     labels_y = torch.ones(latent_real.size(0), device=latent_real.device)

#     # Combine data
#     combined_latents = torch.cat([latent_x, latent_y], dim=0)
#     combined_labels = torch.cat([labels_x, labels_y], dim=0)

#     # Shuffle combined data
#     perm = torch.randperm(combined_latents.size(0))
#     combined_latents = combined_latents[perm]
#     combined_labels = combined_labels[perm]

#     # Forward pass
#     logits = cls_model(combined_latents)
#     loss = citration_cls(logits, combined_labels)

#     # Calculate accuracy
#     predicted = (logits > 0).float()
#     correct = (predicted == combined_labels).sum().item()
#     accuracy = correct / combined_labels.size(0)

#     return loss, accuracy

def train_disc(disc_model, cs_model, optimizer_disc, train_dataloader, val_dataloader, max_epochs, phase, device, args):
    
    disc_model.train()
    cs_model.eval()  # don't train cs_model here

    print("Discriminator training started...")

    best_val_loss = float('inf')

    scaler = torch.cuda.amp.GradScaler()  # Enable AMP
    gradient_accumulation_steps = 8  # Accumulate gradients over multiple steps

    for epoch in range(max_epochs):

        # Training phase
        train_loss_epoch, train_correct, train_total = 0, 0, 0
        optimizer_disc.zero_grad()  # Clear gradients at the start of each epoch

        for step, batch in enumerate(train_dataloader):
            zbg = batch["diffAE_zbg"].to(device)
            zt = batch["diffAE_zt"].to(device)
            c_bg, s_bg = cs_model(zbg)
            c_t, s_t = cs_model(zt)

            with torch.cuda.amp.autocast():
                latent_x, latent_y = select_latentxy(zbg, zt, c_bg, s_bg, c_t, s_t, latent_type=args.latent_type)
                loss, acc = disc_step_loss(latent_x, latent_y, disc_model)
                scaled_loss = loss / gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer_disc)
                scaler.update()
                optimizer_disc.zero_grad()

            train_loss_epoch += loss.item()
            
            batch_size = zbg.size(0) + zt.size(0)
            train_correct += acc * batch_size
            train_total += batch_size

        avg_train_loss = train_loss_epoch / len(train_dataloader)
        avg_train_acc = train_correct / train_total

        if (epoch + 1) % args.disc_train_interval == 0:
            train_log_msg = f"Phase {phase+1} | Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.4f}"
            write_log_to_txt(train_log_msg + "\n", args.results_dir, "disc_train_loss.txt")
            print(train_log_msg)

        # Validation phase
        if (epoch + 1) % args.disc_val_interval == 0:
            avg_val_loss, avg_val_acc = validate_disc(disc_model, cs_model, val_dataloader, device, args)
            
            val_log_msg = f"Phase {phase+1} | Epoch {epoch+1}/{max_epochs} | Validation Loss: {avg_val_loss:.6f} | Validation Acc: {avg_val_acc:.4f}"
            write_log_to_txt(val_log_msg + "\n", args.results_dir, "disc_val_loss.txt")
            print(val_log_msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch,
                    "disc_model": disc_model.state_dict(),
                    #"optimizer_disc": optimizer_disc.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args)
                }, f"{args.results_dir}/checkpoints/disc_best.pth")

                print(f"New best checkpoint saved at epoch {epoch+1}")

    print("Discriminator training finished")


def validate_disc(disc_model, cs_model, val_dataloader, device, args):
    disc_model.eval()

    val_loss_epoch, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for batch in val_dataloader:
            zbg = batch["diffAE_zbg"].to(device)
            zt = batch["diffAE_zt"].to(device)

            with torch.cuda.amp.autocast():
                c_bg, s_bg = cs_model(zbg)
                c_t, s_t = cs_model(zt)

                latent_x, latent_y = select_latentxy(
                    zbg, zt, c_bg, s_bg, c_t, s_t, latent_type=args.latent_type
                )
                loss, acc = disc_step_loss(latent_x, latent_y, disc_model)

            batch_size = zbg.size(0) + zt.size(0)
            val_loss_epoch += loss.item()
            val_correct += acc * batch_size
            val_total += batch_size

    avg_val_loss = val_loss_epoch / len(val_dataloader)
    avg_val_acc = val_correct / val_total

    disc_model.train()  # Optional, reset to training mode

    return avg_val_loss, avg_val_acc

def train_main(cs_model, diff_model, disc_model, optimizer, train_dataloader, val_dataloader, test_image_fixed, max_epochs, phase, device, args):

    print("training start.....")
    cs_model.train()
    disc_model.eval()

    for epoch in range(max_epochs):  # Iterate over epochs
        train_dict = {"total_loss": 0}
        
        for batch in train_dataloader:
            zbg = batch["diffAE_zbg"].to(device, non_blocking=True)
            zt = batch["diffAE_zt"].to(device, non_blocking=True)

            # Forward pass
            c_bg, s_bg = cs_model(zbg)
            c_t, s_t = cs_model(zt)

            # Compute losses
            loss_list = calc_latent_loss(zbg, zt, c_bg, s_bg, c_t, s_t, args)
            latent_x, latent_y = select_latentxy(zbg, zt, c_bg, s_bg, c_t, s_t, latent_type=args.latent_type)
            adv_loss = calc_adv_loss(latent_x, latent_y, disc_model, args)

            total_batch_loss = sum(loss_list.values()) + adv_loss

            # Backpropagation
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            # Accumulate losses
            for loss_name, loss_value in loss_list.items():
                train_dict.setdefault(loss_name, 0)
                train_dict[loss_name] += loss_value.item()

            train_dict.setdefault("loss_adv", 0)
            train_dict["loss_adv"] += adv_loss.item()
            train_dict["total_loss"] += total_batch_loss.item()

        # Compute average loss per epoch
        for key in train_dict.keys():
            train_dict[key] /= len(train_dataloader)

        # Save **train loss** independently if it's the train interval
        if (epoch + 1) % args.train_interval == 0:
            train_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in train_dict.items()])
            write_log_to_txt(f"Phase {phase+1} | Epoch {epoch+1}/{max_epochs} | Train Loss {train_loss_msg}\n", args.results_dir, "train_loss.txt")

        # Run validation and log if it's time
        if (epoch + 1) % args.val_interval == 0:
            val_dict = validate_main(cs_model, disc_model, val_dataloader, args, device)

            # Save **validation loss** independently if it's the val interval
            val_loss_msg = " | ".join([f"{key}: {value:.6f}" for key, value in val_dict.items()])
            write_log_to_txt(f"Phase {phase+1} | Epoch {epoch+1}/{max_epochs} | Validation Loss {val_loss_msg}\n", args.results_dir, "val_loss.txt")

        if (epoch + 1) % args.image_interval == 0:
            # print("showing start.....")
            save_path = f"{args.results_dir}/images/recon_phase_{phase+1}_epoch_{epoch + 1}.png"
            #show_image_results(cs_model, diffAE_model, test_fixed, save_path)
            visualize_eval_recons(cs_model, diff_model, test_image_fixed, T_render=10, save_dir=save_path, is_train=True)
            
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.results_dir}/checkpoints/model_phase_{phase+1}_epoch_{epoch+1}.pth"
            torch.save(cs_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("training finish")

def validate_main(cs_model, disc_model, val_dataloader, args, device):
    """Evaluate the model on the validation dataset."""
    cs_model.eval()

    val_dict = {"total_loss": 0}
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in val_dataloader:
            zbg = batch["diffAE_zbg"].to(device, non_blocking=True)
            zt = batch["diffAE_zt"].to(device, non_blocking=True)

            # Forward pass
            c_bg, s_bg = cs_model(zbg)
            c_t, s_t = cs_model(zt)

            # Compute losses
            loss_list = calc_latent_loss(zbg, zt, c_bg, s_bg, c_t, s_t, args)
            latent_x, latent_y = select_latentxy(zbg, zt, c_bg, s_bg, c_t, s_t, latent_type=args.latent_type)
            adv_loss = calc_adv_loss(latent_x, latent_y, disc_model, args)

            total_batch_loss = sum(loss_list.values()) + adv_loss

            # Accumulate validation losses
            for loss_name, loss_value in loss_list.items():
                val_dict.setdefault(loss_name, 0)
                val_dict[loss_name] += loss_value.item()

            val_dict.setdefault("loss_adv", 0)
            val_dict["loss_adv"] += adv_loss.item()
            val_dict["total_loss"] += total_batch_loss.item()

    # Compute average validation loss
    for key in val_dict.keys():
        val_dict[key] /= num_batches

    cs_model.train()  # Switch back to training mode
    return val_dict

# def calc_images_batch(diff_model, imgs_bg, imgs_t):
#     xT_bg = diff_model.encode_stochastic(imgs_bg, diffAE_zbg, T=T_noise)
#     xT_t = diff_model.encode_stochastic(imgs_t, diffAE_zt, T=T_noise)

#     pred_diffAE_bg = diff_model.render(xT_bg, latents_bg, T=T_render)
#     pred_diffAE_t = diff_model.render(xT_t, latents_t, T=T_render)    