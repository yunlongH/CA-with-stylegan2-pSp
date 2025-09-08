import torch
import torch.nn.functional as F
from criteria.lpips.lpips import LPIPS
from criteria import id_loss

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# id_loss_fn = id_loss.IDLoss().to(device).eval()
# lpips_loss_fn = LPIPS(net_type='alex').to(device).eval()

# criterion_disc = torch.nn.BCEWithLogitsLoss()

def calc_latent_recon_loss(c_bg, s_bg, c_t, s_t, diffAE_zbg, diffAE_zt, args):
    """Compute different loss components for training."""
    loss_list = {
        "loss_recon_bg": args.w_bg * F.mse_loss(c_bg, diffAE_zbg),
        "loss_recon_t": args.w_t * F.mse_loss(c_t + s_t, diffAE_zt),
        "loss_recon_sbg": args.w_sbg * F.mse_loss(s_bg, torch.zeros_like(s_bg)),
        "loss_mmd": args.w_mmd * mmd_loss(c_t, diffAE_zbg, kernel='rbf', sigma=1.0)
    }

    return loss_list

def mmd_loss(x, y, kernel='rbf', sigma=1.0):
    """Compute MMD between two sets of samples using the specified kernel"""
    
    def gaussian_kernel(a, b, sigma):
        a_sq = a.pow(2).sum(dim=1, keepdim=True)
        b_sq = b.pow(2).sum(dim=1, keepdim=True)
        ab = a @ b.t()
        return torch.exp(- (a_sq - 2 * ab + b_sq.t()) / (2 * sigma ** 2))

    def linear_kernel(a, b):
        return a @ b.t()

    if kernel == 'rbf':
        Kxx = gaussian_kernel(x, x, sigma)
        Kyy = gaussian_kernel(y, y, sigma)
        Kxy = gaussian_kernel(x, y, sigma)
    elif kernel == 'linear':
        Kxx = linear_kernel(x, x)
        Kyy = linear_kernel(y, y)
        Kxy = linear_kernel(x, y)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def train_disc_step(latent_real, latent_fake, model, optimizer, criterion):
    model.train()

    # Labels explicitly defined
    labels_real = torch.ones(latent_real.size(0), device=latent_real.device)
    labels_fake = torch.zeros(latent_fake.size(0), device=latent_fake.device)
    
    # Detach explicitly the fake latent vectors
    latent_fake = latent_fake.detach()

    # Forward pass explicitly
    logits_real = model(latent_real)
    logits_fake = model(latent_fake)
    
    # Compute losses explicitly
    loss_real = criterion(logits_real, labels_real)
    loss_fake = criterion(logits_fake, labels_fake)
    
    loss = 0.5 * (loss_fake + loss_real)

    # Backpropagation explicitly
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return explicitly loss values for logging clearly
    loss_D_list = {
        "loss_D_fake": loss_fake,
        "loss_D_real": loss_real,
        "loss_D_total": loss
    }

    return loss_D_list

def calc_adv_loss(latent_fake, model, criterion):
    model.eval()

    labels_real = torch.ones(latent_fake.size(0), device=latent_fake.device)  # explicitly label fake latents as "real"
    logits_fake = model(latent_fake)
    loss_adv = criterion(logits_fake, labels_real)  # explicitly fool discriminator  

    return loss_adv


# def calc_cls_loss(latent_x, latent_y, disc_model):
#     # Prepare labels: latent_x -> class 0, latent_y -> class 1
#     labels_x = torch.zeros(latent_x.size(0), device=latent_x.device)
#     labels_y = torch.ones(latent_y.size(0), device=latent_y.device)

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


# def calc_adv_loss(latent_x, latent_y, disc_model, args):

#     logits_x = disc_model(latent_x)  # Raw outputs (logits) for background latents
#     logits_y = disc_model(latent_y)  # Raw outputs (logits) for target latents

#     loss = BCElogit(logits_y, torch.zeros_like(logits_y))  # latent_y -> class 0 vs. Disc latent_y -> class 1
    

#     return loss  * args.w_cls


# def calc_recon_images(diff_model, imgs_bg, imgs_t, latents_bg, latents_t, T_noise, T_render):

#     xT_bg = diff_model.encode_stochastic(imgs_bg, latents_bg, T=T_noise)
#     xT_t = diff_model.encode_stochastic(imgs_t, latents_t, T=T_noise)
#     recon_bg = diff_model.render(xT_bg, latents_bg, T=T_render)
#     recon_t = diff_model.render(xT_t, latents_t, T=T_render)  

#     recon_bg = (recon_bg * 2) - 1
#     recon_t = (recon_t * 2) - 1

#     return recon_bg, recon_t


# def calc_image_loss(real, pred, args):
#     """Compute image-space loss components for training."""
#     loss_list = {}
#     id_logs = None  # for logging identity improvements if applicable

#     # ID loss
#     if args.w_id > 0:
#         loss_id, sim_improvement, id_logs = id_loss_fn(pred, real, real)
#         loss_list["loss_id"] = args.w_id * loss_id
#         #loss_list["id_improve"] = sim_improvement  # for logging only

#     # Pixel (L2) loss
#     if args.w_pix > 0:
#         loss_pix = F.mse_loss(real, pred)
#         loss_list["loss_pix"] = args.w_pix * loss_pix

#     # LPIPS loss
#     if args.w_lpips > 0:
#         loss_lpips = lpips_loss_fn(pred, real)
#         loss_list["loss_lpips"] = args.w_lpips * loss_lpips

#     return loss_list, id_logs

