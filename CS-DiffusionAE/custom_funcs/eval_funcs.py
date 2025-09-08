import os
import torch
import matplotlib.pyplot as plt
from dataset import SingleImageDataset

def get_fixed_for_test(batch_bg, batch_t, diff_model, device, T_noise=100, T_render=100, idx=1):
    """Precompute and store tensors for fast visualization during training, selecting one sample by idx."""
    with torch.no_grad():
        imgs_bg = batch_bg['img'][idx].unsqueeze(0).to(device)        # [1, C, H, W]
        imgs_t = batch_t['img'][idx].unsqueeze(0).to(device)

        latents_bg = batch_bg['latent'][idx].unsqueeze(0).to(device)  # [1, D]
        latents_t = batch_t['latent'][idx].unsqueeze(0).to(device)

        xT_bg = diff_model.encode_stochastic(imgs_bg, latents_bg, T=T_noise)
        xT_t = diff_model.encode_stochastic(imgs_t, latents_t, T=T_noise)

        pred_diffAE_bg = diff_model.render(xT_bg, latents_bg, T=T_render)
        pred_diffAE_t = diff_model.render(xT_t, latents_t, T=T_render)

        test_fixed = {
            "ori_bg": imgs_bg,
            "ori_t": imgs_t,
            "diffAE_zbg": latents_bg,
            "diffAE_zt": latents_t,
            "xT_bg": xT_bg,
            "xT_t": xT_t,
            "pred_diffAE_bg": pred_diffAE_bg,
            "pred_diffAE_t": pred_diffAE_t
        }

    return test_fixed


def eval_recons(cs_model, diffAE_model, image_fixed, T_render=200, save_dir=None, is_train=False):
    if is_train:
        cs_model.eval()
    
    original_bg = (image_fixed["ori_bg"] + 1)/2
    original_t = (image_fixed["ori_t"] + 1)/2
    
    diffAE_bg = image_fixed["pred_diffAE_bg"]
    diffAE_t = image_fixed["pred_diffAE_t"]


    with torch.no_grad():
        c_bg, s_bg = cs_model(image_fixed["diffAE_zbg"])
        c_t, s_t = cs_model(image_fixed["diffAE_zt"])

        our_latents_bg = diffAE_model.render(image_fixed["xT_bg"], c_bg, T=T_render)
        our_latents_t = diffAE_model.render(image_fixed["xT_t"], (c_t + s_t), T=T_render)
        our_latents_swap_bg = diffAE_model.render(image_fixed["xT_bg"], (c_bg + s_t), T=T_render)
        our_latents_swap_t = diffAE_model.render(image_fixed["xT_t"], c_t, T=T_render)


    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    ax[0, 0].imshow(original_bg[0].cpu().permute(1, 2, 0).numpy())
    ax[0, 0].set_title("Original_bg")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(diffAE_bg[0].cpu().permute(1, 2, 0).numpy())
    ax[0, 1].set_title("diffAE_bg")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(our_latents_bg[0].cpu().permute(1, 2, 0).numpy())
    ax[0, 2].set_title("Our_recon_bg")
    ax[0, 2].axis("off")

    ax[0, 3].imshow(our_latents_swap_bg[0].cpu().permute(1, 2, 0).numpy())
    ax[0, 3].set_title("Our_swap_bg")
    ax[0, 3].axis("off")

    ax[1, 0].imshow(original_t[0].cpu().permute(1, 2, 0).numpy())
    ax[1, 0].set_title("Original_t")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(diffAE_t[0].cpu().permute(1, 2, 0).numpy())
    ax[1, 1].set_title("diffAE_t")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(our_latents_t[0].cpu().permute(1, 2, 0).numpy())
    ax[1, 2].set_title("Our_recon_t")
    ax[1, 2].axis("off")

    ax[1, 3].imshow(our_latents_swap_t[0].cpu().permute(1, 2, 0).numpy())
    ax[1, 3].set_title("Our_swap_t")
    ax[1, 3].axis("off")

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        plt.close(fig)
    elif not is_train:
        plt.show()

    if is_train:
        cs_model.train()

def run_on_batch(diff_model, images_batch, latents, T_noise=250, Trender=100):

    xT = diff_model.encode_stochastic(images_batch, latents, T=T_noise)    
    pred = diff_model.render(xT, latents, T=Trender)

    return latents, xT, pred


def show_diffAE(real, xT, pred, idx=0):

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ori = (real + 1) / 2
    ax[0].imshow(ori[idx].permute(1, 2, 0).cpu())
    ax[0].set_title("Input")
    ax[1].imshow(xT[idx].permute(1, 2, 0).cpu())
    ax[1].set_title("xT")
    ax[2].imshow(pred[idx].permute(1, 2, 0).cpu())
    ax[2].set_title("reconstruction")
