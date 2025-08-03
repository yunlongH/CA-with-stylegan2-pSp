from datasets.datasets import ImageDataset
from datasets.loaders import InfiniteLoader
from datasets.transforms import transforms_registry
from runners.simple_runner import SimpleRunner
import torch.nn.functional as F
import torch

def configure_eval_datasets(images_bg_path,  images_t_path, batch_size=4, shuffle=False):
    print("Loading dataset ... ")
    transform_dict = transforms_registry["face_1024"]().get_transforms()

    test_dataset_X = ImageDataset(images_bg_path, transform_dict["test"])
    test_dataset_Y = ImageDataset(images_t_path, transform_dict["test"])

    testloader_X = InfiniteLoader(
        test_dataset_X,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True,
        is_infinite=False
    )
    testloader_Y = InfiniteLoader(
        test_dataset_Y,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True,
        is_infinite=False
    )

    print(f"Number of test samples: {len(test_dataset_Y)*2}")		
    return testloader_X, testloader_Y

def load_eval_models(editor_ckpt_pth="pretrained_models/sfe_editor_light.pt", 
                simple_config_pth = "configs/simple_inference_e4e.yaml", 
                w_space_encoder="e4e"):

    runner = SimpleRunner(
        editor_ckpt_pth=editor_ckpt_pth,
        simple_config_pth = simple_config_pth,
        w_space_encoder=w_space_encoder
    )

    return runner 
    # sfe_method = runner.inference_runner.method

    # device = runner.inference_runner.device



@torch.inference_mode()
def calculate_delta(sfe_method, x, x_cond, w_space_encoder="e4e"):
    # sample X_E as training input and X'_E as training target
    if w_space_encoder == "pSp":
        _, w_x = sfe_method.pSp_encoder(x, return_latents=True)
        _, w_x_cond = sfe_method.pSp_encoder(x_cond, return_latents=True)
        c_x, s_x = sfe_method.pSp_cs_model(w_x)
        c_y, s_y = sfe_method.pSp_cs_model(w_x_cond)

    elif w_space_encoder == "e4e":
        w_x = sfe_method.e4e_encoder(x) + sfe_method.latent_avg
        w_x_cond = sfe_method.e4e_encoder(x_cond) + sfe_method.latent_avg
        c_x, s_x = sfe_method.e4e_cs_model(w_x)
        c_y, s_y = sfe_method.e4e_cs_model(w_x_cond)

    else:
        raise ValueError("Unsupported w_space_encoder") 
           

    encoder_recon, fx = sfe_method.decoder(
        [c_x + s_x],
        input_is_latent=True,
        randomize_noise=False,
        return_latents=False,
        return_features=True
    )

    encoder_swap, fy = sfe_method.decoder(
        [c_x + s_y], 
        is_stylespace=False,
        input_is_latent=True,
        randomize_noise=False,
        return_features=True
    )

    delta = fx[9] - fy[9]
    
    return delta, encoder_recon, encoder_swap

    
@torch.inference_mode()
def swap_by_delta(sfe_method, x, x_cond, n_iter=1e5, w_space_encoder="e4e"):
    x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    x_resh_cond = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

    delta, encoder_recon, encoder_swap = calculate_delta(sfe_method, x_resh, x_resh_cond, w_space_encoder=w_space_encoder)

    w_recon, predicted_feat = sfe_method.inverter.fs_backbone(x_resh)
    w_recon = w_recon + sfe_method.latent_avg
            
    _, w_feats = sfe_method.decoder(
        [w_recon],
        input_is_latent=True,
        return_features=True,
        is_stylespace=False,
        randomize_noise=False,
        early_stop=64
    )

    w_feat = w_feats[9]  # bs x 512 x 64 x 64 
    
    fused_feat = sfe_method.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
    if delta is None:
        delta = torch.zeros_like(fused_feat)  # inversion case
    
    edited_feat = sfe_method.encoder(torch.cat([fused_feat, delta], dim=1))
    feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

    images, _ = sfe_method.decoder(
        [w_recon],
        input_is_latent=True,
        return_features=True,
        new_features=feats,
        feature_scale=min(1.0, 0.0001 * n_iter),
        is_stylespace=False,
        randomize_noise=False
    )

    return images, encoder_recon, encoder_swap


# @torch.inference_mode()
# def inference_special(self):

#     print("Start inference special")

#     X = self.special_batch_bg
#     Y = self.special_batch_t

#     rec_X = self.method(X)
#     rec_Y = self.method(Y)

#     swap_X2Y, pSp_rec_X, pSp_swap_X2Y = self.swap_by_delta(X, Y)
#     swap_Y2X, pSp_rec_Y, pSp_swap_Y2X = self.swap_by_delta(Y, X)

#     idx = self.config.data.special_idx

#     row1 = torch.stack([X[idx], pSp_rec_X[idx], pSp_swap_X2Y[idx], rec_X[idx], swap_X2Y[idx]], dim=0)
#     row2 = torch.stack([Y[idx], pSp_rec_Y[idx], pSp_swap_Y2X[idx], rec_Y[idx], swap_Y2X[idx]], dim=0)
#     # Assuming row1 and row2 are [5, C, H, W]
#     columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

#     visualize_batch_grid(
#         image_batches=columns,  # list of [2, C, H, W]
#         titles=["Input", "pSp-cs Recon", "pSp-cs Swap", "Refined Recon", "Refined Swap"],
#         save_path=f"{self.log_dir}/images/train_step_{self.global_step}.png"
#     )
