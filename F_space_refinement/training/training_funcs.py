import torch.nn.functional as F
import torch

from models.cs_mlp.mlp3D import MappingNetwork_cs_3Dmlp
from models.cs_mlp.mlp2D import MappingNetwork_cs_2Dmlp
from argparse import Namespace

def load_e4e_cs_model(model_path, device='cuda'):
    
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts = Namespace(**opts)   

    cs_mlp_net = MappingNetwork_cs_3Dmlp(opts).to(device)

    print(f"loading e4e-cs model from path {model_path}")
    if 'state_dict_cs_net' in ckpt:
        cs_mlp_net.load_state_dict(ckpt['state_dict_cs_net'])
    elif 'state_dict_cs_e4e' in ckpt:
        cs_mlp_net.load_state_dict(ckpt['state_dict_cs_e4e'])
    else:
        raise KeyError("Checkpoint does not contain 'state_dict_cs_net' or 'state_dict_cs_e4e'.")
    cs_mlp_net.eval()

    return cs_mlp_net

# def encode_to_CS_latents(sfe_model, cs_mlp_net, images_X, images_Y):

#     w_sfe_x, fused_feat_x, w_e4e_x = encode_to_sfe_latents(sfe_model, images_X)
#     w_sfe_y, fused_feat_y, w_e4e_y = encode_to_sfe_latents(sfe_model, images_Y)

#     latent_c_x, latent_s_x = cs_mlp_net(w_sfe_x)
#     latent_c_y, latent_s_y = cs_mlp_net(w_sfe_y) 	

#     return {
#         "latent_c_x": latent_c_x,
#         "latent_s_x": latent_s_x,
#         "latent_c_y": latent_c_y,
#         "latent_s_y": latent_s_y,
#         "fused_feat_x": fused_feat_x,
#         "fused_feat_y": fused_feat_y,
#         "w_sfe_x": w_sfe_x,
#         "w_sfe_y": w_sfe_y,
#         "w_e4e_x": w_e4e_x,
#         "w_e4e_y": w_e4e_y
#         }


# def swap_CS_from_real(sfe_model, e4e_cs_model, latents, sx_cond_type='hard'):

#     # X to Y
#     e4e_c_x, e4e_s_x = e4e_cs_model(latents['w_e4e_x'])
#     e4e_c_y, e4e_s_y = e4e_cs_model(latents['w_e4e_y']) 

#     if sx_cond_type == 'hard':
#         latent_s_x = torch.zeros_like(latents['latent_s_x'])
#         e4e_s_x = torch.zeros_like(latents['e4e_s_x'])


#     edited_X2Y, e4e_env_X, e4e_etidted_X2Y = editing_on_batch(sfe_model, edited_w_input=latents['latent_c_x'] + latents['latent_s_y'], 
#                                 fused_feat_input=latents['fused_feat_x'], 
#                                 w_e4e_input = e4e_c_x + e4e_s_x, 
#                                 edited_w_e4e_input = e4e_c_x + e4e_s_y,
#                                 editing_degrees=[10],
#                                 return_e4e=True)
#     # X to Y
#     edited_Y2X, e4e_env_Y, e4e_etidted_Y2X = editing_on_batch(sfe_model, edited_w_input=latents['latent_c_y'] + latent_s_x, 
#                                 fused_feat_input=latents['fused_feat_y'], 
#                                 w_e4e_input = e4e_c_y + e4e_s_y, 
#                                 edited_w_e4e_input = e4e_c_y + e4e_s_x,
#                                 editing_degrees=[10],
#                                 return_e4e=True)

#     return edited_X2Y.squeeze(1), edited_Y2X.squeeze(1), e4e_env_X.squeeze(1), e4e_env_Y.squeeze(1), e4e_etidted_X2Y.squeeze(1), e4e_etidted_Y2X.squeeze(1)


# def swap_CS_from_e4e(sfe_model, e4e_cs_model, latents, sx_cond_type='hard'):

#     # X to Y
#     e4e_c_x, e4e_s_x = e4e_cs_model(latents['w_e4e_x'])
#     e4e_c_y, e4e_s_y = e4e_cs_model(latents['w_e4e_y']) 

#     if sx_cond_type == 'hard':
#         latent_s_x = torch.zeros_like(latents['latent_s_x'])
#         e4e_s_x = torch.zeros_like(latents['e4e_s_x'])


#     edited_X2Y = editing_on_batch(sfe_model, edited_w_input=latents['latent_c_x'] + latents['latent_s_y'], 
#                                 fused_feat_input=latents['fused_feat_x'], 
#                                 w_e4e_input = e4e_c_x + e4e_s_x, 
#                                 edited_w_e4e_input = e4e_c_x + e4e_s_y,
#                                 editing_degrees=[10],
#                                 return_e4e=False)
#     # X to Y
#     edited_Y2X = editing_on_batch(sfe_model, edited_w_input=latents['latent_c_y'] + latent_s_x, 
#                                 fused_feat_input=latents['fused_feat_y'], 
#                                 w_e4e_input = e4e_c_y + e4e_s_y, 
#                                 edited_w_e4e_input = e4e_c_y + e4e_s_x,
#                                 editing_degrees=[10],
#                                 return_e4e=False)

#     return edited_X2Y.squeeze(1), edited_Y2X.squeeze(1)

# @torch.no_grad()
# def swap_for_eval(sfe_model, e4e_cs_model,
#                           latent_c_x, latent_s_x, latent_c_y, latent_s_y,
#                           fused_feat_x, fused_feat_y,
#                           w_e4e_x, w_e4e_y,
#                           sx_cond_type='hard'):
#     # Get CS components from w_e4e
#     e4e_c_x, e4e_s_x = e4e_cs_model(w_e4e_x)
#     e4e_c_y, e4e_s_y = e4e_cs_model(w_e4e_y)

#     if sx_cond_type == 'hard':
#         latent_s_x = torch.zeros_like(latent_s_x)
#         e4e_s_x = torch.zeros_like(e4e_s_x)

#     # Compose swapped latent representations
#     X_cs_swap = editing_on_batch(
#         sfe_model,
#         edited_w_input=latent_c_x + latent_s_y,
#         fused_feat_input=fused_feat_x,
#         w_e4e_input=e4e_c_x + e4e_s_x,
#         edited_w_e4e_input=e4e_c_x + e4e_s_y,
#         editing_degrees=[10],
#         return_e4e=False
#     )

#     Y_cs_swap = editing_on_batch(
#         sfe_model,
#         edited_w_input=latent_c_y + latent_s_x,
#         fused_feat_input=fused_feat_y,
#         w_e4e_input=e4e_c_y + e4e_s_y,
#         edited_w_e4e_input=e4e_c_y + e4e_s_x,
#         editing_degrees=[10],
#         return_e4e=False
#     )

#     return X_cs_swap.squeeze(1), Y_cs_swap.squeeze(1)



# def recon_from_latents(sfe_model, w_recon, fused_feat, n_iter=1e5):


#     delta = torch.zeros_like(fused_feat)  # For inversion case

#     edited_feat = sfe_model.method.encoder(torch.cat([fused_feat, delta], dim=1))
#     feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

#     images, _ = sfe_model.method.decoder(
#         [w_recon],
#         input_is_latent=True,
#         return_features=True,
#         new_features=feats,
#         feature_scale=min(1.0, 0.0001 * n_iter),
#         is_stylespace=False,
#         randomize_noise=False
#     )

#     return images




# def latent_editing_sota_batch(sfe_model, inputs_img, orig_latents, editing_name, editing_degrees=[10]):
#     edited_list = []
#     for i, latent in enumerate(orig_latents):
#         edited_latents = sfe_model.get_edited_latent(
#             latent.unsqueeze(0), 
#             editing_name, 
#             editing_degrees, 
#             inputs_img[i].unsqueeze(0)
#         )   ## imagine C + S

#         if edited_latents is None:
#             print(f"WARNING, skip editing {editing_name}")
#             continue

#         is_stylespace = isinstance(edited_latents, tuple)

#         if not is_stylespace:
#             edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)

#         edited_list.append(edited_latents)

#     return edited_list


# def calculate_delta_e4e(sfe_model, w_e4e, edited_w_e4e, is_stylespace, return_e4e=False):
#     e4e_inv, fs_x = sfe_model.method.decoder(
#         [w_e4e],
#         input_is_latent=True,
#         randomize_noise=False,
#         return_latents=False,
#         return_features=True,
#         early_stop=None if return_e4e else 64
#     )

#     e4e_edit, fs_y = sfe_model.method.decoder(
#         [edited_w_e4e] if not is_stylespace else edited_w_e4e,
#         input_is_latent=True,
#         randomize_noise=False,
#         return_latents=False,
#         is_stylespace=is_stylespace,
#         return_features=True,
#         early_stop=None if return_e4e else 64
#     )

#     delta = fs_x[9] - fs_y[9]  # assuming fs_x[9].shape == [B, C, H, W]

#     return delta, e4e_inv, e4e_edit



# def editing_on_batch(
#     sfe_model,
#     edited_w_input,         # Tensor: [B, 18, 512]
#     fused_feat_input,       # Tensor: [B, C, H, W]
#     w_e4e_input,            # Tensor: [B, 18, 512]
#     edited_w_e4e_input,     # Tensor: [B, 18, 512]
#     editing_degrees=[10],
#     return_e4e=False
# ):
#     edited_images = []
#     e4e_inv_images = []
#     e4e_edit_images = []
#     n_iter = 1e5
#     B = edited_w_input.shape[0]

#     for i in range(B):
#         edited_w = edited_w_input[i].unsqueeze(0)          # [1, 18, 512]
#         fused_feat = fused_feat_input[i].unsqueeze(0)      # [1, C, H, W]
#         w_e4e = w_e4e_input[i].unsqueeze(0)                # [1, 18, 512]
#         edited_w_e4e = edited_w_e4e_input[i].unsqueeze(0)  # [1, 18, 512]

#         is_stylespace = isinstance(edited_w, tuple)

#         w_e4e = w_e4e.repeat(len(editing_degrees), 1, 1)
#         delta, e4e_inv, e4e_edit = calculate_delta_e4e(
#             sfe_model, w_e4e, edited_w_e4e, is_stylespace, return_e4e=return_e4e
#         )

#         fused_feat = fused_feat.to(sfe_model.device).repeat(len(editing_degrees), 1, 1, 1)
#         edited_feat = sfe_model.method.encoder(torch.cat([fused_feat, delta], dim=1))
#         edit_features = [None] * 9 + [edited_feat] + [None] * (17 - 9)

#         image_edits, _ = sfe_model.method.decoder(
#             [edited_w],  # ← wrap in list for StyleGAN2
#             input_is_latent=True,
#             new_features=edit_features,
#             feature_scale=min(1.0, 0.0001 * n_iter),
#             is_stylespace=is_stylespace,
#             randomize_noise=False
#         )

#         edited_images.append(image_edits)
#         e4e_inv_images.append(e4e_inv)
#         e4e_edit_images.append(e4e_edit)

#     edited_images = torch.stack(edited_images)         # [B, N, 3, 1024, 1024]
#     e4e_inv_images = torch.stack(e4e_inv_images)
#     e4e_edit_images = torch.stack(e4e_edit_images)

#     if return_e4e:
#         return edited_images, e4e_inv_images, e4e_edit_images
#     return edited_images


# def encode_into_w_space(sfe_model, x, latent_avg=False):
#     x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#     w_recon, predicted_feat = sfe_model.method.inverter.fs_backbone(x)
#     if latent_avg:
#         w_recon = w_recon + sfe_model.method.latent_avg
#     return w_recon, predicted_feat

# def encode_to_fused_feat(sfe_model, w_recon, predicted_feat, latent_avg=False):
#     # x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#     # w_recon, predicted_feat = sfe_model.method.inverter.fs_backbone(x)
#     # w_recon = w_recon + sfe_model.method.latent_avg
#     if latent_avg:
#         w_recon = w_recon + sfe_model.method.latent_avg
#     _, w_feats = sfe_model.method.decoder(
#         [w_recon],
#         input_is_latent=True,
#         return_features=True,
#         is_stylespace=False,
#         randomize_noise=False,
#         early_stop=64
#     )

#     w_feat = w_feats[9]  # bs x 512 x 64 x 64
#     fused_feat = sfe_model.method.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))

#     return fused_feat


# def encode_to_e4e_latents(sfe_model, x):
#     x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#     w_e4e = sfe_model.method.e4e_encoder(x)
#     w_e4e = w_e4e + sfe_model.method.latent_avg
#     return w_e4e


# def encode_to_sfe_latents(sfe_model, x):
#     x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#     w_recon, predicted_feat = sfe_model.method.inverter.fs_backbone(x)
#     w_recon = w_recon + sfe_model.method.latent_avg

#     _, w_feats = sfe_model.method.decoder(
#         [w_recon],
#         input_is_latent=True,
#         return_features=True,
#         is_stylespace=False,
#         randomize_noise=False,
#         early_stop=64
#     )

#     w_feat = w_feats[9]  # bs x 512 x 64 x 64
#     fused_feat = sfe_model.method.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))

#     w_e4e = sfe_model.method.e4e_encoder(x)
#     w_e4e = w_e4e + sfe_model.method.latent_avg

#     return w_recon, fused_feat, w_e4e

