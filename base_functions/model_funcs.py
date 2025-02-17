import torch
from argparse import Namespace
from models.psp import pSp
from models.mlp3D import MappingNetwork_cs_independent
import h5py
from tqdm import tqdm


def load_pSp_cmlp_models(model_path, device, eval_models=True):
    # load model ckpts
    #print('Loading trained checkpoint from path: {}'.format(model_path))
    model_ckpt = torch.load(model_path, map_location='cpu', weights_only=True)  
    opts = model_ckpt['opts']
    opts = Namespace(**opts)

    #### load pSp model including StyleGAN2 generator ####
    pSp_net = pSp(opts, previous_train_ckpt=None).to(device)

    cs_mlp_net = MappingNetwork_cs_independent(opts).to(device)
    print(f"Loading csmlp from path: {model_path}")   
    cs_ckpt  = model_ckpt['state_dict_cs_enc']  
    cs_mlp_net.load_state_dict(cs_ckpt)

    if eval_models:
        pSp_net.eval()
        cs_mlp_net.eval()

    return pSp_net, cs_mlp_net, opts

def process_recon_swap(cs_mlp_net, pSp_net, input_images_bg, input_images_t, opts):

    with torch.no_grad():

        w_pSp_bg = pSp_net.forward(input_images_bg, encode_only=True)  
        w_pSp_t = pSp_net.forward(input_images_t, encode_only=True)  

        latent_bg_c, latent_bg_s = cs_mlp_net(w_pSp_bg, zero_out_silent=opts.zero_out_silent_bg)
        latent_t_c, latent_t_s = cs_mlp_net(w_pSp_t, zero_out_silent=opts.zero_out_silent_t) 
      
        rec_bg_pSp = pSp_net.forward(w_pSp_bg, input_code=True, randomize_noise=False, recon_modle=True)
        rec_t_pSp = pSp_net.forward(w_pSp_t, input_code=True, randomize_noise=False, recon_modle=True)
      
        recon_bg = pSp_net.forward(latent_bg_c, input_code=True, randomize_noise=False, recon_modle=True)
        recon_t = pSp_net.forward(latent_t_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
    
        swap_bg = pSp_net.forward(latent_bg_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
        swap_t = pSp_net.forward(latent_t_c , input_code=True, randomize_noise=False, recon_modle=True)  

        output_latents = {
                    'w_bg_pSp': w_pSp_bg,
                    'w_t_pSp': w_pSp_t,
                    'latent_bg_c':latent_bg_c,
                    'latent_bg_s':latent_bg_s,
                    'latent_t_c':latent_t_c,
                    'latent_t_s':latent_t_s,
                   }
        
        output_images = {
                    'recon_pSp_bg':rec_bg_pSp,
                    'recon_pSp_t':rec_t_pSp,
                    'recon_bg':recon_bg,
                    'recon_t':recon_t,
                    'swap_bg':swap_bg,
                    'swap_t':swap_t
                   }
        return output_latents, output_images 







