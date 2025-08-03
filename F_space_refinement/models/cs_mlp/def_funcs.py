
from argparse import Namespace
# import time
import sys
# import pprint
import numpy as np
from PIL import Image
import torch
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython.display import display
import os
sys.path.append(".")
sys.path.append("..")
import torch.nn as nn
# from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %load_ext autoreload
# %autoreload 2
import math
import torch.nn.functional as F
import random

def GAN_train_inversion(inputs, net):
    
    y_hat, latent = net.forward(inputs.to("cuda").float(), return_latents=True)

    return y_hat, latent

# def GAN_train_inversion(latent, net):
    
#     injected_results, result_latents = net(input_image.to("cuda").float(),
#                 inject_latent=latent_to_inject,
#                 return_latents=True)

#     return y_hat, latent

def styles_inject_func(input_image, net, latent_inject_type='normal', defined_vec_Z=None, define_latents_W=None, latent_mask=None):

    injected_results = []
        
    # get latent vector to inject into our input image, from Z to W+ space including 18 replicates of w with dim of 1 x 512 
    
    if latent_inject_type == 'zero':

        latent_to_inject = None
        vec_to_inject = None

    elif latent_inject_type == 'normal':

        if defined_vec_Z is not None:
            vec_to_inject = defined_vec_Z
        else:
            vec_to_inject = np.random.randn(1, 512).astype('float32')  # Z space
        
        if define_latents_W is not None:
            latent_to_inject = define_latents_W
        else:
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                    input_code=True,
                                    return_latents=True)

    # inject style vector to the generation of image
    injected_results, result_latents = net(input_image.to("cuda").float(),
                latent_mask=latent_mask,
                inject_latent=latent_to_inject,
                return_latents=True)
    
    return injected_results, vec_to_inject, latent_to_inject, result_latents


def experiment_subset_styles_injection(input_image, net, latent_inject_type='normal', defined_vec_Z=None, define_latents_W=None, latent_mask_list=None):
    
    with torch.no_grad():

        injected_results_list = []
        injected_Z = []
        injected_W = []
        input_latents_W = []

        for i in range(len(latent_mask_list)):

            latent_mask = latent_mask_list[i]
            
            injected_results, vec_to_inject, latent_to_inject, result_latents = styles_inject_func(input_image, net, latent_inject_type=latent_inject_type, 
                                                            defined_vec_Z=defined_vec_Z, define_latents_W=define_latents_W, latent_mask=latent_mask)
            
            injected_results_list.append(injected_results)
            injected_Z.append(vec_to_inject)
            injected_W.append(latent_to_inject)
            input_latents_W.append(result_latents)

        return injected_results_list, injected_Z, injected_W, input_latents_W


def visulize_result_images(injected_results_list, original_image=None):
    
    if original_image is None:
        # injected_results_list[i][j], where dim [i] refers to random sampling times and dim [j] refers to different images
        first_image = tensor2im(injected_results_list[0].squeeze())
        res = np.array(first_image.resize((256, 256)))

        for i in  range (len(injected_results_list)):
            
            if i > 0:
                result_image = tensor2im(injected_results_list[i].squeeze())

                res = np.concatenate([res,
                                    np.array(result_image.resize((256, 256)))], axis=1)
                res_im = Image.fromarray(res)     

    else:
        
        res = np.array(original_image.resize((256, 256)))

        for i in  range (len(injected_results_list)):
            
            result_image = tensor2im(injected_results_list[i].squeeze())

            res = np.concatenate([res,
                                np.array(result_image.resize((256, 256)))], axis=1)
            res_im = Image.fromarray(res)
            
    display(res_im)



### function to visualize the Images from folder
def vislize_folder_imgs(image_folder, show_images=True):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]
    n_images = len(image_paths)

    images = []
    n_cols = np.ceil(n_images / 2).astype(int)
    fig = plt.figure(figsize=(20, 4))
    for idx, image_path in enumerate(image_paths):
        ax = fig.add_subplot(2, n_cols, idx + 1)
        img = Image.open(image_path).convert("RGB")
        images.append(img)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)

    if show_images:
        plt.show()
    return images, n_images, n_cols

def transform_to_input_batch(images_list, transform):
    
    transformed_images = [transform(image) for image in images_list] 
    # 'transformed_images' is a transformed version of images_list, each element is convert from Image  to 
    # a torch tensor using torchvision.transforms 

    batched_images = torch.stack(transformed_images, dim=0)
    # batched_images type of image tensors, e.g., 13x3x255x255

    return batched_images

def load_folder_images(image_folder, num_images=4, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get all image paths from the folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

    # Randomly select 'num_images' from the image_paths
    selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    images = []

    for image_path in selected_paths:
        img = Image.open(image_path).convert("RGB")
        images.append(img)

    return images

def transform_images_to_batch(images_list, transforms):
    
    transformed_images = [transforms(image) for image in images_list] 
    # 'transformed_images' is a transformed version of images_list, each element is convert from Image  to 
    # a torch tensor using torchvision.transforms 

    batched_images = torch.stack(transformed_images, dim=0)
    # batched_images type of image tensors, e.g., 13x3x255x255

    return batched_images
#  
def visulize_images_list(images_list):
    res = images_list[0].resize((256, 256))

    for idx, image in enumerate(images_list):
        if idx > 0:
            res = np.concatenate([res, image.resize((256, 256))], axis=1)

    res_im = Image.fromarray(res)

    display(res_im)

def visulize_images_batch(images_batch, axis_dim= 1):

    res = tensor2im(images_batch[0, :, :, :].squeeze()).resize((256, 256))

    for idx, image in enumerate(images_batch):
        if idx > 0:
            image = tensor2im(image).resize((256, 256))

            res = np.concatenate([res, image
                                ], axis=axis_dim)


    res_im = Image.fromarray(res)
    display(res_im)



def visulize_images_paired3(images_list, result_images1, result_images2, axis_dim = 1):
    
    # visulize a series/batch of original images paired with results
    for original_image, result_image1, result_image2 in zip(images_list, result_images1, result_images2):
        
        original_image = tensor2im(original_image)
        result_image1 = tensor2im(result_image1)
        result_image2 = tensor2im(result_image2)

        res = np.concatenate([np.array(original_image.resize((256, 256))),
                            np.array(result_image1.resize((256, 256))),
                            np.array(result_image2.resize((256, 256)))
                            ], axis=axis_dim)
        
        res_im = Image.fromarray(res)

        display(res_im)

def visulize_singleImg(image1):
    
        image1 = tensor2im(image1)
        
        res = np.array(image1.resize((256, 256)))
        
        res_im = Image.fromarray(res)

        display(res_im)

def visulize_singleImg_paired2(image1, image2):
    
        image1 = tensor2im(image1)
        image2 = tensor2im(image2)
        
        res = np.concatenate([np.array(image1.resize((256, 256))),
                            np.array(image2.resize((256, 256)))
                            ], axis=1)
        
        res_im = Image.fromarray(res)

        display(res_im)

def visulize_singleImg_paired3(image1, image2, image3):
    
        image1 = tensor2im(image1)
        image2 = tensor2im(image2)
        image3 = tensor2im(image3)

        res = np.concatenate([np.array(image1.resize((256, 256))),
                            np.array(image2.resize((256, 256))),
                            np.array(image3.resize((256, 256)))
                            ], axis=1)
        
        res_im = Image.fromarray(res)

        display(res_im)

from PIL import Image
import numpy as np

def visualize_batch_images(batch_images, reference_image=None):
    """
    Visualize a batch of images, dynamically concatenating them along axis=1.
    Optionally, add a reference image as the first column.
    
    Args:
        batch_images (list of tensors): A list of image tensors to visualize. Each tensor is processed by `tensor2im`.
        reference_image (tensor, optional): A tensor representing the reference image. Defaults to None.
    
    """
    # Process the images in the batch
    processed_images = [tensor2im(img).resize((256, 256)) for img in batch_images]

    # If a reference image is provided, process and prepend it
    if reference_image is not None:
        ref_image = tensor2im(reference_image).resize((256, 256))
        processed_images.insert(0, ref_image)

    # Convert each processed image to numpy array and concatenate dynamically along axis=1
    res = np.concatenate([np.array(img) for img in processed_images], axis=1)
    
    # Convert the concatenated array back to an image
    res_im = Image.fromarray(res)
    
    # Display the final concatenated image
    display(res_im)
    

def visulize_singleImg_paired4(image1, image2, image3, image4):
    
        image1 = tensor2im(image1)
        image2 = tensor2im(image2)
        image3 = tensor2im(image3)
        image4 = tensor2im(image4)

        res = np.concatenate([np.array(image1.resize((256, 256))),
                            np.array(image2.resize((256, 256))),
                            np.array(image3.resize((256, 256))),
                            np.array(image4.resize((256, 256)))], axis=1)
        
        res_im = Image.fromarray(res)

        display(res_im)

def visulize_singleImg_paired5(image1, image2, image3, image4, image5):
    
        image1 = tensor2im(image1)
        image2 = tensor2im(image2)
        image3 = tensor2im(image3)
        image4 = tensor2im(image4)
        image5 = tensor2im(image5)
        res = np.concatenate([np.array(image1.resize((256, 256))),
                            np.array(image2.resize((256, 256))),
                            np.array(image3.resize((256, 256))),
                            np.array(image4.resize((256, 256))),
                            np.array(image5.resize((256, 256)))], axis=1)
        
        res_im = Image.fromarray(res)

        display(res_im)


def visulize_images_paired4(images_list, result_images1, result_images2,result_images3):
    
    # visulize a series/batch of original images paired with results
    for original_image, result_image1, result_image2, result_image3 in zip(images_list, result_images1, result_images2, result_images3):
        original_image = tensor2im(result_image1)
        result_image1 = tensor2im(result_image1)
        result_image2 = tensor2im(result_image2)
        result_image3 = tensor2im(result_image3)
        res = np.concatenate([np.array(original_image.resize((256, 256))),
                            np.array(result_image1.resize((256, 256))),
                            np.array(result_image2.resize((256, 256))),
                            np.array(result_image3.resize((256, 256)))], axis=1)
        
        res_im = Image.fromarray(res)

        display(res_im)

def load_pretained_pSp_model(model_path, map_location = 'cpu', output_size = 1024):
    
    ckpt = torch.load(model_path, map_location=map_location)
    opts = ckpt['opts']

    # update the training options
    opts['pSp_checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = output_size

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return net, opts, ckpt

def test_latent_same(tensor1, tensor2, number):
    # test if inputcode is equal 
    for i in range (number):

        print('dim ' + str(i), torch.equal(tensor1[i], tensor2[i]))

        # if not torch.equal(tensor1[i], tensor2[i]):
        #     print('not same')

def visulize_images_paired(original_image, result_image, axis_dim = 0):
    
    # visulize a series/batch of original images paired with results
    # for original_image, result_image in zip(images_list, result_images):
    original_image = tensor2im(original_image.squeeze())    
    result_image = tensor2im(result_image.squeeze())
    res = np.concatenate([np.array(original_image.resize((256, 256))),
                        np.array(result_image.resize((256, 256)))], axis=axis_dim)
    res_im = Image.fromarray(res)

    display(res_im)
    
def visulize_inject_paired(transformed_image, result_batch1, opts):
    input_vis_image = log_input_image(transformed_image, opts)
    output_vis_image = tensor2im(result_batch1)

    # result_batch[0].squeeze(0).shape

    vis_image = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                            np.array(output_vis_image.resize((256, 256)))], axis=1)

    res_image = Image.fromarray(vis_image)

    display(res_image)

def evaluate_model(cs_mlp_net, pSp_net, x_bg, x_t, opts):

    with torch.no_grad():

        rec_x_bg_pSp, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True)
        rec_x_t_pSp, w_t_pSp = pSp_net.forward(x_t, return_latesnts=True) 

        latent_bg_c, latent_bg_s = cs_mlp_net(w_bg_pSp, zero_out_silent=opts.zero_out_silent_bg)
        latent_t_c, latent_t_s = cs_mlp_net(w_t_pSp, zero_out_silent=opts.zero_out_silent_t) 
        
        rec_x_bg = pSp_net.forward(latent_bg_c, input_code=True, randomize_noise=True, recon_modle=True)
        rec_x_t = pSp_net.forward(latent_t_c + latent_t_s, input_code=True, randomize_noise=True, recon_modle=True)	

        swap_x_bg = pSp_net.forward(latent_bg_c + 1.1*latent_t_s, input_code=True, randomize_noise=True, recon_modle=True)
        swap_x_t = pSp_net.forward(latent_t_c, input_code=True, randomize_noise=True, recon_modle=True)

        output_latents = {
                    'w_bg_pSp': w_bg_pSp,
                    'w_t_pSp': w_t_pSp,
                    'latent_bg_c':latent_bg_c,
                    'latent_bg_s':latent_bg_s,
                    'latent_t_c':latent_t_c,
                    'latent_t_s':latent_t_s,
                   }
        
        output_images = {
                    'recon_pSp_bg':rec_x_bg_pSp,
                    'recon_pSp_t':rec_x_t_pSp,
                    'recon_bg':rec_x_bg,
                    'recon_t':rec_x_t,
                    'recon_bg_swap':swap_x_bg,
                    'recon_t_swap':swap_x_t
                   }
        return output_images, output_latents


def evaluate_combined_model(cs_mlp_net, pSp_net, x_bg, x_t, opts):

    with torch.no_grad():

        rec_x_bg_pSp, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True)
        rec_x_t_pSp, w_t_pSp = pSp_net.forward(x_t, return_latents=True) 

        latent_bg, latent_bg_c, latent_bg_s = cs_mlp_net(w_bg_pSp, zero_out_silent=opts.zero_out_silent_bg, zero_sbg=True)
        latent_t, latent_t_c, latent_t_s = cs_mlp_net(w_t_pSp, zero_out_silent=opts.zero_out_silent_t) 

        rec_x_bg = pSp_net.forward(latent_bg, input_code=True, randomize_noise=True, recon_modle=True)
        rec_x_t = pSp_net.forward(latent_t, input_code=True, randomize_noise=True, recon_modle=True)	

        swap_bg = cs_mlp_net.combined(torch.cat([latent_bg_c, latent_t_s], dim=-1))
        swap_t = cs_mlp_net.combined(torch.cat([latent_t_c, torch.zeros_like(latent_t_c).to(latent_t_c.device)], dim=-1))

        swap_x_bg = pSp_net.forward(swap_bg, input_code=True, randomize_noise=True, recon_modle=True)
        swap_x_t = pSp_net.forward(swap_t, input_code=True, randomize_noise=True, recon_modle=True)

        output_latents = {
                    'w_bg_pSp': w_bg_pSp,
                    'w_t_pSp': w_t_pSp,
                    'latent_bg_c':latent_bg_c,
                    'latent_bg_s':latent_bg_s,
                    'latent_t_c':latent_t_c,
                    'latent_t_s':latent_t_s,
                   }
        
        output_images = {
                    'recon_pSp_bg':rec_x_bg_pSp,
                    'recon_pSp_t':rec_x_t_pSp,
                    'recon_bg':rec_x_bg,
                    'recon_t':rec_x_t,
                    'recon_bg_swap':swap_x_bg,
                    'recon_t_swap':swap_x_t
                   }
        return output_images, output_latents

def shift_with_avg(pSp_net, codes):
        # normalize with respect to the center of an average face
    shifted_codes = codes + pSp_net.latent_avg.repeat(codes.shape[0], 1, 1)
    return shifted_codes

def center_with_avg(pSp_net, codes):
        # normalize with respect to the center of an average face

    shifted_codes = codes - pSp_net.latent_avg.repeat(codes.shape[0], 1, 1)
    return shifted_codes

def evaluate_scaled_st(cs_mlp_net, pSp_net, x_bg, x_t, opts):

    with torch.no_grad():

        rec_x_bg_pSp, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True) 
        rec_x_t_pSp, w_t_pSp = pSp_net.forward(x_t, return_latents=True) 

        norm_w_bg_pSp = center_with_avg(pSp_net, w_bg_pSp)
        norm_w_t_pSp = center_with_avg(pSp_net, w_t_pSp)

        latent_bg_c, latent_bg_s = cs_mlp_net(norm_w_bg_pSp, zero_out_silent=opts.zero_out_silent_bg)
        latent_t_c, latent_t_s = cs_mlp_net(norm_w_t_pSp, zero_out_silent=opts.zero_out_silent_t) 

        latent_bg_s = opts.s_scale_lambda * math.sqrt(512)*F.normalize(latent_bg_s, p=2, dim=2)
        latent_t_s = opts.s_scale_lambda * math.sqrt(512)*F.normalize(latent_t_s, p=2, dim=2)

        latent_bg_rec = shift_with_avg(pSp_net, latent_bg_c)
        latent_t_rec = shift_with_avg(pSp_net, latent_t_c + latent_t_s)

        rec_x_bg = pSp_net.forward(latent_bg_rec, input_code=True, randomize_noise=True, recon_modle=True)
        rec_x_t = pSp_net.forward(latent_t_rec, input_code=True, randomize_noise=True, recon_modle=True)
        latent_bg_swap = shift_with_avg(pSp_net, latent_bg_c + latent_t_s)
        latent_t_swap = shift_with_avg(pSp_net, latent_t_c )
        swap_x_bg = pSp_net.forward(latent_bg_swap, input_code=True, randomize_noise=True, recon_modle=True)
        swap_x_t = pSp_net.forward(latent_t_swap, input_code=True, randomize_noise=True, recon_modle=True)

        output_latents = {
                    'w_bg_pSp': w_bg_pSp,
                    'w_t_pSp': w_t_pSp,
                    'latent_bg_c':latent_bg_c,
                    'latent_bg_s':latent_bg_s,
                    'latent_t_c':latent_t_c,
                    'latent_t_s':latent_t_s,
                   }
        
        output_images = {
                    'recon_pSp_bg':rec_x_bg_pSp,
                    'recon_pSp_t':rec_x_t_pSp,
                    'recon_bg':rec_x_bg,
                    'recon_t':rec_x_t,
                    'recon_bg_swap':swap_x_bg,
                    'recon_t_swap':swap_x_t
                   }
        return output_images, output_latents
    

def evaluate_learn_scaled_st(cs_mlp_net, pSp_net, x_bg, x_t, opts):

    with torch.no_grad():

        rec_x_bg_pSp, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True) 
        rec_x_t_pSp, w_t_pSp = pSp_net.forward(x_t, return_latents=True) 
        
        norm_w_bg_pSp = center_with_avg(pSp_net, w_bg_pSp)
        norm_w_t_pSp = center_with_avg(pSp_net, w_t_pSp)

        latent_bg_c, latent_bg_s = cs_mlp_net(norm_w_bg_pSp, zero_out_silent=opts.zero_out_silent_bg)
        latent_t_c, latent_t_s = cs_mlp_net(norm_w_t_pSp, zero_out_silent=opts.zero_out_silent_t) 

        latent_bg_s = opts.s_scale_lambda * math.sqrt(512)*latent_bg_s
        latent_t_s = opts.s_scale_lambda * math.sqrt(512)*latent_t_s

        latent_bg_rec = shift_with_avg(pSp_net, latent_bg_c)
        latent_t_rec = shift_with_avg(pSp_net, latent_t_c + latent_t_s)

        rec_x_bg = pSp_net.forward(latent_bg_rec, input_code=True, randomize_noise=True, recon_modle=True)
        rec_x_t = pSp_net.forward(latent_t_rec, input_code=True, randomize_noise=True, recon_modle=True)		
        latent_bg_swap = shift_with_avg(pSp_net, latent_bg_c + latent_t_s)
        latent_t_swap = shift_with_avg(pSp_net, latent_t_c )
        swap_x_bg = pSp_net.forward(latent_bg_swap, input_code=True, randomize_noise=True, recon_modle=True)
        swap_x_t = pSp_net.forward(latent_t_swap, input_code=True, randomize_noise=True, recon_modle=True)

        output_latents = {
                    'w_bg_pSp': w_bg_pSp,
                    'w_t_pSp': w_t_pSp,
                    'latent_bg_c':latent_bg_c,
                    'latent_bg_s':latent_bg_s,
                    'latent_t_c':latent_t_c,
                    'latent_t_s':latent_t_s,
                   }
        
        output_images = {
                    'recon_pSp_bg':rec_x_bg_pSp,
                    'recon_pSp_t':rec_x_t_pSp,
                    'recon_bg':rec_x_bg,
                    'recon_t':rec_x_t,
                    'recon_bg_swap':swap_x_bg,
                    'recon_t_swap':swap_x_t
                   }
        return output_images, output_latents

class EMA:
    def __init__(self, model, decay=0.9999):
        """
        Initialize EMA for the given model.
        :param model: The model to track.
        :param decay: EMA decay rate.
        """
        self.model = model
        self.decay = decay
        self.ema_model = self.clone_model(model)
        self.ema_model.eval()  # EMA model is used only for evaluation

    @staticmethod
    def clone_model(model):
        """
        Create a clone of the given model with identical parameters.
        """
        cloned_model = type(model)(model.opts)  # Reinitialize with the same options
        cloned_model.load_state_dict(model.state_dict())  # Copy parameters
        return cloned_model

    @torch.no_grad()
    def update(self):
        """
        Update the EMA parameters with the current model parameters.
        """
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data * (1 - self.decay))

        # Update buffers (e.g., BatchNorm running stats) if applicable
        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(model_buffer)
  

def load_ema_model(model_path, device='cpu'):
        
        # load model ckpts
        print('Loading trained checkpoint from path: {}'.format(model_path))
        model_ckpt = torch.load(model_path, map_location=device, weights_only=True)  
        opts = model_ckpt['opts']
        opts = Namespace(**opts)

        # if opts.save_training_data:
        #     global_step = model_ckpt['global_step']
        #     optimizer = model_ckpt['optimizer']
        #     best_val_loss = model_ckpt['best_val_loss']
        
        #### load pSp model including StyleGAN2 generator ####
        # pSp_ckpt = model_ckpt['state_dict_pSp']
        latent_aveg = model_ckpt['latent_avg']
        pSp_net = pSp(opts, previous_train_ckpt=None).to(device)
        		# Estimate latent_avg via dense sampling if latent_avg is not available
        if pSp_net.latent_avg is None:
            pSp_net.latent_avg = latent_aveg
        pSp_net.eval()
        pSp_net.cuda()  

        #pSp_ckpt = prev_train_ckpt['state_dict_pSp']
        cs_ckpt = model_ckpt['state_dict_cs_enc']
        ema_ckpt = model_ckpt['state_dict_cs_enc_ema']  # Correct key for EMA model
        global_step = model_ckpt['global_step']

        from models.mlp3D import MappingNetwork_cs_independent

        # Initialize cs_mlp_net
        cs_mlp_net = MappingNetwork_cs_independent(opts).to(device)

        # Initialize EMA for cs_mlp_net
        ema_cs_mlp_net = EMA(cs_mlp_net, decay=opts.ema_decay)
        ema_cs_mlp = ema_cs_mlp_net.ema_model.to(device)

        ema_cs_mlp.load_state_dict(ema_ckpt)
        ema_cs_mlp.eval()
        ema_cs_mlp.cuda()   
        
        return pSp_net, ema_cs_mlp, opts

def load_sparsity_model(model_path, device='cpu'):
        
        # load model ckpts
        print('Loading trained checkpoint from path: {}'.format(model_path))
        model_ckpt = torch.load(model_path, map_location=device, weights_only=True)  
        opts = model_ckpt['opts']
        opts = Namespace(**opts)
        print('training_step: ', model_ckpt['global_step'])

        if opts.output_size:
            opts.stylegan_size = opts.output_size
        else:
            opts.stylegan_size = 1024
        pSp_net = pSp(opts, previous_train_ckpt=None).to(device)
        		# Estimate latent_avg via dense sampling if latent_avg is not available
        print('Loading trained checkpoint from path: {}'.format(model_path))
        # if pSp_net.latent_avg is not None:
        #     print('pSp_net.latent_avg is not None')

        pSp_net.eval()
        # pSp_net.cuda()  

        ##########  load cs_mlp model ##########
        cs_ckpt  = model_ckpt['state_dict_cs_enc']	
        from models.mlp3D import MappingNetwork_cs_sparsity
        cs_mlp_net = MappingNetwork_cs_sparsity(opts).to(device)
        cs_mlp_net.load_state_dict(cs_ckpt)
        cs_mlp_net.eval()
        # cs_mlp_net.cuda()    

        return pSp_net, cs_mlp_net, opts


def load_c_s1_s2_model(model_path, device='cpu'):
        
        # load model ckpts
        print('Loading trained checkpoint from path: {}'.format(model_path))
        model_ckpt = torch.load(model_path, map_location=device, weights_only=True)  
        opts = model_ckpt['opts']
        opts = Namespace(**opts)
        print('training_step: ', model_ckpt['global_step'])

        if opts.output_size:
            opts.stylegan_size = opts.output_size
        else:
            opts.stylegan_size = 1024
        pSp_net = pSp(opts, previous_train_ckpt=None).to(device)
        		# Estimate latent_avg via dense sampling if latent_avg is not available
        print('Loading trained checkpoint from path: {}'.format(model_path))
        # if pSp_net.latent_avg is not None:
        #     print('pSp_net.latent_avg is not None')

        pSp_net.eval()
        # pSp_net.cuda()  

        ##########  load cs_mlp model ##########
        cs_ckpt  = model_ckpt['state_dict_cs_enc']	
        from models.mlp3D import MappingNetwork_c_s1_s2
        cs_mlp_net = MappingNetwork_c_s1_s2(opts).to(device)	
        cs_mlp_net.load_state_dict(cs_ckpt)
        cs_mlp_net.eval()
        # cs_mlp_net.cuda()    

        return pSp_net, cs_mlp_net, opts

def load_pretained_pSp_net(model_cspSp_path, map_location = 'cpu', output_size = 1024, pSp_ckpt_path = None):
    
    cspSp_ckpt = torch.load(model_cspSp_path, map_location=map_location, weights_only=True)
    opts = cspSp_ckpt['opts']

    opts = Namespace(**opts)
    opts.output_size = output_size
    if pSp_ckpt_path is not None:
        opts.pSp_checkpoint_path = pSp_ckpt_path 
        pSp_net = pSp(opts)
    else: 
        pSp_net = pSp(opts)

    pSp_net.eval()
    pSp_net.cuda()

    
    return pSp_net, opts



def transform_to_input_image(image, img_transforms, opts):
    
    if opts.label_nc == 0:
        input_image = image.convert("RGB")
    else:
        input_image = image.convert("L")

    transformed_image = img_transforms(input_image)   
    
    return transformed_image


# def init_networks(self, prev_train_ckpt):
# 		pSp_ckpt = None
# 		cs_ckpt = None
# 		if prev_train_ckpt is not None:
# 			pSp_ckpt = prev_train_ckpt['state_dict_pSp']
# 			cs_ckpt  = prev_train_ckpt['state_dict_cs_enc']	
# 			self.global_step = prev_train_ckpt['global_step'] + 1

# 		self.pSp_net = pSp(self.opts, pSp_ckpt).to(self.device)

# 		if self.opts.mlp_model=='normal':
# 			self.cs_mlp_net = MappingNetwork_cs(512, self.opts.n_layers_mlp).to(self.device)
# 		elif self.opts.mlp_model=='individal':
# 			self.cs_mlp_net = MappingNetwork_cs_individal(512, self.opts.n_layers_mlp).to(self.device)
# 		elif self.opts.mlp_model=='pyramid':
# 			self.cs_mlp_net = MappingNetwork_cs_pyramid(512, self.opts.n_layers_mlp).to(self.device)
# 		elif self.opts.mlp_model=='sparsity':
# 			self.cs_mlp_net = MappingNetwork_cs_sparsity(self.opts).to(self.device)
# 		else:
# 			raise Exception('errors in loading mlp model')			
		
# 		if cs_ckpt is not None:
# 			print('Loading cs encoder from previous checkpoint...')
# 			self.cs_mlp_net.load_state_dict(cs_ckpt)
# 			print(f'Resuming training from step {self.global_step}')

# def load_pretained_csmlp_net(model_cspSp_path, map_location = 'cpu'):

#     print('Loading cs encoder from checkpoint: ' + model_cspSp_path)

#     cspSp_ckpt = torch.load(model_cspSp_path, map_location=map_location, weights_only=True)
#     opts = cspSp_ckpt['opts']
    

#     opts = Namespace(**opts)

#     print(opts.mlp_model)

#     if opts.mlp_model=='normal':
#         cs_mlp_net = MappingNetwork_cs(512, opts.n_layers_mlp)
#     elif opts.mlp_model=='individal':
#         cs_mlp_net = MappingNetwork_cs_individal(512, opts.n_layers_mlp)
#     elif opts.mlp_model=='pyramid':
#         cs_mlp_net = MappingNetwork_cs_pyramid(512, opts.n_layers_mlp)
#     elif opts.mlp_model=='sparsity':
#         cs_mlp_net = MappingNetwork_cs_sparsity(opts)
#     else:
#         raise Exception('errors in loading mlp model')		               
    
#     ckpt_cs_enc = cspSp_ckpt['state_dict_cs_enc']
#     cs_mlp_net.load_state_dict(ckpt_cs_enc)

#     cs_mlp_net.eval()
#     cs_mlp_net.cuda()    
    
#     return cs_mlp_net, opts

# def load_pretained_csmlp_pos_net(model_cspSp_path, map_location = 'cpu', output_size = 1024, pSp_ckpt_path = None):

#     print('Loading cs encoder from checkpoint: ' + model_cspSp_path)

#     cspSp_ckpt = torch.load(model_cspSp_path, map_location=map_location, weights_only=True)
#     opts = cspSp_ckpt['opts']
 
#     if 'pos_s_only' not in opts: 
#         opts['pos_s_only'] = False               
    
#     opts = Namespace(**opts)
#     ckpt_cs_enc = cspSp_ckpt['state_dict_cs_enc']

#     cs_mlp_net = MappingNetwork_cs_pos_out(512, opts.n_layers_mlp, opts.mlp_act_fn, opts.pos_s_only)
    
#     cs_mlp_net.load_state_dict(ckpt_cs_enc)

#     cs_mlp_net.eval()
#     cs_mlp_net.cuda()    
    
#     return cs_mlp_net, opts


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    



def visualize_latentmat(mat_roi):

    yticks=[0, 17]

    plt.figure(figsize=(15,10))
    ax = plt.gca()
    im = ax.matshow(mat_roi)
    ax.set_yticks(yticks)    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.5)
    
    plt.colorbar(im, cax=cax)


def hard_threshold(x, threshold):
    return torch.where(torch.abs(x) < threshold, torch.tensor(0.0, device=x.device), x)

def soft_threshold(x, threshold):
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))

def calculate_output_sparsity(output, zero_threshold=0.0):
    """
    Calculate sparsity of output activations.

    Args:
        output (torch.Tensor): The output tensor for which sparsity is calculated.
        zero_threshold (float): Threshold below which values are considered zero.

    Returns:
        sparsity (float): The fraction of elements that are effectively zero.
        zero_count (int): Number of zero-valued elements.
        total_count (int): Total number of elements.
    """
    zero_count = (output.abs() <= zero_threshold).sum().item()   #If zero_threshold=0.0, only values that are exactly zero will be counted as sparse. For any other zero_threshold > 0, values with absolute magnitude less than or equal to the threshold will be considered sparse
    total_count = output.numel()
    sparsity = zero_count / total_count if total_count > 0 else 0
    return sparsity, zero_count, total_count



def show_latent_map(latent_matrix, img_idx=0, show_height=18, show_weight=512, zero_threshold=0.0):
    """
    Displays a portion of the latent matrix with values below `zero_threshold` (absolute value) set to zero, after calculating sparsity.

    Args:
        latent_matrix (torch.Tensor): The latent matrix to display, with dimensions [N, H, W] or [H, W].
        img_idx (int): Index for selecting the image slice in case of 3D tensor.
        show_height (int): Number of rows to display.
        show_weight (int): Number of columns to display.
        zero_threshold (float): Absolute threshold below which values are set to zero.

    Returns:
        None. Displays the processed matrix as an image.
    """
    
    # Select a slice of the matrix based on dimensionality
    if latent_matrix.dim() == 3:
        show_matrix = latent_matrix[img_idx, :show_height, :show_weight]
    else:
        show_matrix = latent_matrix[:show_height, :show_weight]
    
    # Calculate sparsity information before zeroing out values
    sparsity, zero_count, total_count = calculate_output_sparsity(latent_matrix, zero_threshold=zero_threshold)

    # Zero out values with absolute magnitude below the specified threshold
    show_matrix = torch.where(show_matrix.abs() < zero_threshold, torch.tensor(0.0, device=show_matrix.device), show_matrix)

    # Convert to numpy for display if necessary
    show_matrix = show_matrix.cpu().numpy() if show_matrix.is_cuda else show_matrix.numpy()

    print(f"Output Sparsity: {sparsity:.4f} ({zero_count} of {total_count})")

    # Plotting the matrix
    plt.figure(figsize=(12, 4))
    plt.imshow(show_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("")
    plt.xlabel("feature dimension (50 of 512)")
    plt.ylabel("style dimension (18)")
    plt.show()


# results/alternative_cls/cls10ep_lr0.01_lamb0.1/checkpoints/iteration_147600.pt
def perfrom_decompose(input_images_bg, input_images_t, model_path, device):
    pSp_net, cs_mlp_net, opts = load_sparsity_model(model_path, device=device)
    with torch.no_grad():
        # rec_x_t_pSp, w_t_pSp = pSp_net.forward(input_images_t, return_latents=True)  
        
        # latent_t_c, latent_t_s = csmlp_net(w_t_pSp, zero_out_silent=opts.zero_out_silent_t)

        # rec_x_t_pSp1 = pSp_net.forward(w_t_pSp, input_code=True, randomize_noise=True, recon_modle=True)
        # rec_x_t = pSp_net.forward(latent_t_c + latent_t_s, input_code=True, randomize_noise=True, recon_modle=True)

        w_pSp_bg = pSp_net.forward(input_images_bg, encode_only=True)  
        w_pSp_t = pSp_net.forward(input_images_t, encode_only=True)  

        latent_bg_c, latent_bg_s = cs_mlp_net(w_pSp_bg, zero_out_silent=opts.zero_out_silent_bg)
        latent_t_c, latent_t_s = cs_mlp_net(w_pSp_t, zero_out_silent=opts.zero_out_silent_t) 

        recon_bg = pSp_net.forward(latent_bg_c, input_code=True, randomize_noise=False, recon_modle=True)
        recon_t = pSp_net.forward(latent_t_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)

        swap_bg = pSp_net.forward(latent_bg_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
        swap_t = pSp_net.forward(latent_t_c , input_code=True, randomize_noise=False, recon_modle=True)    

    return recon_bg, recon_t, swap_bg, swap_t
