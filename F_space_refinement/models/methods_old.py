import math
import sys
import pickle
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator
from models.hyperinverter.stylegan2_ada import Discriminator 
from utils.class_registry import ClassRegistry
from utils.common_utils import get_keys
from utils.model_utils import toogle_grad
from configs.paths import DefaultPathsClass
from argparse import Namespace
from training.loggers import BaseTimer
from runners.model_funcs import load_e4e_cs_model, load_pSp_cmlp_models
from utils.stylegan2_ada_legacy import load_network_pkl as legacy_load

sys.path.append("./utils")
methods_registry = ClassRegistry()

@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path", "w_space_encoder", "dataset"))
class FSEFull(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 checkpoint_path=None,
                 inverter_pth=None,
                 w_space_encoder=None,
                 dataset = None):
        super(FSEFull, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "w_space_encoder": w_space_encoder,
            "dataset" : dataset, 
            "stylegan_size": 1024,
        }
        # instantiate a fresh DefaultPathsClass for *this* dataset
        per_dataset_paths = DefaultPathsClass(dataset=dataset)

        # merge its fields into opts
        self.opts.update({k: v for k, v in per_dataset_paths})
        self.opts = Namespace(**self.opts)

        self.device = device
        self.inverter_pth = inverter_pth

        self.encoder = self.set_encoder()
        
        if self.opts.w_space_encoder == "e4e":
            print("Using e4e as w encoder, loading e4e-cs model ....")
            self.e4e_cs_model = load_e4e_cs_model(model_path=self.opts.e4e_cs_path, device=self.device)
        elif self.opts.w_space_encoder == "pSp":
            print("Using pSp as w encoder, loading pSp-cs model ....")
            self.pSp_encoder, self.pSp_cs_model, self.pSp_cs_opts = load_pSp_cmlp_models(model_path=self.opts.pSp_cs_path, psp_path=self.opts.psp_path, device=self.device, eval_models=True)   
        else:
            raise ValueError(
                f"Unknown training w encoder type: {self.opts.w_space_encoder}"
            )   
        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()
        
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    def load_disc(self):
        path = self.opts.stylegan_weights_pkl
        print("Loading discriminator from", path)

        # decide which loader to use
        ds = self.opts.dataset.lower()
        use_legacy = "afhq" in ds

        if use_legacy:
            # vendored legacy loader (keeps scripted module, eval+no-grad)
            with open(path, 'rb') as f:
                ckpt = legacy_load(f)
            D = ckpt['D'].eval().requires_grad_(False).to(self.device).float()
            self.discriminator = D
            print("✅ discriminator loaded via vendored legacy loader")

        else:
            # plain pickle + fresh class instantiation
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            D_original = ckpt["D"].float()
            # re-create the same architecture
            disc = Discriminator(**D_original.init_kwargs)
            disc.load_state_dict(D_original.state_dict())
            disc = disc.to(self.device).eval().requires_grad_(False)
            self.discriminator = disc
            print("✅ discriminator re-instantiated via state_dict")

    def load_disc_from_ckpt(self, ckpt):
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        if "discriminator" in unique_keys:
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
        else:
            print("Can not find Discriminator weights in checkpoint, leave default weights.")

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print(f"Loading from checkpoint: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.inverter.load_state_dict(get_keys(ckpt, "inverter"), strict=True)
        else:
            print(f"Loading Discriminator and Inverter from Inverter checkpoint: {self.inverter_pth}")
            ckpt = torch.load(self.inverter_pth, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.inverter.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        # print("Loading Decoder from", self.opts.stylegan_weights)
        # ckpt = torch.load(self.opts.stylegan_weights)
        # self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.e4e_path))
        e4e_ckpt = torch.load(self.opts.e4e_path, map_location='cpu', weights_only=True)
        self.decoder.load_state_dict(get_keys(e4e_ckpt, 'decoder'), strict=True)
        self.latent_avg = e4e_ckpt['latent_avg'].to(self.device)
        self.decoder = self.decoder.eval().to(self.device)

        toogle_grad(self.decoder, False)

        if self.opts.w_space_encoder == "e4e":
            print("Loading E4E from", self.opts.e4e_path)
            ckpt = torch.load(self.opts.e4e_path, map_location="cpu")
            self.e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.e4e_encoder = self.e4e_encoder.eval().to(self.device)
            toogle_grad(self.e4e_encoder, False)
        elif self.opts.w_space_encoder == "pSp":
            toogle_grad(self.pSp_encoder, False)
        else:
            raise ValueError(
                f"Unknown training w encoder type: {self.opts.w_space_encoder}"
            )                        
    def set_encoder(self):
        self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18) 
        if self.opts.w_space_encoder == "e4e":
            self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        return feat_editor  # trainable part
    
    def forward(self, x, return_latents=False, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            w_recon = w_recon + self.latent_avg
                    
            _, w_feats = self.decoder(
                [w_recon],
                input_is_latent=True,
                return_features=True,
                is_stylespace=False,
                randomize_noise=False,
                early_stop=64
            )

            w_feat = w_feats[9]  # bs x 512 x 64 x 64 
            
            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            delta = torch.zeros_like(fused_feat)  # inversion case

        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )

        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images


@methods_registry.add_to_registry("fse_inverter", stop_args=("self", "checkpoint_path", "dataset"))
class FSEInverter(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 checkpoint_path=None,
                 dataset=None):
        super(FSEInverter, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "dataset": dataset,
            "stylegan_size": 1024
        }
        # instantiate a fresh DefaultPathsClass for *this* dataset
        per_dataset_paths = DefaultPathsClass(dataset=dataset)

        # merge its fields into opts
        self.opts.update({k: v for k, v in per_dataset_paths})
        self.opts = Namespace(**self.opts)

        self.device = device
        self.encoder = self.set_encoder()

        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        path = self.opts.stylegan_weights_pkl
        print("Loading discriminator from", path)

        # decide which loader to use
        ds = self.opts.dataset.lower()
        use_legacy = "afhq" in ds

        if use_legacy:
            # vendored legacy loader (keeps scripted module, eval+no-grad)
            with open(path, 'rb') as f:
                ckpt = legacy_load(f)
            D = ckpt['D'].eval().requires_grad_(False).to(self.device).float()
            self.discriminator = D
            print("✅ discriminator loaded via vendored legacy loader")

        else:
            # plain pickle + fresh class instantiation
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            D_original = ckpt["D"].float()
            # re-create the same architecture
            disc = Discriminator(**D_original.init_kwargs)
            disc.load_state_dict(D_original.state_dict())
            disc = disc.to(self.device).eval().requires_grad_(False)
            self.discriminator = disc
            print("✅ discriminator re-instantiated via state_dict")


    def load_disc_from_ckpt(self, ckpt):
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        if "discriminator" in unique_keys:
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
        else:
            print("Can not find Discriminator weights in checkpoint, leave default weights.")

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading  from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

        print("Loading decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)

    def set_encoder(self):
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18)
        return inverter  # trainable part
    
    def forward(self, x, return_latents=False, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        w_recon, predicted_feat = self.encoder.fs_backbone(x)
        w_recon = w_recon + self.latent_avg
                
        _, w_feats = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )

        w_feat = w_feats[9]  # bs x 512 x 64 x 64 
        fused_feat = self.encoder.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        feats = [None] * 9 + [fused_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )
        
        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                w_feat = w_feat.cpu()
            return images, w_recon, fused_feat, w_feat
        return images
