import torch

torch.set_float32_matmul_precision('medium')

from config import *
from dataset import *
import pandas as pd
import json
import os
import copy
import csv
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from experiment import LitModel
import torchvision.utils as vutils
from templates import *
from pytorch_lightning.loggers import CSVLogger
from model.mlp2D import Discriminator

class ZipLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for each in zip(*self.loaders):
            yield each

class EqualizedLinear(nn.Module):
    """
    Equalized Learning-Rate Linear Layer with initialized weight scaling.
    """
    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedWeight(nn.Module):
    """
    Implements learning-rate equalized weight scaling, based on Progressive GANs,
    with weights initialized to `N(0, 1)` and scaled by `c`.
    """
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

class MappingNetwork_cs(nn.Module):
    """
    Mapping Network with Dual Outputs (`c` and `s`), mapping `z` into two separate spaces
    through parallel networks with equalized learning-rate linear layers and LeakyReLU.
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        features=conf.style_ch
        n_layers=conf.n_layers
        # self.fnormalize=conf.fnormalize
        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        #print('z size::::::::::::::::::', z.shape)
        # if self.fnormalize:
        #     z = F.normalize(z, dim=1)

        c = self.net_c(z)
        s = self.net_s(z)

        return c, s

class MlpModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()

        if conf.seed is not None:
            pl.seed_everything(conf.seed)
        self.automatic_optimization = False
        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf
        self.discriminator = Discriminator(input_dim=512)
            
        self.model = LitModel(conf)
        state = torch.load(f'checkpoints/{conf.encoder_name}/last.ckpt', map_location='cpu')
        print(f"Loaded pretrained model from checkpoints/{conf.encoder_name}/last.ckpt' successfully.")
        
        self.model.load_state_dict(state['state_dict'], strict=False)
        self.model.eval()
        self.model.ema_model.eval()

        # load the latent stats
        if conf.manipulate_znormalize:
            print('loading latent stats ...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean',
                                    state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None


        self.mlp = MappingNetwork_cs(conf)  # Original MLP
        self.ema_mlp = copy.deepcopy(self.mlp)  # EMA version

        #self.mse_loss = nn.MSELoss()
        self.lr = conf.lr

    def state_dict(self):
        # Save only specific components
        return {
            "mlp": self.mlp.state_dict(),
            "ema_mlp": self.ema_mlp.state_dict(),
            "conf": self.conf
        }

    def load_state_dict(self, state_dict, strict=True):
        # Load only the saved components
        self.mlp.load_state_dict(state_dict["mlp"], strict=strict)
        self.ema_mlp.load_state_dict(state_dict["ema_mlp"], strict=strict)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond


    def setup(self, stage=None) -> None:
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################
        

    def train_dataloader(self):
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataset_bg, dataset_t = self.conf.make_mlp_dataset(path_bg="datasets/ffhq256_mlp/ffhq256_bg.lmdb",
                                   path_t ="datasets/ffhq256_mlp/ffhq256_glass.lmdb" )
        
        train_data = [dataset_bg, dataset_t]

        # Create independent loaders for each dataset in self.train_data
        dataloader = []
        for each in train_data:
            dataloader.append(
                conf.make_loader(each, shuffle=True, drop_last=True))
        
        # Zip the loaders together
        dataloader = ZipLoader(dataloader)

        return dataloader

    def val_dataloader(self):
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        # Create datasets
        dataset_bg, dataset_t = self.conf.make_mlp_dataset(
            path_bg="datasets/ffhq256_mlp/ffhq256_test_bg.lmdb",
            path_t="datasets/ffhq256_mlp/ffhq256_test_glass.lmdb"
        )

        val_data = [dataset_bg, dataset_t]
        # Create dataloaders for each dataset
        dataloader = []
        for each in val_data:
            dataloader.append(
                conf.make_loader(each, shuffle=True, drop_last=True))
        
        # Zip the loaders together
        dataloader = ZipLoader(dataloader)

        # Combine dataloaders using ZipLoader
        return dataloader


    @property
    def batch_size(self):
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws
    
    # def validation_step(self, batch, batch_idx):
    #     """
    #     Compute validation losses and log metrics.
    #     """
    #     # Unpack the batches
    #     bg_batch, t_batch = batch
    #     bg_imgs = bg_batch['img']
    #     t_imgs = t_batch['img']

    #     # Forward pass through the encoder
    #     cond_bg = self.model.encode(bg_imgs)
    #     cond_t = self.model.encode(t_imgs)

    #     # Apply optional normalization
    #     if self.conf.manipulate_znormalize:
    #         cond_bg = self.normalize(cond_bg)
    #         cond_t = self.normalize(cond_t)

    #     # Forward pass through the MLP
    #     c_bg, s_bg = self.mlp(cond_bg)
    #     c_t, s_t = self.mlp(cond_t)

    #     with torch.no_grad():  # No gradients during validation
    #         # ----------------------------
    #         # Compute Discriminator Loss
    #         # ----------------------------
    #         real_labels = torch.ones(cond_bg.shape[0], 1, device=self.device)
    #         fake_labels = torch.zeros(c_t.shape[0], 1, device=self.device)

    #         d_real = self.discriminator(cond_bg)  # Real images
    #         d_fake = self.discriminator(c_t)  # Fake images

    #         d_loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
    #         d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
    #         d_loss = (d_loss_real + d_loss_fake) * 0.5

    #         # ----------------------------
    #         # Compute Generator Loss (MLP)
    #         # ----------------------------
    #         g_loss = F.binary_cross_entropy_with_logits(self.discriminator(c_t), real_labels)

    #         # Compute Reconstruction Losses
    #         loss_bg = F.mse_loss(c_bg, cond_bg)
    #         loss_t = F.mse_loss(c_t + s_t, cond_t)
    #         loss_sbg = F.mse_loss(s_bg, torch.zeros_like(s_bg))

    #         total_loss = loss_bg + loss_t + loss_sbg + self.conf.adv_weight * g_loss
    #         aux_st_loss = F.mse_loss(s_t, torch.zeros_like(s_t))

    #     # Logging validation metrics
    #     self.log("val_d_loss", d_loss, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_d_loss_real", d_loss_real, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_d_loss_fake", d_loss_fake, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_g_loss", g_loss, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_loss_bg", loss_bg, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_loss_t", loss_t, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_loss_sbg", loss_sbg, prog_bar=True, on_epoch=True, on_step=False)
    #     self.log("val_aux_st_loss", aux_st_loss, prog_bar=True, on_epoch=True, on_step=False)

    #     return total_loss



    def training_step(self, batch, batch_idx):
        """
        Manually step optimizers because we have multiple optimizers.
        """
        bg_batch, t_batch = batch  # Unpack the batches
        bg_imgs = bg_batch['img']
        t_imgs = t_batch['img']

        cond_bg = self.model.encode(bg_imgs)
        cond_t = self.model.encode(t_imgs)

        if self.conf.manipulate_znormalize:
            cond_bg = self.normalize(cond_bg)
            cond_t = self.normalize(cond_t)

        c_bg, s_bg = self.mlp(cond_bg)
        c_t, s_t = self.mlp(cond_t)

        # Get optimizers manually
        opt_mlp, opt_d = self.optimizers()

        # ----------------------------
        # Step 1: Train Discriminator
        # ----------------------------
        opt_d.zero_grad()
        real_labels = torch.ones(cond_bg.shape[0], 1, device=self.device)
        fake_labels = torch.zeros(c_t.shape[0], 1, device=self.device)

        d_real = self.discriminator(cond_bg.detach())  # Real images
        d_fake = self.discriminator(c_t.detach())  # Fake images

        d_loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        self.manual_backward(d_loss)  # Backprop manually
        opt_d.step()

        # ----------------------------
        # Step 2: Train MLP (Generator)
        # ----------------------------
        opt_mlp.zero_grad()
        g_loss = F.binary_cross_entropy_with_logits(self.discriminator(c_t), real_labels)

        loss_bg = F.mse_loss(c_bg, cond_bg)
        loss_t = F.mse_loss(c_t + s_t, cond_t)
        loss_sbg = F.mse_loss(s_bg, torch.zeros_like(s_bg))

        total_loss = (loss_bg + loss_t + loss_sbg + self.conf.adv_weight * g_loss)* self.conf.w_factor
        aux_st_loss = F.mse_loss(s_t, torch.zeros_like(s_t))

        self.manual_backward(total_loss)  # Backprop manually
        opt_mlp.step()

        # Logging
        self.log("train_d_loss", d_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_d_loss_real", d_loss_real, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_d_loss_fake", d_loss_fake, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_g_loss", g_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_loss_bg", loss_bg, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_loss_t", loss_t, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_loss_sbg", loss_sbg, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_aux_st_loss", aux_st_loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return total_loss 




    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
 
        # Apply EMA (Exponential Moving Average)
        ema(self.mlp, self.ema_mlp, self.conf.ema_decay)

    # def configure_optimizers(self):
    #     optim = torch.optim.Adam(self.mlp.parameters(),
    #                              lr=self.conf.lr,
    #                              weight_decay=self.conf.weight_decay)

    #     return optim

    def configure_optimizers(self):
        mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.conf.lr_mlp, weight_decay=self.conf.weight_decay)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.conf.lr_d)

        return [mlp_optimizer, d_optimizer]

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def train_mlp(conf: TrainConfig, gpus):

    model = MlpModel(conf)

    if not os.path.exists(conf.results_path):
        os.makedirs(conf.results_path)
    checkpoint = ModelCheckpoint(
        dirpath=f'{conf.results_path}',
        save_last=True,
        save_top_k=-1,
        every_n_epochs=50,
        filename="mlp-{epoch:02d}"
        # every_n_train_steps=conf.save_every_samples //
        # conf.batch_size_effective,
    )
        # Add a CSV logger
    csv_logger = CSVLogger(save_dir=conf.results_path, name="logs")

    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.results_path,
    #                                          name=None,
    #                                          version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    # trainer = pl.Trainer(
    #     #max_steps=conf.max_steps // conf.batch_size_effective,
    #     max_epochs = conf.max_epochs,
    #     #resume_from_checkpoint=resume,
    #     devices=gpus,
    #     accelerator="gpu" if len(gpus) > 0 else "cpu",
    #     precision=16 if conf.fp16 else 32,
    #     callbacks=[checkpoint,],
    #     #replace_sampler_ddp=True,
    #     logger=[csv_logger],
    #     accumulate_grad_batches=conf.accum_batches,
    #     plugins=plugins,
    #     check_val_every_n_epoch=1,  # Run validation every epoch
    #     # val_check_interval=0.25     # (Optional) Run validation every 25% of an epoch
    # )
    trainer = pl.Trainer(
        max_epochs=conf.max_epochs,
        devices=gpus,
        accelerator="gpu" if len(gpus) > 0 else "cpu",
        precision=16 if conf.fp16 else 32,
        callbacks=[checkpoint],
        logger=[csv_logger],
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
        check_val_every_n_epoch=0,  # Disables validation
        # Do not include val_check_interval
    )

    trainer.fit(model)
