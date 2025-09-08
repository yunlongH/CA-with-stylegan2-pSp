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
from criteria import id_loss
from criteria.lpips.lpips import LPIPS

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

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf

        if self.conf.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.conf.id_lambda > 0:	
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
            
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
    
    # def save_images_result(self, bg_imgs, t_imgs, recon_bg, recon_t, swap_bg, swap_t):
    #     # Concatenate images along the batch dimension
    #     num_images = bg_imgs.size(0)  # Assuming all tensors have the same batch size
    #     images = torch.cat([bg_imgs, t_imgs, recon_bg, recon_t, swap_bg, swap_t], dim=0)

    #     # Reshape the concatenated tensor to organize each set of images in a separate row
    #     images_per_row = images.view(6, num_images, *images.shape[1:])
    #     reshaped_images = images_per_row.transpose(0, 1).reshape(-1, *images.shape[1:])

    #     # Create a grid with the reshaped images
    #     grid = vutils.make_grid(reshaped_images, nrow=num_images, normalize=True)

    #     # Ensure the directory exists
    #     save_dir = os.path.join(self.conf.results_path, "images")
    #     os.makedirs(save_dir, exist_ok=True)

    #     # Save the grid image
    #     save_path = os.path.join(self.conf.results_path, f"images/validation_results_epoch_{self.current_epoch}.png")
    #     vutils.save_image(grid, save_path)
    #     print(f"Saved validation grid to: {save_path}")

    def save_images_result(self, bg_imgs, t_imgs, recon_bg, recon_t, swap_bg, swap_t):
        # Concatenate the images along the batch dimension
        images = torch.cat([bg_imgs, t_imgs, recon_bg, recon_t, swap_bg, swap_t], dim=0)

        # Calculate the number of images per row
        num_images = bg_imgs.size(0)  # Assuming both bg_imgs and t_imgs have the same batch size

        # Create a grid with the images
        grid = vutils.make_grid(images, nrow=num_images, normalize=True)

        # Ensure the directory exists
        save_dir = os.path.join(self.conf.results_path, "images")
        os.makedirs(save_dir, exist_ok=True)

        # Save the grid image
        save_path = os.path.join(save_dir, f"validation_results_epoch_{self.current_epoch}.png")
        vutils.save_image(grid, save_path)
        print(f"Saved validation grid to: {save_path}")

    
        # Log to TensorBoard
        #self.logger.experiment.add_image(f"validation_results_epoch_{self.current_epoch}", grid, global_step=self.current_epoch)

    def calc_image_loss(self, x, x_hat):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        # Calculate pSp id, rec, lpips losses for images
        if self.conf.id_lambda > 0:			
            loss_id, sim_improvement, id_logs = self.id_loss(x_hat, x, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.conf.id_lambda

        if self.conf.pix_lambda > 0:
            loss_pix = F.mse_loss(x, x_hat)
            loss_dict['loss_pix'] = float(loss_pix)
            loss += loss_pix * self.conf.pix_lambda

        if self.conf.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(x_hat, x)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.conf.lpips_lambda

        loss_dict['loss'] = float(loss)

        return loss, loss_dict, id_logs
    
    def validation_step(self, batch, batch_idx):
        # Unpack combined batches
        bg_batch, t_batch = batch
        bg_imgs = bg_batch['img']  # Background images
        t_imgs = t_batch['img']  # Target images

        # Compute features
        cond_bg = self.model.encode(bg_imgs)
        cond_t = self.model.encode(t_imgs)
        
        if self.conf.manipulate_znormalize == True:
            cond_bg = self.normalize(cond_bg)
            cond_t = self.normalize(cond_t)

        # Forward pass through MLP
        c_bg, s_bg = self.mlp(cond_bg)
        c_t, s_t = self.mlp(cond_t)

        if self.conf.scaled_silent:
            #s_bg = self.conf.scaled_s_factor * math.sqrt(512) * F.normalize(s_bg, dim=1)
            s_t = self.conf.scaled_s_factor * math.sqrt(512) * F.normalize(s_t, dim=1)

        # Compute losses
        loss_bg =  self.conf.w_factor * F.mse_loss(c_bg, cond_bg)
        loss_t = self.conf.w_factor * F.mse_loss(c_t + s_t, cond_t)
        loss_sbg = self.conf.w_factor * self.conf.sbg_lambda * F.mse_loss(s_bg, torch.zeros_like(s_bg))
        
        # Combine losses if needed
        total_loss = loss_bg + loss_t + loss_sbg

        aux_st_loss = self.conf.w_factor * F.l1_loss(s_t, torch.zeros_like(s_t))

        # Log individual and total losses
        self.log("val_loss_bg", loss_bg, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_loss_t", loss_t, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_loss_sbg", loss_sbg, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_aux_st_loss", aux_st_loss, prog_bar=True, on_epoch=True, on_step=False)
        # # Save and log images for the first batch
        # if batch_idx == 0:
        #     num_images = 6  # Number of images to visualize
        #     self.save_images_result(cond_bg, cond_t, bg_imgs, t_imgs, c_bg, s_bg, c_t, s_t, num_images)

        return total_loss



    def training_step(self, batch, batch_idx):
        """
        batch: a tuple containing two batches (bg_batch and t_batch)
        """
        current_epoch = self.trainer.current_epoch

        bg_batch, t_batch = batch  # Unpack the batches

        # Extract images from the batches
        bg_imgs = bg_batch['img']  # Background images
        t_imgs = t_batch['img']  # Target images (images with glasses)
                
        cond_bg = self.model.encode(bg_imgs)  # Extract diffAE latent for background
        cond_t = self.model.encode(t_imgs)  # Extract diffAE latent for target

        if self.conf.manipulate_znormalize == True:
            cond_bg = self.normalize(cond_bg)
            cond_t = self.normalize(cond_t)

        # Forward pass through the MLP
        c_bg, s_bg = self.mlp(cond_bg)           # Output for background images
        c_t, s_t = self.mlp(cond_t)   

        # s_bg = self.conf.scaled_s_factor * math.sqrt(512) * F.normalize(s_bg, dim=1)
        if self.conf.scaled_silent:
            #s_bg = self.conf.scaled_s_factor * math.sqrt(512) * F.normalize(s_bg, dim=1)
            s_t = self.conf.scaled_s_factor * math.sqrt(512) * F.normalize(s_t, dim=1)

        if self.conf.train_on_imgs:

            if self.conf.manipulate_znormalize == True:
                xT_bg = self.model.encode_stochastic(bg_imgs, self.denormalize(cond_bg), T=250)
                xT_t = self.model.encode_stochastic(t_imgs, self.denormalize(cond_t), T=250)
                recon_bg = self.model.render(xT_bg, self.denormalize(c_bg), T=100)
                recon_t = self.model.render(xT_t, self.denormalize(c_t + s_t), T=100)          
                swap_bg = self.model.render(xT_bg, self.denormalize(c_bg + s_t), T=100)
                swap_t = self.model.render(xT_t, self.denormalize(c_t), T=100)
            else:
                xT_bg = self.model.encode_stochastic(bg_imgs, cond_bg, T=250)
                xT_t = self.model.encode_stochastic(t_imgs, cond_t, T=250)
                recon_bg = self.model.render(xT_bg, c_bg, T=100)
                recon_t = self.model.render(xT_t, c_t + s_t, T=100)          
                swap_bg = self.model.render(xT_bg, c_bg + s_t, T=100)
                swap_t = self.model.render(xT_t, c_t, T=100)      

            loss_img_bg, _, id_logs_bg = self.calc_image_loss(bg_imgs, recon_bg)
            loss_img_t, _, id_logs_t = self.calc_image_loss(t_imgs, recon_t)

        if current_epoch % self.conf.save_image_interval == 0:
            self.save_images_result(bg_imgs, t_imgs, recon_bg, recon_t, swap_bg, swap_t)

        else:
            loss_img_bg = 0
            loss_img_t = 0
        
        # Compute losses
        loss_bg =  self.conf.w_factor * F.mse_loss(c_bg, cond_bg)
        loss_t = self.conf.w_factor * F.mse_loss(c_t + s_t, cond_t)
        loss_sbg = self.conf.w_factor * self.conf.sbg_lambda * F.mse_loss(s_bg, torch.zeros_like(s_bg))
        
        # Combine losses if needed
        total_loss = loss_bg + loss_t + loss_sbg + loss_img_bg + loss_img_t
        aux_st_loss = self.conf.w_factor * F.l1_loss(s_t, torch.zeros_like(s_t))

        # Log metrics only at the end of the epoch
        self.log('train_loss_bg', loss_bg, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_loss_t', loss_t, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_loss_sbg', loss_sbg, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_loss_img_bg', loss_img_bg, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_loss_img_t', loss_img_t, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_aux_st_loss", aux_st_loss, prog_bar=True, on_epoch=True, on_step=False)

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
 
        # Apply EMA (Exponential Moving Average)
        ema(self.mlp, self.ema_mlp, self.conf.ema_decay)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.mlp.parameters(),
                                 lr=self.conf.lr,
                                 weight_decay=self.conf.weight_decay)
        return optim

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

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.results_path,
                                             name=None,
                                             version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        #max_steps=conf.max_steps // conf.batch_size_effective,
        max_epochs = conf.max_epochs,
        #resume_from_checkpoint=resume,
        devices=gpus,
        accelerator="gpu" if len(gpus) > 0 else "cpu",
        precision=16 if conf.fp16 else 32,
        callbacks=[checkpoint,],
        #replace_sampler_ddp=True,
        logger=[tb_logger, csv_logger],
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
        check_val_every_n_epoch=1,  # Run validation every epoch
        # val_check_interval=0.25     # (Optional) Run validation every 25% of an epoch
    )

    trainer.fit(model)
