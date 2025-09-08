import os
import lmdb
import torch
import pickle
from tqdm import tqdm
from pathlib import Path

# from templates import *
from dataset import ImageDataset
from custom_funcs.model_funcs import load_diffusion_model

device = 'cuda:0'
model, conf = load_diffusion_model(device)

# conf = ffhq256_autoenc()
# model = LitModel(conf)
# state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
# model.load_state_dict(state['state_dict'], strict=False)
# model.ema_model.eval()
# model.ema_model.to(device)

image_folder = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs/test_bg"
lmdb_path = "datasets/preprocessed/ffhq_bg_glasses/test_bg.lmdb"

image_size = conf.img_size
dataset = ImageDataset(image_folder, image_size, exts=['jpg', 'JPG', 'png'], do_augment=False)

# Estimate map size (safely overestimate to 10GB)
map_size = 10 * 1024 * 1024 * 1024  # 10 GB

env = lmdb.open(lmdb_path, map_size=map_size)

with env.begin(write=True) as txn:
    for i in tqdm(range(len(dataset)), desc="Processing Images"):
        sample = dataset[i]
        img_tensor = sample['img']       # [C, H, W]
        batch = img_tensor[None]         # [1, C, H, W]
        filename = sample['filename']    # e.g. '00012.png'

        with torch.no_grad():
            latent = model.encode(batch.to(device))

        entry = {
            'filename': filename,
            'img': img_tensor.numpy(),             # float32 [C, H, W]
            'latent': latent.cpu().numpy(),        # whatever shape your latent is
        }

        txn.put(key=filename.encode(), value=pickle.dumps(entry))

print(f"Encodings and processed images saved to {lmdb_path}")
