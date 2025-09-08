import os
import h5py
import torch
from tqdm import tqdm
from pathlib import Path
from templates import *
from dataset import ImageDataset  # Assuming your custom dataset is here

device = 'cuda:0'
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

image_folder = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs/test_bg"
hdf5_file_path = "datasets/preprocessed/ffhq_bg_glasses/test_bg.h5"

image_size = conf.img_size
dataset = ImageDataset(image_folder, image_size, exts=['jpg', 'JPG', 'png'], do_augment=False)

with h5py.File(hdf5_file_path, "w") as hdf5_file:
    encodings_group = hdf5_file.create_group("encodings")
    images_group = hdf5_file.create_group("images")
    filename_list = []

    for i in tqdm(range(len(dataset)), desc="Processing Images"):
        sample = dataset[i]
        img_tensor = sample['img']  # [C, H, W]
        batch = img_tensor[None]    # [1, C, H, W]
        filename = sample['filename']

        with torch.no_grad():
            latent = model.encode(batch.to(device))
        latent_np = latent.cpu().numpy()

        # Save encoding
        encodings_group.create_dataset(filename, data=latent_np)

        # Save processed image tensor as float32
        images_group.create_dataset(filename, data=img_tensor.numpy(), dtype='float32')

        filename_list.append(filename)

    hdf5_file.create_dataset("filenames", data=[f.encode() for f in filename_list])

print(f"Encodings and processed images saved to {hdf5_file_path}")
