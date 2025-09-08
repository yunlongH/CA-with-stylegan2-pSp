import os
import h5py
import torch
from tqdm import tqdm
from pathlib import Path
from templates import *


device = 'cuda:0'
conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)


image_folder = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_cs/test_bg"
hdf5_file_path = "datasets/encoded_ffhq_test_bg.h5"

image_size = conf.img_size  # Use the configured image size
dataset = ImageDataset(image_folder, image_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
# Create HDF5 file


with h5py.File(hdf5_file_path, "w") as hdf5_file:
    encodings_group = hdf5_file.create_group("encodings")
    filename_list = []

    for i in tqdm(range(len(dataset)), desc="Processing Images"):
        sample = dataset[i]  # Load image
        batch = sample['img'][None]  # Convert to batch format
        filename = sample['filename']  # Get filename

        # Encode the image using the model
        with torch.no_grad():
            latent = model.encode(batch.to(device))  # Send to GPU
            #print(latent.shape)  # Debugging step to see what it returns
        # Convert tensor to numpy for saving
        cond_numpy = latent.cpu().numpy()

        # Save encoding using filename as key
        encodings_group.create_dataset(filename, data=cond_numpy)
        filename_list.append(filename)

    # Save filenames separately (optional)
    hdf5_file.create_dataset("filenames", data=[f.encode() for f in filename_list])

print(f"Encodings saved in {hdf5_file_path}")