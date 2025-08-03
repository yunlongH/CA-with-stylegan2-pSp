import os
import torch
import matplotlib.pyplot as plt
from dataset import SingleImageDataset
from PIL import Image, ImageDraw
from IPython.display import display, HTML
from io import BytesIO
import base64
import random
import numpy as np


def tensor2im(var):
    """
    Converts a PyTorch tensor to a PIL image.
    Automatically detects and rescales from [-1, 1] or [0, 1] to [0, 255].
    """
    var = var.cpu().detach()
    var = var.clamp(0, 1) if var.min() >= 0 else var.clamp(-1, 1)
    if var.min() >= 0:
        var = var
    else:
        var = (var + 1) / 2.0  # map [-1,1] -> [0,1]
    
    var = var.transpose(0, 2).transpose(0, 1).numpy()
    var = (var * 255.0).astype(np.uint8)
    return Image.fromarray(var)


def display_and_save_images(images, img_size=(256, 256), save_path=None, filename="combined_image.png", padding=20, is_display=True):
    """
    Displays multiple images in a single row with white spacing and saves them as a single combined image.

    Args:
        images: List of PIL.Image or tensors to be displayed and saved.
        img_size: Tuple (width, height) for resizing images (default: 256x256).
        save_path: Directory path to save the combined image. If None, image is not saved.
        filename: Name of the saved image file (default: "combined_image.png").
        padding: Space (in pixels) between images (default: 20px).
    """
    # Convert tensors to PIL if needed
    images = [tensor2im(img) if not isinstance(img, Image.Image) else img for img in images]
    
    # Resize all images
    images = [img.resize(img_size) for img in images]

    # Compute the final size including padding
    total_width = sum(img.width for img in images) + (padding * (len(images) - 1))
    max_height = max(img.height for img in images)

    # Create a blank white image
    combined_image = Image.new("RGB", (total_width, max_height), "white")

    # Paste images with padding
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + padding

    # Save the combined image if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        save_filepath = os.path.join(save_path, filename)
        combined_image.save(save_filepath)
        print(f"Saved combined image: {save_filepath}")
    if is_display:
        # Display inline in Jupyter Notebook
        buffer = BytesIO()
        combined_image.save(buffer, format="PNG")  
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        display(HTML(f"<img src='data:image/png;base64,{img_str}' style='margin:5px;'>"))


def save_single_image(img_tensor, save_path, filename, img_size=(256, 256)):
    """
    Save a single tensor as a clean image for FID computation.
    """
    img = tensor2im(img_tensor).resize(img_size)
    os.makedirs(save_path, exist_ok=True)
    img.save(os.path.join(save_path, filename))


import os
import subprocess

def set_notebook_env_ids(CODE_DIR, cuda_version="12.5"):
    # Save original PATH and LD_LIBRARY_PATH
    original_path = os.environ.get("PATH", "")
    original_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    # Set CUDA environment variables
    os.environ["CUDA_HOME"] = f"/usr/local/cuda-{cuda_version}"
    os.environ["PATH"] = f"/usr/local/cuda-{cuda_version}/bin:{original_path}"
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda-{cuda_version}/lib64:{original_ld_library_path}"

    print(f"CUDA environment variables set for CUDA {cuda_version}")

    # Query GPU type
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
    except Exception as e:
        print(f"Error querying GPU: {e}")
        return

    # GPU to arch mapping
    arch_map = {
        "P100": "6.0",
        "V100": "7.0",
        "A100": "8.0",
        "A40": "8.6",
        "L40": "8.9",
    }

    # Set TORCH_CUDA_ARCH_LIST
    torch_arch = None
    for key, arch in arch_map.items():
        if key in gpu_name:
            torch_arch = arch
            break

    if torch_arch is None:
        print(f"Unknown GPU: {gpu_name}")
        return

    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch
    print(f"Set TORCH_CUDA_ARCH_LIST to {torch_arch}")

    # Change to CODE_DIR
    os.chdir(CODE_DIR)
    notebook_path = os.getcwd()
    print('Current working directory is:\n', notebook_path)

