import torch
import random
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
from IPython.display import display, HTML
class AlignerCantFindFaceError(Exception):
    pass

class MaskerCantFindFaceError(Exception):
    pass


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def tensor2im_no_tfm(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def printer(obj, tabs=0):
    for (key, value) in obj.items():
        try:
            _ = value.items()
            print(" " * tabs + str(key) + ":")
            printer(value, tabs + 4)
        except:
            print(f" " * tabs + str(key) + " : " + str(value))


def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name) + 1] == name + '.'}
    return d_filt


def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)


def visualize_batch_grid(image_batches, titles=None, row_indices=None, save_path=None):
    """
    Show or save a grid of images with aligned column titles.
    Each row = one sample across all versions (original, x_T, rec...).
    
    Args:
        image_batches (List[Tensor]): List of [B, C, H, W] tensors
        titles (List[str]): Column titles (same length as image_batches)
        row_indices (List[int]): Indices from batch to visualize as rows (e.g., [0, 2, 3])
        save_path (str): Optional path to save the image
    """
    def norm(img):
        return (img.clamp(-1, 1) + 1) / 2

    num_versions = len(image_batches)
    batch_size = image_batches[0].shape[0]

    # Default: show top-4 or less if batch is small
    if row_indices is None:
        max_show = min(4, batch_size)
        row_indices = list(range(max_show))

    n_rows = len(row_indices)

    fig, axes = plt.subplots(n_rows, num_versions, figsize=(3 * num_versions, 3 * n_rows))

    if n_rows == 1:
        axes = axes[None, :]  # ensure 2D shape

    for i, row_idx in enumerate(row_indices):
        for j in range(num_versions):
            img = norm(image_batches[j][row_idx]).detach().permute(1, 2, 0).cpu().numpy()
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0 and titles:
                ax.set_title(titles[j], fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_batch_grid_adjust(image_batches, titles=None, row_indices=None, save_path=None,
                         wspace=0.0, hspace=0.05, fontsize=14):
    """
    Show or save a grid of images with aligned column titles.
    Each row = one sample across all versions (original, x_T, rec...).

    Args:
        image_batches (List[Tensor]): List of [B, C, H, W] tensors
        titles (List[str]): Column titles (same length as image_batches)
        row_indices (List[int]): Indices from batch to visualize as rows (e.g., [0, 2, 3])
        save_path (str): Optional path to save the image
        wspace (float): Horizontal (column) spacing
        hspace (float): Vertical (row) spacing
    """
    def norm(img):
        return (img.clamp(-1, 1) + 1) / 2

    num_versions = len(image_batches)
    batch_size = image_batches[0].shape[0]

    if row_indices is None:
        max_show = min(4, batch_size)
        row_indices = list(range(max_show))

    n_rows = len(row_indices)

    fig, axes = plt.subplots(n_rows, num_versions, figsize=(3 * num_versions, 3 * n_rows))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)  # ✅ Control layout spacing

    if n_rows == 1:
        axes = axes[None, :]  # ensure 2D shape

    for i, row_idx in enumerate(row_indices):
        for j in range(num_versions):
            img = norm(image_batches[j][row_idx]).detach().permute(1, 2, 0).cpu().numpy()
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0 and titles:
                ax.set_title(titles[j], fontsize=fontsize)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def display_and_save_images(images, img_size=(256, 256), save_path=None, filename="combined_image.png", padding=20):
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

    # Display inline in Jupyter Notebook
    buffer = BytesIO()
    combined_image.save(buffer, format="PNG")  
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    display(HTML(f"<img src='data:image/png;base64,{img_str}' style='margin:5px;'>"))



def process_to_images(image_folder):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

    import random
    from PIL import Image
    # Randomly select 'num_images' from the image_paths
    selected_paths = random.sample(image_paths, min(4, len(image_paths)))

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
        