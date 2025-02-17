import torch
from torch.utils.data import Dataset, DataLoader
from datasets.images_dataset import ImagesDataset
from configs import data_configs
import h5py
from tqdm import tqdm
import os

def configure_datasets(opts):
    """
    Configures and loads datasets based on `opts.dataset_type`.

    Args:
        opts: Namespace containing dataset configurations.

    Returns:
        Tuple containing (train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset).
    """
    if opts.dataset_type not in data_configs.DATASETS.keys():
        raise ValueError(f"{opts.dataset_type} is not a valid dataset_type")

    print(f"Loading dataset for {opts.dataset_type}")
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args["transforms"](opts).get_transforms()

    train_bg_dataset = ImagesDataset(
        source_root=dataset_args["train_bg_source_root"],
        target_root=dataset_args["train_bg_target_root"],
        source_transform=transforms_dict["transform_source"],
        target_transform=transforms_dict["transform_gt_train"],
        opts=opts,
    )

    train_t_dataset = ImagesDataset(
        source_root=dataset_args["train_t_source_root"],
        target_root=dataset_args["train_t_target_root"],
        source_transform=transforms_dict["transform_source"],
        target_transform=transforms_dict["transform_gt_train"],
        opts=opts,
    )

    test_bg_dataset = ImagesDataset(
        source_root=dataset_args["test_bg_source_root"],
        target_root=dataset_args["test_bg_target_root"],
        source_transform=transforms_dict["transform_source"],
        target_transform=transforms_dict["transform_test"],
        opts=opts,
    )

    test_t_dataset = ImagesDataset(
        source_root=dataset_args["test_t_source_root"],
        target_root=dataset_args["test_t_target_root"],
        source_transform=transforms_dict["transform_source"],
        target_transform=transforms_dict["transform_test"],
        opts=opts,
    )

    print(f"Number of training background samples: {len(train_bg_dataset)}")
    print(f"Number of test target samples: {len(test_bg_dataset)}")

    return train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset


def create_dataloaders(opts, seed=99):
    """
    Creates DataLoaders from configured datasets.

    Args:
        opts: Namespace containing dataloader configurations.

    Returns:
        Tuple containing (train_bg_dataloader, train_t_dataloader, test_bg_dataloader, test_t_dataloader).
    """
    train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset = configure_datasets(opts)
    
    train_bg_dataloader = DataLoader(
        train_bg_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=int(opts.workers),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed)  # Ensure reproducibility for shuffling
    )

    train_t_dataloader = DataLoader(
        train_t_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=int(opts.workers),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed)  # Ensure reproducibility for shuffling
    )

    test_bg_dataloader = DataLoader(
        test_bg_dataset,
        batch_size=opts.test_batch_size,
        shuffle=False,
        num_workers=int(opts.test_workers),
        drop_last=True,
    )

    test_t_dataloader = DataLoader(
        test_t_dataset,
        batch_size=opts.test_batch_size,
        shuffle=False,
        num_workers=int(opts.test_workers),
        drop_last=True,
    )

    return train_bg_dataloader, train_t_dataloader, test_bg_dataloader, test_t_dataloader



def perform_inference_to_hdf5(bg_dataloader, t_dataloader, cs_mlp_net, pSp_net, save_path="latents.h5", device="cuda"):
    """
    Perform inference on background and target images, extract latents using the existing computation flow,
    and save them to an HDF5 file with separate datasets for each latent type.

    Args:
        bg_dataloader (DataLoader): Background images DataLoader.
        t_dataloader (DataLoader): Target images DataLoader.
        cs_mlp_net (nn.Module): The mapping network for latent processing.
        pSp_net (nn.Module): The StyleGAN encoder network.
        save_path (str): Path to save the HDF5 file.
        device (str): Device (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with h5py.File(save_path, "w") as f:
        # Create datasets to store all latents as continuous arrays
        latent_bg_c_list, latent_bg_s_list, w_pSp_bg_list = [], [], []
        latent_t_c_list, latent_t_s_list, w_pSp_t_list = [], [], []
        bg_labels_list, t_labels_list = [], []

        with torch.no_grad():
            # Process background images
            for x_bg, _ in tqdm(bg_dataloader, desc="Processing BG Latents"):
                x_bg = x_bg.to(device).float()

                # Compute latents (your existing computation)
                _, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True)
                latent_bg_c, latent_bg_s = cs_mlp_net(w_bg_pSp, zero_out_silent=False)

                # Append to lists
                latent_bg_c_list.append(latent_bg_c.cpu())
                latent_bg_s_list.append(latent_bg_s.cpu())
                w_pSp_bg_list.append(w_bg_pSp.cpu())
                bg_labels_list.append(torch.zeros(latent_bg_c.shape[0], dtype=torch.long))  # Label 0 for BG

            # Process target images
            for x_t, _ in tqdm(t_dataloader, desc="Processing T Latents"):
                x_t = x_t.to(device).float()

                # Compute latents (your existing computation)
                _, w_t_pSp = pSp_net.forward(x_t, return_latents=True)
                latent_t_c, latent_t_s = cs_mlp_net(w_t_pSp, zero_out_silent=False)

                # Append to lists
                latent_t_c_list.append(latent_t_c.cpu())
                latent_t_s_list.append(latent_t_s.cpu())
                w_pSp_t_list.append(w_t_pSp.cpu())
                t_labels_list.append(torch.ones(latent_t_c.shape[0], dtype=torch.long))  # Label 1 for T

        # Convert lists to tensors
        latent_bg_c_all = torch.cat(latent_bg_c_list, dim=0)
        latent_bg_s_all = torch.cat(latent_bg_s_list, dim=0)
        w_pSp_bg_all = torch.cat(w_pSp_bg_list, dim=0)
        latent_t_c_all = torch.cat(latent_t_c_list, dim=0)
        latent_t_s_all = torch.cat(latent_t_s_list, dim=0)
        w_pSp_t_all = torch.cat(w_pSp_t_list, dim=0)
        bg_labels_all = torch.cat(bg_labels_list, dim=0)
        t_labels_all = torch.cat(t_labels_list, dim=0)

        # Store as HDF5 datasets
        f.create_dataset("latent_bg_c", data=latent_bg_c_all.numpy())
        f.create_dataset("latent_bg_s", data=latent_bg_s_all.numpy())
        f.create_dataset("w_pSp_bg", data=w_pSp_bg_all.numpy())

        f.create_dataset("latent_t_c", data=latent_t_c_all.numpy())
        f.create_dataset("latent_t_s", data=latent_t_s_all.numpy())
        f.create_dataset("w_pSp_t", data=w_pSp_t_all.numpy())

        f.create_dataset("bg_labels", data=bg_labels_all.numpy())
        f.create_dataset("t_labels", data=t_labels_all.numpy())

    print(f"Saved latents to {save_path}.")

class HDF5LatentClassifierDataset(Dataset):
    def __init__(self, hdf5_path, label_0="latent_bg_c", label_1="latent_t_c"):
        """
        Args:
            hdf5_path (str): Path to HDF5 file.
            label_0 (str): Key for background latents (e.g., "latent_bg_c", "latent_bg_s", "w_pSp_bg").
            label_1 (str): Key for target latents (e.g., "latent_t_c", "latent_t_s", "w_pSp_t").
        """
        self.hdf5_path = hdf5_path
        self.label_0 = label_0
        self.label_1 = label_1

        # Open HDF5 to get dataset sizes
        with h5py.File(hdf5_path, "r") as f:
            self.bg_len = f[self.label_0].shape[0]  # Number of BG samples
            self.t_len = f[self.label_1].shape[0]  # Number of T samples
            self.total_len = self.bg_len + self.t_len  # Total samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as f:
            if index < self.bg_len:
                # Load background latent (label = 0)
                latent_c = torch.tensor(f[self.label_0][index], dtype=torch.float32).flatten()
                label = torch.tensor(0, dtype=torch.long)
            else:
                # Load target latent (label = 1)
                target_index = index - self.bg_len  # Adjust index
                latent_c = torch.tensor(f[self.label_1][target_index], dtype=torch.float32).flatten()
                label = torch.tensor(1, dtype=torch.long)

        return latent_c, label  # Shape: (9216,), (1,)

class HDF5LatentDataset(Dataset):
    def __init__(self, hdf5_path, latent_keys):
        """
        Args:
            hdf5_path (str): Path to HDF5 file.
            latent_keys (list): List of keys specifying which latents to return.
        """
        self.hdf5_path = hdf5_path
        self.latent_keys = latent_keys

        # Open HDF5 to get dataset size
        with h5py.File(hdf5_path, "r") as f:
            self.data_size = f[self.latent_keys[0]].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as f:
            latents = [torch.tensor(f[key][index], dtype=torch.float32) for key in self.latent_keys]
        return latents

def reproduce_latent_hdf5(cs_mlp_net, pSp_net, train_hdf5_path, val_hdf5_path, opts, seed, device):
    """
    Collects and saves latents into HDF5 files for classifier training and validation.
    """

    # Create dataloaders
    train_bg_dataloader, train_t_dataloader, test_bg_dataloader, test_t_dataloader = create_dataloaders(opts, seed)
    
    print("Collecting HDF5 for classifier training...")
    perform_inference_to_hdf5(train_bg_dataloader, train_t_dataloader, cs_mlp_net, pSp_net, save_path=train_hdf5_path, device=device)
    # Load and verify training samples
    with h5py.File(train_hdf5_path, "r") as f:
        latent_bg_c = torch.from_numpy(f["latent_bg_c"][:]).to(device)
        latent_t_c = torch.from_numpy(f["latent_t_c"][:]).to(device)
    print(f"Collected {latent_bg_c.shape[0]} bg samples, {latent_t_c.shape[0]} target samples")

    print("Collecting HDF5 for classifier validation...")
    perform_inference_to_hdf5(test_bg_dataloader, test_t_dataloader, cs_mlp_net, pSp_net, save_path=val_hdf5_path, device=device)
    # Load and verify validation samples
    with h5py.File(val_hdf5_path, "r") as f:
        latent_bg_c = torch.from_numpy(f["latent_bg_c"][:]).to(device)
        latent_t_c = torch.from_numpy(f["latent_t_c"][:]).to(device)
    print(f"Collected {latent_bg_c.shape[0]} bg samples, {latent_t_c.shape[0]} target samples")



def get_latent_dataloader(train_hdf5_path, val_hdf5_path, args, label_0="latent_bg_c", label_1="latent_t_c"):
    """
    Loads latent data from HDF5 files and creates DataLoaders.

    Args:
        train_hdf5_path (str): Path to training HDF5 file.
        val_hdf5_path (str): Path to validation HDF5 file.
        args: Argument object containing batch_size and seed.
        label_0 (str): Latent representation for background (default: "latent_bg_c").
        label_1 (str): Latent representation for target (default: "latent_t_c").

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_dataset = HDF5LatentClassifierDataset(train_hdf5_path, label_0=label_0, label_1=label_1)
    val_dataset = HDF5LatentClassifierDataset(val_hdf5_path, label_0=label_0, label_1=label_1)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

