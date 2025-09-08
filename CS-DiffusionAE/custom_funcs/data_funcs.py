import torch
from dataset import LatentDataset, LMDBDataset  
from torch.utils.data import DataLoader

def get_dataloaders(args):
    """Initialize and return train and validation dataloaders."""
    train_dataset = LatentDataset(
        "datasets/encoded_latents_ffhq/encoded_ffhq_train_bg.h5",
        "datasets/encoded_latents_ffhq/encoded_ffhq_train_glass.h5"
    )

    val_dataset = LatentDataset(
        "datasets/encoded_latents_ffhq/encoded_ffhq_test_bg.h5",
        "datasets/encoded_latents_ffhq/encoded_ffhq_test_glass.h5"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def config_dataset_path(dataset_type="ffhq"):
    
    if dataset_type=="ffhq":
        train_path_bg = "datasets/preprocessed/ffhq_bg_glasses/train_bg.lmdb"
        train_path_t = "datasets/preprocessed/ffhq_bg_glasses/train_t.lmdb"
        val_path_bg = "datasets/preprocessed/ffhq_bg_glasses/test_bg.lmdb"
        val_path_t = "datasets/preprocessed/ffhq_bg_glasses/test_t.lmdb" 

    return train_path_bg, train_path_t, val_path_bg, val_path_t


def load_lmdb_dataset(lmdb_path_bg, lmdb_path_t, load_image=True, shuffle=True):
    """Initialize and return train and validation dataloaders."""

    dataset_bg = LMDBDataset(lmdb_path_bg, load_image=load_image)
    dataset_t = LMDBDataset(lmdb_path_t, load_image=load_image)
    loader_bg = DataLoader(dataset_bg, batch_size=4, shuffle=shuffle, num_workers=4)
    loader_t = DataLoader(dataset_t, batch_size=4, shuffle=shuffle, num_workers=4)

    return loader_bg, loader_t



