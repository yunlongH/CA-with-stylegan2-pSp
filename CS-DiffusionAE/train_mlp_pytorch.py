import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import csv
from config import *
from dataset import *
from torch.utils.data import DataLoader
import copy


# Equalized Linear Layer
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 1 / in_features**0.5)
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


# Mapping Network
class MappingNetwork_cs(nn.Module):
    def __init__(self, features, n_layers):
        super().__init__()
        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z):
        return self.net_c(z), self.net_s(z)


# MLP Model
class MlpModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.mlp = MappingNetwork_cs(features=conf.style_ch, n_layers=conf.n_layers)
        self.ema_mlp = copy.deepcopy(self.mlp)
        self.mse_loss = nn.MSELoss()
        self.ema_decay = conf.ema_decay

    def forward(self, cond_bg, cond_t):
        c_bg, s_bg = self.mlp(cond_bg)
        c_t, s_t = self.mlp(cond_t)
        return c_bg, s_bg, c_t, s_t

    def update_ema(self):
        for ema_param, param in zip(self.ema_mlp.parameters(), self.mlp.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data * (1 - self.ema_decay))



def train(conf, model, dataloaders, device):
    train_loader, val_loader = dataloaders
    optimizer = optim.Adam(model.mlp.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # Prepare log directories
    train_log_path = os.path.join(conf.logdir, "train_logs.csv")
    val_log_path = os.path.join(conf.logdir, "val_logs.csv")
    os.makedirs(conf.logdir, exist_ok=True)

    # CSV headers
    with open(train_log_path, 'w') as f:
        csv.writer(f).writerow(['epoch', 'train_loss_bg', 'train_loss_t', 'train_loss'])

    with open(val_log_path, 'w') as f:
        csv.writer(f).writerow(['epoch', 'val_loss_bg', 'val_loss_t'])

    # Training Loop
    for epoch in range(conf.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            bg_imgs, t_imgs = batch
            cond_bg, cond_t = bg_imgs.to(device), t_imgs.to(device)

            optimizer.zero_grad()
            c_bg, s_bg, c_t, s_t = model(cond_bg, cond_t)

            # Compute loss
            loss_bg = model.mse_loss(c_bg, cond_bg)
            loss_t = model.mse_loss(c_t + s_t, cond_t)
            loss_sbg = model.mse_loss(s_bg, torch.zeros_like(s_bg))
            loss = loss_bg + loss_t + loss_sbg

            loss.backward()
            optimizer.step()
            model.update_ema()  # Update EMA model

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")

        # Save train loss
        with open(train_log_path, 'a') as f:
            csv.writer(f).writerow([epoch, loss_bg.item(), loss_t.item(), avg_train_loss])

        # Validation Step
        validate(model, val_loader, device, epoch, val_log_path, conf)


def validate(model, val_loader, device, epoch, val_log_path, conf):
    model.eval()
    val_loss_bg, val_loss_t = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            bg_imgs, t_imgs = batch
            cond_bg, cond_t = bg_imgs.to(device), t_imgs.to(device)

            c_bg, s_bg, c_t, s_t = model(cond_bg, cond_t)
            val_loss_bg += model.mse_loss(c_bg, cond_bg).item()
            val_loss_t += model.mse_loss(c_t + s_t, cond_t).item()

        val_loss_bg /= len(val_loader)
        val_loss_t /= len(val_loader)
        print(f"Epoch {epoch}, Val Loss BG: {val_loss_bg:.4f}, Val Loss T: {val_loss_t:.4f}")

        # Save validation loss
        with open(val_log_path, 'a') as f:
            csv.writer(f).writerow([epoch, val_loss_bg, val_loss_t])


def get_dataloaders(conf):
    train_dataset_bg, train_dataset_t = conf.make_mlp_dataset(
        path_bg="datasets/ffhq256_mlp/ffhq256_bg.lmdb",
        path_t="datasets/ffhq256_mlp/ffhq256_glass.lmdb"
    )

    val_dataset_bg, val_dataset_t = conf.make_mlp_dataset(
        path_bg="datasets/ffhq256_mlp/ffhq256_test_bg.lmdb",
        path_t="datasets/ffhq256_mlp/ffhq256_test_glass.lmdb"
    )

    train_loader = DataLoader(
        list(zip(train_dataset_bg, train_dataset_t)), batch_size=conf.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        list(zip(val_dataset_bg, val_dataset_t)), batch_size=conf.batch_size, shuffle=False
    )

    return train_loader, val_loader



if __name__ == "__main__":
    conf = TrainConfig()  # Replace with your configuration setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MlpModel(conf).to(device)
    dataloaders = get_dataloaders(conf)

    train(conf, model, dataloaders, device)
