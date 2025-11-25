
# 📘 CS-StyleGAN & CS-DiffusionAE  
### Official PyTorch Implementation of  
**Learning Common and Salient Generative Factors Between Two Image Datasets**

<p align="center">
  <img src="CA_problems_examples.png" width="800px"/>
  <br>
  Our framework learns <i>common</i> and <i>salient</i> generative factors for 
  <b><span style="color:#3467eb;">Contrastive Analysis (CA)</span></b> on high-quality images.
  It handles not only the classic <b>Background / Target</b> setting, 
  but also harder scenarios such as 
  <b>Multiple-Attributes</b> and <b>Multiple-Salient</b> domains.
</p>

---

# ⭐ Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Pretrained Models](#pretrained-models)
- [CS-StyleGAN Training](#cs-stylegan-training)
- [CS-DiffusionAE](#cs-diffusionae)
- [CS-Asyrp (h-space)](#cs-asyrp-h-space)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Contact](#contact)

---

# 🧠 Introduction

This repository contains the official PyTorch implementation of **CS-StyleGAN** and **CS-DiffusionAE**, two generative models designed to *separate common and salient generative factors* between two related datasets.

Given:
- **X (background)** dataset  
- **Y (target)** dataset  

Our models learn:
- **C** → common latent factors shared by X and Y  
- **S** → salient latent factors unique to X or Y  

Applications include:
- Salient attribute transfer  
- Controlled semantic editing  
- Contrastive generative analysis

---

# 🔧 Installation

Install dependencies with:

```bash
pip install -r requirements.txt


### Dataset Setup

Before training, set the dataset paths in `./configs/paths_config.py`.
You need four folders:

* Background (X) images for training
* Target (Y) images for training
* Background (X) images for testing
* Target (Y) images for testing

Supported input: **.png** images

**Example (`paths_config.py`):**

```python
dataset_paths = {
    'ffhq_bg_train': 'path/to/your/background/train/images',
    'ffhq_glass_train': 'path/to/your/target/train/images',
    'ffhq_bg_test': 'path/to/your/background/test/images',
    'ffhq_glass_test': 'path/to/your/target/test/images',
}
```

*Here, `ffhq_bg` and `ffhq_glass` could refer to FFHQ images without and with glasses, respectively.*

---

### Pretrained Models

Download the pretrained models:

| Path                                                                                                              | Description                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [Pretrained pSp](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing)              | pSp model trained on FFHQ                                                                                   |
| [Pretrained StyleGAN2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing)        | StyleGAN2 pretrained on FFHQ ([rosinality implementation](https://github.com/rosinality/stylegan2-pytorch)) |
| [Pretrained pSp\_on\_BraTS](https://drive.google.com/file/d/1nqXMxZV4B_W5GTRE-pk6iTc3wkswgNd_/view?usp=sharing)   | pSp trained on BraTS2023                                                                                    |
| [Pretrained StyleGAN2\_BraTS](https://drive.google.com/file/d/1KjEzuKW4-t62EuyRhJIOVChXh8q1WoaV/view?usp=sharing) | StyleGAN2 pretrained on BraTS2023                                                                           |

For more pretrained models used in the paper, please contact to the author via yunlong.he@telecom-paris.fr.
---

### Training stage 1:

Example command:

```bash
python training_scripts/train.py \
    --dataset_type=ffhq_glasses \
    --stylegan_weights=./psp_ffhq_encode.pt \
    --pSp_checkpoint_path=./stylegan2-ffhq1024.pt \
    --exp_dir=results/baseline/
```

---

### Training stage 2:

Example command:

```bash
cd ./F_space refinement \
python scripts/train.py \
    exp.exp_dir=./experiments/ \
    data.dataset=ffhq_glasses \
    exp.config_dir=configs \
    exp.config=fse_cs_editor_train.yaml \
    exp.name=fse_cs_editor_train/pSp_encoder/ablation/ffhq_glasses_cs1s2 \
    methods_args.fse_full.inverter_pth=./pretrained_models/sfe_inverter_light.pt \
    train.train_runner=fse_editor_cs1s2 \
    train.start_step=300000 \
    train.direction=two_directions \
    train.log_step=2000 \
    train.val_step=2000 \
    train.checkpoint_step=10000 \
    data.special_idx=0 \
    model.w_space_encoder=pSp \
    model.stylegan_size=1024 \
```

---

## CS-DiffusionAE

The code for training the **CS-DiffusionAE** model can be found in `./CS-DiffusionAE/`.

### Dataset

For faster training, it is recommended to preprocess the image data into **LMDB** format using:

```bash
python ./CS-DiffusionAE/train_cs/preprocess_lmdb.py
```

Otherwise, you can also use datasets in other formats (e.g., .png image folders) by modifying the dataset input part in `./CS-DiffusionAE/train_cs/train_common_salient_baseline.py`.

### Training
First, download the pretrained models from [diffae](https://github.com/phizaz/diffae).

Example command for training **without image losses** (faster training but suboptimal results):

```bash
cd ./F_space refinement \
# train with only latent loss (baseline)
python train_cs/train_common_salient_baseline.py \
    --results_dir=./results/layers2/lr=0.0001 \
    --n_layers=2 \
    --learning_rate=0.0001 \
    --w_bg=10.0 \
    --w_t=10.0 \
    --w_sbg=10.0 \
```

Alternatively, modify `train_common_salient_full.py` to train the CS model **with image losses**.  
In this case, you may need fewer training steps (around **4000**) to achieve good performance.

## CS-Asyrp (h-space of U-Net)

All codes and implementations about CS-Diffusion on h-space be found at https://github.com/ZiqianLiu666/Asyrp-h_space.

## Acknowledgments
This code borrows heavily from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), [StyleFeatureEditor](https://github.com/ControlGenAI/StyleFeatureEditor), [Asyrp](https://github.com/kwonminki/Asyrp_official), and [diffae](https://github.com/phizaz/diffae)

---

## Contact
If you have any questions about the code, implementation details, or pretrained models, please feel free to contact me at **ylh.icandoit@gmail.com**.


---
