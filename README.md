# CS-StyleGAN

**Learning Common and Salient Generative Factors Between Two Image Datasets**

This repository contains the official PyTorch implementation for CS-StyleGAN, as described in our paper:
*\[Learning Common and Salient Generative Factors Between Two Image Datasets]*

---

## Overview

CS-StyleGAN learns *common* and *salient* features between two image datasets (for example, faces with and without glasses).

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

*(Generate this with `pip freeze > requirements.txt` after installing your environment.)*

---

## Dataset Setup

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

## Pretrained Models

Download the pretrained models:

| Path                                                                                                              | Description                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [Pretrained pSp](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing)              | pSp model trained on FFHQ                                                                                   |
| [Pretrained StyleGAN2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing)        | StyleGAN2 pretrained on FFHQ ([rosinality implementation](https://github.com/rosinality/stylegan2-pytorch)) |
| [Pretrained pSp\_on\_BraTS](https://drive.google.com/file/d/1nqXMxZV4B_W5GTRE-pk6iTc3wkswgNd_/view?usp=sharing)   | pSp trained on BraTS2023                                                                                    |
| [Pretrained StyleGAN2\_BraTS](https://drive.google.com/file/d/1KjEzuKW4-t62EuyRhJIOVChXh8q1WoaV/view?usp=sharing) | StyleGAN2 pretrained on BraTS2023                                                                           |

---

## Training

Example command:

```bash
python training_scripts/train.py \
    --dataset_type=ffhq_glasses \
    --stylegan_weights=./psp_ffhq_encode.pt \
    --pSp_checkpoint_path=./stylegan2-ffhq1024.pt \
    --exp_dir=results/baseline/
```

---

## Results

Additional results and qualitative examples are available in the `results/` directory and in the Supplementary Material.
Our method can transfer detailed characteristics (such as the shape or shade of sunglasses) between datasets.
For quantitative results, please see the main paper and Supplementary Material.

---

## Citation

If you use this code, please cite our work:


---

## Contact

For questions or issues, please open an issue or contact:


---

Let me know if you want an **inference/evaluation** section, a troubleshooting guide, or anything else added!
