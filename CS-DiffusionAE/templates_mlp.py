from templates import *


# def ffhq128_autoenc_cls():
#     conf = ffhq128_autoenc_130M()
#     conf.train_mode = TrainMode.manipulate
#     conf.manipulate_mode = ManipulateMode.celebahq_all
#     conf.manipulate_znormalize = True
#     conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
#     conf.batch_size = 32
#     conf.lr = 1e-3
#     conf.total_samples = 300_000
#     # use the pretraining trick instead of contiuning trick
#     conf.pretrain = PretrainConfig(
#         '130M',
#         f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
#     )
#     conf.name = 'ffhq128_autoenc_cls'
#     return conf


# def ffhq256_autoenc_cls():
#     '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
#     conf = ffhq256_autoenc()
#     conf.train_mode = TrainMode.manipulate
#     conf.manipulate_mode = ManipulateMode.celebahq_all
#     conf.manipulate_znormalize = True
#     conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
#     conf.batch_size = 32
#     conf.lr = 1e-3
#     conf.total_samples = 300_000
#     # use the pretraining trick instead of contiuning trick
#     conf.pretrain = PretrainConfig(
#         '130M',
#         f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
#     )
#     conf.name = 'ffhq256_autoenc_cls'
#     return conf

def ffhq256_autoenc_mlp():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.diffusion
    #conf.manipulate_mode = ManipulateMode.celebahq_all
    #conf.results_dir = 'results'
    conf.manipulate_znormalize = False
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 4
    conf.n_layers = 12
    conf.lr = 1e-2
    conf.encoder_name = 'ffhq256_autoenc'
    conf.results_path = 'results/new_version/orn'
    conf.scaled_s=False
    conf.scale_s_type = 'tanh'
    conf.train_on_imgs=False
    conf.save_image_interval = 1
    conf.lpips_lambda=0.8
    conf.id_lambda = 0.4
    conf.pix_lambda = 1.0
    conf.w_factor = 1.0
    conf.sbg_lambda = 1.0
    conf.scaled_s_factor=0.3
    conf.scaled_silent=False

    # conf.total_samples = 300_000
    conf.max_epochs = 800
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc'
    return conf


    # - scale_type (str): The scaling method to use. Options include:
    #     "max": Scale by the maximum absolute value.
    #     "clip": Clip values to [-1, 1].
    #     "tanh": Apply tanh to scale values to [-1, 1].
    #     "zscore": Normalize to zero mean and unit variance.
    #     "l2": Normalize by the L2 norm.