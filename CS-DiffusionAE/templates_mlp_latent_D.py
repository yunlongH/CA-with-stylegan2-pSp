from templates import *


def ffhq256_autoenc_mlp():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.diffusion
    #conf.manipulate_mode = ManipulateMode.celebahq_all
    #conf.results_dir = 'results'
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 4
    conf.n_layers = 8
    conf.lr_mlp = 1e-2
    conf.lr_d = 1e-2
    conf.encoder_name = 'ffhq256_autoenc'
    conf.results_path = 'results/with_adv/normalize/bs4'
    conf.w_factor = 1.0
    conf.adv_weight=0.5
    # conf.total_samples = 300_000
    conf.max_epochs = 600
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc'
    return conf

