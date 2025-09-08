from templates import *


def ffhq256_autoenc_mlp():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.diffusion
    #conf.manipulate_mode = ManipulateMode.celebahq_all
    #conf.results_dir = 'results'
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 8
    conf.n_layers = 12
    conf.lr = 1e-3
    conf.encoder_name = 'ffhq256_autoenc'
    conf.results_path = 'results/baseline/znorm'
    conf.w_factor = 1.0
    # conf.sbg_lambda = 1.0

    # conf.total_samples = 300_000
    conf.max_epochs = 800
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc'
    return conf

