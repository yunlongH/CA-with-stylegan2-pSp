from templates_mlp_latent import *
from experiment_mlp_latent import *

if __name__ == '__main__':
    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = [0]
    conf = ffhq256_autoenc_mlp()
    train_mlp(conf, gpus=gpus)

    # after this you can do the manipulation!
