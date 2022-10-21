from .gan_based import *
from .vae_based import *


class AutoGenerativeModel:

    GENERATIVE_MODEL_MAPPING_NAMES = {
        "gan": GAN,
        "wgan_gp": WGANGP,
        "dcgan": DCGAN,
        "mgan": MGAN,
        "infogan": InfoGAN,
        "infogan_large": InfoGANLarge,
        "vae": VAE,
        "vae_gan": VAEGAN,
    }

    @classmethod
    def get_class_name(cls, gm_name):
        return cls.GENERATIVE_MODEL_MAPPING_NAMES[gm_name]
