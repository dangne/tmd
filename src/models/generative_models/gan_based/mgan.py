from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from .gan import GAN, Generator, Discriminator


class MGAN(GAN):
    def __init__(
        self,
        latent_dim: int = 100,
        emb_dim: int = 768,
        g_hidden_dims: List[int] = [768],
        d_hidden_dims: List[int] = [768],
        g_lr: float = 5e-5,
        d_lr: float = 5e-5,
        dropout_rate: float = 0.1,
        use_scheduler: bool = False,  
        warmup_proportion: float = 0.1,
        k: int = 10,
        step_size: float = 1.0,
        num_steps: int = 100,
        **kwargs,
    ):

        super().__init__()

        self.save_hyperparameters()

        self.gen = Generator(
            self.hparams.latent_dim,
            self.hparams.g_hidden_dims,
            self.hparams.emb_dim,
            self.hparams.dropout_rate,
        )
        self.dis = Discriminator(
            self.hparams.emb_dim, self.hparams.d_hidden_dims, self.hparams.dropout_rate
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError