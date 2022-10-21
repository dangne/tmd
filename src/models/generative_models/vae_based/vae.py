from logging import log
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from models.generative_models.gm import GenerativeModel


class VAE(GenerativeModel):
    def __init__(
        self,
        latent_dim: int = 100,
        emb_dim: int = 768,
        hidden_dims: List[int] = [768],
        beta: float = 1.0,
        lr: float = 5e-5,
        dropout_rate: float = 0.1,
        use_scheduler: bool = False,
        warmup_proportion: float = 0.1,
        **kwargs,
    ):

        super().__init__()

        self.save_hyperparameters()

        # Build Encoder
        hidden_dims = [emb_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        hidden_dims.reverse()
        hidden_dims = [latent_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        self.decoder(z)
    
    def encode(self, real_embs):
        hidden_rep = self.encoder(real_embs)
        mu = self.fc_mu(hidden_rep)
        logvar = self.fc_var(hidden_rep)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def sample(self, mu, logvar):
        z = self.sample_z(mu, logvar)
        return self.decoder(z)

    def reconstruct(self, real_embs, **kwargs):
        fake_embs = self.decode(self.sample_z(*self.encode(real_embs)))
        rec_losses = torch.norm(real_embs - fake_embs, dim=-1)
        return fake_embs, rec_losses

    def training_step(self, batch, batch_idx):
        real_embs = batch["clean_emb"]

        # Forward
        mu, logvar = self.encode(real_embs)
        fake_embs = self.decode(self.sample_z(mu, logvar))

        # Compute loss 
        rec_loss = torch.norm(fake_embs - real_embs, dim=-1).mean()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = rec_loss + self.hparams.beta * kld_loss

        # Update FTD state
        self.train_ftd.update(real_embs, fake_embs)

        # Log results
        self.log(f"{self.state}/loss", loss)
        self.log(f"{self.state}/rec_loss", rec_loss)
        self.log(f"{self.state}/kld_loss", kld_loss)
        self.log(f"{self.state}/ftd", self.train_ftd, on_step=False, on_epoch=True)
    
        return {"loss": loss}

    def configure_optimizers(self):
        if self.hparams.e_opt == "adam":
            e_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.e_lr)
        else:
            e_opt = torch.optim.SGD(self.encoder.parameters(), lr=self.hparams.e_lr)

        if self.hparams.d_opt == "adam":
            d_opt = torch.optim.Adam(self.decoder.parameters(), lr=self.hparams.d_lr)
        else:
            d_opt = torch.optim.SGD(self.decoder.parameters(), lr=self.hparams.d_lr)

        if self.hparams.dis_opt == "adam":
            dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.dis_lr)
        else:
            dis_opt = torch.optim.SGD(self.discriminator.parameters(), lr=self.hparams.dis_lr)

        if self.hparams.use_scheduler:
            num_training_steps = (
                len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
            )
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)
            e_sched = get_linear_schedule_with_warmup(
                e_opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            d_sched = get_linear_schedule_with_warmup(
                d_opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            dis_sched = get_linear_schedule_with_warmup(
                dis_opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            return [e_opt, d_opt, dis_opt], [e_sched, d_sched, dis_sched] 
        return [e_opt, d_opt, dis_opt], [] 