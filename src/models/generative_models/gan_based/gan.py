from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from models.generative_models.gm import GenerativeModel


class Generator(nn.Module):
    def __init__(
        self, latent_dim=100, hidden_dims=[768], output_dim=768, dropout_rate=0.1
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        layers = []
        hidden_dims = [latent_dim] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)

    """
    # TODO: Current implementation is ugly, improve in the future
    def reconstruct(self, real_embs, k, step_size, num_steps):
        # Initialize K random z candidates
        z_candidates = torch.randn(
            real_embs.shape[0], k, self.latent_dim, device=real_embs.device
        )
        z_candidates.requires_grad = True

        optimizer = torch.optim.SGD([z_candidates], lr=step_size)

        # Perform L GD steps
        for step in range(num_steps):
            fake_embs = self.forward(z_candidates)  # batch_size x k x 768
            reconstruction_losses = torch.norm(
                real_embs.unsqueeze(1) - fake_embs, dim=-1
            )  # batch_size x k
            loss = reconstruction_losses.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Find optimal z
        fake_embs = self.forward(z_candidates)  # batch_size x k x 768
        reconstruction_losses = torch.norm(
            real_embs.unsqueeze(1) - fake_embs, dim=-1
        )  # batch_size x k
        argmin = torch.argmin(reconstruction_losses, dim=-1, keepdim=True).unsqueeze(-1)
        argmin = argmin.expand(real_embs.shape[0], 1, real_embs.shape[-1])
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)

        return reconstructed_emb.detach(), reconstruction_losses.detach()
    """

    def reconstruct(self, real_embs, k, **kwargs):
        with torch.no_grad():
            # Method 3: Sample several points from the z prior, choose the optimal one
            batch_size, emb_dim = real_embs.shape
            device = real_embs.device

            # Sample candidate z
            z = torch.randn(batch_size, k, self.latent_dim, device=device)  # (bs, k, latent_dim)
            z = z.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)

            # Compute candidate fake embeddings
            fake_embs = self.forward(z)  # (bs*k, emb_dim)
            fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

            # Choose the optimal fake embedding for each real_emb
            dists = self.dist(fake_embs, real_embs.unsqueeze(1), metric="cosine")  # (bs, k)
            argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
            argmin = argmin.expand(batch_size, 1, emb_dim)
            reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
            reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

            return reconstructed_emb, reconstruction_losses


class Discriminator(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[768], dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_dim] + hidden_dims
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )

        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_emb):
        input_emb = self.input_dropout(input_emb)
        last_emb = self.layers(input_emb)
        logits = self.logit(last_emb)
        probs = self.sigmoid(logits)
        return last_emb, logits, probs


class GAN(GenerativeModel):
    def __init__(
        self,
        latent_dim: int = 100,
        emb_dim: int = 768,
        hidden_dims: List[int] = None,
        g_hidden_dims: List[int] = [768],
        d_hidden_dims: List[int] = [768],
        lr: float = None,
        g_lr: float = 5e-5,
        d_lr: float = 5e-5,
        optimizer: str = None,
        g_optimizer: str = "adam",
        d_optimizer: str = "adam",
        dropout_rate: float = 0.1,
        use_scheduler: bool = False,  
        warmup_proportion: float = 0.1,
        k: int = 30,
        step_size: float = 1.0,
        num_steps: int = 100,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if hidden_dims is not None:
            self.hparams.g_hidden_dims = self.hparams.d_hidden_dims = self.hparams.hidden_dims
        if lr is not None:
            self.hparams.g_lr = self.hparams.d_lr = self.hparams.lr
        if optimizer is not None:
            self.hparams.g_optimizer = self.hparams.d_optimizer = self.hparams.optimizer

        self.gen = Generator(
            self.hparams.latent_dim,
            self.hparams.g_hidden_dims,
            self.hparams.emb_dim,
            self.hparams.dropout_rate,
        )
        self.dis = Discriminator(
            self.hparams.emb_dim, self.hparams.d_hidden_dims, self.hparams.dropout_rate
        )

    def forward(self, z):
        return self.gen(z)

    def reconstruct(self, real_embs, k=None, step_size=None, num_steps=None, **kwargs):
        if k is None: k = self.hparams.k
        if step_size is None: step_size = self.hparams.step_size
        if num_steps is None: num_steps = self.hparams.num_steps
        return self.gen.reconstruct(real_embs, k, step_size, num_steps)

    def accuracy(self, probs, labels):
        preds = torch.round(probs)
        acc = (preds == labels).float().mean()
        return acc

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_embs = batch["clean_emb"]
        batch_size = real_embs.shape[0]

        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.type_as(real_embs)

        # Train the Generator
        if optimizer_idx == 0:
            # Define real label
            real_labels = torch.ones(batch_size, 1)
            real_labels = real_labels.type_as(real_embs)

            # Generate fake embeddings
            fake_embs = self.gen(z)

            # Compute Discriminator's predictions
            _, _, d_fake_probs = self.dis(fake_embs)

            # Compute loss
            g_loss = F.binary_cross_entropy(d_fake_probs, real_labels)

            # Log results
            self.log(f"{self.state}/g_loss", g_loss)

            return {"loss": g_loss}

        # Train the Discriminator
        if optimizer_idx == 1:
            # Define real and fake labels
            real_labels = torch.ones(batch_size, 1)
            real_labels = real_labels.type_as(real_embs)
            fake_labels = torch.zeros(batch_size, 1)
            fake_labels = fake_labels.type_as(real_embs)

            # Generate fake embeddings 
            fake_embs = self.gen(z).detach()

            # Compute Discriminator's predictions
            _, _, d_real_probs = self.dis(real_embs)
            _, _, d_fake_probs = self.dis(fake_embs)

            # Compute loss
            d_real_loss = F.binary_cross_entropy(d_real_probs, real_labels)
            d_fake_loss = F.binary_cross_entropy(d_fake_probs, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Compute Discrimintor's accuracy
            d_real_acc = self.accuracy(d_real_probs, real_labels)
            d_fake_acc = self.accuracy(d_fake_probs, fake_labels)
            d_acc = (d_real_acc + d_fake_acc) / 2

            # Log results
            self.log(f"{self.state}/d_real_loss", d_real_loss)
            self.log(f"{self.state}/d_fake_loss", d_fake_loss)
            self.log(f"{self.state}/d_loss", d_loss)
            self.log(f"{self.state}/d_real_acc", d_real_acc)
            self.log(f"{self.state}/d_fake_acc", d_fake_acc)
            self.log(f"{self.state}/d_acc", d_acc)

            return {"loss": d_loss}

    def configure_optimizers(self):
        if self.hparams.g_optimizer == "adam":
            g_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.g_lr)
        else:
            g_optimizer = torch.optim.SGD(self.gen.parameters(), lr=self.hparams.g_lr)

        if self.hparams.d_optimizer == "adam":
            d_optimizer = torch.optim.Adam(self.dis.parameters(), lr=self.hparams.d_lr)
        else:
            d_optimizer = torch.optim.SGD(self.dis.parameters(), lr=self.hparams.d_lr)

        if self.hparams.use_scheduler:
            num_training_steps = (
                len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
            )
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)
            g_scheduler = get_linear_schedule_with_warmup(
                g_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            d_scheduler = get_linear_schedule_with_warmup(
                d_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

        return [g_optimizer, d_optimizer], []
