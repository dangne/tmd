from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from scipy.stats import truncnorm

from .gan import GAN, Generator


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

    def dist(self, x, y, metric="cosine"):
        if metric == "cosine":
            return 1-F.cosine_similarity(x, y, dim=-1)
        elif metric == "l2":
            return torch.norm(x - y, dim=-1)
        else:
            raise ValueError(f"metric {metric} is not supported")

    # TODO: Current implementation is ugly, improve in the future
    def reconstruct1(self, real_embs, k, step_size, num_steps, **kwargs):
        # Initialize K random z candidates
        with torch.enable_grad():
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

        # TODO: We should not detach in here. It should be base on the caller to disable/enable grad.
        return reconstructed_emb.detach(), reconstruction_losses.detach()

    def reconstruct2(self, real_embs, k, **kwargs):
        raise NotImplementedError

    def reconstruct3(self, real_embs, k, **kwargs):
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

    def reconstruct4(self, real_embs, k, threshold=1, **kwargs):
        # Method 4: Sample several z from the truncated normal, choose the optimal one
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        # Sample candidate z
        z = truncnorm.rvs(-threshold, threshold, size=(batch_size, k, self.latent_dim))  # (bs, k, latent_dim)
        z = torch.as_tensor(z, dtype=torch.float32, device=device)
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

    def reconstruct5(self, real_embs, k=1, threshold=1, **kwargs):
        # Method 5: Sample 1 random z from the truncated normal (not choosing the optimal)
        assert k == 1, f"k must be 1 for this method. Got k={k}"
        return self.reconstruct4(real_embs, k=1, threshold=threshold)

    def reconstruct(self, real_embs, k, method=1, **kwargs):
        # There are several ways to find the best z
        # Method 1: Use PGD where z is boundded in the region of a normal distribution
        # Method 2: Use PGD where z is guided with the underlying prior distribution?
        # Method 3: Sample several z from the normal prior, choose the optimal one
        # Method 4: Sample several z from the truncated normal prior, choose the optimal one
        # Method 5: Sample a single z from the truncated normal prior

        with torch.no_grad():
            if method == 1:
                return self.reconstruct1(real_embs, k, **kwargs)
            elif method == 2:
                return self.reconstruct2(real_embs, k, **kwargs)
            elif method == 3:
                return self.reconstruct3(real_embs, k, **kwargs) 
            elif method == 4:
                return self.reconstruct4(real_embs, k, **kwargs) 
            elif method == 5:
                return self.reconstruct5(real_embs, k, **kwargs) 
            else:
                raise ValueError(f"method {method} is not supported")


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
        self.last_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, input_emb):
        input_emb = self.input_dropout(input_emb)
        last_emb = self.layers(input_emb)
        score = self.last_layer(last_emb)
        return last_emb, score


class WGANGP(GAN):
    def __init__(
        self,
        latent_dim: int = 100,
        emb_dim: int = 768,
        hidden_dims: List[int] = [768],
        g_hidden_dims: List[int] = [768],
        d_hidden_dims: List[int] = [768],
        lr: float = None,
        g_lr: float = 5e-5,
        d_lr: float = 5e-5,
        gp_weight: float = 10,
        dropout_rate: float = 0.1,
        use_scheduler: bool = False,
        warmup_proportion: float = 0.1,
        method: int = 3,
        k: int = 10,
        step_size: float = 1.0,
        num_steps: int = 100,
        threshold: float = 1.0,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.save_hyperparameters()

        if hidden_dims is not None:
            self.hparams.g_hidden_dims = self.hparams.d_hidden_dims = self.hparams.hidden_dims
        if lr is not None:
            self.hparams.g_lr = self.hparams.d_lr = self.hparams.lr

        self.gen = Generator(
            self.hparams.latent_dim,
            self.hparams.g_hidden_dims,
            self.hparams.emb_dim,
            self.hparams.dropout_rate,
        )
        self.dis = Discriminator(
            self.hparams.emb_dim, self.hparams.d_hidden_dims, self.hparams.dropout_rate
        )

    def reconstruct(self, real_embs, k=None, step_size=None, num_steps=None, method=None, threshold=None, **kwargs):
        if k is None: k = self.hparams.k
        if step_size is None: step_size = self.hparams.step_size
        if num_steps is None: num_steps = self.hparams.num_steps
        if method is None: method = self.hparams.method
        if threshold is None: threshold = self.hparams.threshold

        return self.gen.reconstruct(real_embs, k, method, step_size=step_size, num_steps=num_steps, threshold=threshold)

    def gp_loss(self, real_embs, fake_embs):
        eps = torch.rand(real_embs.shape[0], 1, device=self.device)
        interpolate_embs = eps * real_embs + (1 - eps) * fake_embs
        interpolate_embs.requires_grad_(True)
        _, d_interpolate_scores = self.dis(interpolate_embs)
        grads = torch.autograd.grad(
            d_interpolate_scores,
            interpolate_embs,
            grad_outputs=torch.ones_like(d_interpolate_scores),
            retain_graph=True,
            create_graph=True,
        )[0]
        gp_loss = ((torch.norm(grads, dim=-1) - 1) ** 2).mean()
        return gp_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_embs = batch["clean_emb"]
        batch_size = real_embs.shape[0]

        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.type_as(real_embs)

        # Train the Generator
        if optimizer_idx == 0:
            # Generate fake embeddings
            fake_embs = self.gen(z)

            # Compute Discriminator's predictions
            _, d_fake_scores = self.dis(fake_embs)

            # Compute loss
            g_loss = -d_fake_scores.mean()

            # Log results
            self.log(f"{self.state}/g_loss", g_loss)

            return {"loss": g_loss}

        # Train the Discriminator
        if optimizer_idx == 1:
            # Generate fake embeddings 
            fake_embs = self.gen(z).detach()

            # Compute Discriminator's predictions
            _, d_real_scores = self.dis(real_embs)
            _, d_fake_scores = self.dis(fake_embs)

            # Compute loss
            d_real_loss = -d_real_scores.mean()
            d_fake_loss = -d_fake_scores.mean()
            d_gp_loss = self.gp_loss(real_embs, fake_embs)
            d_loss = d_real_loss + d_fake_loss + self.hparams.gp_weight * d_gp_loss

            # Log results
            self.log(f"{self.state}/d_real_loss", d_real_loss)
            self.log(f"{self.state}/d_fake_loss", d_fake_loss)
            self.log(f"{self.state}/d_gp_loss", d_gp_loss)
            self.log(f"{self.state}/d_loss", d_loss)

            return {"loss": d_loss}
