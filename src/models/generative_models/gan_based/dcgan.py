import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

from models.language_models.lm import LanguageModel

from .gan import GAN


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.gen = nn.Sequential(
            # input size: bs x 100 x 1
            nn.ConvTranspose1d(latent_dim, 512, 32, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # current size: bs x 512 x 32
            nn.ConvTranspose1d(512, 384, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            # current size: bs x 384 x 64
            nn.ConvTranspose1d(384, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # current size: bs x 256 x 128
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # current size: bs x 128 x 256
            nn.ConvTranspose1d(128, 1, 5, stride=3, padding=1, bias=False),
            nn.Tanh(),
            # current size: bs x 1 x 768
        )

    def forward(self, z):
        # input size: bs x 100
        z = z.unsqueeze(dim=-1)  # bs x 100 x 1
        output = self.gen(z)  # bs x 1 x 768
        return output.squeeze(dim=1)  # bs x 768

    def dist(self, x, y, metric="cosine"):
        if metric == "cosine":
            return 1-F.cosine_similarity(x, y, dim=-1)
        elif metric == "l2":
            return torch.norm(x - y, dim=-1)
        else:
            raise ValueError(f"metric {metric} is not supported")

    def reconstruct1(self, real_embs, k, step_size, num_steps, **kwargs):
        # Initialize K random z candidates
        with torch.enable_grad():
            z_candidates = torch.randn(real_embs.shape[0], k, self.latent_dim, device=real_embs.device)  # bs x k x 100
            z_candidates = z_candidates.view(real_embs.shape[0]*k, self.latent_dim)  # bs*k x 100
            z_candidates.requires_grad = True
    
            optimizer = torch.optim.SGD([z_candidates], lr=step_size)
    
            # Perform L GD steps
            for step in range(num_steps):
                fake_embs = self.forward(z_candidates)  # bs*k x 768
                fake_embs = fake_embs.view(real_embs.shape[0], k, real_embs.shape[-1])  # bs x k x 768
                reconstruction_losses = torch.norm(real_embs.unsqueeze(1) - fake_embs, dim=-1)  # bs x k
                loss = reconstruction_losses.sum()
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        fake_embs = self.forward(z_candidates)  # bs*k x 768
        fake_embs = fake_embs.view(real_embs.shape[0], k, real_embs.shape[-1])  # bs x k x 768
        reconstruction_losses = torch.norm(real_embs.unsqueeze(1) - fake_embs, dim=-1)  # bs x k

        # Find optimal z
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
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            # input size: bs x 1 x 768
            nn.Conv1d(1, 128, 5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current size: bs x 128 x 256
            nn.Conv1d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # current size: bs x 256 x 128
            nn.Conv1d(256, 384, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2, inplace=True),
            # current size: bs x 384 x 64
            nn.Conv1d(384, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # current size: bs x 512 x 32
            nn.Conv1d(512, 768, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2, inplace=True),
            # current size: bs x 768 x 16
            nn.Conv1d(768, 1, 16, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # current size: bs x 1 x 1
        )

    def forward(self, x):
        # input size: bs x 768
        x = x.unsqueeze(dim=1)  # bs x 1 x 768
        output = self.dis(x)  # bs x 1 x 1
        return output.squeeze(dim=-1)  # bs x 1


class DCGAN(GAN):
    def __init__(
        self, 
        latent_dim: int=100,
        emb_dim: int = 768,
        g_hidden_dims: int = [256, 512, 768],
        d_hidden_dims: int = [768, 512, 256],
        lr: float = None,
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        optimizer: str = None,
        g_optimizer: str = "adam",
        d_optimizer: str = "adam",
        method: int = 3,
        k: int = 30,
        step_size: float = 1,
        num_steps: int = 10,
        threshold: float = 1.0,
        d_step_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if lr is not None:
            self.hparams.g_lr = self.hparams.d_lr = self.hparams.lr
        if optimizer is not None:
            self.hparams.g_optimizer = self.hparams.d_optimizer = self.hparams.optimizer

        self.gen = Generator(latent_dim)
        self.dis = Discriminator()

    def reconstruct(self, real_embs, k=None, step_size=None, num_steps=None, method=None, threshold=None, **kwargs):
        if k is None: k = self.hparams.k
        if step_size is None: step_size = self.hparams.step_size
        if num_steps is None: num_steps = self.hparams.num_steps
        if method is None: method = self.hparams.method
        if threshold is None: threshold = self.hparams.threshold

        return self.gen.reconstruct(real_embs, k, method, step_size=step_size, num_steps=num_steps, threshold=threshold)

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
        if optimizer_idx == 0 and batch_idx % self.hparams.d_step_ratio == 0:
            # Define real label
            real_labels = torch.ones(batch_size, 1)
            real_labels = real_labels.type_as(real_embs)

            # Generate fake embeddings
            fake_embs = self.gen(z)

            # Compute Discriminator's predictions
            d_fake_probs = self.dis(fake_embs)

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
            d_real_probs = self.dis(real_embs)
            d_fake_probs = self.dis(fake_embs)

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
        g_lr = self.hparams.g_lr
        d_lr = self.hparams.d_lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=g_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=d_lr, betas=(b1, b2))

        return [opt_g, opt_d], []
