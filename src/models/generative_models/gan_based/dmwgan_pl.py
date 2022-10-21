import torch
import torch.nn as nn
import torch.nn.functional as F

from . import WGAN


class Generator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class DMWGANPL(WGAN):
    def __init__(
        self,
        num_g: int = 10,
        latent_dim: int = 100,
        g_lr: float = 2e-4,
        g_b1: float = 0.5,
        g_b2: float = 0.5,
        d_lr: float = 2e-4,
        d_b1: float = 0.5,
        d_b2: float = 0.5,
        e_lr: float = 2e-4,
        e_b1: float = 0.9,
        e_b2: float = 0.999,
        p_lr: float = 1e-3,
        p_b1: float = 0.5,
        p_b2: float = 0.5,
        gp_weight: float = 10,
        en_weight: float = 1,
        pg_q_lr: float = 0.01,
        pg_temp: float = 1,
        k: int = 10,
        step_size: float = 1,
        num_steps: int = 10,
        **kwargs,
    ):

        self.save_hyperparameters()

        self.gen = Generator()
        self.dis = Discriminator()
        self.enc = Encoder()
        self.pc = nn.Parameter(torch.ones(num_g))

    def forward(self, z):
        pass

    def reconstruct(self, real_embs):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_embs = batch["clean_emb"]
        batch_size = real_embs.shape[0]

        # Sample z
        z = torch.randn(batch_size, self.hparams.latent_dim)  # (bs, latent_dim)
        z = z.type_as(real_embs)

        # Sample generator id
        c = torch.multinomial(self.pc, batch_size, replacement=True)  # (bs)
        c = c.to(self.device)

        # Generate fake embeddings
        fake_embs = None  # TODO

        # Train the Generator
        if optimizer_idx == 0:
            d_fake_embs, d_fake_scores = self.dis(fake_embs)
            gen_loss = -d_fake_scores.mean()
            e_fake_logits = self.enc(d_fake_embs)
            en_loss = F.cross_entropy(e_fake_logits, c)
            g_loss = gen_loss + self.hparams.l1 * en_loss

            # Log results
            self.log(f"{self.state}/gen_loss", gen_loss)
            self.log(f"{self.state}/en_loss", en_loss)
            self.log(f"{self.state}/g_loss", g_loss)

            return {"loss": g_loss}

        fake_embs = fake_embs.detach()

        # Train the Discriminator
        if optimizer_idx == 1:
            pass

        # Train the Encoder
        if optimizer_idx == 2:
            pass

        # Train the Prior Learning
        if optimizer_idx == 3:
            pass

    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        g_b1 = self.hparams.g_b1
        g_b2 = self.hparams.g_b2
        d_lr = self.hparams.d_lr
        d_b1 = self.hparams.d_b1
        d_b2 = self.hparams.d_b2
        e_lr = self.hparams.e_lr
        e_b1 = self.hparams.e_b1
        e_b2 = self.hparams.e_b2
        p_lr = self.hparams.p_lr
        p_b1 = self.hparams.p_b1
        p_b2 = self.hparams.p_b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=g_lr, betas=(g_b1, g_b2))
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=d_lr, betas=(d_b1, d_b2))
        opt_e = torch.optim.Adam(self.enc.parameters(), lr=e_lr, betas=(e_b1, e_b2))
        opt_p = torch.optim.Adam(self.pc.parameters(), lr=p_lr, betas=(p_b1, p_b2))

        return [opt_g, opt_d, opt_e, opt_p], []
