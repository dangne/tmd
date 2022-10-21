import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup
from scipy.stats import truncnorm

from .gan import GAN


class Generator(nn.Module):
    def __init__(self, latent_dim=100, code_dim=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.input_dim = latent_dim + code_dim

        self.gen = nn.Sequential(
            # input size: (bs, latent_dim+code_dim, 1)
            nn.ConvTranspose1d(self.input_dim, 512, 32, stride=1, padding=0, bias=False),
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

    def forward(self, z, c, make_one_hot=True):
        # input size: ((bs, latent_dim), (bs))
        if make_one_hot:
            c = F.one_hot(c, self.code_dim)  # (bs, code_dim)
        inp = torch.cat([z, c], dim=-1).unsqueeze(dim=-1)  # (bs, latent_dim+code_dim, 1)
        output = self.gen(inp)  # (bs, 1, 768)
        return output.squeeze(dim=1)  # (bs, 768)

    def dist(self, x, y, metric="cosine"):
        if metric == "cosine":
            return 1-F.cosine_similarity(x, y, dim=-1)
        elif metric == "l2":
            return torch.norm(x - y, dim=-1)
        else:
            raise ValueError(f"metric {metric} is not supported")

    def reconstruct1(self, real_embs, c, k, step_size=1.0, num_steps=10, metric="l2", **kwargs):
        # Method 1: Use SGD to find optimize z that minize the reconstruction loss
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        with torch.enable_grad():
            # Initialize K random z candidates
            z_candidates = torch.randn(batch_size, k, self.latent_dim, device=device)  # (bs, k, latent_dim)
            z_candidates = z_candidates.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)
            z_candidates.requires_grad = True
    
            # Duplicate c for candidate z
            c = c.unsqueeze(dim=-1)  # (bs, 1)
            c = c.expand(batch_size, k)  # (bs, k)
            c = c.reshape(batch_size*k)  # (bs*k)
    
            # Define optimizer
            optimizer = torch.optim.SGD([z_candidates], lr=step_size)
    
            # Perform L GD steps
            for step in range(num_steps):
                fake_embs = self.forward(z_candidates, c)  # (bs*k, emb_dim)
                fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
                reconstruction_losses = self.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)
                loss = reconstruction_losses.sum()
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        fake_embs = self.forward(z_candidates, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
        reconstruction_losses = self.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)

        # Find optimal z
        argmin = torch.argmin(reconstruction_losses, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)

        return reconstructed_emb, reconstruction_losses

    def reconstruct2(self, real_embs, c, **kwargs):
        raise NotImplementedError

    def reconstruct3(self, real_embs, c, k, metric, **kwargs):
        # Method 3: Sample several points from the z prior, choose the optimal one
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        # Sample candidate z
        z = torch.randn(batch_size, k, self.latent_dim, device=device)  # (bs, k, latent_dim)
        z = z.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)

        # Duplicate c for candidate z
        c = c.unsqueeze(dim=-1)  # (bs, 1)
        c = c.expand(batch_size, k)  # (bs, k)
        c = c.reshape(batch_size*k)  # (bs*k)

        # Compute candidate fake embeddings
        fake_embs = self.forward(z, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

        # Choose the optimal fake embedding for each real_emb
        dists = self.dist(fake_embs, real_embs.unsqueeze(1), metric=metric)  # (bs, k)
        argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
        reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

        return reconstructed_emb, reconstruction_losses

    def reconstruct4(self, real_embs, c, k, metric, threshold=1, **kwargs):
        # Method 4: Sample several z from the truncated normal, choose the optimal one
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        # Sample candidate z
        z = truncnorm.rvs(-threshold, threshold, size=(batch_size, k, self.latent_dim))  # (bs, k, latent_dim)
        z = torch.as_tensor(z, dtype=torch.float32, device=device)
        z = z.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)

        # Duplicate c for candidate z
        c = c.unsqueeze(dim=-1)  # (bs, 1)
        c = c.expand(batch_size, k)  # (bs, k)
        c = c.reshape(batch_size*k)  # (bs*k)

        # Compute candidate fake embeddings
        fake_embs = self.forward(z, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

        # Choose the optimal fake embedding for each real_emb
        dists = self.dist(fake_embs, real_embs.unsqueeze(1), metric=metric)  # (bs, k)
        argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
        reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

        return reconstructed_emb, reconstruction_losses

    def reconstruct5(self, real_embs, c, k=1, threshold=1, **kwargs):
        # Method 5: Sample 1 random z from the truncated normal (not choosing the optimal)
        assert k == 1, f"k must be 1 for this method. Got k={k}"
        return self.reconstruct4(real_embs, c, k=1, threshold=threshold)

    def reconstruct(self, real_embs, c, k, method=3, **kwargs):
        # There are several ways to find the best z
        # Method 1: Use PGD where z is boundded in the region of a normal distribution
        # Method 2: Use PGD where z is guided with the underlying prior distribution?
        # Method 3: Sample several z from the normal prior, choose the optimal one
        # Method 4: Sample several z from the truncated normal prior, choose the optimal one
        # Method 5: Sample a single z from the truncated normal prior

        with torch.no_grad():
            if method == 1:
                return self.reconstruct1(real_embs, c, k, **kwargs)
            elif method == 2:
                return self.reconstruct2(real_embs, c, k, **kwargs)
            elif method == 3:
                return self.reconstruct3(real_embs, c, k, **kwargs) 
            elif method == 4:
                return self.reconstruct4(real_embs, c, k, **kwargs) 
            elif method == 5:
                return self.reconstruct5(real_embs, c, k, **kwargs) 
            else:
                raise ValueError(f"method {method} is not supported")


class Discriminator(nn.Module):
    def __init__(self, code_dim=10):
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
            #nn.Conv1d(768, 1, 16, stride=1, padding=0, bias=False),
            #nn.Sigmoid()
            # current size: bs x 1 x 1
        )

        self.dis_head = nn.Conv1d(768, 1, 16, stride=1, padding=0, bias=False)
        self.enc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12288, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, code_dim)
        )

    def forward(self, x):
        # input size: bs x 768
        x = x.unsqueeze(dim=1)  # (bs, 1, 768)
        reps = self.dis(x)  # (bs, 768, 16)

        d_logits = self.dis_head(reps).squeeze(-1)  # (bs, 1)
        d_probs = torch.sigmoid(d_logits)  # (bs, 1)

        e_logits = self.enc_head(reps)  # (bs, code_dim)
        return reps, d_logits, d_probs, e_logits


class InfoGAN(GAN):
    def __init__(
        self,
        latent_dim: int = 100,
        code_dim: int = 10,
        lr: float = None,
        b1: float = None,
        b2: float = None,
        g_lr: float = 2e-4,
        g_b1: float = 0.5,
        g_b2: float = 0.999,
        d_lr: float = 2e-4,
        d_b1: float = 0.5,
        d_b2: float = 0.999,
        p_lr: float = 1e-3,
        p_b1: float = 0.5,
        p_b2: float = 0.999,
        reg_prior_weight: float = 100,
        reg_info_weight: float = 1.0,
        k: int = 30,
        step_size: float = 1,
        num_steps: int = 10,
        method: int = 3,
        threshold: float = 1.0,
        metric: str = "l2",
        opt_weight: float = 1.0,
        num_warmup_steps: int = 1000,
        d_step_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if lr is not None:
            self.hparams.g_lr = self.hparams.d_lr = self.hparams.lr
        if b1 is not None:
            self.hparams.g_b1 = self.hparams.d_b1 = self.hparams.b1
        if b2 is not None:
            self.hparams.g_b2 = self.hparams.d_b2 = self.hparams.b2

        self.gen = Generator(latent_dim, code_dim)
        self.dis = Discriminator(code_dim)

        self.prior = nn.Parameter(torch.ones(code_dim)/code_dim)

        self.register_buffer("pl_counter", torch.ones(1, dtype=torch.float32))

    def forward(self, z, c):
        return self.gen(z, c)

    def reconstruct6(self, real_embs, c, k, step_size=1.0, num_steps=10, metric="l2", opt_weight=1, **kwargs):
        # Note that for this reconstruction strategy, c is a distribution so it has shape (bs, code_dim)
        # Method 6: Use SGD to find optimize z that minize the reconstruction loss with constraint Q(c|G(z, c_t)) = one_hot(argmax(Q(c|t))) (hard-label constraint)
        # Method 7: Use SGD to find optimize z that minize the reconstruction loss with constraint Q(c|G(z, c_t)) = Q(c|t) (soft-label constraint)
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        with torch.enable_grad():
            # Initialize K random z candidates
            z_candidates = torch.randn(batch_size, k, self.hparams.latent_dim, device=device)  # (bs, k, latent_dim)
            z_candidates = z_candidates.view(batch_size*k, self.hparams.latent_dim)  # (bs*k, latent_dim)
            z_candidates.requires_grad = True
    
            # Duplicate c for candidate z
            c = c.unsqueeze(dim=1)  # (bs, 1, code_dim)
            c = c.expand(batch_size, k, self.hparams.code_dim)  # (bs, k, code_dim)
            c = c.reshape(batch_size*k, self.hparams.code_dim)  # (bs*k, code_dim)
    
            # Define optimizer
            optimizer = torch.optim.SGD([z_candidates], lr=step_size)
    
            # Perform L GD steps
            for step in range(num_steps):
                c_ = torch.argmax(c, dim=-1)  # (bs*k)
                fake_embs = self.forward(z_candidates, c_)  # (bs*k, emb_dim)
                _, _, _, e_logits = self.dis(fake_embs)  # (bs*k, code_dim)
                reg_loss = -(torch.softmax(e_logits, dim=-1) * torch.log(c)).sum(dim=-1).mean()

                fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
                reconstruction_losses = self.gen.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)
                loss = reconstruction_losses.sum() - self.hparams.opt_weight * reg_loss
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        c_ = torch.argmax(c, dim=-1)  # (bs*k)
        fake_embs = self.forward(z_candidates, c_)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
        reconstruction_losses = self.gen.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)

        # Find optimal z
        argmin = torch.argmin(reconstruction_losses, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)

        return reconstructed_emb, reconstruction_losses

    def reconstruct8(self, real_embs, c, k, metric, **kwargs):
        # Method 8: Sample several points from the z prior and the c prior, choose the optimal one
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        # Sample candidate z
        z = torch.randn(batch_size, k, self.hparams.latent_dim, device=self.device)  # (bs, k, latent_dim)
        z = z.view(batch_size*k, self.hparams.latent_dim)  # (bs*k, latent_dim)

        # Sample candidate c
        probs = torch.softmax(self.prior, dim=-1)
        c = torch.multinomial(probs, batch_size*k, replacement=True)  # (bs*k)
        c = c.to(self.device)

        # Compute candidate fake embeddings
        fake_embs = self.forward(z, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

        # Choose the optimal fake embedding for each real_emb
        dists = self.gen.dist(fake_embs, real_embs.unsqueeze(1), metric=metric)  # (bs, k)
        argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
        reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

        return reconstructed_emb, reconstruction_losses

    # TODO: improve the way to pass arguments
    def reconstruct(self, real_embs, k=None, step_size=None, num_steps=None, method=None, threshold=None, metric=None, opt_weight=None, **kwargs):
        if k is None: k = self.hparams.k
        if step_size is None: step_size = self.hparams.step_size
        if num_steps is None: num_steps = self.hparams.num_steps
        if method is None: method = self.hparams.method
        if threshold is None: threshold = self.hparams.threshold
        if metric is None: metric = self.hparams.metric
        if opt_weight is None: opt_weight = self.hparams.opt_weight

        # Compute c using the Encoder network
        # There are several ways to approximate the cluster c which x belong to
        # Method 1: Employing the Encoder network q(c|x)
        # Method 2: Using clustering method, e.g., K-NN
        # Method 3: Gradient descent w.r.t c and z simultaneously
        _, _, _, e_logits = self.dis(real_embs)
        c = torch.argmax(e_logits, dim=-1)  # (bs)

        # TODO: Improve later
        if method == 6:
            c = torch.softmax(e_logits, dim=-1)  # (bs, code_dim)
            return self.reconstruct6(real_embs, c, k, step_size, num_steps, metric, opt_weight)
        elif method == 8:
            return self.reconstruct8(real_embs, c, k, metric, **kwargs) 
        else:
            return self.gen.reconstruct(real_embs, c, k, method, step_size=step_size, num_steps=num_steps, threshold=threshold, metric=metric)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_embs = batch["clean_emb"]  # (bs, emb_dim)
        batch_size = real_embs.shape[0]

        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim)  # (bs, latent_dim)
        z = z.type_as(real_embs)

        # Sample latent code (discrete)
        probs = torch.softmax(self.prior, dim=-1)
        c = torch.multinomial(probs, batch_size, replacement=True)  # (bs)
        c = c.to(self.device)

        # Train the Generator
        if optimizer_idx == 0 and batch_idx % self.hparams.d_step_ratio == 0:
            # Define real label
            real_labels = torch.ones(batch_size, 1)  # (bs, 1)
            real_labels = real_labels.type_as(real_embs)

            # Generate fake embeddings
            fake_embs = self.gen(z, c)  # (bs, emb_dim)

            # Compute Discriminator's predictions
            _, _, d_fake_probs, e_fake_logits = self.dis(fake_embs)  # (bs, 1), (bs, code_dim)

            # Compute loss 
            g_gan_loss = F.binary_cross_entropy(d_fake_probs, real_labels)
            g_info_loss = F.cross_entropy(e_fake_logits, c)
            g_loss = g_gan_loss + self.hparams.reg_info_weight * g_info_loss

            # Log results
            self.log(f"{self.state}/g_gan_loss", g_gan_loss)
            self.log(f"{self.state}/g_info_loss", g_info_loss)
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
            fake_embs = self.gen(z, c).detach()  # (bs, emb_dim)

            # Compute Discriminator's predictions
            _, _, d_real_probs, _ = self.dis(real_embs)  # (bs, 1)
            _, _, d_fake_probs, e_fake_logits = self.dis(fake_embs)  # (bs, 1), (bs, code_dim)

            # Compute loss
            d_real_loss = F.binary_cross_entropy(d_real_probs, real_labels)
            d_fake_loss = F.binary_cross_entropy(d_fake_probs, fake_labels)
            d_gan_loss = (d_real_loss + d_fake_loss) / 2
            d_info_loss = F.cross_entropy(e_fake_logits, c)
            d_loss = d_gan_loss + self.hparams.reg_info_weight * d_info_loss

            # Compute Discrimintor's accuracy
            d_real_preds = torch.round(d_real_probs)
            d_fake_preds = torch.round(d_fake_probs)
            d_real_acc = (d_real_preds == real_labels).float().mean()
            d_fake_acc = (d_fake_preds == fake_labels).float().mean()
            d_acc = (d_real_acc + d_fake_acc) / 2

            # Compute Encoder's accuracy
            e_preds = torch.argmax(e_fake_logits, dim=-1)  # (bs)
            e_acc = (e_preds == c).float().mean()

            # Log results
            self.log(f"{self.state}/d_real_loss", d_real_loss)
            self.log(f"{self.state}/d_fake_loss", d_fake_loss)
            self.log(f"{self.state}/d_gan_loss", d_gan_loss)
            self.log(f"{self.state}/d_info_loss", d_info_loss)
            self.log(f"{self.state}/d_loss", d_loss)
            self.log(f"{self.state}/d_real_acc", d_real_acc)
            self.log(f"{self.state}/d_fake_acc", d_fake_acc)
            self.log(f"{self.state}/d_acc", d_acc)
            self.log(f"{self.state}/e_acc", e_acc)

            return {"loss": d_loss}

        # Train the Prior Distribution
        if optimizer_idx == 2:
            # Compute Discriminator's predictions
            _, _, _, e_real_logits = self.dis(real_embs)  # (bs, code_dim)

            # Update scheduler
            self.pl_counter *= 0.999

            # Compute loss
            p_reg_loss = -(torch.softmax(self.prior, dim=-1) * torch.log_softmax(self.prior, dim=-1)).sum(dim=-1)
            p_ce_loss = -(torch.softmax(e_real_logits, dim=-1) * torch.log_softmax(self.prior, dim=-1)).sum(dim=-1).mean()
            p_loss = p_ce_loss - self.pl_counter * self.hparams.reg_prior_weight * p_reg_loss

            # Log results
            self.log(f"{self.state}/p_reg_loss", p_reg_loss)
            self.log(f"{self.state}/p_ce_loss", p_ce_loss)
            self.log(f"{self.state}/p_loss", p_loss)
            # TODO: Find a better way to plot the prior distribution
            for i in range(len(self.prior)):
                self.log(f"{self.state}/c{i}", self.prior[i])

            return {"loss": p_loss}

    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        g_b1 = self.hparams.g_b1
        g_b2 = self.hparams.g_b2
        d_lr = self.hparams.d_lr
        d_b1 = self.hparams.d_b1
        d_b2 = self.hparams.d_b2
        p_lr = self.hparams.p_lr
        p_b1 = self.hparams.p_b1
        p_b2 = self.hparams.p_b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=g_lr, betas=(g_b1, g_b2))
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=d_lr, betas=(d_b1, d_b2))
        opt_p = torch.optim.Adam([self.prior], lr=p_lr, betas=(p_b1, p_b2))

        p_scheduler = get_constant_schedule_with_warmup(opt_p, num_warmup_steps=self.hparams.num_warmup_steps)

        return [opt_g, opt_d, opt_p], [p_scheduler]
