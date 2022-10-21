from logging import log
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

#from .ftd import FTD


class Discriminator(nn.Module):
    def __init__(self, emb_dim=768, hidden_dims=[768], dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [emb_dim] + hidden_dims
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
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


class VAEGAN(pl.LightningModule):
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

        # Build Discriminator
        self.discriminator = Discriminator(emb_dim, hidden_dims, dropout_rate)

        # Build Encoder
        hidden_dims = [emb_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
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
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.decoder = nn.Sequential(*layers)

        #self.train_ftd = FTD()
        #self.val_ftd = FTD()

        self.automatic_optimization = False

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

    def reconstruct(self, real_embs, logger=None, **kwargs):
        fake_embs = self.decode(self.sample_z(*self.encode(real_embs)))
        rec_losses = torch.norm(real_embs - fake_embs, dim=-1)
        return fake_embs, rec_losses

    def training_step(self, batch, batch_idx):
        e_opt, d_opt, dis_opt, = self.optimizers()
        if self.hparams.use_scheduler:
            e_sched, d_sched, dis_sched = self.lr_schedulers()

        x = batch["bert_sentence_embs"]

        # Optimize Encoder
        # Sample z and z_p
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        x_tilde = self.decode(z)

        d_x_feat, _, d_x_probs = self.discriminator(x)
        d_x_tilde_feat, _, d_x_tilde_probs = self.discriminator(x_tilde)

        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        e_loss = F.mse_loss(d_x_tilde_feat, d_x_feat) + kld
        e_opt.zero_grad(set_to_none=True)
        self.manual_backward(e_loss, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
        e_opt.step()

        # Optimize Decoder
        bce = F.binary_cross_entropy
        y_reals = torch.ones_like(d_x_probs)
        y_fakes = torch.zeros_like(d_x_probs)

        x_tilde = self.decode(z.detach())
        z_p = torch.randn_like(z.detach())
        x_p = self.decode(z_p)

        d_x_tilde_feat, _, d_x_tilde_probs = self.discriminator(x_tilde)
        d_x_p_feat, _, d_x_p_probs = self.discriminator(x_p)

        d_loss = bce(d_x_tilde_probs, y_reals) + bce(d_x_p_probs, y_reals) + F.mse_loss(d_x_tilde_feat, d_x_feat)
        d_opt.zero_grad(set_to_none=True)
        self.manual_backward(d_loss, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.5)
        d_opt.step()

        # Optimize Discriminator
        _, _, d_x_tilde_probs = self.discriminator(x_tilde.detach())
        _, _, d_x_p_probs = self.discriminator(x_p.detach())

        dis_loss = bce(d_x_probs, y_reals) + bce(d_x_tilde_probs, y_fakes) + bce(d_x_p_probs, y_fakes)
        dis_opt.zero_grad(set_to_none=True)
        self.manual_backward(dis_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        dis_opt.step()

        if self.hparams.use_scheduler:
            d_sched.step()
            e_sched.step()
            dis_sched.step()

        # Evaluation stuffs
        # Discriminator's accuracy
        n = d_x_probs.shape[0]
        d_real_preds = (d_x_probs > 0.5).float()
        d_fake_preds = (torch.cat([d_x_tilde_probs, d_x_p_probs], dim=0) > 0.5).float()
        d_real_acc = (d_real_preds == torch.ones_like(d_real_preds)).float().mean()
        d_fake_acc = (d_fake_preds == torch.zeros_like(d_fake_preds)).float().mean()
        d_acc = (n*d_real_acc + n*2*d_fake_acc) / (n*3)

        self.log("e_loss", e_loss)
        self.log("d_loss", d_loss)
        self.log("dis_loss", dis_loss)
        self.log("d_acc", d_acc)
        self.log("d_real_acc", d_real_acc)
        self.log("d_fake_acc", d_fake_acc)

        # Fréchet Text Distance
        #self.train_ftd.update(x, x_tilde)
        #self.log("train_ftd", self.train_ftd, on_step=False, on_epoch=True)

    def shared_eval_step(self, batch, batch_idx):
        # Evaluate the reconstruction performance
        real_embs = batch["bert_sentence_embs"]
        with torch.enable_grad():
            reconstructed_embs, reconstruction_losses = self.reconstruct(real_embs)

        rec_loss = reconstruction_losses.mean()
        self.log("rec_loss", rec_loss, on_step=False, on_epoch=True)

        #self.val_ftd.update(real_embs, reconstructed_embs)
        #self.log("val_ftd", self.val_ftd, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx)

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


"""
class VAEGANv2(pl.LightningModule):
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

        # Build Discriminator
        self.discriminator = Discriminator(emb_dim, hidden_dims, dropout_rate)

        # Build Encoder
        hidden_dims = [emb_dim*4] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        hidden_dims[0] = emb_dim
        hidden_dims.reverse()
        hidden_dims = [latent_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.decoder = nn.Sequential(*layers)

        self.train_ftd = FTD()
        self.val_ftd = FTD()

        self.automatic_optimization = False

    def forward(self, z):
        self.decoder(z)
    
    def encode(self, enc_input):
        hidden_rep = self.encoder(enc_input)
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

    def reconstruct(self, real_embs, enc_input, logger=None, **kwargs):
        fake_embs = self.decode(self.sample_z(*self.encode(enc_input)))
        rec_losses = torch.norm(real_embs - fake_embs, dim=-1)
        return fake_embs, rec_losses

    def training_step(self, batch, batch_idx):
        e_opt, d_opt, dis_opt, = self.optimizers()
        if self.hparams.use_scheduler:
            e_sched, d_sched, dis_sched = self.lr_schedulers()

        x = batch["bert_sentence_embs"]
        x1 = batch["bert_sentence_embs_1"]
        x2 = batch["bert_sentence_embs_2"]
        x3 = batch["bert_sentence_embs_3"]
        enc_input = torch.cat([x, x1, x2, x3], dim=-1)

        x = x.to(self.device)
        enc_input = enc_input.to(self.device)

        # Optimize Encoder
        # Sample z and z_p
        mu, logvar = self.encode(enc_input)
        z = self.sample_z(mu, logvar)
        x_tilde = self.decode(z)

        d_x_feat, _, d_x_probs = self.discriminator(x)
        d_x_tilde_feat, _, d_x_tilde_probs = self.discriminator(x_tilde)

        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        e_loss = F.mse_loss(d_x_tilde_feat, d_x_feat) + kld
        e_opt.zero_grad(set_to_none=True)
        self.manual_backward(e_loss, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
        e_opt.step()

        # Optimize Decoder
        bce = F.binary_cross_entropy
        y_reals = torch.ones_like(d_x_probs)
        y_fakes = torch.zeros_like(d_x_probs)

        x_tilde = self.decode(z.detach())
        z_p = torch.randn_like(z.detach())
        x_p = self.decode(z_p)

        d_x_tilde_feat, _, d_x_tilde_probs = self.discriminator(x_tilde)
        d_x_p_feat, _, d_x_p_probs = self.discriminator(x_p)

        d_loss = bce(d_x_tilde_probs, y_reals) + bce(d_x_p_probs, y_reals) + F.mse_loss(d_x_tilde_feat, d_x_feat)
        d_opt.zero_grad(set_to_none=True)
        self.manual_backward(d_loss, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.5)
        d_opt.step()

        # Optimize Discriminator
        _, _, d_x_tilde_probs = self.discriminator(x_tilde.detach())
        _, _, d_x_p_probs = self.discriminator(x_p.detach())

        dis_loss = bce(d_x_probs, y_reals) + bce(d_x_tilde_probs, y_fakes) + bce(d_x_p_probs, y_fakes)
        dis_opt.zero_grad(set_to_none=True)
        self.manual_backward(dis_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        dis_opt.step()

        if self.hparams.use_scheduler:
            d_sched.step()
            e_sched.step()
            dis_sched.step()

        # Evaluation stuffs
        # Discriminator's accuracy
        n = d_x_probs.shape[0]
        d_real_preds = (d_x_probs > 0.5).float()
        d_fake_preds = (torch.cat([d_x_tilde_probs, d_x_p_probs], dim=0) > 0.5).float()
        d_real_acc = (d_real_preds == torch.ones_like(d_real_preds)).float().mean()
        d_fake_acc = (d_fake_preds == torch.zeros_like(d_fake_preds)).float().mean()
        d_acc = (n*d_real_acc + n*2*d_fake_acc) / (n*3)

        self.log("e_loss", e_loss)
        self.log("d_loss", d_loss)
        self.log("dis_loss", dis_loss)
        self.log("d_acc", d_acc)
        self.log("d_real_acc", d_real_acc)
        self.log("d_fake_acc", d_fake_acc)

        # Fréchet Text Distance
        self.train_ftd.update(x, x_tilde)
        self.log("train_ftd", self.train_ftd, on_step=False, on_epoch=True)

    def shared_eval_step(self, batch, batch_idx):
        # Evaluate the reconstruction performance
        real_embs = batch["bert_sentence_embs"]
        x1 = batch["bert_sentence_embs_1"]
        x2 = batch["bert_sentence_embs_2"]
        x3 = batch["bert_sentence_embs_3"]
        enc_input = torch.cat([real_embs, x1, x2, x3], dim=-1)

        real_embs = real_embs.to(self.device)
        enc_input = enc_input.to(self.device)

        with torch.enable_grad():
            reconstructed_embs, reconstruction_losses = self.reconstruct(real_embs, enc_input)

        rec_loss = reconstruction_losses.mean()
        self.log("rec_loss", rec_loss, on_step=False, on_epoch=True)

        self.val_ftd.update(real_embs, reconstructed_embs)
        self.log("val_ftd", self.val_ftd, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx)

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
"""