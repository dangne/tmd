from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.argparse import add_argparse_args

from models.language_models import LanguageModel


class GenerativeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lm = None

    def set_lm(self, lm: LanguageModel):
        self.lm = lm

    @abstractmethod
    def reconstruct(self, embs, **kwargs):
        pass

    @abstractmethod
    def forward(self, embs):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def eval_rec_loss(self, embs):
        with torch.enable_grad():
            rec_embs, rec_losses = self.reconstruct(embs)
        rec_loss = rec_losses.mean().detach()
        return rec_loss, rec_embs
    
    def eval_generalization(self, embs, labels):
        logits = self.lm.classify_embs(embs)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        generalization = (preds == labels).float().mean()
        return generalization

    def shared_eval_step(self, batch, batch_idx):
        # This method still make several assumptions on the data
        embs = batch["clean_emb"] 
        labels = batch["label"] 

        rec_loss, rec_embs = self.eval_rec_loss(embs)
        generalization = self.eval_generalization(rec_embs, labels)

        self.log(f"{self.state}/rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log(f"{self.state}/generalization", generalization, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx)

    @abstractmethod
    def configure_optimizers(self):
        pass

    def on_train_start(self):
        self.lm.to("cpu")

    def on_validation_start(self):
        self.lm.to(self.device)

    def on_test_start(self):
        self.lm.to(self.device)

    def shared_checkpoint_hook(self, checkpoint):
        # Hacky stuff to avoid saving/loading the pretrained LM in the checkpoint
        # TODO: Anyway better way than this?
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            if "lm" in key:
                del checkpoint['state_dict'][key]

    def on_load_checkpoint(self, checkpoint):
        self.shared_checkpoint_hook(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        self.shared_checkpoint_hook(checkpoint)

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        return add_argparse_args(cls, parent_parser, **kwargs)

    @property
    def state(self):
        return self.trainer.state.stage
