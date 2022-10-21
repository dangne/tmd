from abc import abstractmethod
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        raise NotImplementedError

    def setup(self):
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []

    def train_dataloader(self, batch_size=None, num_workers=None):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size if batch_size is None else batch_size,
            num_workers=self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self, batch_size=None, num_workers=None):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.train_batch_size if batch_size is None else batch_size,
            num_workers=self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory=True,
        )

    def test_dataloader(self, batch_size=None, num_workers=None):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.train_batch_size if batch_size is None else batch_size,
            num_workers=self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory=True,
        )
