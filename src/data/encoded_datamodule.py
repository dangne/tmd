import os
import sys
import warnings
import random

import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from . import DataModule

sys.path.append("/root/manifold_defense/src/")
from models import AutoLanguageModel


class PEDataset(Dataset):
    def __init__(self, clean_embs, labels):
        self.clean_embs = clean_embs
        self.labels = labels

    def __len__(self):
        return len(self.clean_embs)

    def __getitem__(self, idx):
        return {"clean_emb": self.clean_embs[idx], "label": self.labels[idx]}


class EncodedDataModule(DataModule):
    """
    Note to the future me:
    If you want to load the data from cache, you can just provide the 'data' and 'lm' arguments.
    If you want to create new encoded data, you must provide specifically which pretrained LM to be used.
    """

    def __init__(
        self,
        data="imdb",
        data_root="/root/manifold_defense/data/processed/",
        lm="bert",
        lm_path="/root/manifold_defense/models/bert-base-uncased-imdb/",
        save_name: str = None,
        is_encoded: bool = True,
        use_cache: bool = True,
        setup_batch_size: int = 512,
        train_batch_size: int = 1024,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 4,
        setup_device: str = "cuda",
        layer: int = -1,
        max_length: int = 256,
        num_samples_for_pe: int = 1000000,
        seed=42,
        **kwargs,
    ):
        super().__init__()
        if data.replace("_", "").replace("-", "") not in lm_path.replace(
            "_", ""
        ).replace("-", ""):
            warnings.warn(
                f"There maybe a mismatch between dataset ({data}) and pretrained language model ({lm_path})."
            )
        self.save_hyperparameters()
        pl.seed_everything(seed)

    def get_encode_fn(self, lm):
        def encode_fn(examples):
            with torch.no_grad():
                text_pairs = None
                texts = examples["text"]

                if self.hparams.layer == -1:
                    sentence_embs = lm.text2emb(texts, text_pairs)
                    examples["clean_emb"] = sentence_embs.cpu().detach().numpy()
                else:
                    inputs = lm.tokenizer(
                        texts,
                        text_pairs,
                        padding="max_length",
                        truncation=True,
                        max_length=self.hparams.max_length,
                        return_tensors="pt",
                    )
                    inputs = {
                        k: v.to(self.hparams.setup_device) for k, v in inputs.items()
                    }
                    outputs = lm(**inputs, output_hidden_states=True)
                    examples["clean_emb"] = (
                        outputs.hidden_states[self.hparams.layer].cpu().detach().numpy()
                    )
                return examples

        return encode_fn

    def encode(self, dataset, lm):
        # Encode the data using a LM
        encode_fn = self.get_encode_fn(lm)
        encoded_dataset = dataset.map(
            encode_fn, batched=True, batch_size=self.hparams.setup_batch_size
        )
        return encoded_dataset

    def setup(self):
        if self.hparams.is_encoded:
            loaded = False
            if self.hparams.use_cache:
                try:
                    print(f"Load data from cache ({self.cache_path})")
                    datasets = load_from_disk(self.cache_path)
                    loaded = True
                except:
                    print("Cache not found.")

            if not loaded:
                print("Setup data from scratch.")
                lm = AutoLanguageModel.get_class_name(
                    self.hparams.lm
                ).from_pretrained(self.hparams.lm_path)
                lm.eval()
                lm.to(self.hparams.setup_device)

                datasets = load_dataset(self.hparams.data)

                # To reduce computational cost, we only work on a subset of the original yelp dataset
                if self.hparams.data == "yelp_polarity":
                    random.seed(self.hparams.seed)
                    l = len(datasets["train"])
                    datasets["train"] = datasets["train"].select(random.sample(range(l), 25000))

                # Process PE case (subsample)
                if self.hparams.layer != -1:
                    random.seed(self.hparams.seed)
                    for k, v in datasets.items():
                        num_chosen_ids = min(len(v), self.hparams.num_samples_for_pe // self.hparams.max_length)
                        chosen_ids = random.sample(range(len(v)), num_chosen_ids)
                        datasets[k] = v.select(chosen_ids)

                datasets = self.encode(datasets, lm)

                # Process PE case. For each split:
                # 1. Merge all token embedding into a single tensor, then split into separate vectors
                # 2. Construct new Dataset
                if self.hparams.layer != -1:
                    for k, v in datasets.items():
                        breakpoint()
                        embs = torch.cat([torch.vstack(e) for e in v["clean_emb"]])
                        labels = (
                            v["label"]
                            .unsqueeze(-1)
                            .expand(v["label"].shape[0], self.hparams.max_length)
                            .reshape(-1)
                        )

                datasets.save_to_disk(self.cache_path)

            datasets.set_format(type="torch", columns=["clean_emb", "label"])
        else:
            datasets = load_dataset(self.hparams.data)

        if self.hparams.layer != -1:
            tmp = {}
            for k, v in datasets.items():
                embs = torch.cat([torch.vstack(e) for e in v["clean_emb"]])
                labels = (
                    v["label"]
                    .unsqueeze(-1)
                    .expand(v["label"].shape[0], self.hparams.max_length)
                    .reshape(-1)
                )
                embs = embs[torch.randperm(len(labels))][:self.hparams.num_samples_for_pe].clone()
                labels = labels[torch.randperm(len(labels))][:self.hparams.num_samples_for_pe].clone()

                tmp[k] = PEDataset(embs, labels)
            datasets = tmp
        else:
            train_dataset, val_dataset = (
                datasets["train"]
                .train_test_split(0.1, seed=self.hparams.seed)
                .values()
                )
            datasets["train"] = train_dataset
            datasets["validation"] = val_dataset

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]

    @property
    def cache_path(self):
        if self.hparams.save_name is None:
            if self.hparams.layer == -1:
                filename = (
                    f"{self.hparams.lm}_encoded_{self.hparams.data.replace('-', '_')}"
                )
            else:
                filename = f"{self.hparams.lm}_encoded_{self.hparams.data.replace('-', '_')}_layer{self.hparams.layer}"
            path = os.path.join(self.hparams.data_root, filename)
            return path
        else:
            return os.path.join(self.hparams.data_root, self.hparams.save_name)
