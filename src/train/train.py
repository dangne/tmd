import argparse
import os
import sys

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

sys.path.append("/root/manifold_defense/src")
from data import EncodedDataModule
from models import AutoGenerativeModel, AutoLanguageModel


def main(args):
    # Load the dataset
    print(f"Loading Dataset: {args.data}")
    dm = EncodedDataModule(
        data=args.data, data_root=args.data_root, lm=args.lm,
        lm_path=args.lm_path,
        setup_device="cuda:"+args.gpus[0],
        seed=args.seed,
    )
    dm.setup()

    # Define the model
    print(f"Loading LM: {args.lm}")
    lm = AutoLanguageModel.get_class_name(args.lm).from_pretrained(args.lm_path)
    args.emb_dim = lm.emb_dim

    print(f"Loading GM: {args.gm}")
    gm = AutoGenerativeModel.get_class_name(args.gm)(**vars(args))
    gm.set_lm(lm)

    # Define loggers
    wandb_logger = WandbLogger(
        save_dir=args.output_dir,
        group=args.group,
        job_type=args.job_type,
        tags=args.tags,
    )

    # Run the training loop
    trainer = Trainer.from_argparse_args(
        args,
        benchmark=True,
        default_root_dir=args.output_dir,
        logger=wandb_logger,
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
    )

    trainer.fit(gm, dm.train_dataloader(), dm.val_dataloader())
    trainer.test(gm, dm.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data & Models
    parser.add_argument("--data", type=str, default="imdb",)
    parser.add_argument("--data_root", type=str, default="/root/manifold_defense/data/processed/")
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--lm_path", type=str, default="/root/manifold_defense/models/bert-base-uncased-imdb")
    parser.add_argument("--gm", type=str, default="infogan")
    args = parser.parse_known_args()[0]

    # Other stuffs
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=os.path.join("/root/manifold_defense/outputs", f"{args.gm}_{args.lm}_{args.data}"))
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--job_type", type=str)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser = AutoGenerativeModel.get_class_name(args.gm).add_argparse_args(parser)
    parser = EncodedDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        config=args,
        group=args.group,
        job_type=args.job_type,
        tags=args.tags,
        dir=args.output_dir,
        notes=f"Train {AutoGenerativeModel.get_class_name(args.gm).__name__}",
    )

    seed_everything(args.seed, workers=True)

    main(args)
