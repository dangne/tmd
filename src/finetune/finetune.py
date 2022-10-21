import argparse
import os
import random

import torch
import wandb
from datasets import load_dataset, load_metric, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler

from tqdm import tqdm

num_labels = {"imdb": 2, "ag_news": 4, "yelp_polarity": 2}


def get_tokenize_function(tokenizer, max_length, dataset):
    tokenize_function = None
    def tokenize_function(examples):
        input_dict = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
        examples.update(input_dict)
        return examples
    return tokenize_function


def prepare_data(args):
    raw_datasets = load_dataset(args.dataset)

    # To reduce computational cost, we only work on a subset of the original yelp dataset
    if args.dataset == "yelp_polarity":
        random.seed(42)
        l = len(raw_datasets["train"])
        raw_datasets["train"] = raw_datasets["train"].select(random.sample(range(l), 25000))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.model_max_length,
    )
    tokenize_function = get_tokenize_function(tokenizer, args.model_max_length, args.dataset)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    # Data splitting 
    train_dataset, eval_dataset = tokenized_datasets["train"].train_test_split(0.1, seed=42).values()
    test_dataset = tokenized_datasets["test"]

    # Make dataloaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2,
        collate_fn=data_collator,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2,
        collate_fn=data_collator,
        pin_memory=True
    )

    return tokenizer, train_dataloader, eval_dataloader, test_dataloader


def rename(k):
    if k == "label":
        return "labels"
    else:
        return k

def main(args):
    tokenizer, train_dataloader, eval_dataloader, test_dataloader = prepare_data(args)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if args.ckpt_path is not None:
        model.from_pretrained(args.ckpt_path)
        tokenizer.from_pretrained(args.ckpt_path)

    if args.mode == "train":
        optimizer = AdamW(model.parameters(), lr=args.lr)
        num_training_steps = args.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        global_step = 0
        for epoch in range(args.num_epochs):
            wandb.log({"epoch": epoch})
            for batch in train_dataloader:
                batch = {rename(k): v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                if global_step % 100 == 0:
                    with torch.no_grad():
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        labels = batch["labels"]

                        loss = loss.item()
                        acc = (preds == labels).float().mean().item()

                        wandb.log({"train/loss": loss})
                        wandb.log({"train/acc": acc})
                        progress_bar.set_postfix({"loss": loss, "acc": acc})

                global_step += 1

            # Evaluate on val set
            metric = load_metric("accuracy")
            model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            acc = metric.compute()["accuracy"]
            wandb.log({"validate/acc": acc})

            # Save checkpoint
            model.save_pretrained(args.output_dir + f"/epoch_{epoch}")
            tokenizer.save_pretrained(args.output_dir + f"/epoch_{epoch}")

    # Evaluate on test set
    metric = load_metric("accuracy")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    acc = metric.compute()["accuracy"]
    if args.mode == "train":
        wandb.log({"test/acc": acc})
    print(f"Accuracy on test set: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=3e-05)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_max_length", type=int, default=128)
    parser.add_argument("--project", type=str)
    parser.add_argument("--entity", type=str)
    parser.add_argument("--tags", type=str, nargs="+", default=["finetune"])
    args = parser.parse_args()
    args.num_labels = num_labels[args.dataset]
    print(args)

    if args.mode == "train":
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            config=args,
        )
        args.output_dir = f"{args.dataset}/{run.id}"
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
