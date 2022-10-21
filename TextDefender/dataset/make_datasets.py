import os
import argparse
import random
from datasets import load_dataset


def preprocess(examples):
    examples['text'] = [' '.join(text.split('\t')) for text in examples['text']]
    return examples


def main(args):
    if args.dataset == "sst2":
        dataset = load_dataset("glue", args.dataset)
        dataset = dataset.rename_column("sentence", "text")
    else:
        dataset = load_dataset(args.dataset)

    # Prototyping
    if args.dataset == "yelp_polarity":
        random.seed(42)
        dataset["train"] = dataset["train"].select(random.sample(range(len(dataset["train"])), 25000))

    dataset = dataset.map(preprocess, batched=True)

    dataset_name = "".join(args.dataset.split("_"))

    if args.dataset in ["imdb", "snli", "yelp_polarity"]:
        train, dev = dataset["train"].train_test_split(args.dev_ratio, seed=args.seed).values()
        test = dataset["test"]
    else:
        train = dataset["train"]
        dev = dataset["validation"]
        test = dataset["validation"]

    os.makedirs(dataset_name, exist_ok=True)

    with open(f"{dataset_name}/train.txt", "w") as f:
        texts = train['text']
        labels = train['label']
        for i in range(len(train)):
            f.write(f"{texts[i]}\t{labels[i]}\n")

    with open(f"{dataset_name}/dev.txt", "w") as f:
        texts = dev['text']
        labels = dev['label']
        for i in range(len(dev)):
            f.write(f"{texts[i]}\t{labels[i]}\n")

    with open(f"{dataset_name}/test.txt", "w") as f:
        texts = test['text']
        labels = test['label']
        for i in range(len(test)):
            f.write(f"{texts[i]}\t{labels[i]}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yelp_polarity")
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)

