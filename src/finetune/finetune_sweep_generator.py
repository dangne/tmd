import sys
import yaml


def lr(model):
    if model == "bert":
        return {'values': [1e-6, 3e-6, 5e-6, 7e-6, 9e-6, 1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4]}
    if model == "bert-large":
        return {'values': [1e-6, 2e-6, 3e-6, 5e-6, 6e-6, 7e-6]}
    elif model == "roberta":
        return {'values': [1e-6, 3e-6, 5e-6, 7e-6, 9e-6, 1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4]}
    elif model == "roberta-large":
        return {'values': [3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]}
    elif model in ["xlnet", "xlnet-large"]:
        return {'values': [1e-6, 3e-6, 5e-6, 7e-6, 9e-6, 1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4]}


if __name__ == "__main__":
    project = "none"
    entity = "none"

    datasets = ["ag_news", "imdb", "yelp_polarity"]
    lms = ["bert", "roberta", "xlnet"]
    lm_path = {
        "bert": "/root/manifold_defense/models/bert-base-uncased",
        "bert-large": "/root/manifold_defense/models/bert-large-uncased",
        "roberta": "/root/manifold_defense/models/roberta-base",
        "roberta-large": "/root/manifold_defense/models/roberta-large",
        "xlnet": "/root/manifold_defense/models/xlnet-base-cased",
        "xlnet-large": "/root/manifold_defense/models/xlnet-large-cased",
    }
    model_max_lengths = {"ag_news": 128, "imdb": 256, "yelp_polarity": 256}
    train_batch_size = 64
    num_epochs = 10
    method = "grid"

    for dataset in datasets:
        for lm in lms:
            config = {
                'project': project,
                'entity': entity,
                'command': [
                   '${env}',
                   '${interpreter}',
                   '${program}',
                   '${args}',
                   f'--dataset={dataset}',
                   f'--model_name={lm_path[lm]}',
                   f'--train_batch_size={train_batch_size}',
                   f'--model_max_length={model_max_lengths[dataset]}',
                   f'--num_epochs={num_epochs}',
                ],
                'description': '',
                'method': method,
                'metric': {'goal': 'maximize', 'name': 'validate/acc'},
                'name': f'Finetune {lm} on {dataset} Dataset',
                'parameters': {
                    'lr': lr(lm),
                },
                'program': 'finetune.py'
            }
            with open(f"{lm}_{dataset}_sweep.yaml", "w") as f:
                yaml.dump(config, f)
