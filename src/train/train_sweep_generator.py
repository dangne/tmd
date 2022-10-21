import sys
import yaml


def get_parameters(dataset, lm, gm):
    if gm == "infogan":
        return {
            'g_lr': {'values': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 3e-4, 5e-4]},
            'd_lr': {'values': [7e-5, 9e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4, 1e-3]},
            'p_lr': {'values': [0, 1e-4, 1e-3, 1e-2]},
            'latent_dim': {'values': [50, 100, 200]},
            'code_dim': {'values': [10, 20, 30, 40]},
            'd_step_ratio': {'values': [1, 2, 3, 4, 5]},
        }
    elif gm == "dcgan":
        return {
            'g_lr': {'values': [
                1e-06,
                2e-06,
                3e-06,
                4e-06,
                5e-06,
                6e-06,
                7e-06,
                8e-06, 9e-06, 1e-05,
                2e-05,
                3e-05,
                4e-05,
                5e-05,
                6e-05,
                7e-05,
                8e-05,
                9e-05,
                1e-04,
                2e-04,
                3e-04,
                4e-04,
                5e-04,
                6e-04,
                7e-04,
                8e-04,
                9e-04,
            ]},
            'd_lr': {'values': [
                1e-06,
                2e-06,
                3e-06,
                4e-06,
                5e-06,
                6e-06,
                7e-06,
                8e-06,
                9e-06,
                1e-05,
                2e-05,
                3e-05,
                4e-05,
                5e-05,
                6e-05,
                7e-05,
                8e-05,
                9e-05,
                1e-04,
                2e-04,
                3e-04,
                4e-04,
                5e-04,
                6e-04,
                7e-04,
                8e-04,
                9e-04,
            ]},
            'latent_dim': {'values': [50, 75, 100, 125, 150, 175, 200]},
            'd_step_ratio': {'values': [1, 2, 3, 4, 5]},
            'optimizer': {'values': ["agam", "sgd"]},
        }


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
        "xlnet": "/root/manifold_defense/models/xlnet-large-cased",
    }
    gms = ["infogan"]
    gpus = "0,"
    train_batch_size = 1024
    val_batch_size = 512
    test_batch_size = 512

    for dataset in datasets:
        for lm in lms:
            for gm in gms:
                config = {
                    'project': project,
                    'entity': entity,
                    'command': [
                       '${env}',
                       '${interpreter}',
                       '${program}',
                       '${args}', f'--data={dataset}', f'--lm={lm}', f'--lm_path={lm_path[lm] + "-" + dataset}', f'--gm={gm}', f'--gpus={gpus}',
                       f'--train_batch_size={train_batch_size}',
                       f'--val_batch_size={val_batch_size}',
                       f'--test_batch_size={test_batch_size}',
                    ],
                    'description': '',
                    'method': 'random',
                    'metric': {'goal': 'minimize', 'name': 'validate/rec_loss'},
                    'name': f'Hyperparameter Search for {gm} on {lm}-encoded {dataset} Dataset',
                    'parameters': get_parameters(dataset, lm, gm),
                    'program': 'train.py'
                }
                with open(f"{gm}_{lm}_{dataset}_sweep.yaml", "w") as f:
                    yaml.dump(config, f)
