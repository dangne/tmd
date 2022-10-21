import argparse
import os


def init_dependent_args(args):
    args.model_paths = {
        "bert": "/root/manifold_defense/models/bert-base-uncased",
        "roberta": "/root/manifold_defense/models/roberta-base",
        "xlnet": "/root/manifold_defense/models/xlnet-base-cased",
        "albert": "/root/manifold_defense/models/albert-base-v2",
    }
    args.dataset_paths = {
        "imdb": "/root/TextDefender/dataset/imdb",
        "agnews": "/root/TextDefender/dataset/agnews",
        "sst2": "/root/TextDefender/dataset/sst2",
        "yelppolarity": "/root/TextDefender/dataset/yelppolarity",
    }
    args.max_seq_lens = {"imdb": 256, "agnews": 128, "sst2": 64, "yelppolarity": 256}
    args.do_lower_cases = {"bert": "True", "roberta": "True", "xlnet": "False", "albert": "True"}

    # DNE's training parameters
    args.dir_alpha = {"imdb": 0.1, "agnews": 1.0, "sst2": 1.0, "yelppolarity": 0.1}
    args.dir_decay = {"imdb": 0.1, "agnews": 0.5, "sst2": 0.5, "yelppolarity": 0.1}

    # SAFER's training parameters
    args.safer_perturbation_set = {
        "imdb": "/root/TextDefender/cache/embed/imdb/perturbation_constraint_pca0.8_100.pkl", 
        "agnews": "/root/TextDefender/cache/embed/agnews/perturbation_constraint_pca0.8_100.pkl",
        "sst2": "/root/TextDefender/cache/embed/sst2/perturbation_constraint_pca0.8_100.pkl",
        "yelppolarity": "/root/TextDefender/cache/embed/yelppolarity/perturbation_constraint_pca0.8_100.pkl",
    }

    # FreeLB++'s training parameters
    args.adv_learning_rate = {"imdb": 10, "agnews": 30, "sst2": 30, "yelppolarity": 10}

def main(args):
    template = open(args.template_path, "r").read()

    init_dependent_args(args)

    f = open(f"{args.output_dir}/run.sh", "w")
    f.write("#!/bin/bash\n\n")
    for dataset_name in args.dataset_names:
        for model_type in args.model_types:
            for training_type in args.training_types:
                model_path = args.model_paths[model_type]
                dataset_path = args.dataset_paths[dataset_name]
                max_seq_len = args.max_seq_lens[dataset_name]
                do_lower_case = args.do_lower_cases[model_type]

                job_name = f"train_{dataset_name}_{training_type}_{model_type}"
                cmd = f"python main.py --mode=train --model_type={model_type} --model_name_or_path={model_path} --dataset_name={dataset_name} --dataset_path={dataset_path} --training_type={training_type} --max_seq_len={max_seq_len} --do_lower_case={do_lower_case}"
                if training_type == "dne":
                    cmd += f" --dir_alpha={args.dir_alpha[dataset_name]} --dir_decay={args.dir_decay[dataset_name]}"
                if training_type == "safer":
                    cmd += f" --safer-perturbation-set={args.safer_perturbation_set[dataset_name]}"
                if training_type == "freelb":
                    cmd += f" --adv-learning-rate={args.adv_learning_rate[dataset_name]}"

                g = open(job_name + ".sh", "w")
                g.write(template.format(partition=args.partition, workspace=args.workspace, job_name=job_name, cmd=cmd))
                g.close()

                f.write(f"sbatch {job_name}.sh\n")
            f.write("\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", type=str, default="/root/TextDefender/tmd/template_sbatch.txt")
    parser.add_argument("--output_dir", type=str, default="/root/TextDefender/tmd/train_scripts")
    parser.add_argument("--partition", type=str, default="applied")
    parser.add_argument("--workspace", type=str, default="/root/TextDefender")

    parser.add_argument("--model_types", type=str, nargs="+", default=["bert", "roberta", "xlnet"])
    parser.add_argument("--dataset_names", type=str, nargs="+", default=["yelppolarity"])
    parser.add_argument("--training_types", type=str, nargs="+", default=["ascc", "dne", "safer"])

    args = parser.parse_args()
    main(args)
