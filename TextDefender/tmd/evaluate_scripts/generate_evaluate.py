import os
import argparse
from collections import defaultdict


def init_dependent_args(args):
    args.model_paths = {
        ("bert", "agnews"): "/root/manifold_defense/models/bert-base-uncased-agnews",
        ("bert", "imdb"): "/root/manifold_defense/models/bert-base-uncased-imdb",
        ("bert", "snli"): "/root/manifold_defense/models/bert-base-uncased-snli",
        ("bert", "snli_pe_small"): "/root/manifold_defense/models/bert-base-uncased-snli",
        ("bert", "sst2"): "/root/manifold_defense/models/bert-base-uncased-sst2",
        ("bert", "yelppolarity"): "/root/manifold_defense/models/bert-base-uncased-yelppolarity",
        ("roberta", "agnews"): "/root/manifold_defense/models/roberta-base-agnews",
        ("roberta", "imdb"): "/root/manifold_defense/models/roberta-base-imdb",
        ("roberta", "sst2"): "/root/manifold_defense/models/roberta-base-sst2",
        ("roberta", "yelppolarity"): "/root/manifold_defense/models/roberta-base-yelppolarity",
        ("xlnet", "agnews"): "/root/manifold_defense/models/xlnet-base-cased-agnews",
        ("xlnet", "imdb"): "/root/manifold_defense/models/xlnet-base-cased-imdb",
        ("xlnet", "sst2"): "/root/manifold_defense/models/xlnet-base-cased-sst2",
        ("xlnet", "yelppolarity"): "/root/manifold_defense/models/xlnet-base-cased-yelppolarity",
        ("bert-large", "imdb"): "/root/manifold_defense/models/bert-large-uncased-imdb",
        ("bert-large", "agnews"): "/root/manifold_defense/models/bert-large-uncased-agnews",
        ("roberta-large", "imdb"): "/root/manifold_defense/models/roberta-large-imdb",
        ("roberta-large", "agnews"): "/root/manifold_defense/models/roberta-large-agnews",
        ("albert", "agnews"): "/root/manifold_defense/models/albert-base-v2-agnews",
        ("albert", "imdb"): "/root/manifold_defense/models/albert-base-v2-imdb",
        ("albert", "sst2"): "/root/manifold_defense/models/albert-base-v2-sst2",
    }

    args.dataset_paths = {
        "imdb": "/root/TextDefender/dataset/imdb",
        "agnews": "/root/TextDefender/dataset/agnews",
        "sst2": "/root/TextDefender/dataset/sst2",
        "snli": "/root/TextDefender/dataset/snli",
        "snli_pe_small": "/root/TextDefender/dataset/snli",
        "yelppolarity": "/root/TextDefender/dataset/yelppolarity",
    }
    args.max_seq_lens = {"imdb": 256, "agnews": 128, "snli": 64, "snli_pe_small": 64, "sst2": 64, "yelppolarity": 256}
    args.do_lower_cases = {"bert": "True", "roberta": "True", "xlnet": "False", "bert-large": "True", "roberta-large": "True", "albert": "True"}
    args.gm_paths = defaultdict(lambda: [])
    with open(args.top_10_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            if line[0] == "#":
                current_key = line.split()[1]
                #gm, model, dataset = current_key.split("_")
                parts = current_key.split("_")
                gm, model, dataset = "_".join(parts[:-2]), parts[-2], parts[-1]
                if dataset == "agnews":
                    dataset = "ag_news"
                if dataset == "yelppolarity":
                    dataset = "yelp_polarity"
            else:
                if len(args.gm_paths[current_key]) < args.top_n:
                    gm_dir = f"/root/manifold_defense/outputs/{gm}_{model}_{dataset}/manifold-defense/{line}/checkpoints/"
                    gm_ckpts = os.listdir(gm_dir)
                    assert len(gm_ckpts) == 1, f"Found {len(gm_ckpts)} GM checkpoint(s)"
                    gm_path = gm_dir + gm_ckpts[0]
                    args.gm_paths[current_key].append(gm_path)


def handle_special_case(dataset_name):
    if dataset_name in ["snli_pe", "snli_pe_small"]:
        return "snli"
    else:
        return dataset_name


def main(args):
    template = open(args.template_path, "r").read()

    init_dependent_args(args)

    f = open(f"{args.output_dir}/run.sh", "w")
    f.write("#!/bin/bash\n\n")
    for dataset_name in args.dataset_names:
        for model_type in args.model_types:
            for gm_type in args.gm_types:
                for training_type in args.training_types:
                    for tmd_layer in args.tmd_layers:
                        model_path = args.model_paths[(model_type, dataset_name)]
                        dataset_path = args.dataset_paths[dataset_name]
                        max_seq_len = args.max_seq_lens[dataset_name]
                        do_lower_case = args.do_lower_cases[model_type]

                        job_name = f"evaluate_{dataset_name}_{training_type}_{model_type}_{args.use_epoch}_recmethod_{args.rec_method}"
                        cmd = f"python main.py --mode=evaluate --model_type={model_type} --model_name_or_path={model_path} --dataset_name={handle_special_case(dataset_name)} --dataset_path={dataset_path} --training_type={training_type} --max_seq_len={max_seq_len} --do_lower_case={do_lower_case} --method={args.rec_method}"

                        if training_type == "tmd":
                            for idx, gm_path in enumerate(args.gm_paths[f"{gm_type}_{model_type}_{dataset_name}"]):
                                gm_id = gm_path.split("/")[6]
                                if tmd_layer != -1:
                                    new_job_name = job_name + f"_{gm_type}_{idx}_{gm_id}_{tmd_layer}"
                                    new_cmd = cmd + f" --gm={gm_type} --gm_path={gm_path} --tmd_layer={tmd_layer}"
                                else:
                                    new_job_name = job_name + f"_{gm_type}_{idx}_{gm_id}"
                                    new_cmd = cmd + f" --gm={gm_type} --gm_path={gm_path}"

                                #g = open(f"{args.output_dir}/{new_job_name}.sh", "w")
                                #g.write(template.format(partition=args.partition, workspace=args.workspace, job_name=new_job_name, cmd=new_cmd))
                                #g.close()

                                #f.write(f"sbatch {new_job_name}.sh\n")
                                f.write(f"echo {new_job_name}\n")
                                f.write(f"{new_cmd}\n")
                        else:
                            #g = open(job_name+".sh", "w")
                            #g.write(template.format(partition=args.partition, workspace=args.workspace, job_name=job_name, cmd=cmd))
                            #g.close()

                            #f.write(f"sbatch {job_name}.sh\n")
                            f.write(f"echo {job_name}\n")
                            f.write(f"{cmd}\n")
                f.write("\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", type=str, default="/root/TextDefender/tmd/template_sbatch.txt")
    parser.add_argument("--top_10_path", type=str, default="/root/TextDefender/tmd/top_10_candidates.txt")
    parser.add_argument("--output_dir", type=str, default="/root/TextDefender/tmd/evaluate_scripts")
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--partition", type=str, default="research")
    parser.add_argument("--workspace", type=str, default="/root/TextDefender")

    parser.add_argument("--use_epoch", type=str, default=None)
    parser.add_argument("--model_types", type=str, nargs="+", default=["bert", "roberta", "xlnet"])
    parser.add_argument("--gm_types", type=str, nargs="+", default=["infogan"])
    parser.add_argument("--dataset_names", type=str, nargs="+", default=["imdb", "agnews", "yelppolarity"])
    parser.add_argument("--training_types", type=str, nargs="+", default=["ascc"])
    parser.add_argument("--tmd_layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--rec_method", type=int, default=3)

    args = parser.parse_args()
    main(args)
