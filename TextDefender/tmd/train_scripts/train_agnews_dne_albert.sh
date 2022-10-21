#!/bin/bash
#SBATCH --partition=research
#SBATCH --output=/lustre/scratch/client/vinai/users/dangnm12/slurm/log/%x-%j.out
#SBATCH --error=/lustre/scratch/client/vinai/users/dangnm12/slurm/log/%x-%j.out
#SBATCH --job-name=train_agnews_dne_albert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-user=v.dangnm12@vinai.io
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


srun --container-image=/lustre/scratch/client/vinai/users/dangnm12/setup/docker_images/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
     --container-mounts=/lustre/scratch/client/vinai/users/dangnm12/:/root/ \
     --container-workdir=/root/ \
     /bin/bash -c \
     "
     export HTTP_PROXY=http://proxytc.vingroup.net:9090/
     export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
     export http_proxy=http://proxytc.vingroup.net:9090/
     export https_proxy=http://proxytc.vingroup.net:9090/
     export HF_DATASETS_CACHE=/root/cache/huggingface/datasets
     export TRANSFORMERS_CACHE=/root/cache/huggingface/transformers
     export TOKENIZERS_PARALLELISM=false
     export TFHUB_CACHE_DIR=/root/cache/tfhub_modules

     source /opt/conda/bin/activate

     cd /root
     conda activate /root/miniconda3/envs/textdef

     cd /root/TextDefender
     python main.py --mode=train --model_type=albert --model_name_or_path=/root/manifold_defense/models/albert-base-v2 --dataset_name=agnews --dataset_path=/root/TextDefender/dataset/agnews --training_type=dne --max_seq_len=128 --do_lower_case=True --dir_alpha=1.0 --dir_decay=0.5
     "
