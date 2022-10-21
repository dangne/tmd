#!/bin/bash

echo evaluate_imdb_ascc_bert_None_recmethod_3
python main.py --mode=evaluate --model_type=bert --model_name_or_path=/root/manifold_defense/models/bert-base-uncased-imdb --dataset_name=imdb --dataset_path=/root/TextDefender/dataset/imdb --training_type=ascc --max_seq_len=256 --do_lower_case=True --method=3

echo evaluate_imdb_ascc_roberta_None_recmethod_3
python main.py --mode=evaluate --model_type=roberta --model_name_or_path=/root/manifold_defense/models/roberta-base-imdb --dataset_name=imdb --dataset_path=/root/TextDefender/dataset/imdb --training_type=ascc --max_seq_len=256 --do_lower_case=True --method=3

echo evaluate_imdb_ascc_xlnet_None_recmethod_3
python main.py --mode=evaluate --model_type=xlnet --model_name_or_path=/root/manifold_defense/models/xlnet-base-cased-imdb --dataset_name=imdb --dataset_path=/root/TextDefender/dataset/imdb --training_type=ascc --max_seq_len=256 --do_lower_case=False --method=3

echo evaluate_agnews_ascc_bert_None_recmethod_3
python main.py --mode=evaluate --model_type=bert --model_name_or_path=/root/manifold_defense/models/bert-base-uncased-agnews --dataset_name=agnews --dataset_path=/root/TextDefender/dataset/agnews --training_type=ascc --max_seq_len=128 --do_lower_case=True --method=3

echo evaluate_agnews_ascc_roberta_None_recmethod_3
python main.py --mode=evaluate --model_type=roberta --model_name_or_path=/root/manifold_defense/models/roberta-base-agnews --dataset_name=agnews --dataset_path=/root/TextDefender/dataset/agnews --training_type=ascc --max_seq_len=128 --do_lower_case=True --method=3

echo evaluate_agnews_ascc_xlnet_None_recmethod_3
python main.py --mode=evaluate --model_type=xlnet --model_name_or_path=/root/manifold_defense/models/xlnet-base-cased-agnews --dataset_name=agnews --dataset_path=/root/TextDefender/dataset/agnews --training_type=ascc --max_seq_len=128 --do_lower_case=False --method=3

echo evaluate_yelppolarity_ascc_bert_None_recmethod_3
python main.py --mode=evaluate --model_type=bert --model_name_or_path=/root/manifold_defense/models/bert-base-uncased-yelppolarity --dataset_name=yelppolarity --dataset_path=/root/TextDefender/dataset/yelppolarity --training_type=ascc --max_seq_len=256 --do_lower_case=True --method=3

echo evaluate_yelppolarity_ascc_roberta_None_recmethod_3
python main.py --mode=evaluate --model_type=roberta --model_name_or_path=/root/manifold_defense/models/roberta-base-yelppolarity --dataset_name=yelppolarity --dataset_path=/root/TextDefender/dataset/yelppolarity --training_type=ascc --max_seq_len=256 --do_lower_case=True --method=3

echo evaluate_yelppolarity_ascc_xlnet_None_recmethod_3
python main.py --mode=evaluate --model_type=xlnet --model_name_or_path=/root/manifold_defense/models/xlnet-base-cased-yelppolarity --dataset_name=yelppolarity --dataset_path=/root/TextDefender/dataset/yelppolarity --training_type=ascc --max_seq_len=256 --do_lower_case=False --method=3

