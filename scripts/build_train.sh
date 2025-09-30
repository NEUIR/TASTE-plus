python src/build_train.py \
    --data_name Amazon \
    --train_file data/beauty/train.txt \
    --item_file data/beauty/item.txt \
    --item_ids_file data/beauty/item.jsonl \
    --output train.jsonl \
    --output_dir data/beauty \
    --split_num 243 \
    --sample_num 100 \
    --seed 42 \
    --tokenizer google-t5/t5-base

python src/build_train.py \
    --data_name Amazon \
    --train_file data/beauty/valid.txt \
    --item_file data/beauty/item.txt \
    --item_ids_file data/beauty/item.jsonl \
    --output valid.jsonl \
    --output_dir data/beauty \
    --split_num 243 \
    --sample_num 100 \
    --seed 42 \
    --tokenizer google-t5/t5-base
