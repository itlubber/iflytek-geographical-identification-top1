# 数据生成1
python finetune_step1_dataprocess_ner.py

# 数据生成2
python finetune_step2_doccano.py \
--doccano_file ./mid_data/train.json \
--task_type "ext" \
--splits 1.0 0.0 0.0 \
--save_dir ./final_data/ \
--negative_ratio 3

python finetune_step2_doccano.py \
--doccano_file ./mid_data/dev.json \
--task_type "ext" \
--splits 0.0 1.0 0.0 \
--save_dir ./final_data/ \
--negative_ratio 0

python finetune_step2_doccano.py \
--doccano_file ./mid_data/test.json \
--task_type "ext" \
--splits 0.0 0.0 1.0 \
--save_dir ./final_data/ \
--negative_ratio 0

# finetune训练
CUDA_VISIBLE_DEVICES=3 python finetune_step3_train.py