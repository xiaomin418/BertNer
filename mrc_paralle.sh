#!/bin/bash

export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
DATA_DIR="/home/xm/document/BertMRC/ner2mrc/mrc_paralle"
BERT_DIR="/home/xm/document/PreTrainMode/chinese_roberta_wwm_large_ext_pytorch"

MAXLENGTH=128
WEIGHT_SPAN=0.1
lr=1e-5
OPTIMIZER="adamw"
OUTPUT_DIR="/home/xm/document/BertMRC/train_logs/mrc_paralle_${OPTIMIZER}_lr${lr}_maxlen${MAXLENGTH}_spanw${WEIGHT_SPAN}"
mkdir -p $OUTPUT_DIR

nohup python trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLENGTH \
--batch_size 1 \
--gpus="4,5" \
--precision=16 \
--progress_bar_refresh_rate 100 \
--lr $lr \
--workers 0 \
--distributed_backend=d p \
--val_check_interval 0.25 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--max_epochs 10 \
--chinese \
--span_loss_candidates "pred_and_gold" \
--weight_span $WEIGHT_SPAN \
--mrc_dropout 0.3 \
--warmup_steps 5000 \
--gradient_clip_val 5.0 \
--final_div_factor 20 > ${OUTPUT_DIR}/train_log.txt & tail -f ${OUTPUT_DIR}/train_log.txt
