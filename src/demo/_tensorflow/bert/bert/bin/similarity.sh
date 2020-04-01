#!/bin/bash

ROOT_DIR='/home/zhangjiacheng/ai_testing/zhangjiacheng/bert'
PYTHON='/apps/python/python3/bin/python3'

DATA_DIR='/home/zhangjiacheng/data/bert'

################################################################
cd $ROOT_DIR/src

echo"test Chinese text similarity"

$PYTHON classifier.py \
	--task_name='sim'  \
	--do_train=True \
	--do_eval=True \
	--do_predict=True \
	--data_dir=$DATA_DIR/news_data/ \
	--vocab_file=$ROOT_DIR/conf/vocab.txt \
	--bert_config_file=$ROOT_DIR/conf/bert_config.json \
	--init_checkpoint=$DATA_DIR/chinese_L-12_H-768_A-12/bert_model.ckpt \
	--output_dir=$DATA_DIR/output/sim/ \
	--train_batch_size=2 \
	--eval_batch_size=2 \
	--predict_batch_size=2 \
	--learning_rate=5e-5 \
	--num_train_epochs=3.0 \
	--save_checkpoints_steps=450

