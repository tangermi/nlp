#!/bin/bash
ROOT_DIR='/apps/dev/ai_nlp_testing'
DATA_DIR='/apps/data/ai_nlp_testing'
LOG_DIR='/apps/logs/ai_nlp_testing'
PYTHON='/apps/python/python3/bin/python3'

mkdir -p $LOG_DIR
cd $ROOT_DIR/src

$PYTHON index.py --config conf.similarity.conf --log_file $LOG_DIR/log
