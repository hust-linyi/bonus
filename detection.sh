#!/usr/bin/env bash

raw_data_dir=YOUR_RAWDATA_DIR # entire images
processed_data_dir=YOUR_PROCESSEDDATA_DIR # patch images
save_dir=YOUR_SAVED_DIR # experiments
description='_'
ratio=2.00
dataset='MO'
thresh=0.65
k=3
update_freq=30

cd ./detection
python main.py --random-seed -1 --lr 0.0001 --batch-size 64 --epochs 100 \
  --raw-data-dir ${raw_data_dir} --processed-data-dir ${processed_data_dir} --save_dir ${save_dir} \
  --gpus 4 \
  --ratio ${ratio} \
  --threshold ${thresh} \
  --update-freq ${update_freq} \
  --k-neighbors ${k} \
  --description ${description} \
  --dataset ${dataset}