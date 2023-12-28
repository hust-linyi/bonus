#!/usr/bin/env bash
cd ./segmentation

dataset=MO # MO, CPM, CONIC
ratio=2.00
description=_
data_dir=YOUR_DATA_DIR # patched images
label_dir=YOUR_LABEL_DIR # ground truth
save_dir=YOUR_SAVE_DIR # experiments
gpus=4
random_seed=-1
epochs=80
batch_size=32
threshold=0.5

fg=0.55
bg=0.05
aff_weight=0.1
path_radius=8

python train_coarse.py --dataset ${dataset} --ratio ${ratio} --description ${description} --data-dir ${data_dir} --label-dir ${label_dir} \
--save-dir ${save_dir} --gpus ${gpus} --random-seed ${random_seed} --epochs ${epochs} --batch-size ${batch_size}
python test_coarse.py --dataset ${dataset} --ratio ${ratio} --description ${description} --data-dir ${data_dir} --label-dir ${label_dir} \
--save-dir ${save_dir} --gpus ${gpus} --random-seed ${random_seed} --threshold ${threshold}

python train_fine.py --dataset ${dataset} --ratio ${ratio} --description ${description} --data-dir ${data_dir} --label-dir ${label_dir} \
--save-dir ${save_dir} --gpus ${gpus} --random-seed ${random_seed} --epochs ${epochs} --batch-size ${batch_size} \
--fg ${fg} --bg ${bg} --aff-weight ${aff_weight} --path-radius ${path_radius}
python test_fine.py --dataset ${dataset} --ratio ${ratio} --description ${description} --data-dir ${data_dir} --label-dir ${label_dir} \
--save-dir ${save_dir} --gpus ${gpus} --random-seed ${random_seed} --threshold ${threshold}

python train_self.py --dataset ${dataset} --ratio ${ratio} --description ${description} --data-dir ${data_dir} --label-dir ${label_dir} \
--save-dir ${save_dir} --gpus ${gpus} --random-seed ${random_seed} --epochs ${epochs} --batch-size ${batch_size} \
--data-thresh ${threshold} --aff-weight ${aff_weight} --path-radius ${path_radius}
python test_self.py --dataset ${dataset} --ratio ${ratio} --description ${description} --data-dir ${data_dir} --label-dir ${label_dir} \
--save-dir ${save_dir} --gpus ${gpus} --random-seed ${random_seed} --threshold ${threshold}