#!/usr/bin/env bash


run_scripts="codes/train_search.py"
dataset_file='ml100k.txt'

CUDA_VISIBLE_DEVICES=0 python ${run_scripts} --use_gpu=True --gpu=$1 --dataset_path=${dataset_file1}
