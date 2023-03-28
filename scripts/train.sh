#!/bin/sh
export DIR="$(dirname "$(pwd)")"
source activate transformer
export PYTHONPATH=${PYTHONPATH}:${DIR}

export CUDA_VISIBLE_DEVICES=4,6
export num_gpus=2
export MASTER_PORT=1234
export config_file=/home/linzihao/config/train_gpt2.py
#SBATCH --nodelist=dgx[056]

torchrun --standalone --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} train.py config/train_gpt2.py