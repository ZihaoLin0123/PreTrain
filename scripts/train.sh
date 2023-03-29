#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:8
#SBATCH --mail-user=linzihao@idea.edu.cn

export DIR="$(dirname "$(pwd)")"
source activate transformer
export PYTHONPATH=${PYTHONPATH}:${DIR}


export num_gpus=8
export MASTER_PORT=1234
export config_file=/home/pengyue/config/train_gpt2.py

python -m torch.distributed.run --standalone --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} train.py config/train_gpt2.py
