#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=vanke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:hgx:4


export DIR="$(dirname "$(pwd)")"
source activate transformer
export PYTHONPATH=${PYTHONPATH}:${DIR}


export num_gpus=4
export MASTER_PORT=1234
export config_file=/home/pengyue/PreTrain/config/train_gpt2.py

# torchrun --standalone --nproc_per_node=${num_gpus} train.py config/train_gpt2.py

python -m torch.distributed.run --standalone --nproc_per_node=${num_gpus} train.py config/train_gpt2.py
