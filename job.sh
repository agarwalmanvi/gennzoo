#!/bin/bash
#SBATCH -o ./out/Hz100_lr005.out
#SBATCH --job-name=Hz100_lr005
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
python ./xor_1.py