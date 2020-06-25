#!/bin/bash
#SBATCH -o ./out/%a.out
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=1GB
python ./spike_source_array_pygenn.py