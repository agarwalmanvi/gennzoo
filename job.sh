#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -o ./out/random_cfg0.out
#SBATCH --job-name=random_cfg0
python ./xor_cluster.py ./config/Random/cfg0.ini