#!/bin/bash
lr='01'
wmax='1'
tussen='_'
jobname=$lr$tussen$wmax
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH -o ./out/$jobname.out
#SBATCH --job-name=$jobname
python ./xor_1.py