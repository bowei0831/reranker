#!/bin/bash
#SBATCH --job-name=merge-data
#SBATCH --partition=gpNCHC_LLM
#SBATCH --account=GOV112003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/peter831/test/logs/merge_%j.out
#SBATCH --error=/home/peter831/test/logs/merge_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

python /home/peter831/test/src/merge_data.py
