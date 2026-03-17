#!/bin/bash
#SBATCH --job-name=test-model
#SBATCH --partition=gpNCHC_LLM
#SBATCH --account=GOV112003
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/peter831/test/logs/test_model_%j.out
#SBATCH --error=/home/peter831/test/logs/test_model_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

python /home/peter831/test/scripts/test_model.py
