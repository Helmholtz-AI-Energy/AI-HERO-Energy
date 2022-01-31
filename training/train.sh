#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_training
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=./baseline_training.txt

export CUDA_CACHE_DISABLE=1

group_name=energy_challenge
group_workspace=/hkfs/work/workspace/scratch/bh6321-${group_name}
data_dir=/hkfs/work/workspace/scratch/bh6321-energy_challenge/data

source ${group_workspace}/energy_baseline_env/bin/activate
python -u ${group_workspace}/AI-HERO-Energy/training.py --data_dir ${data_dir} --save_dir ${PWD}

