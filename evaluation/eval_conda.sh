#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_evaluation_conda
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=./baseline_eval_conda.txt

export CUDA_CACHE_DISABLE=1

group_name=energy_challenge
group_workspace=/hkfs/work/workspace/scratch/bh6321-${group_name}

data_dir=/hkfs/work/workspace/scratch/bh6321-energy_challenge/data
forecast_path=${group_workspace}/AI-HERO-Energy/forecasts.csv

source /hkfs/work/workspace/scratch/im9193-conda/conda/etc/profile.d/conda.sh
conda activate ${group_workspace}/energy_baseline_conda_env
python3 -u ${group_workspace}/AI-HERO-Energy/evaluation.py --save_dir "$PWD" --data_dir ${data_dir} --forecast_path ${forecast_path}
