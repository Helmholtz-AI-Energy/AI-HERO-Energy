#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_forecast
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=./baseline_forecast.txt

export CUDA_CACHE_DISABLE=1

group_name=energy_challenge
group_workspace=/hkfs/work/workspace/scratch/bh6321-${group_name}

data_dir=/hkfs/work/workspace/scratch/bh6321-energy_challenge/data
weights_path=${group_workspace}/AI-HERO-Energy/energy_baseline.pt

source ${group_workspace}/energy_baseline_env/bin/activate

nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv --loop-ms=500 -f job_nvidia_smi.csv &
python3 -u ${group_workspace}/AI-HERO-Energy/forecast.py --save_dir "$PWD" --data_dir ${data_dir} --weights_path ${weights_path} --force_cpu

