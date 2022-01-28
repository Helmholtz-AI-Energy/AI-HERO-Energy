# AI-HERO-Energy
Baseline model and job submission templates for the use-case "Energy" of the AI-HERO hackathon 2022 on energy efficient AI

# Haicore Setup
 
For your setup please substitute `<YOUR_GROUP_NAME>` by your group ID.

    /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>

## Clone baseline to your group workspace
    cd /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>
    git clone https://github.com/Helmholtz-AI-Energy/AI-HERO-Energy.git


## Set up your environment (either virtualenv OR conda)

### virtualenv (allows python 3.6 or 3.8)

	# go to your workspace
	cd /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>

	# load python
	module load devel/python/3.8

	# create virtual environment
	python -m venv health_baseline_env
	source health_baseline_env/bin/activate
	pip install -U pip
	pip install -r /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>/Helmholtz-AI-Energy/AI-HERO-Energy/requirements.txt
	
### conda (at own risk!)

	# load conda module
	source /hkfs/work/workspace/scratch/im9193-conda/conda/etc/profile.d/conda.sh
	
	# create new environment in your workspace (you can specifiy any python version you want)
	conda create --prefix /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>/energy_baseline_conda_env python==3.8.0
	
	# activate env and install requirements
	conda activate /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>/energy_baseline_conda_env
	pip install -U pip
	pip install -r /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>/Helmholtz-AI-Energy/AI-HERO-Energy/requirements.txt


## Execute training

    cd /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>/Helmholtz-AI-Energy/AI-HERO-Energy/
    sbatch train.sh


## Useful commands for monitoring your jobs on Haicore

List your active jobs and check their status, time and nodes:

    squeue

A more extensive list of all your jobs in a specified time frame, including the consumed energy per job:

    sacct --format User,Account,JobID,JobName,ConsumedEnergy,NodeList,Elapsed,State -S 2022-02-0108:00:00 -E 2022-02-0314:00:00

Print the sum of your overall consumed energy (fill in your user ID):

    sacct -X -o ConsumedEnergy --starttime 2022-02-0108:00:00 --endtime 2022-02-0314:00:00 --user <YOUR USER ID> |awk '{sum+=$1} END {print sum}'

Open a new bash shell on the node your job is running on and use regular Linux commands for monitoring:

    srun --jobid <YOUR JOB ID> --overlap --pty /bin/bash
    htop
    watch -n 0.1 nvidia-smi
    exit  # return to the regular Haicore environment

Cancel / kill a job:
    
    scancel <YOUR JOB ID>

Find more information here: https://wiki.bwhpc.de/e/BwForCluster_JUSTUS_2_Slurm_HOWTO#How_to_view_information_about_submitted_jobs.3F

## Test the evaluation script with your dataloader and model

Adapt the files run_eval.py and eval.sh \
Afterwards submit the evaluation as another job:
    
    cd /hkfs/work/workspace/scratch/im9193-<YOUR_GROUP_NAME>/Helmholtz-AI-Energy/AI-HERO-Energy
    sbatch eval.sh
