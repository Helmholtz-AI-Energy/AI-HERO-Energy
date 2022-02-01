AI-HERO Energy Challenge on energy efficient AI - 7-day forecast of electric load
=============================================================================================

This repository provides the code for the baseline. It contains a full training pipeline in Pytorch
including data loading, training an LSTM and evaluating.
You are free to clone this baseline and build from here.
The repository also includes necessary bash scripts for submitting to HAICORE. 

The following sections contain detailed explanations about the structure, how to adapt the templates
and how to get going on HAICORE.


Table of contents
=================

<!--ts-->
   * [Data](#data)
   * [Structure of the Skeleton Code](#structure-of-the-skeleton-code)
   * [HAICORE Setup](#haicore-setup)
     * [Clone from GitHub](#clone-the-skeleton-code)
     * [Virtual Environment](#set-up-your-environment)
       * [venv](#venv-allows-python-36-or-38)
       * [conda](#conda-experimental)
   * [Training on HAICORE](#training-on-haicore)
   * [Monitoring Jobs](#useful-commands-for-monitoring-your-jobs-on-haicore)
   * [Test your Submission](#test-the-final-evaluation)
<!--te-->

# Data

The train and validation data is available in the following workspace on the cluster:

    /hkfs/work/workspace/scratch/bh6321-energy_challenge/data

It consists of electric load data from 14 cities in Lower Saxony. Each dataset is a csv-file with the columns:
* Load [MWh]: Integrated hourly load
* Time [s]: UTC time stamp in the format "yyyy-mm-dd hh:mm:ss"
* City: [City identifier](#list-of-city-identifiers)

The challenge involves making daily 7-day forecasts with a rolling window approach using the previous 7 days as input.
The datasets are continuous, complete (NaN-free) lists of hourly load sorted first by city and then by time stamp.
The training data covers the years 2014-2017 (368,256 hours), the validation data covers 2018 and test data 2019 (122,640 hours each).
Thus, most hours are used multiple times as input data and label and need to be loaded appropriately 
(i.e. for each possible 336h (14-day) period, where the first half is input and the second the label). 
The `CustomLoadDataset` class from this repository in `dataset.py` already implements the correct loading of the data.
Test data is not available during the development.


# Structure of the skeleton code

The content of the different files is as follows:

- `dataset.py`: Implements a Pytorch Dataset that loads the challenge data
- `model.py`: Implements an LSTM network
- `training.py`: Implements a training pipeline
- `forecast.py`: Implements inference for a given model and saves predictions as csv
- `training`: directory holding bash scripts for submitting on HAICORE, see [Training on HAICORE](#training-on-haicore)
- `evaluation`: directory holding bash scripts for testing the evaluation on HAICORE, see [Test your Submission](#test-the-final-evaluation)

# HAICORE Setup

The HAICORE cluster is organized in workspaces. Each group got its own workspace assigned that is named after your group ID (e.g. E1).
In this workspace you will develop your code, create your virtual environment, save models and preprocessed versions of data and so on.
Once you're logged in to HAICORE, your first step is going to your group workspace.
For the following steps please substitute `<YOUR_GROUP_ID>` by your group ID.


    cd /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>

### Clone the skeleton code

Clone this repository to your workspace. 

    cd /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>
    git clone https://github.com/Helmholtz-AI-Energy/AI-HERO-Energy.git

### Set up your environment

For your virtual environment you can use either venv or conda. Using venv is the standard and recommended way on HAICORE and works most reliably.
However, only python versions 3.6 and 3.8 are pre-installed. If you want to use a different python version you will need to use conda.
Using conda on the cluster is experimental and is therefore at your own risk.
Follow either the venv or the conda instructions to create a virtual environment. Optionally, you can install the requirements.txt from this repo if you want to build on it.

#### venv (allows python 3.6 or 3.8)

	# go to your workspace
	cd /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>

	# load python
	module load devel/python/3.8

	# create virtual environment
	python -m venv energy_baseline_env
	source energy_baseline_env/bin/activate
	pip install -U pip
	pip install -r /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>/AI-HERO-Energy/requirements.txt
	
#### conda (experimental)

	# load conda module
	source /hkfs/work/workspace/scratch/im9193-conda/conda/etc/profile.d/conda.sh
	
	# create new environment in your workspace (you can specifiy any python version you want)
	conda create --prefix /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>/energy_baseline_conda_env python==3.8.0
	
	# activate env and install requirements
	conda activate /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>/energy_baseline_conda_env
	pip install -U pip
	pip install -r /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>/AI-HERO-Energy/requirements.txt


# Training on HAICORE

Submitting to HAICORE is done via the `sbatch` command. It requires a bash script that will be executed on the GPU nodes.
In the folder `training` you will find the bash script that starts training the baseline model. 
Depending on what virtual environment you chose, you need to use `train.sh` or `train_conda.sh`.
In the script you also see the defined sbatch flags. Except for the `--partition=haicore-gpu4` flag (this partition is reserved for the hackathon, on other partitions your jobs would be pending)
you can adapt all other flags if you want. Find more information about `sbatch` here: https://slurm.schedmd.com/sbatch.html.

In the script you need to adapt the path to your group workspace in line 13.

    cd /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>/AI-HERO-Energy
    sbatch training/train.sh


# Useful commands for monitoring your jobs on HAICORE

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
    exit  # return to the regular HAICORE environment

Cancel / kill a job:
    
    scancel <YOUR JOB ID>

Find more information here: https://wiki.bwhpc.de/e/BwForCluster_JUSTUS_2_Slurm_HOWTO#How_to_view_information_about_submitted_jobs.3F

# Test the final Evaluation

The final ranking depends on the consumed energy for development as well as running inference and additionally on
the Mean Absolute Scaled Error (MASE, i.e. your MAE divided by the MAE of the reference implementation) on the test set.
You can have a look at the calculation of the ranks in `ranking.py`, however, it is not needed in your final repository.
In order to allow us to run your model on the test set, you need to adapt the evaluation files from this repository.
The most important one is `forecast.py`. It will load your model, run inference and save the predictions as a csv-file.
It is in your responsibility that this file loads the model and weights you intend to submit. 
You can test the script by just running it, it will automatically predict on the validation set during the development.
In the folder `evaluation` you find the corresponding bash scripts `forecast.sh` or `forecast_conda.sh` to be able to test the
evaluation on HAICORE like this:
    
    cd /hkfs/work/workspace/scratch/bh6321-<YOUR_GROUP_ID>/AI-HERO-Energy
    sbatch evaluation/forecast.sh

In the bash scripts you again need to adapt the paths to your workspace and also insert the correct model weights path.
Each line of the csv-file should contain the 7x24=168 values of a single forecast without a header.

After the csv is created, the MASE is calculated using `evaluation.py`, which again will use the validation set during development.
You do not need to adapt this file, for evaluating on the test set the organizers will use their own copy of this file.
Nevertheless, you can test if your created csv-file works by running the appropriate bash script:

     sbatch evaluation/eval.sh

For that you need to adapt the group workspace in line 13. Both scripts write their outputs to your current directory by default.

For calculating the groups' final scores the mentioned files need to work. That means, that your workspace needs to contain the virtual environment that is loaded, the code as well as model weights.
To make the submission FAIR you additionally have to provide your code on GitHub (with a requirements file that reproduces your full environment), and your weights uploaded to Zenodo.
You can complete your submission here: https://ai-hero-hackathon.de/.
We will verify your results by also downloading everything from GitHub and Zenodo to a clean new workspace and check whether the results match.

# List of city identifiers
'h' : Hannover,  'bs' : Braunschweig, 'ol' : Oldenburg , 'os': Osnabrück, 'wob' : Wolfsburg, 'go' : Göttingen , 'sz' : Salzgitter, 'hi' : Hildesheim , 'del' : Delmenhorst, 'lg': Lüneburg,  'whv'  : Wilhelmshaven, 'ce' : Celle, 'hm' : Hameln, 'el' : Lingen(Ems)
