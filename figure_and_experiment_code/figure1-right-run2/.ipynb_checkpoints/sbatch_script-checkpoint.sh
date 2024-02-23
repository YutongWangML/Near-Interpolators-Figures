#!/bin/bash
#SBATCH --job-name=near_interpol
#SBATCH --partition=standard
#SBATCH --time=0-01:00:00
#SBATCH --mem=8GB
#SBATCH --output=/home/yutongw/near_interpol.log
#SBATCH --mail-user=yutongw@umich.edu
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=vvh0

source /home/yutongw/.bash_profile
source /home/yutongw/.bashrc
conda activate ml
echo $CONDA_DEFAULT_ENV
pwd

jupyter nbconvert --to script compute.ipynb
python compute.py $1
