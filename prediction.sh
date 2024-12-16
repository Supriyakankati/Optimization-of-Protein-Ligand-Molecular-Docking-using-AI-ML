#!/bin/bash

#SBATCH --mail-user=supriya.kankati@sjsu.edu
#SBATCH --mail-type=COMPLETE,FAIL
#SBATCH --job-name=medusa
#SBATCH --output=output/output_%j.out
#SBATCH --error=error/error_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00    
##SBATCH --mem-per-cpu=2000
##SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

python prediction.py
