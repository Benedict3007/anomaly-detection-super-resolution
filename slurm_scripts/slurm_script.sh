#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=best_drct_dc2_auc_3_early_stopping_via_slurm.sh
#SBATCH --output=/europa/hpc-homes/bd6102s/logs/slurm/output-%x.%j.out
#SBATCH --error=/europa/hpc-homes/bd6102s/logs/slurm/output-%x.%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=eighty
#SBATCH --mem=36G 
#SBATCH --cpus-per-task=5
    
/europa/hpc-homes/bd6102s/miniconda/bin/python /europa/hpc-homes/bd6102s/main.py 
