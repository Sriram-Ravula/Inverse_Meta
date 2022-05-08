#!/bin/bash
#SBATCH -o ''
#SBATCH -A 'IFML'
#SBATCH -p 'v100'
#SBATCH -J ''
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 16

cd /scratch/04703/sravula
cd Inverse_Meta

module load conda
conda activate inverse_meta

python3 main_new.py --config --doc