#!/bin/bash
#SBATCH -o ''
#SBATCH -p 'gtx'
#SBATCH -J ''
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 16

cdw

source /work/04703/sravula/maverick2/software/venvs/ncsnv2/bin/activate

cd Inverse_Meta

python3 main_new.py --config --doc