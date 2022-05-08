#!/bin/bash
#SBATCH -A 'IFML'
#SBATCH -o ''
#SBATCH -p 'rtx'
#SBATCH -J ''
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 16

module load python3/3.9.2
module load gcc/9.1.0
module load cuda/11.3

cd /scratch1/04703/sravula/

source venvs/ncsnv2/bin/activate

cd Inverse_Meta

python3 main_new.py --config --doc