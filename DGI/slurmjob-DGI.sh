#!/bin/bash -l
#SBATCH --job-name="dgi-reddit"
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

conda activate s3gc
srun python3 /home/stud68/GDL_Code/DGI/DGI.py reddit

