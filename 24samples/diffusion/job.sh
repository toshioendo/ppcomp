#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:10:00

# export OMP_NUM_THREADS=4
./diffusion 100

