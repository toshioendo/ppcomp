#!/bin/sh
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=00:10:00

export OMP_NUM_THREADS=16
./mm 2000 2000 2000
