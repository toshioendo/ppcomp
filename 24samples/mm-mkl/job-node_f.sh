#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:10:00

export OMP_NUM_THREADS=192
./mm 4000 4000 4000
