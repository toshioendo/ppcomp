#!/bin/sh
#$ -cwd
#$ -l cpu_8=2
#$ -l h_rt=0:10:00

module load cuda intel-mpi

mpirun -n 16 -npernode 8 ./neicomm
