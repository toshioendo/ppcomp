#!/bin/sh
#$ -cwd
#$ -l cpu_16=2
#$ -l h_rt=0:10:00

module load intel-mpi

mpiexec -n 32 -npernode 16 ./pi 100000000
