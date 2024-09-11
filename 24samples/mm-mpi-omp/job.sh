#!/bin/sh
#$ -cwd
#$ -l cpu_16=2
#$ -l h_rt=0:10:00

module load intel-mpi

export OMP_NUM_THREADS=16
mpiexec -n 2 -ppn 1 ./mm 2000 2000 2000
