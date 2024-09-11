#!/bin/sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=0:10:00

module load intel-mpi

mpiexec -n 2 ./test
