#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=0:10:00

module load cuda intel-mpi

mpiexec -n 8 -ppn 4 ./mm 10000 10000 10000
