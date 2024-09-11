#!/bin/sh
#$ -cwd
#$ -l node_q=2
#$ -l h_rt=0:10:00

module load cuda intel-mpi

mpiexec -n 2 -ppn 1 ./mm 10000 10000 10000
