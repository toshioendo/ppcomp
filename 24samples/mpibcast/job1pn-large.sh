#!/bin/sh
#$ -cwd
#$ -l cpu_8=32
#$ -l h_rt=0:10:00

module load intel-mpi

mpiexec -n 12 -ppn 1 ./mpibcast
mpiexec -n 24 -ppn 1 ./mpibcast
mpiexec -n 32 -ppn 1 ./mpibcast
