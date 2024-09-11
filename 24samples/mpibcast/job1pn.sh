#!/bin/sh
#$ -cwd
#$ -l cpu_8=16
#$ -l h_rt=0:10:00

module load intel-mpi

mpiexec -n 2 -ppn 1 ./mpibcast
mpiexec -n 4 -ppn 1 ./mpibcast
mpiexec -n 8 -ppn 1 ./mpibcast
mpiexec -n 12 -ppn 1 ./mpibcast
mpiexec -n 16 -ppn 1 ./mpibcast
