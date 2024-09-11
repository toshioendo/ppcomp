#!/bin/sh
#$ -cwd
#$ -l f_node=4
#$ -l h_rt=0:10:00

. /etc/profile.d/modules.sh
module load cuda
module load openmpi

mpirun -n 2 -npernode 2 ./mpibcast
mpirun -n 4 -npernode 4 ./mpibcast
mpirun -n 8 -npernode 4 ./mpibcast
mpirun -n 16 -npernode 4 ./mpibcast
