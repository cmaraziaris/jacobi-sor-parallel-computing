#!/bin/bash

# JobName #
#PBS -N Jseq

#Which Queue to use #
#PBS -q N10C80

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:20:00

# How many nodes and tasks per node
#PBS -l select=4:ncpus=8:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
<<<<<<< HEAD:Project/Jacobi_mpi/PBSseqJ.sh
mpirun -np 4  jacobi_mpi.x < input
=======
mpirun -np 1  jacobi_serial_opt.x < input
>>>>>>> 0b30caa445f1026217e9dbfaab38c505453aef27:Project/Jacobi_mpi/PBSseq.sh
