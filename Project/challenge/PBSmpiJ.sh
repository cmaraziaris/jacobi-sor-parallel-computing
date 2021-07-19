#!/bin/bash

# JobName #
#PBS -N Challenge

#Which Queue to use #
#PBS -q N10C80

# Max VM size #

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:02:00

# How many nodes and tasks per node
# Example 4 nodes, 8 cores/node, 8 mpiprocs/node => 32 procs on 32 cores
#PBS -l select=10:ncpus=8:mpiprocs=8:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun  jacobiMPI.x < input
