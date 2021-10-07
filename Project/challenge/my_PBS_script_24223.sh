#!/bin/bash

#PBS -N J_challenge_64_840
#PBS -q N10C80
#PBS -l walltime=00:20:00
#PBS -l select=8:ncpus=8:mpiprocs=8:mem=16400000kb
cd $PBS_O_WORKDIR
mpirun jacobi_mpi.x < input
