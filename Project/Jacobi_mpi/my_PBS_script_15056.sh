#!/bin/bash

#PBS -N J_mpi_9_16i
#PBS -q N10C80
#PBS -l walltime=00:20:00
#PBS -l select=3:ncpus=8:mpiprocs=3:mem=16400000kb
cd $PBS_O_WORKDIR
mpirun jacobi_mpi.x < input
