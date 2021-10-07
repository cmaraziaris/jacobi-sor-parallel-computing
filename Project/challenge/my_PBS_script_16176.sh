#!/bin/bash

#PBS -N J_seq_1_840
#PBS -q N10C80
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=1:mem=16400000kb
cd $PBS_O_WORKDIR
mpirun -np 1 jacobi_serial_opt.x < input
