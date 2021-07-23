#!/bin/bash

mpicc jacobi_mpi.c -o jacobi_mpi.x -lm -O3
qsub PBSseqJ.sh