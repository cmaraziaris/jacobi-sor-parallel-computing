#!/bin/bash

mpicc jacobi_serial_opt.c -o jacobi_serial_opt.x -lm -O3
qsub PBSseqJ.sh