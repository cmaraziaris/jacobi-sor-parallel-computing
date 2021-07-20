#!/bin/bash

mpicc jacobi_serial.c -o jacobi_serial.x -lm -O3
qsub PBSseqJ.sh