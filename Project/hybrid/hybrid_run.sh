#!/bin/bash

# Usage: $ bash hybrid_run.sh <ARRAY_SIZE> <# NODES> <# PROC PER NODE> <# THREADS PER NODE>"
# Example run: $ bash hybrid_run.sh 3360 2 2 4

HYBRID_SRC_NAME="jacobi_mpi"

echo
echo ">>> Started hybrid_run.sh"
echo ">>> AVAILABLE ARG OPTIONS"
echo ">>> 1st arg - array/side size: (840, 1680, 3360, 6720, 13440, 26880)"
echo ">>> 2nd arg - nodes: {1, 2, ..., 10}"
echo ">>> 3rd arg - processes per node: (1, 2, 4, 8)"
echo ">>> 4th arg - threads per node: (2, 4, 8)"
echo 

show_exit_msg()
{
	echo "[!!] >>> Usage: $ bash hybrid_run.sh <ARRAY_SIZE> <# NODES> <# PROCS PER NODE> <# THREADS PER PROC>"
	echo
	echo ">>> Finishing global_run.sh"
	echo
}

if [ $# -ne 4 ]; then
	show_exit_msg
	exit 1
fi

array_size=$1
nodes=$2
procs=$3
num_threads=$4

total_procs=$(( $nodes * $procs ))
total_threads=$(( $total_procs * $num_threads ))

echo ">>> Args given:"
echo ">>> Array size: ${array_size}"
echo ">>> Nodes: ${nodes}"
echo ">>> Processes per node: ${procs}"
echo ">>> Threads per process: ${num_threads}"
echo ">>> [TOTAL]"
echo ">>> Total Processes: ${total_procs} : valid range is {4, 9, 16, 25, 36, 49, 64, 80}"
echo ">>> Total Threads: ${total_threads}"
echo 

mpicc -fopenmp ${HYBRID_SRC_NAME}.c -o ${HYBRID_SRC_NAME}.x -lm -O3
prog_type="hybrid"
run_c="mpirun ${HYBRID_SRC_NAME}.x < input"

nodes_c="#PBS -l select=${nodes}:ncpus=8:mpiprocs=${procs}:ompthreads=${num_threads}:mem=16400000kb"
shell_c="#!/bin/bash"
job_name_c="#PBS -N J_${prog_type}_N${nodes}_P${procs}_T${num_threads}_${array_size}"
queue_c="#PBS -q N10C80"
wall_time_c="#PBS -l walltime=00:20:00"
working_dir_c="cd \$PBS_O_WORKDIR"

final_sh_input="$shell_c\n\n$job_name_c\n$queue_c\n$wall_time_c\n$nodes_c\n$working_dir_c\n$run_c\n"

RANDOM_NAME="$RANDOM"

# Run actual commands
printf "$array_size,$array_size\n0.8\n1.0\n1e-13\n50\n" > input
printf "$final_sh_input" > my_PBS_script_${RANDOM_NAME}.sh
sleep 2
qsub my_PBS_script_${RANDOM_NAME}.sh
sleep 2
rm -f my_PBS_script_${RANDOM_NAME}.sh


echo
echo ">>> Finishing hybrid_run.sh"
echo

exit 0