#!/bin/bash

# Usage:       $ bash global_run.sh <NUM_OF_TOTAL_PROCESSES> <ARRAY_SIZE>
# Example run: $ bash global_run.sh 25 3360

SERIAL_SRC_NAME="jacobi_mpi"
MPI_SRC_NAME="jacobi_mpi"

echo
echo ">>> Started global_run.sh"
echo ">>> AVAILABLE ARG OPTIONS"
echo ">>> 1st arg - processes: (1 [serial], 4, 16, 25, 36, 49, 64, 80)"
echo ">>> 2nd arg - array/side size: (840, 1680, 3360, 6720, 13440, 26880)"
echo 

show_exit_msg()
{
	echo "[!!] >>> Usage: $ bash global_run.sh <NUM_OF_TOTAL_PROCESSES> <ARRAY_SIZE>"
	echo
	echo ">>> Finishing global_run.sh"
	echo
}


# if [ $# -ne 2 ]; then
# 	show_exit_msg
# 	exit 1
# fi

procs=$1
array_size=$2
num_threads=$3

echo ">>> Args given:"
echo ">>> Processes: ${procs}"
echo ">>> Array size: ${array_size}"
echo 

if [[ $procs -eq 1 ]]; then
	
	mpicc -fopenmp ${SERIAL_SRC_NAME}.c -o ${SERIAL_SRC_NAME}.x -lm -O3
	prog_type="seq"
	nodes_c="#PBS -l select=1:ncpus=1:ompthreads=${num_threads}:mem=16400000kb"
	run_c="mpirun -np 1 ${SERIAL_SRC_NAME}.x < input"
	

else

	mpicc -fopenmp ${MPI_SRC_NAME}.c -o ${MPI_SRC_NAME}.x -lm -O3
	prog_type="hybrid"
	run_c="mpirun ${MPI_SRC_NAME}.x < input"

	procs_arr=(4 9 16 25 36 49 64 80)
	sel_arr=(1 3 4 5 6 7 8 8)
	mpi_procs_arr=(4 3 4 5 6 7 8 10)

	select_arg=0

	for (( i = 0; i < ${#procs_arr[@]}; i++ )); do
		if [[ "${procs_arr[$i]}" == "$procs" ]]; then
			select_arg=${sel_arr[$i]}
			mpi_procs_arg=${mpi_procs_arr[$i]}
			break
		fi
	done

	if [[ select_arg -eq 0 ]]; then
		show_exit_msg
		exit 2
	fi

	nodes_c="#PBS -l select=${select_arg}:ncpus=8:mpiprocs=${mpi_procs_arg}:ompthreads=${num_threads}:mem=16400000kb"
fi

shell_c="#!/bin/bash"
job_name_c="#PBS -N J_${prog_type}_${procs}_${array_size}"
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
echo ">>> Finishing global_run.sh"
echo

exit 0