#!/bin/bash

# Usage:       $ bash global_run.sh <NUM_OF_GPUs> <ARRAY_SIZE>
# Example run: $ bash global_run.sh 1 3360

echo
echo ">>> Started global_run.sh"
echo ">>> AVAILABLE ARG OPTIONS"
echo ">>> 1st arg - GPUs: (1, 2)"
echo ">>> 2nd arg - array/side size: (840, 1680, 3360, 6720, 13440, 26880)"
# echo ">>> 3rd arg - CONVERGENCE CHECK: 1 or 0"
echo 

show_exit_msg()
{
	echo "[!!] >>> Usage: $ bash global_run.sh <NUM_OF_GPUs> <ARRAY_SIZE>"
	echo
	echo ">>> Finishing global_run.sh"
	echo
}


# if [ $# -ne 3 ]; then
if [ $# -ne 2 ]; then
	show_exit_msg
	exit 1
fi

gpus=$1
array_size=$2
# conv_check=$3

echo ">>> Args given:"
echo ">>> GPUs: ${gpus}"
echo ">>> Array size: ${array_size}"
# echo ">>> Conv-Check: ${conv_check}"
echo 

# if [ "$3" -eq 1 ]; then
# 	mpicc ${MPI_SRC_NAME}.c -o ${MPI_SRC_NAME}.x -lm -O3 -D CONVERGE_CHECK_TRUE=1
# else 
# 	mpicc ${MPI_SRC_NAME}.c -o ${MPI_SRC_NAME}.x -lm -O3
# fi


shell_c="#!/bin/bash"

# if [ "$conv_check" -eq 1 ]; then
# 	job_name_c="#PBS -N J_CUDA_${gpus}_${array_size}_CONV"
# else 
# 	job_name_c="#PBS -N J_CUDA_${gpus}_${array_size}"
# fi

job_name_c="#PBS -N J_CUDA_${gpus}_${array_size}"
queue_c="#PBS -q GPUq"
wall_time_c="#PBS -l walltime=00:20:00"
working_dir_c="cd \$PBS_O_WORKDIR"
nodes_c="#PBS -l select=1:ncpus=1:ngpus=${gpus} -lplace=excl"
run_c="nvprof --print-summary-per-gpu ./Jacobi_cuda < input"

final_sh_input="$shell_c\n\n$job_name_c\n$queue_c\n$wall_time_c\n$nodes_c\n$working_dir_c\n$run_c\n"

RANDOM_NAME="$RANDOM"

# Run actual commands
make rebuild
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