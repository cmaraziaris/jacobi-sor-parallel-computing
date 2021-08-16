#!/bin/bash

# Usage:       $ bash serial_run.sh <ARRAY_SIZE>
# Example run: $ bash serial_run.sh 3360

SERIAL_SRC_NAME="jacobi_serial_opt"

echo
echo ">>> Started serial_run.sh"
echo ">>> AVAILABLE ARG OPTIONS"
echo ">>> 1st arg - array/side size: (840, 1680, 3360, 6720, 13440, 26880)"
echo 

show_exit_msg()
{
	echo "[!!] >>> Usage: $ bash serial_run.sh <ARRAY_SIZE>"
	echo
	echo ">>> Finishing serial_run.sh"
	echo
}


if [ $# -ne 1 ]; then
	show_exit_msg
	exit 1
fi

array_size=$1

echo ">>> Args given:"
echo ">>> Array size: ${array_size}"
echo 
	
mpicc ${SERIAL_SRC_NAME}.c -o ${SERIAL_SRC_NAME}.x -lm -O3
prog_type="seq"
nodes_c="#PBS -l select=1:ncpus=8:mem=16400000kb"
run_c="mpirun -np 1 ${SERIAL_SRC_NAME}.x < input"


shell_c="#!/bin/bash"
job_name_c="#PBS -N J_${prog_type}_${array_size}"
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
echo ">>> Finishing serial_run.sh"
echo

exit 0