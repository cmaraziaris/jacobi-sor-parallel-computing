
# printf "\n--SOF--\n"

echo "> Start of sh <"
echo "> Available sizes (1st arg) are: (840, 1680, 3360, 6720, 13440, 26880)"
echo "> Give a 2nd argument to run both Sequential and Parallel with the same input"
echo ">"

if [ $# -eq 0 ]; then
    array_size=840
else
    array_size=$1
fi

printf "$array_size,$array_size\n0.8\n1.0\n1e-13\n50\n" > input

if [ $# -eq 2 ]; then
    echo "Serial with input $array_size x $array_size is:"
    mpicc jacobi_serial_opt.c -o jacobi_serial_opt.x -lm -O3
    qsub PBSseq.sh
fi

echo "Parallel with input $array_size x $array_size is:"
mpicc jacobi_mpi.c -o jacobi_mpi.x -lm -O3
qsub PBSmpi.sh

echo ">"
echo "> End of sh <"

# printf "\n--EOF--\n"