
#include <mpi.h>

double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount,
                     double *u, double deltaX, double deltaY, double alpha);


void read_and_bcast_problem_config(int myRank, int *n, int *m, double *alpha, double *relax, double *tol, int *mits, MPI_Comm *comm_cart);


void calc_error_with_reduce(double xLeft, double yBottom, int local_n, int local_m, int n, int m, int myRank,
                            double *u_old, double deltaX, double deltaY, double alpha, MPI_Comm *comm_cart);

void calc_error_with_gather(double xLeft, double yBottom, int local_n, int local_m, int n, int m, int myRank, int numProcs,
                            double *u_old, double deltaX, double deltaY, double alpha, MPI_Comm *comm_cart);