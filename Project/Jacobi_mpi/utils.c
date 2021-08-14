#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX, YY) u[(YY)*maxXCount + (XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount - 1); y++)
    {
        fY = yStart + (y - 1) * deltaY;
        for (x = 1; x < (maxXCount - 1); x++)
        {
            fX = xStart + (x - 1) * deltaX;
            localError = U(x, y) - (1.0 - fX * fX) * (1.0 - fY * fY);
            error += localError * localError;
        }
    }
    return error;
    //return sqrt(error) / ((maxXCount - 2) * (maxYCount - 2));
}

void read_and_bcast_problem_config(int myRank, int *n, int *m, double *alpha, double *relax, double *tol, int *mits, MPI_Comm *comm_cart)
{
    // make sure only the root process will read input configurations
    if (myRank == 0) {
        //    printf("Input n,m - grid dimension in x,y direction:\n");
        scanf("%d,%d", n, m);
        //    printf("Input alpha - Helmholtz constant:\n");
        scanf("%lf", alpha);
        //    printf("Input relax - successive over-relaxation parameter:\n");
        scanf("%lf", relax);
        //    printf("Input tol - error tolerance for the iterrative solver:\n");
        scanf("%lf", tol);
        //    printf("Input mits - maximum solver iterations:\n");
        scanf("%d", mits);

        printf("-> %d, %d, %g, %g, %g, %d\n", *n, *m, *alpha, *relax, *tol, *mits);
    }

    // Broadcast configuration to every process in the grid
    MPI_Bcast(n, 1, MPI_INT, 0, *comm_cart);
    MPI_Bcast(m, 1, MPI_INT, 0, *comm_cart);
    MPI_Bcast(alpha, 1, MPI_DOUBLE, 0, *comm_cart);
    MPI_Bcast(relax, 1, MPI_DOUBLE, 0, *comm_cart);
    MPI_Bcast(tol, 1, MPI_DOUBLE, 0, *comm_cart);
    MPI_Bcast(mits, 1, MPI_INT, 0, *comm_cart);
}


void calc_error_with_reduce(double xLeft, double yBottom, int local_n, int local_m, int n, int m, int myRank,
                            double *u_old, double deltaX, double deltaY, double alpha, MPI_Comm *comm_cart)
{
    double absolute_error, local_absolute_error;

    // u_old holds the full local solution
    local_absolute_error = checkSolution(xLeft, yBottom, local_n + 2, local_m + 2, u_old, deltaX, deltaY, alpha);
    
    MPI_Reduce(&local_absolute_error, &absolute_error, 1, MPI_DOUBLE, MPI_SUM, 0, *comm_cart);

    if (myRank == 0)
    {
        absolute_error = sqrt(absolute_error) / (n * m);
        printf("The error of the iterative solution is %g\n", absolute_error);
    }
}

void calc_error_with_gather(double xLeft, double yBottom, int local_n, int local_m, int n, int m, int myRank, int numProcs,
                            double *u_old, double deltaX, double deltaY, double alpha, MPI_Comm *comm_cart)
{
    //  Re-assemble u_all (we name it: u_final)
    double *u_all = NULL;

    // Define Block datatype
    // Block is (supposed to be) the initial local_n x local_m matrix,
    // aka actual elements *without* halo rows/cols
    MPI_Datatype block_t;
    MPI_Type_vector(local_m, local_n, local_n+2, MPI_DOUBLE, &block_t);
    MPI_Type_commit(&block_t);
    
    // gather all the u-matrices in the u_all matrix and get ready to reassemble it
    if (myRank == 0)
    {
        u_all = calloc(numProcs *local_m*(local_n+2), sizeof(double)); // u_all : Global solution array
        if (u_all == NULL)
        {
            printf("Not enough memory for u_all matrix\n"); 
            return;
        }
    }

    MPI_Gather(u_old + (local_n+3), 1, block_t, u_all, 1, block_t, 0, *comm_cart);
    free(u_old);
    MPI_Barrier(*comm_cart);  // Make sure every process has free'd u_old
    
    if (myRank == 0)
    {
        #define INDEX(y) (y*(n+2))

        double *u_final = calloc((n+2)*(m+2), sizeof(double));

        int index = 0;
        // Let the root process re-assemble the matrix.
        for (int x = 1; x < n+1; x+=local_n) {  // traverse blocks in the x axis
            for (int y = 1; y < m+1; y++) {     // traverse blocks in the y axis

                memcpy(&u_final[INDEX(y)+x], &u_all[index], local_n*sizeof(double));
                // continue to the next local_n elements that are construct a part of the next row
                index += local_n; 
            }
        }
        free(u_all);

        // u_final holds the full solution
        double absoluteError = checkSolution(xLeft, yBottom, n + 2, m + 2, u_final, deltaX, deltaY, alpha);
        
        absoluteError = sqrt(absoluteError) / (n * m);
        printf("The error of the gathered solution is %g\n", absoluteError);

        free(u_final);
    }
}