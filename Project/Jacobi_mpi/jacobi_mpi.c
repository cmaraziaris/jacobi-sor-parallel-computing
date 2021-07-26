/************************************************************
 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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
    return sqrt(error) / ((maxXCount - 2) * (maxYCount - 2));
}

int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    double *u, *u_old, *u_all, *tmp;
    int allocCount, localAlloc;
    int iterationCount, maxIterationCount;
    double t1, t2;

    int myRank, numProcs;
    int prevRank, nextRank;
    double error_sum;

    // init MPI and get comm size
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Create Cartesian topology (NxN)
    MPI_Comm comm_cart;
    
    int periodic[2] = { 0, 0 };
    int dims[2];
    dims[0] = dims[1] = (int) sqrt(numProcs);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &comm_cart);
    MPI_Comm_rank(comm_cart, &myRank);


    // find neighbours (create cartesian topology)
    int south, north, east, west;
    MPI_Cart_shift(comm_cart, 0, 1, &north, &south);  // North --> Upper Row, South --> Lower Row
    MPI_Cart_shift(comm_cart, 1, 1, &west, &east);    // West  --> Left Col,  East  --> Right Col

    // make sure only the root process will read input configurations
    if (myRank == 0) {
        //    printf("Input n,m - grid dimension in x,y direction:\n");
        scanf("%d,%d", &n, &m);
        //    printf("Input alpha - Helmholtz constant:\n");
        scanf("%lf", &alpha);
        //    printf("Input relax - successive over-relaxation parameter:\n");
        scanf("%lf", &relax);
        //    printf("Input tol - error tolerance for the iterrative solver:\n");
        scanf("%lf", &tol);
        //    printf("Input mits - maximum solver iterations:\n");
        scanf("%d", &mits);

        printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);
    }

    // Broadcast configuration to all of the processes
    MPI_Bcast(&n, 1, MPI_INT, 0, comm_cart);
    MPI_Bcast(&m, 1, MPI_INT, 0, comm_cart);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, comm_cart);
    MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, comm_cart);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, comm_cart);
    MPI_Bcast(&mits, 1, MPI_INT, 0, comm_cart);

    // // block dimensions for children processes
    // int localAlloCountN = (int)ceil((double)n / (double)numProcs);
    // int localAlloCountM = (int)ceil((double)m / (double)numProcs);
    // int localAllocCount = (localAlloCountN + 2) * (localAlloCountM + 2);

    // Block dimensions for worker processes is n/sqrt(p) x m/sqrt(p) per process.
    int local_n = (int) ceil((double)n / sqrt((double)numProcs));
    int local_m = (int) ceil((double)m / sqrt((double)numProcs));

    // Define Row datatype
    MPI_Datatype row_t;
    MPI_Type_contiguous(local_m, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // Define Col datatype
    MPI_Datatype col_t;
    MPI_Type_vector(local_n, 1, local_m, MPI_DOUBLE, &col_t);
    MPI_Type_commit(&col_t);

    // allAllocCount = n*m;
    // allocCount = myRank == 0 
    //                     ? n * m                                             // root proc array-u dimensions 
    //                     : (local_n) + 2) * (local_m + 2) ;  // child proc array-u dimensions, remember to add the extra halo rows

    // Those two calls also zero the boundary elements
    // u = (double *)calloc(allocCount, sizeof(double)); //reverse order
    // u_old = myRank == 0 ? NULL : (double *)calloc(allocCount, sizeof(double));
    if (myRank == 0) {
        // main process
        u_all = (double *)calloc(n * m, sizeof(double));
    }
    // child process
    u = (double *)calloc(((local_n+2)*(local_m+2)), sizeof(double));
    u_old = (double *)calloc(((local_n+2)*(local_m+2)), sizeof(double));

    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", n + 2, m + 2);
        exit(1);
    }
    
    // Scatter the blocks
    // [harry] : Den exei nohma to na metaferoume mhdenika blocks (afou einai idi 0 to worker/root buffer logw calloc), 
    // mporoume na kalesoume mono Gather sto telos
    // MPI_Scatter(u_all, local_n * local_m, MPI_DOUBLE, u_old, local_n * local_m, MPI_DOUBLE, 0, comm_cart);
    /////

    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight - xLeft) / (local_n - 1);
    double deltaY = (yUp - yBottom) / (local_m - 1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;

    t1 = MPI_Wtime();

    int maxXCount = local_n + 2;
    int maxYCount = local_m + 2;

    double cx = 1.0 / (deltaX * deltaX);
    double cy = 1.0 / (deltaY * deltaY);
    double cc = -2.0 * (cx + cy) - alpha;
    double div_cc = 1.0 / cc;
    double cx_cc = 1.0 / (deltaX * deltaX) * div_cc;
    double cy_cc = 1.0 / (deltaY * deltaY) * div_cc;
    double c1 = (2.0 + alpha) * div_cc;
    double c2 = 2.0 * div_cc;

    double fX_sq[local_n], fY_sq[local_m], updateVal;

    // Optimize
    for (int x = 0; x < local_n; x++) {
        fX_sq[x] = (xLeft + x * deltaX) * (xLeft + x * deltaX);
    }
    for (int y = 0; y < local_m; y++) {
        fY_sq[y] = (yBottom + y * deltaY) * (yBottom + y * deltaY);
    }

    int indices[maxYCount];
    for (int i = 0; i < maxYCount; i++) {
        indices[i] = i * maxXCount;
    }
    
    int tag = 666; // random tag
    
    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {
        iterationCount++;

        /*************************************************************
         * Performs one iteration of the Jacobi method and computes
         * the residual value.
         *
         * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
         * are BOUNDARIES and therefore not part of the solution.
         *************************************************************/

        error = 0.0;

        // MPI_Request req_send_n, req_send_s, req_send_e, req_send_w;
        // MPI_Request req_recv_n, req_recv_s, req_recv_e, req_recv_w;

        MPI_Request req_send[4], req_recv[4];

        // TODO halo swap --> check this on paper
        MPI_Irecv(&u[1], 1, row_t, north, tag, comm_cart, &req_recv[0]);
        MPI_Irecv(&u[(local_m+2)*(local_n+1)+1], 1, row_t, south, tag, comm_cart, &req_recv[1]);
        MPI_Irecv(&u[local_m + 2], 1, col_t, west, tag, comm_cart, &req_recv[2]);
        MPI_Irecv(&u[local_m + 2 + local_m + 1], 1, col_t, east, tag, comm_cart, &req_recv[3]);

        MPI_Isend(&u[local_m + 2 + 1], 1, row_t, north, tag, comm_cart, &req_send[0]);
        MPI_Isend(&u[(local_m+2) * (local_n) + 1], 1, row_t, south, tag, comm_cart, &req_send[1]);
        MPI_Isend(&u[local_m + 2 + 1], 1, col_t, west, tag, comm_cart, &req_send[2]);
        MPI_Isend(&u[local_m + 2 + local_m], 1, col_t, east, tag, comm_cart, &req_send[3]);


        // TODO: Calculate inner values
        for (int y = 1; y < (maxYCount - 1); y++)
        {
            for (int x = 1; x < (maxXCount - 1); x++)
            {
                double fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];

                updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
                            (u_old[indices[y-1] + x] + u_old[indices[y+1] + x]) * cy_cc +
                            u_old[indices[y] + x] +
                            c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);

                u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
                error += updateVal * updateVal;
            }
        }

        // Make sure we got the Halos from the neighbours
        MPI_Waitall(4, req_recv, MPI_STATUSES_IGNORE);

        //TODO: Calculate outer values

        // all reduce - to sum errors for all processes
        MPI_Allreduce(&error, &error_sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        error = sqrt(error_sum) / (n * m);

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;

        MPI_Waitall(4, req_send, MPI_STATUSES_IGNORE);

    }

    t2 = MPI_Wtime();
    if (myRank == 0) {
        printf("Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1);
    }

    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    int max_msec;

    MPI_Reduce(&msec, &max_msec, 1, MPI_INT, MPI_MAX, 0, comm_cart);

    if (myRank == 0) {
        printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
        printf("Residual %g\n", error);
    }


    // TODO: Re-assemble u_all


    // u_old holds the solution after the most recent buffers swap
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n + 2, m + 2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha);
    printf("The error of the iterative solution is %g\n", absoluteError);

    MPI_Finalize();
    return 0;
}