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
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "utils.h"

int main(int argc, char **argv)
{
    int n, m, mits, myRank, numProcs, iterationCount, maxIterationCount;
    double alpha, tol, relax, maxAcceptableError, error, *u, *u_old, *tmp;
    double t1, t2, error_sum;

    // init MPI and get comm size
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Create Cartesian topology (NxN)
    MPI_Comm comm_cart;
    
    int periodic[2] = { 0, 0 };
    int dims[2] = { 0, 0 };

    MPI_Dims_create(numProcs, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &comm_cart);
    MPI_Comm_rank(comm_cart, &myRank);

    fprintf(stderr, "\nProcs: %d Dims[0]: %d, Dims[1]: %d\n", numProcs, dims[0], dims[1]);

    read_and_bcast_problem_config(myRank, &n, &m, &alpha, &relax, &tol, &mits, &comm_cart);

    int local_n = n / dims[0];
    int local_m = m / dims[1];

    // Define Row datatype
    MPI_Datatype row_t;
    MPI_Type_contiguous(local_m, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // Define Col datatype
    MPI_Datatype col_t;
    MPI_Type_vector(local_n, 1, local_m+2, MPI_DOUBLE, &col_t);
    MPI_Type_commit(&col_t);

    // Store worker blocks in u, u_old
    u = calloc(((local_n + 2) * (local_m + 2)), sizeof(double));
    u_old = calloc(((local_n + 2) * (local_m + 2)), sizeof(double));

    if (u == NULL || u_old == NULL) {
        printf("Not enough memory for two %ix%i matrices\n", local_n + 2, local_m + 2); exit(1);
    }
    
    int tag = 666; // random tag

    // Install persistent communication handles among processes
    MPI_Request req_send_u_old[4], req_recv_u_old[4], req_send_u[4], req_recv_u[4];
    MPI_Request *req_send = req_send_u_old, *req_recv = req_recv_u_old;

    // find neighbours (from cartesian topology)
    int south, north, east, west;
    MPI_Cart_shift(comm_cart, 0, 1, &north, &south);  // North --> Upper Row, South --> Lower Row
    MPI_Cart_shift(comm_cart, 1, 1, &west, &east);    // West  --> Left Col,  East  --> Right Col

    // Persistent comms targeting u_old
    MPI_Recv_init(&u_old[1], 1, row_t, north, tag, comm_cart, &req_recv_u_old[0]);
    MPI_Recv_init(&u_old[(local_m+2)*(local_n+1)+1], 1, row_t, south, tag, comm_cart, &req_recv_u_old[1]);
    MPI_Recv_init(&u_old[local_m + 2], 1, col_t, west, tag, comm_cart, &req_recv_u_old[2]);
    MPI_Recv_init(&u_old[local_m + 2 + local_m + 1], 1, col_t, east, tag, comm_cart, &req_recv_u_old[3]);

    MPI_Send_init(&u_old[local_m + 2 + 1], 1, row_t, north, tag, comm_cart, &req_send_u_old[0]);
    MPI_Send_init(&u_old[(local_m+2) * (local_n) + 1], 1, row_t, south, tag, comm_cart, &req_send_u_old[1]);
    MPI_Send_init(&u_old[local_m + 2 + 1], 1, col_t, west, tag, comm_cart, &req_send_u_old[2]);
    MPI_Send_init(&u_old[local_m + 2 + local_m], 1, col_t, east, tag, comm_cart, &req_send_u_old[3]);

    // Persistent comms targeting u
    MPI_Recv_init(&u[1], 1, row_t, north, tag, comm_cart, &req_recv_u[0]);
    MPI_Recv_init(&u[(local_m+2)*(local_n+1)+1], 1, row_t, south, tag, comm_cart, &req_recv_u[1]);
    MPI_Recv_init(&u[local_m + 2], 1, col_t, west, tag, comm_cart, &req_recv_u[2]);
    MPI_Recv_init(&u[local_m + 2 + local_m + 1], 1, col_t, east, tag, comm_cart, &req_recv_u[3]);

    MPI_Send_init(&u[local_m + 2 + 1], 1, row_t, north, tag, comm_cart, &req_send_u[0]);
    MPI_Send_init(&u[(local_m+2) * (local_n) + 1], 1, row_t, south, tag, comm_cart, &req_send_u[1]);
    MPI_Send_init(&u[local_m + 2 + 1], 1, col_t, west, tag, comm_cart, &req_send_u[2]);
    MPI_Send_init(&u[local_m + 2 + local_m], 1, col_t, east, tag, comm_cart, &req_send_u[3]);
    ///////////////////////////

    // Setup problem variables
    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight - xLeft) / (n - 1);
    double deltaY = (yUp - yBottom) / (m - 1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start, diff;

    MPI_Barrier(comm_cart);

    start = clock();
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

    double fX_sq[local_n], fY_sq[local_m], updateVal, fX_dot_fY_sq;

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

    /*************************************************************
     * Performs one iteration of the Jacobi method and computes
     * the residual value.
     *
     * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
     * are BOUNDARIES and therefore not part of the solution.
     *************************************************************/

    /* Iterate as long as it takes to meet the convergence criterion */
#ifdef CONVERGE_CHECK_TRUE
    while (iterationCount++ < maxIterationCount && error > maxAcceptableError)
#else
    while (iterationCount++ < maxIterationCount)
#endif
    {
        // Begin Halo swap
        MPI_Startall(4, req_recv);
        MPI_Startall(4, req_send);

        error = 0.0;

        // Calculate inner values
        for (int y = 2; y < (maxYCount - 2); y++)
        {
            for (int x = 2; x < (maxXCount - 2); x++)
            {
                fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];

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

        // Calculate outer values

        // for x from 1 to maxXCount-2
        // y = 1
        // y = maxYCount - 2
        // estimate outer rows
        for (int x = 1; x < (maxXCount - 1); x++)
        {
            int y = 1;
            fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];

            updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
                        (u_old[indices[y-1] + x] + u_old[indices[y+1] + x]) * cy_cc +
                        u_old[indices[y] + x] +
                        c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);

            u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
            error += updateVal * updateVal;

            y = maxYCount - 2;
            fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];

            updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
                        (u_old[indices[y-1] + x] + u_old[indices[y+1] + x]) * cy_cc +
                        u_old[indices[y] + x] +
                        c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);

            u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
            error += updateVal * updateVal;
        }

        // for y from 1 to maxYCount-2
        // x = 1
        // x = maxXCount - 2
        // estimate outer columns
        for (int y = 1; y < (maxYCount - 1); y++)
        {
            int x = 1;
            fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];
            // TODO : replace x where needed with constant
            updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
                        (u_old[indices[y-1] + x] + u_old[indices[y+1] + x]) * cy_cc +
                        u_old[indices[y] + x] +
                        c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);

            u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
            error += updateVal * updateVal;

            x = maxXCount - 2;
            fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];

            updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
                        (u_old[indices[y-1] + x] + u_old[indices[y+1] + x]) * cy_cc +
                        u_old[indices[y] + x] +
                        c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);

            u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
            error += updateVal * updateVal;
        }

    #ifdef CONVERGE_CHECK_TRUE
        // all reduce - to sum errors for all processes
        MPI_Allreduce(&error, &error_sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        error = sqrt(error_sum) / (n * m);
    #endif

        MPI_Waitall(4, req_send, MPI_STATUSES_IGNORE);

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;

        req_recv = (req_recv == req_recv_u_old) ? req_recv_u : req_recv_u_old;
        req_send = (req_send == req_send_u_old) ? req_send_u : req_send_u_old;
    }

    t2 = MPI_Wtime();

    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    int max_msec;

    // Get the maximum time among every process-worker
    MPI_Reduce(&msec, &max_msec, 1, MPI_INT, MPI_MAX, 0, comm_cart);

    // Get the maximum MPI time among every process-worker
    double final_time, local_final_time = t2 - t1;
    MPI_Reduce(&local_final_time, &final_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

#ifndef CONVERGE_CHECK_TRUE
    // Reduce - to sum errors for all processes
    MPI_Reduce(&error, &error_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
    error = sqrt(error_sum) / (n * m);
#endif

    if (myRank == 0) {
        printf("Iterations=%3d\nElapsed MPI Wall time is %f\n", iterationCount, final_time);
        printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
        printf("Residual %g\n", error);
    }

    free(u);

    // Calculate absolute error method #1
    calc_error_with_reduce(xLeft, yBottom, local_n, local_m, n, m, myRank, u_old, deltaX, deltaY, alpha, &comm_cart);

    // Calculate absolute error method #2
    calc_error_with_gather(xLeft, yBottom, local_n, local_m, n, m, myRank, numProcs, u_old, deltaX, deltaY, alpha, &comm_cart);

    MPI_Finalize();
    return 0;
}
