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

    int dims[2];
    if (numProcs == 80)
    {
        dims[0] = 8;
        dims[1] = 10;
    }
    else
    {
        dims[0] = dims[1] = (int) sqrt(numProcs);
    }
    
    // Create Cartesian topology (NxN)
    MPI_Comm comm_cart;
    
    int periodic[2] = { 0, 0 };

    // fprintf(stderr, "\n%d : %d\n", numProcs, dims[0]);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &comm_cart);
    MPI_Comm_rank(comm_cart, &myRank);
    // printf("\nHello from Proccess with Rank: %d\n", myRank);
    // fprintf(stderr, "\n[NEIGHB] Hi, this is Rank %d, with Neighbours: N->%d, S->%d, W->%d, E->%d\n", myRank, north, south, west, east);

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

    int local_n, local_m;
    if (numProcs != 80)
    {
        // Block dimensions for worker processes is n/sqrt(p) x m/sqrt(p) per process.
        local_n = (int) ceil((double)n / sqrt((double)numProcs));
        local_m = (int) ceil((double)m / sqrt((double)numProcs));
    }
    else
    {
        local_n = n / 8;
        local_m = m / 10;
    }

    // printf("\n \n %d local_m : %d local_n: %d\n\n", myRank, local_m, local_n);

    // Define Row datatype
    // NOTE: Probably not needed - comms can be done via local_x * MPI_DOUBLE instead of 1 * row_t
    // but it's probably cleaner and more symmetric w/ row_t - not sure if faster though - TODO: CHECK it
    MPI_Datatype row_t;
    MPI_Type_contiguous(local_m, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // Define Col datatype
    MPI_Datatype col_t;
    MPI_Type_vector(local_n, 1, local_m+2, MPI_DOUBLE, &col_t);
    MPI_Type_commit(&col_t);

    // Store worker blocks in u, u_old
    u = (double *) calloc(((local_n + 2) * (local_m + 2)), sizeof(double));
    u_old = (double *) calloc(((local_n + 2) * (local_m + 2)), sizeof(double));

    if (u == NULL || u_old == NULL) {
        printf("Not enough memory for two %ix%i matrices\n", local_n + 2, local_m + 2); exit(1);
    }
    
    int tag = 666; // random tag

    // find neighbours (from cartesian topology)
    int south, north, east, west;
    MPI_Cart_shift(comm_cart, 0, 1, &north, &south);  // North --> Upper Row, South --> Lower Row
    MPI_Cart_shift(comm_cart, 1, 1, &west, &east);    // West  --> Left Col,  East  --> Right Col


    // Install persistent communication handles
    MPI_Request req_send_u_old[4], req_recv_u_old[4], req_send_u[4], req_recv_u[4];
    MPI_Request *req_send = req_send_u_old, *req_recv = req_recv_u_old;

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

    // #ifdef CONVERGE_CHECK_TRUE
    //     #define CHECK (iterationCount < maxIterationCount && error > maxAcceptableError)
    // #elif

    // #endif
    /* Iterate as long as it takes to meet the convergence criterion */
    #ifdef CONVERGE_CHECK_TRUE
        while (iterationCount < maxIterationCount && error > maxAcceptableError)
    #else
        while (iterationCount < maxIterationCount)
    #endif
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

        // Begin Halo swap
        MPI_Startall(4, req_recv);
        MPI_Startall(4, req_send);

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
    double absolute_error, local_absolute_error;

    // u_old holds the full local solution
    local_absolute_error = checkSolution(xLeft, yBottom,
                                        local_n + 2, local_m + 2,
                                        u_old,
                                        deltaX, deltaY,
                                        alpha);
    
    MPI_Reduce(&local_absolute_error, &absolute_error, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
    absolute_error = sqrt(absolute_error) / (n * m);

    if (myRank == 0)
        printf("The error of the iterative solution is %g\n", absolute_error);

    // Calculate absolute error method #2

    //  Re-assemble u_all (we name it: u_final)

    // Define Block datatype
    // Block is (supposed to be) the initial local_n x local_m matrix,
    // aka actual elements *without* halo rows/cols
    MPI_Datatype block_t;
    MPI_Type_vector(local_m, local_n, local_n+2, MPI_DOUBLE, &block_t);
    MPI_Type_commit(&block_t);
    
    // gather all the u-matrices in the u_all matrix and get ready to reassemble it
    if (myRank == 0)
    {
        u_all = (double *)calloc(numProcs *local_m*(local_n+2), sizeof(double)); // u_all : Global solution array
        if (u_all == NULL)
        {
            printf("Not enough memory for u_all matrix\n"); 
            exit(1);
        }
    }

    MPI_Gather(&u_old[local_n+3], 1, block_t, u_all, 1, block_t, 0, comm_cart);
    free(u_old);
    
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
        double absoluteError = checkSolution(xLeft, yBottom,
                                         n + 2, m + 2,
                                         u_final,
                                         deltaX, deltaY,
                                         alpha);
        
        absoluteError = sqrt(absoluteError) / (n * m);
        printf("The error of the gathered solution is %g\n", absoluteError);

        free(u_final);
    }

    MPI_Finalize();
    return 0;
}