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
    double error_sum, total_error;

    // init MPI and get comm rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);


    // find neighbours (create cartesian topology)
    // TODO

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
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mits, 1, MPI_INT, 0, MPI_COMM_WORLD);


    
    // block dimensions for children processes
    int localAlloCountN = (int)ceil((double)n/(double)numProcs);
    int localAlloCountM = (int)ceil((double)m/(double)numProcs);
    allAllocCount = n*m;
    localAllocCount = (localAlloCountN + 2) * (localAlloCountM + 2);
    // allocCount = myRank == 0 
    //                     ? n * m                                             // root proc array-u dimensions 
    //                     : (localAlloCountN) + 2) * (localAlloCountM + 2) ;  // child proc array-u dimensions, remember to add the extra halo rows

    // Those two calls also zero the boundary elements
    // u = (double *)calloc(allocCount, sizeof(double)); //reverse order
    // u_old = myRank == 0 ? NULL : (double *)calloc(allocCount, sizeof(double));
    if (myRank == 0) {
        // main process
        u_all = (double *)calloc(n * m, sizeof(double));
    }
    // child process
    u = (double *)calloc(localAllocCount, sizeof(double));
    u_old = (double *)calloc(localAllocCount, sizeof(double));
    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", n + 2, m + 2);
        exit(1);
    }
    

    // Scatter the blocks
    MPI_Scatter(u_all, localAllocCountN * localAlloCountM, MPI_DOUBLE, u_old, localAllocCountN * localAlloCountM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /////


    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight - xLeft) / (localAlloCountN - 1);
    double deltaY = (yUp - yBottom) / (localAlloCountM - 1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;

    MPI_Init(NULL, NULL);
    t1 = MPI_Wtime();

    int maxXCount = localAlloCountN + 2;
    int maxYCount = localAlloCountM + 2;

    double cx = 1.0 / (deltaX * deltaX);
    double cy = 1.0 / (deltaY * deltaY);
    double cc = -2.0 * (cx + cy) - alpha;
    double div_cc = 1.0 / cc;
    double cx_cc = 1.0 / (deltaX * deltaX) * div_cc;
    double cy_cc = 1.0 / (deltaY * deltaY) * div_cc;
    double c1 = (2.0 + alpha) * div_cc;
    double c2 = 2.0 * div_cc;

    double fX_sq[localAlloCountN], fY_sq[localAlloCountM], updateVal;

    // Optimize
    for (int x = 0; x < localAlloCountN; x++) {
        fX_sq[x] = (xLeft + x * deltaX) * (xLeft + x * deltaX);
    }
    for (int y = 0; y < localAlloCountM; y++) {
        fY_sq[y] = (yBottom + y * deltaY) * (yBottom + y * deltaY);
    }

    int indices[maxYCount];
    for (int i = 0; i < maxYCount; i++) {
        indices[i] = i * maxXCount;
    }
    

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

        // TODO halo swap

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

        // all reduce - to sum errors for all processes

        total_error = sqrt(error_sum) / (n * m);

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    printf("Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1);
    MPI_Finalize();

    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
    printf("Residual %g\n", error);

    // u_old holds the solution after the most recent buffers swap
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n + 2, m + 2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha);
    printf("The error of the iterative solution is %g\n", absoluteError);

    return 0;
}
