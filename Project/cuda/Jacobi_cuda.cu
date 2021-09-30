#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

int maxXCount = 1;
int maxYCount = 1;

// declare constant-device variables
__constant__ double cx_cc;
__constant__ double cy_cc;
__constant__ double c1;
__constant__ double c2;
__constant__ double relax;

#define CUDA_SAFE_CALL(call)                                                  \
  {                                                                           \
    cudaError err = call;                                                     \
    if (cudaSuccess != err) {                                                 \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#define FRACTION_CEILING(numerator, denominator) \
  ((numerator + denominator - 1) / denominator)

double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount,
                     double *u, double deltaX, double deltaY, double alpha);



__global__ void kernel(double *u, double *u_old, int n, int m, double fX_sq[], double fY_sq[], int indices[]) {
  // calculate x and y before do the following line
  int x = find x coordinate 
  int y = find y coordinate 


  double fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];

  double updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
              (u_old[indices[y - 1] + x] + u_old[indices[y + 1] + x]) * cy_cc +
              u_old[indices[y] + x] +
              c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) -
              c2 * (fX_dot_fY_sq - 1.0);

  u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
}

int main(int argc, char **argv) {
  int n, m, mits;
  double alpha, tol;
  double maxAcceptableError;
  double error;
  double *u, *u_old, *tmp;
  int allocCount;
  int iterationCount, maxIterationCount;
  double t1, t2;

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

  allocCount = (n + 2) * (m + 2);
  // Those two calls also zero the boundary elements
  CUDA_SAFE_CALL(cudaMallocManaged(&u, allocCount*sizeof(double))); // reserve memory in global unified address space
  CUDA_SAFE_CALL(cudaMallocManaged(&u_old, allocCount * sizeof(double))); // reserve memory in global unified address space


  maxIterationCount = mits;
  maxAcceptableError = tol;

  // Solve in [-1, 1] x [-1, 1]
  double xLeft = -1.0, xRight = 1.0;
  double yBottom = -1.0, yUp = 1.0;

  double deltaX = (xRight - xLeft) / (n - 1);
  double deltaY = (yUp - yBottom) / (m - 1);

  iterationCount = 0;
  error = HUGE_VAL;
  clock_t start = clock(), diff;

//   t1 = MPI_Wtime();

  maxXCount = n + 2;
  maxYCount = m + 2;


  double fX_sq[n], fY_sq[m];
  int indices[maxYCount];

  double cx = 1.0 / (deltaX * deltaX);
  double cy = 1.0 / (deltaY * deltaY);
  double cc = -2.0 * (cx + cy) - alpha;
  double div_cc = 1.0 / cc;

  cx_cc = 1.0 / (deltaX * deltaX) * div_cc;
  cy_cc = 1.0 / (deltaY * deltaY) * div_cc;
  c1 = (2.0 + alpha) * div_cc;
  c2 = 2.0 * div_cc;
//   TODO: pass this values as symbols into GPU (check guide) 

  // Optimize
  for (int x = 0; x < n; x++) {
    fX_sq[x] = (xLeft + x * deltaX) * (xLeft + x * deltaX);
  }
  for (int y = 0; y < m; y++) {
    fY_sq[y] = (yBottom + y * deltaY) * (yBottom + y * deltaY);
  }

  for (int i = 0; i < maxYCount; i++) {
    indices[i] = i * maxXCount;
  }

  // set blocks and threads/block TODO: make it more generic
  int threads_per_block = 1;
  int blocks = 1; 

  /* Iterate as long as it takes to meet the convergence criterion */
  while (iterationCount < maxIterationCount && error > maxAcceptableError) {
    iterationCount++;

    /*************************************************************
     * Performs one iteration of the Jacobi method and computes
     * the residual value.
     *
     * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
     * are BOUNDARIES and therefore not part of the solution.
     *************************************************************/

    error = 0.0;

    ///////////////////////////////////////////////////
    kernel<<<blocks, threads_per_block>>>(u, u_old, n, m, fX_sq, fY_sq, indices)
    estimate the error : error += updateVal * updateVal;

    error = sqrt(error) / (n * m);
    ///////////////////////////////////////////
    

    // Swap the buffers
    tmp = u_old;
    u_old = u;
    u = tmp;
  }

//   t2 = MPI_Wtime();
  printf("Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount,
         t2 - t1);

  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
  printf("Residual %g\n", error);

  // u_old holds the solution after the most recent buffers swap
  double absoluteError =
      checkSolution(xLeft, yBottom, n + 2, m + 2, u_old, deltaX, deltaY, alpha);
  printf("The error of the iterative solution is %g\n", absoluteError);

  return 0;
}

/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount,
                     double *u, double deltaX, double deltaY, double alpha) {
#define U(XX, YY) u[(YY)*maxXCount + (XX)]
  int x, y;
  double fX, fY;
  double localError, error = 0.0;

  for (y = 1; y < (maxYCount - 1); y++) {
    fY = yStart + (y - 1) * deltaY;
    for (x = 1; x < (maxXCount - 1); x++) {
      fX = xStart + (x - 1) * deltaX;
      localError = U(x, y) - (1.0 - fX * fX) * (1.0 - fY * fY);
      error += localError * localError;
    }
  }
  return sqrt(error) / ((maxXCount - 2) * (maxYCount - 2));
}
