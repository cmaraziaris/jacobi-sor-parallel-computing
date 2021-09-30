#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

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

int maxXCount = 1;
int maxYCount = 1;

// declare constant-device variables
__constant__ int n, m;
__constant__ double relax, cx_cc, cy_cc, c1, c2, xLeft, xRight, yBottom, yUp, deltaX, deltaY;

int h_n, h_m;
double h_relax, h_cx_cc, h_cy_cc, h_c1, h_c2, h_xLeft, h_xRight, h_yBottom, h_yUp, h_deltaX, h_deltaY;

// ON-HOST FUNCTIONS

// solution checker
double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount,
                     double *u, double deltaX, double deltaY, double alpha);

void initGPU(void) {
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(n, &h_n, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(m, &h_m, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(relax, &h_relax, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(cx_cc, &h_cx_cc, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(cy_cc, &h_cy_cc, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(c1, &h_c1, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(c2, &h_c2, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(xLeft, &h_xLeft, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(xRight, &h_xRight, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(yBottom, &h_yBottom, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(yUp, &h_yUp, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(deltaX, &h_deltaX, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(deltaY, &h_deltaY, sizeof(double), 0, cudaMemcpyHostToDevice));
}

// ON-DEVICE FUNCTIONS


__global__ void kernel(double *u, double *u_old) {
  // calculate x and y before do the following line
  int ti = threadIdx.x + blockIdx.x * blockDim.x; // get thread id
  int x = (ti % m);
  int y = (ti / n);

  double fX = (xLeft + (x-1) * deltaX);
  double fX_sq = fX*fX;

  double fY = (yBottom + (y-1) * deltaY);
  double fY_sq = fY*fY;

  double fX_dot_fY_sq = fX_sq * fY_sq;

  int indices[3] = {
    (y-1)*maxXCount, 
    y*maxXCount, 
    (y+1)*maxXCount
  };

  __shared__ double shared_mem[blockDim.x][4]; // left and right elements
  shared_mem[threadIdx.x][0] = threadIdx.x > 0 ? shared_mem[threadIdx.x - 1] : 0; // left halo
  shared_mem[threadIdx.x][1] = threadIdx.x < blockDim.x-1 ? shared_mem[threadIdx.x + 1] : 0; // right halo
  shared_mem[threadIdx.x][0] = threadIdx.x > 0 ? shared_mem[threadIdx.x - 1] : 0;
  shared_mem[threadIdx.x][0] = threadIdx.x > 0 ? shared_mem[threadIdx.x - 1] : 0;
  
  __syncthreads();

  double updateVal = (u_old[indices[1] + x - 1] + u_old[indices[1] + x + 1]) * cx_cc +
              (u_old[indices[0] + x] + u_old[indices[2] + x]) * cy_cc +
              u_old[indices[1] + x] +
              c1 * (1.0 - fX_sq - fY_sq + fX_dot_fY_sq) -
              c2 * (fX_dot_fY_sq - 1.0);

  
  u[indices[1] + x] = u_old[indices[1] + x] - relax * updateVal;
}

int main(int argc, char **argv) {
  int mits;
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
  xLeft = -1.0, xRight = 1.0;
  yBottom = -1.0, yUp = 1.0;

  deltaX = (xRight - xLeft) / (n - 1);
  deltaY = (yUp - yBottom) / (m - 1);

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

  // pass_values_to_gpu();

  // set blocks and threads/block TODO: make it more generic
  const int BLOCK_SIZE = 1024;
  dim3 dimBl(BLOCK_SIZE);
  dim3 dimGr(FRACTION_CEILING(n*m, BLOCK_SIZE));

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

    // run kernel
    kernel<<<dimGr, dimBl>>>(u, u_old);
    
    // estimate the error : error += updateVal * updateVal;

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // error = sqrt(error) / (n * m);
    
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
