#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestamp.h"

// Uncomment to ommit convergence check (Always perform "mits" iterations)
#define CONVERGE_CHECK_TRUE

#define CUDA_SAFE_CALL(call)                                                  \
  {                                                                           \
    cudaError err = call;                                                     \
    if (cudaSuccess != err)                                                   \
    {                                                                         \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#define FRACTION_CEILING(numerator, denominator) ((numerator + denominator - 1) / denominator)

// Declare constant-device variables, for faster access
__constant__ int n, m, maxXCount, maxYCount;
__constant__ double relax, cx_cc, cy_cc, c1, c2, xLeft, xRight, yBottom, yUp, deltaX, deltaY;

// Host variables corresponding to __constant__ device variables.
int h_n, h_m, h_maxXCount, h_maxYCount;
double h_relax, h_cx_cc, h_cy_cc, h_c1, h_c2, h_xLeft, h_xRight, h_yBottom, h_yUp, h_deltaX, h_deltaY;

__host__ void initGPU(void);
__host__ double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount, double *u, double deltaX, double deltaY, double alpha);

// Main kernel : Each thread is assigned only 1 element, computes and stores
// the updated value after 1 jacobi iteration.
__global__ void one_jacobi_iteration(double *u, double *u_old, double *error_matrix)
{
  // Global Thread ID (ti) corresponds to exactly 1 of the n*m elements of u.
  int ti = threadIdx.x + blockIdx.x * blockDim.x;

  if (ti >= n * m) // Required in cases where the number of elements
    return;        // is *not* a multiple of threads per block (aka 1024) eg. 1680x1680/1024=2756.25 -> 2757 blocks

  int x = (ti % m);
  int y = (ti / n);

  // Shared memory array, used to store every element from u_old
  // that will be needed for every compuation performed in this *block*.
  extern __shared__ double u_tmp[];

  // We spawn n*m threads,
  // map "index" from indexing n*m elements -> (n+2)*(m+2) elements, including halos.
  // "index" points to the element position in the enhanced u_old array ( of size (n+2) * (m+2) ).
  int index = ti + (m + 2) + 2 * (ti / m + 1) - 1;

  // In this phase, every thread brings its element and its upper and lower neighbors to the shared mem.
  // Special care is provided to "edge" points, that also need to load "halo points".

  if (threadIdx.x == 0) {                     // 1st element of the block ("edge" point)
    u_tmp[blockDim.x + 2] = u_old[index - 1]; // Load center left "halo point"
  }

  if (threadIdx.x == blockDim.x - 1) {            // Last element ("edge" point)
    u_tmp[2 * blockDim.x + 3] = u_old[index + 1]; // Load center right "halo point"
  }

  u_tmp[1 + threadIdx.x] = u_old[index - (m + 2)];    // Load upper
  u_tmp[blockDim.x + 3 + threadIdx.x] = u_old[index]; // Load center
  u_tmp[2 * blockDim.x + 5 + threadIdx.x] = u_old[index + (m + 2)]; // Load lower

  double fX = (xLeft + (x - 1) * deltaX), fY = (yBottom + (y - 1) * deltaY);
  double fX_sq = fX * fX, fY_sq = fY * fY;
  double fX_dot_fY_sq = fX_sq * fY_sq;
  int tmp_index = ((blockDim.x + 2) * 3);  // Points to the beginning of the error array.

  __syncthreads();

  // Calculate!
  double updateVal = (u_tmp[blockDim.x + threadIdx.x + 2] + u_tmp[blockDim.x + threadIdx.x + 4]) * cx_cc + // left, right
                     (u_tmp[1 + threadIdx.x] + u_tmp[2 * blockDim.x + threadIdx.x + 5]) * cy_cc +          // up, down
                     u_tmp[blockDim.x + threadIdx.x + 3] +                                                 // self
                     c1 * (1.0 - fX_sq - fY_sq + fX_dot_fY_sq) -
                     c2 * (fX_dot_fY_sq - 1.0);

  // Update u
  u[index] = u_tmp[blockDim.x + threadIdx.x + 3] - relax * updateVal;

#ifdef CONVERGE_CHECK_TRUE  
  // Update error
  u_tmp[tmp_index + threadIdx.x] = updateVal * updateVal;
  
  int stride = blockDim.x / 2;

  __syncthreads();
  // In this phase, the block-wide error is calculated.
  // That is, every thread-local error is reduced to a global block-error sum.

  while (stride > 0)  // Perform a tree-like reduction in O(log(blockDim)) steps.
  {   
    if (threadIdx.x > stride)
      break; 

    u_tmp[tmp_index + threadIdx.x] += u_tmp[tmp_index + threadIdx.x + stride - 1];
    stride >>= 1;
  }

  if (threadIdx.x == 0) {  // Reduction finished -> store block error to grid error array.
    error_matrix[blockIdx.x] = u_tmp[tmp_index];
  }
#endif
}

// Perform a tree-like reduction to "error_matrix" elements, in O(log(stride)) steps.
__global__ void kernel_reduce_error(double *error_matrix, int stride)
{
  int ti = threadIdx.x + blockIdx.x * blockDim.x; // get thread id

  if (ti >= stride) // Required in cases where the number of elements
    return;         // is *not* a multiple of threads per block (aka 1024) eg. 1680x1680/1024=2756.25 -> 2757 blocks

  error_matrix[ti] = error_matrix[ti] + error_matrix[ti + stride];
}


int main(int argc, char **argv)
{
  int mits, allocCount, iterationCount, maxIterationCount, stride;
  double alpha, tol, maxAcceptableError, error;
  double *u, *u_old, *tmp, *error_matrix;

  scanf("%d,%d", &h_n, &h_m);
  scanf("%lf", &alpha);
  scanf("%lf", &h_relax);
  scanf("%lf", &tol);
  scanf("%d", &mits);
  printf("-> %d, %d, %g, %g, %g, %d\n", h_n, h_m, alpha, h_relax, tol, mits);

  allocCount = (h_n + 2) * (h_m + 2);

  // Allocate arrays in both CPU and GPU memory.
  double *h_u, *h_u_old, *h_error_matrix;
  h_u = (double *) calloc(allocCount, sizeof(double));
  h_u_old = (double *) calloc(allocCount, sizeof(double));
  h_error_matrix = (double *) calloc(allocCount, sizeof(double));
  
  CUDA_SAFE_CALL(cudaMalloc(&u, allocCount * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&u_old, allocCount * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&error_matrix, allocCount * sizeof(double))); 
  CUDA_SAFE_CALL(cudaMemcpy(u, h_u, allocCount * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(u_old, h_u_old, allocCount * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(error_matrix, h_error_matrix, allocCount * sizeof(double), cudaMemcpyHostToDevice));

  maxIterationCount = mits;
  maxAcceptableError = tol;

  // Solve in [-1, 1] x [-1, 1]
  h_xLeft = h_yBottom = -1.0;
  h_xRight = h_yUp = 1.0;

  h_deltaX = (h_xRight - h_xLeft) / (h_n - 1);
  h_deltaY = (h_yUp - h_yBottom) / (h_m - 1);

  iterationCount = 0;
  error = HUGE_VAL;

  timestamp t_start;
  t_start = getTimestamp();

  h_maxXCount = h_n + 2;
  h_maxYCount = h_m + 2;

  double cx = 1.0 / (h_deltaX * h_deltaX);
  double cy = 1.0 / (h_deltaY * h_deltaY);
  double cc = -2.0 * (cx + cy) - alpha;
  double div_cc = 1.0 / cc;

  h_cx_cc = 1.0 / (h_deltaX * h_deltaX) * div_cc;
  h_cy_cc = 1.0 / (h_deltaY * h_deltaY) * div_cc;
  h_c1 = (2.0 + alpha) * div_cc;
  h_c2 = 2.0 * div_cc;

  initGPU();  // Pass constant values to GPU.

  // Set blocks and threads per block.
  int BLOCK_SIZE = 128;
  printf("GPU Threads used per block: %d\n", BLOCK_SIZE);
  dim3 dimBl(BLOCK_SIZE);
  dim3 dimGr(FRACTION_CEILING(h_n * h_m, BLOCK_SIZE));

  /* Iterate as long as it takes to meet the convergence criterion */
#ifdef CONVERGE_CHECK_TRUE
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
#else
    while (iterationCount < maxIterationCount)
#endif
  {
    iterationCount++;

    // Run kernel
    one_jacobi_iteration<<<dimGr, dimBl, ((BLOCK_SIZE + 2) * 3 + BLOCK_SIZE + 1) * sizeof(double)>>>(u, u_old, error_matrix);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef CONVERGE_CHECK_TRUE
    stride = h_n * h_m / 2;
    while (stride > 0)  // Calculate the residual error.
    {
      int BLOCK_SIZE = ( (stride < 128 ? stride : 128) );
      dim3 dimBl(BLOCK_SIZE);
      dim3 dimGr(FRACTION_CEILING(stride, BLOCK_SIZE));

      kernel_reduce_error<<<dimGr, dimBl>>>(error_matrix, stride);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      stride >>= 1;
    }

    CUDA_SAFE_CALL(cudaMemcpy(h_error_matrix, error_matrix, sizeof(double), cudaMemcpyDeviceToHost));
    error = sqrt(h_error_matrix[0]) / (h_n * h_m);
#endif
  
    // Swap the buffers
    tmp = u_old;
    u_old = u;
    u = tmp;
  }

  float msec = getElapsedtime(t_start);

  printf("Time taken: %f seconds\n", msec / 1000.0);
  printf("Iterations: %d\n", iterationCount);
  printf("Residual: %g\n", error); // :(

  CUDA_SAFE_CALL(cudaMemcpy(h_u_old, u_old, allocCount * sizeof(double), cudaMemcpyDeviceToHost));

  // u_old holds the solution after the most recent buffers swap
  double absoluteError =
      checkSolution(h_xLeft, h_yBottom, h_n + 2, h_m + 2, h_u_old, h_deltaX, h_deltaY, alpha);
  printf("The error of the iterative solution is %g\n", absoluteError);

  free(h_u);
  free(h_u_old);
  free(h_error_matrix);

  CUDA_SAFE_CALL(cudaFree(u));
  CUDA_SAFE_CALL(cudaFree(u_old));
  CUDA_SAFE_CALL(cudaFree(error_matrix));
  return 0;
}

/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount,
                     double *u, double deltaX, double deltaY, double alpha)
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

void initGPU(void)
{
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(n, &h_n, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(m, &h_m, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(maxXCount, &h_maxXCount, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(maxYCount, &h_maxYCount, sizeof(int), 0, cudaMemcpyHostToDevice));
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
