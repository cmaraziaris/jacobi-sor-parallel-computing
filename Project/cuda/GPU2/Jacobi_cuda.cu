#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "timestamp.h"

#define CONVERGE_CHECK_TRUE 1

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

#define FRACTION_CEILING(numerator, denominator) \
  ((numerator + denominator - 1) / denominator)

// declare constant-device variables
__constant__ int n, m, maxXCount, maxYCount;
__constant__ double relax, cx_cc, cy_cc, c1, c2, xLeft, xRight, yBottom, yUp, deltaX, deltaY;

int h_n, h_m, h_maxXCount, h_maxYCount;
double h_relax, h_cx_cc, h_cy_cc, h_c1, h_c2, h_xLeft, h_xRight, h_yBottom, h_yUp, h_deltaX, h_deltaY;

// ON-HOST FUNCTIONS

// solution checker
double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount,
                     double *u, double deltaX, double deltaY, double alpha);

void initGPU(void);
void initGPUs(void);
double get_residual_error(double *error_matrix, int error_elements);

// ON-DEVICE FUNCTIONS

__global__ void kernel(double *u, double *u_old, double *error_matrix, int offset)
{
  // calculate x and y before do the following line
  int ti = threadIdx.x + blockIdx.x * blockDim.x; // get thread id

  if (ti >= (n * m) / 2) // Required in cases where the number of elements
    return;        // is *not* a multiple of threads per block (aka 1024) eg. 1680x1680/1024=2756.25 -> 2757 blocks

  ti += offset;
  int x = (ti % m);
  int y = (ti / n);

  extern __shared__ double u_tmp[];

  // we spawn n*m threads,
  // map "index" from indexing n*m elements -> (n+2)*(m+2) elements, including halos
  int index = ti + (m + 2) + 2 * (ti / m + 1) - 1;

  if (threadIdx.x == 0) {                     // 1st element
    u_tmp[blockDim.x + 2] = u_old[index - 1]; // center left
  }

  if (threadIdx.x == blockDim.x - 1) {            // last element
    u_tmp[2 * blockDim.x + 3] = u_old[index + 1]; // center right
  }

  u_tmp[1 + threadIdx.x] = u_old[index - (m + 2)];    // upper
  u_tmp[blockDim.x + 3 + threadIdx.x] = u_old[index]; // center
  u_tmp[2 * blockDim.x + 5 + threadIdx.x] = u_old[index + (m + 2)]; // lower

  double fX = (xLeft + (x - 1) * deltaX), fY = (yBottom + (y - 1) * deltaY);
  double fX_sq = fX * fX, fY_sq = fY * fY;
  double fX_dot_fY_sq = fX_sq * fY_sq;
  int tmp_index = ((blockDim.x + 2) * 3);

  __syncthreads();

  // do calculations
  double updateVal = (u_tmp[blockDim.x + threadIdx.x + 2] + u_tmp[blockDim.x + threadIdx.x + 4]) * cx_cc + // left, right
                     (u_tmp[1 + threadIdx.x] + u_tmp[2 * blockDim.x + threadIdx.x + 5]) * cy_cc +          // up, down
                     u_tmp[blockDim.x + threadIdx.x + 3] +                                                 // self
                     c1 * (1.0 - fX_sq - fY_sq + fX_dot_fY_sq) -
                     c2 * (fX_dot_fY_sq - 1.0);
  
// #ifdef CONVERGE_CHECK_TRUE
//   error_matrix[ti] = updateVal * updateVal;
// #endif
  u_tmp[tmp_index + threadIdx.x] = updateVal * updateVal;
  u[index] = u_tmp[blockDim.x + threadIdx.x + 3] - relax * updateVal;
#ifdef CONVERGE_CHECK_TRUE  
  int stride = blockDim.x / 2;

  __syncthreads();

  while (stride > 0)
  {   
    if (threadIdx.x > stride) // Required in cases where the number of elements
      break;         // is *not* a multiple of threads per block (aka 1024) eg. 1680x1680/1024=2756.25 -> 2757 blocks

    u_tmp[tmp_index + threadIdx.x] += u_tmp[tmp_index + threadIdx.x + stride - 1];
    stride >>= 1;
  }

  if (threadIdx.x == 0)
  {
    int index = ( offset == 0 ? 0 : blockDim.x );
    error_matrix[index + blockIdx.x] = u_tmp[tmp_index + 0];
  }
#endif
}


// NOTE: na valoume kai ton ari8miti apo to stride
__global__ void kernel_reduce_error(double *error_matrix, int stride)
{
  int ti = threadIdx.x + blockIdx.x * blockDim.x; // get thread id

  if (ti >= stride) // Required in cases where the number of elements
    return;         // is *not* a multiple of threads per block (aka 1024) eg. 1680x1680/1024=2756.25 -> 2757 blocks

  error_matrix[ti] = error_matrix[ti] + error_matrix[ti + stride];
}

int main(int argc, char **argv)
{
  int mits, allocCount, iterationCount, maxIterationCount;
  double alpha, tol, maxAcceptableError, error;
  double *u, *u_old, *tmp, *error_matrix;

  scanf("%d,%d", &h_n, &h_m);
  scanf("%lf", &alpha);
  scanf("%lf", &h_relax);
  scanf("%lf", &tol);
  scanf("%d", &mits);
  printf("-> %d, %d, %g, %g, %g, %d\n", h_n, h_m, alpha, h_relax, tol, mits);

  allocCount = (h_n + 2) * (h_m + 2);

  // set blocks and threads/block TODO: make it more generic
  int BLOCK_SIZE = 128;
  printf("GPU Threads used per block: %d\n", BLOCK_SIZE);
  dim3 dimBl(BLOCK_SIZE);
  dim3 dimGr(FRACTION_CEILING((h_n * h_m)/2, BLOCK_SIZE));

  // Those two calls also zero the boundary elements
  CUDA_SAFE_CALL(cudaMallocManaged(&u, allocCount * sizeof(double)));            // reserve memory in global unified address space
  CUDA_SAFE_CALL(cudaMallocManaged(&u_old, allocCount * sizeof(double)));        // reserve memory in global unified address space
#ifdef CONVERGE_CHECK_TRUE
  CUDA_SAFE_CALL(cudaMallocManaged(&error_matrix, 2*((int) dimGr.x) * sizeof(double))); // reserve memory in global unified address space
#endif

  int N = allocCount / 2;
  // int N1 = h_n*h_m/2;
  // CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaMemPrefetchAsync(u_old, N*sizeof(double), 0, NULL));
  CUDA_SAFE_CALL(cudaMemPrefetchAsync(u, N*sizeof(double), 0, NULL));
  // CUDA_SAFE_CALL(cudaSetDevice(1));
  CUDA_SAFE_CALL(cudaMemPrefetchAsync(u_old + N, N*sizeof(double), 1, NULL));
  CUDA_SAFE_CALL(cudaMemPrefetchAsync(u + N, N*sizeof(double), 1, NULL));

#ifdef CONVERGE_CHECK_TRUE
  int N1 = ((int) dimGr.x);
  CUDA_SAFE_CALL(cudaMemPrefetchAsync(error_matrix, N1*sizeof(double), 0, NULL));
  CUDA_SAFE_CALL(cudaMemPrefetchAsync(error_matrix + N1, N1*sizeof(double), 1, NULL));
#endif

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

  // pass_values_to_gpu();
  initGPUs();

  /* Iterate as long as it takes to meet the convergence criterion */
#ifdef CONVERGE_CHECK_TRUE
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
#else
    while (iterationCount < maxIterationCount)
#endif
  {
    iterationCount++;

    # pragma omp parallel for
    for (int gpu = 0; gpu < 2; ++gpu)
    {
      CUDA_SAFE_CALL(cudaSetDevice(gpu));
      kernel<<<dimGr, dimBl, ((BLOCK_SIZE + 2) * 3 + BLOCK_SIZE + 1) * sizeof(double)>>>(u, u_old, error_matrix, gpu*(h_n*h_m/2)); //xd /bruh
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

  #ifdef CONVERGE_CHECK_TRUE
    error = get_residual_error(error_matrix, 2*((int) dimGr.x));
    error = sqrt(error) / (h_n * h_m);
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

  // u_old holds the solution after the most recent buffers swap
  double absoluteError =
      checkSolution(h_xLeft, h_yBottom, h_n + 2, h_m + 2, u_old, h_deltaX, h_deltaY, alpha);
  printf("The error of the iterative solution is %g\n", absoluteError);

  return 0;
}

double get_residual_error(double *error_matrix, int error_elements) {

  int stride = error_elements / 4;
  while (stride > 0)
  {
    int BLOCK_SIZE = ( (stride < 128 ? stride : 128) );
    dim3 dimBl(BLOCK_SIZE);
    dim3 dimGr(FRACTION_CEILING(stride, BLOCK_SIZE));

    # pragma omp parallel for
    for (int gpu = 0; gpu < 2; ++gpu)
    {
      CUDA_SAFE_CALL(cudaSetDevice(gpu));
      kernel_reduce_error<<<dimGr, dimBl>>>(error_matrix + gpu * (error_elements/2), stride);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    stride >>= 1;
  }

  return error_matrix[0] + error_matrix[error_elements/2];
}

void initGPUs(void) {
  CUDA_SAFE_CALL(cudaSetDevice(0));
  initGPU();
  CUDA_SAFE_CALL(cudaSetDevice(1));
  initGPU();
}

void initGPU(void)
{ // bruh
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