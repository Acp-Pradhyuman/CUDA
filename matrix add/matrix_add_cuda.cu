#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000  // 10K
#define TILE_SIZE 16

// CUDA Kernel to perform matrix addition
__global__ void matrix_addition(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

int main() {
    int n = N;  // Size of the matrix
    size_t size = n * n * sizeof(double);
    
    // Host matrices
    double *A = (double*)malloc(size);
    double *B = (double*)malloc(size);
    double *C = (double*)malloc(size);

    // Initialize matrices A and B with some values (e.g., random values)
    // srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // A[i * n + j] = rand() % 1000 + 1;
            // B[i * n + j] = rand() % 1000 + 1;
            A[i * n + j] = i;
            B[i * n + j] = N - i;
        }
    }

    // Device matrices
    double *d_A, *d_B, *d_C;
    
    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Measure time for parallel execution on the GPU
    double start_time = (double)clock();
    
    matrix_addition<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for GPU to finish before accessing the result
    cudaDeviceSynchronize();
    
    double end_time = (double)clock();
    double parallel_time = (end_time - start_time) / CLOCKS_PER_SEC;
    
    // Copy data from devive to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // printf("Parallel (CUDA) Execution C[%d][%d] = %lf\n", N-1, N-1, C[N-1][N-1]);
    printf("Parallel (CUDA) Execution C[%d] = %lf\n", N-1, C[(N-1) * N + (N-1)]);
    printf("Parallel (CUDA) Execution Time: %lf seconds\n", parallel_time);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    
    return 0;
}