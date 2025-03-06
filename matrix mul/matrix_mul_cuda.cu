#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000  // Matrix size (1K)
#define TILE_SIZE 16  // Tile size for block size

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply_kernel(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double value = 0.0;
        for (int k = 0; k < n; k++) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

// Host function for matrix multiplication
void matrix_multiply_cuda(double *A, double *B, double *C, int n) {
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, n * n * sizeof(double));
    cudaMalloc((void **)&d_B, n * n * sizeof(double));
    cudaMalloc((void **)&d_C, n * n * sizeof(double));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Record start time
    double start = (double)clock() / CLOCKS_PER_SEC;

    // Launch the kernel
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Synchronize the device to ensure kernel completes
    cudaDeviceSynchronize();

    // Record end time
    double end = (double)clock() / CLOCKS_PER_SEC;

    // Check for errors in kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the result matrix from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Access the last element of C (C[N-1][N-1] in 1D array format)
    printf("CUDA C[%d][%d] = %f\n", N-1, N-1, C[(N-1) * N + (N-1)]);

    // Print execution time
    double cuda_time = end - start;
    printf("CUDA Time: %f seconds\n", cuda_time);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    double *A, *B, *C;

    // Allocate memory for matrices on host
    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // A[i * N + j] = (double)rand() / RAND_MAX;
            // B[i * N + j] = (double)rand() / RAND_MAX;
            A[i * N + j] = i + 1;
            B[i * N + j] = N - i;
        }
    }

    // Perform matrix multiplication using CUDA
    matrix_multiply_cuda(A, B, C, N);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}