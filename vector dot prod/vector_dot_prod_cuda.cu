#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 10000000  // Define the size of the vectors
#define THREADS_PER_BLOCK 256

// CUDA kernel for partial dot product
__global__ void vector_dot_kernel(double *A, double *B, double *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];  // Calculate individual products
    }
}

// CUDA kernel to sum the partial results
__global__ void reduce_sum(double *C, double *sum) {
    __shared__ double shared_data[THREADS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N) {
        shared_data[tid] = C[idx];
    } else {
        shared_data[tid] = 0.0;
    }
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Store the block's result in the global memory
    if (tid == 0) {
        atomicAdd(sum, shared_data[0]);
    }
}

int main() {
    double *A, *B, *C;
    double *d_A, *d_B, *d_C, *d_sum;
    double sum = 0.0;

    // Allocate memory for the vectors
    A = (double*)malloc(N * sizeof(double));
    B = (double*)malloc(N * sizeof(double));
    C = (double*)malloc(N * sizeof(double));

    // Initialize the vectors with random values
    for (int i = 0; i < N; i++) {
        // A[i] = rand() % 1000 + 1;  // Random values between 1 and 1000
        // B[i] = rand() % 1000 + 1;
        A[i] = i + 1;
        B[i] = N - i;
    }

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_C, N * sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));

    // Copy input vectors from host to device
    cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize sum on device
    cudaMemcpy(d_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel for partial dot product
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_dot_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);

    // Launch kernel to reduce the partial results
    reduce_sum<<<blocks, THREADS_PER_BLOCK>>>(d_C, d_sum);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float parallel_time = 0.0f;
    cudaEventElapsedTime(&parallel_time, start, stop);

    // Copy the final sum from device to host
    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    printf("CUDA Dot Product: %f\n", sum);
    printf("CUDA Time: %f seconds\n", parallel_time / 1000.0);

    // Cleanup
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_sum);

    return 0;
}