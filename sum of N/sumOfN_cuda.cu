#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000  // Number of elements (10 million)
#define THREADS_PER_BLOCK 256

// CUDA kernel to sum the array elements in parallel
__global__ void sum_kernel(double *data, double *result, int n) {
    __shared__ double shared_data[THREADS_PER_BLOCK];

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int index = thread_id;

    // Initialize shared memory with 0
    shared_data[threadIdx.x] = (index < n) ? data[index] : 0.0;

    // Synchronize threads within a block
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Only one thread in each block writes the sum of the block to the global result
    if (threadIdx.x == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

int main() {
    double *data = (double*)malloc(N * sizeof(double));
    double *d_data, *d_result, result = 0.0;

    // // Seed the random number generator and generate input data
    // srand(time(NULL));
    // FILE *file = fopen("input_data.txt", "w");
    for (int i = 0; i < N; i++) {
        // data[i] = ((double)rand() / RAND_MAX) * 1000.0;  // Random number between 0 and 1000
        // fprintf(file, "%f\n", data[i]);
        data[i] = i + 1;
    }
    // fclose(file);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_data, N * sizeof(double));
    cudaMalloc((void**)&d_result, sizeof(double));
    cudaMemcpy(d_data, data, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int block_size = THREADS_PER_BLOCK;
    int grid_size = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch CUDA kernel to compute the sum
    clock_t start_time = clock();
    sum_kernel<<<grid_size, block_size>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
    clock_t end_time = clock();

    // Copy the result back to the host
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Print the sum and execution time
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Parallel Sum: %f\n", result);
    printf("Parallel Time: %f seconds\n", time_taken);

    // Free memory
    free(data);
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}