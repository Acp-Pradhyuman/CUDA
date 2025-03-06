# CUDA: Parallel Programming for GPU-Based Systems

## Overview

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to harness the processing power of NVIDIA GPUs to perform general-purpose computing tasks. CUDA provides a rich set of tools, libraries, and APIs to write parallel code in C, C++, and Fortran, enabling significant speedups in computationally intensive applications by offloading workloads to the GPU.

## Key Features of CUDA

- **GPU-Accelerated Parallelism**: CUDA enables the parallelization of computational tasks on the GPU, allowing for massive performance improvements in applications such as scientific computing, machine learning, and image processing.
- **GPU Memory Model**: CUDA operates on a GPU's memory, allowing fine-grained control over memory hierarchy and access patterns.
- **Thread Management**: CUDA provides the ability to control thousands of threads and their execution on the GPU, facilitating high-performance computing tasks.
- **Scalable**: CUDA can scale efficiently as the size of the problem grows, with the GPU capable of executing many threads concurrently.
- **Portable**: CUDA is supported on all CUDA-enabled NVIDIA GPUs and is available across multiple platforms.

## Why Use CUDA?

CUDA provides a way to accelerate applications by offloading compute-heavy tasks to the GPU. This is particularly useful for applications with:

1. **Improved Performance**: By utilizing the parallel architecture of GPUs, CUDA can deliver significant speedups for data-heavy computations.
2. **Scalability**: As the size of the problem grows, CUDA provides a highly scalable solution to exploit the power of GPUs for large-scale data processing.
3. **Ease of Use**: CUDA offers a straightforward API and integration with existing C, C++, and Fortran codebases, allowing developers to add parallelism with minimal changes.
4. **Parallel Control**: CUDA provides fine-grained control over the GPU, allowing users to customize thread execution, memory access, and synchronization.

## How CUDA Works

CUDA enables the parallel execution of kernels (functions executed by threads on the GPU). Here's a simple example of a CUDA kernel that performs vector addition:

### Example: Vector Addition in CUDA

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1000;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    size_t size = N * sizeof(int);
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "C[0] = " << h_C[0] << ", C[N-1] = " << h_C[N-1] << std::endl;

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

## Explanation of the Code

### `__global__`
This is a CUDA kernel declaration. The `__global__` keyword marks a function to be executed on the GPU.

### `vectorAdd<<<blocksPerGrid, threadsPerBlock>>>`
This syntax launches the `vectorAdd` kernel with a grid of blocks, each containing a number of threads. The threads execute the `vectorAdd` function in parallel on the GPU.

### `threadIdx`, `blockIdx`, and `blockDim`
These are built-in variables in CUDA that provide information about the thread's position within its block and the block's position within the grid.

## Key Concepts in CUDA

### 1. CUDA Kernels
A kernel is a function that runs on the GPU. Each thread in a kernel executes the same code but operates on different data.

```cpp
__global__ void kernelFunction() {
    // Kernel code executed by each thread
}
```

### 2. Memory Management
CUDA provides several memory spaces:

- **Global memory**: Accessible by all threads.
- **Shared memory**: Shared by threads within a block.
- **Local memory**: Private to each thread.
- **Constant memory**: Read-only memory for all threads.

### 3. Thread Hierarchy
CUDA threads are organized into blocks, and blocks are organized into grids. The organization helps in parallelizing the workload effectively.

- **Threads**: The smallest unit of execution.
- **Blocks**: A group of threads.
- **Grids**: A group of blocks.

### 4. Synchronization
CUDA provides synchronization mechanisms to ensure safe access to memory and control the execution of threads:

- **`__syncthreads()`**: Synchronizes threads within a block.
- **Device-wide synchronization**: Handled by CUDA runtime, e.g., `cudaDeviceSynchronize()`.

## Performance Considerations

While CUDA allows for tremendous performance gains, there are several important factors to consider:

- **Memory Access Patterns**: Proper memory access patterns can significantly improve performance, particularly for global memory.
- **Thread Divergence**: Threads in a warp (group of 32 threads) should execute the same instructions to maximize performance.
- **Occupancy**: The number of threads per block should be optimized for the specific GPU to achieve maximum occupancy.

## Conclusion

CUDA is an excellent choice for parallelizing compute-heavy applications, especially those that can leverage the massively parallel architecture of GPUs. By using CUDA, developers can unlock the power of NVIDIA GPUs to accelerate tasks and improve performance.

For more information on CUDA, visit the official documentation: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

---
## CUDA Toolkit Installation

To install the CUDA Toolkit on both Linux and Windows Subsystem for Linux (WSL), follow the steps below:

### Step 1: Update Your System
Before installing the CUDA Toolkit, ensure your system is up-to-date:

```bash
sudo apt update
sudo apt upgrade -y
```
### Step 2: Install the NVIDIA CUDA Toolkit
Install the CUDA Toolkit using the following command:

```bash
sudo apt install nvidia-cuda-toolkit
```
This command installs the necessary CUDA tools, libraries, and compilers.

---

### Step 3: Verify the Installation
After installation, verify that the CUDA Toolkit is installed correctly by checking the version of `nvcc`, the NVIDIA CUDA compiler:

```bash
nvcc --version
```
You should see output similar to the following, indicating the installed version of the CUDA Toolkit:
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on ...
Cuda compilation tools, release X.X, VX.X.X
```

---

### Step 4: Set Up Environment Variables (Optional)
If needed, set up environment variables for CUDA by adding the following lines to your `~/.bashrc` or `~/.zshrc` file:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Then, reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```
### Step 5: Test CUDA (Optional)
To ensure CUDA is working correctly, compile and run a sample CUDA program:

```bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```
If the installation is successful, you should see detailed information about your GPU.

---

### Notes for WSL Users
- Ensure the NVIDIA driver is installed on your Windows host system. WSL uses the driver installed on Windows to interact with the GPU.
- The CUDA Toolkit installed via `sudo apt install nvidia-cuda-toolkit` on WSL may not include all components available on a native Linux installation. For advanced use cases, consider installing CUDA directly from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

For more detailed instructions, refer to the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Acp-Pradhyuman/CUDA/blob/main/LICENSE.txt) file for details.