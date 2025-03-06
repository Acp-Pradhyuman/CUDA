#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000  // Matrix size (1K)

void matrix_multiply_serial(double **A, double **B, double **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double **A, **B, **C;
    
    // Allocate memory for matrices
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }
    
    // Initialize matrices A and B with random values
    // srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // A[i][j] = (double)rand() / RAND_MAX;
            // B[i][j] = (double)rand() / RAND_MAX;
            A[i][j] = i + 1;
            B[i][j] = N - i;
        }
    }
    
    // Record start time
    clock_t start = clock();
    
    // Perform matrix multiplication
    matrix_multiply_serial(A, B, C);
    
    // Record end time
    clock_t end = clock();
    
    // Print execution time
    double serial_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Serial C[%d][%d] = %f\n", N-1, N-1, C[N-1][N-1]);
    printf("Serial Time: %f seconds\n", serial_time);
    
    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}