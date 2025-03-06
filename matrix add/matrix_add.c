#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000  // 10K

void matrix_addition(double **A, double **B, double **C) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

int main() {
    // Initialize matrices
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    // Fill matrices A and B with some values (example: random values)
    // srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // A[i][j] = rand() % 1000 + 1;
            // B[i][j] = rand() % 1000 + 1;
            A[i][j] = i;
            B[i][j] = N - i;
        }
    }

    // Measure time for serial execution
    clock_t start = clock();
    
    matrix_addition(A, B, C);
    
    clock_t end = clock();
    double serial_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Serial Execution C[%d][%d] = %lf\n", N-1, N-1, C[N-1][N-1]);
    printf("Serial Execution Time: %lf seconds\n", serial_time);

    // Free memory
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