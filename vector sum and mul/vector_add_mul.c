#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000  // Number of elements (10 million)

// Serial function for vector addition
void vector_add(double *A, double *B, double *C) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// Serial function for vector multiplication
void vector_multiply(double *A, double *B, double *C) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * B[i];
    }
}

int main() {
    double *A, *B, *C;
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

    // Time serial vector addition
    clock_t start = clock();
    vector_add(A, B, C);
    clock_t end = clock();
    double add_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Vector Addition C[%d] = %f\n", 9999999, C[9999999]);
    printf("Serial Vector Addition Time: %f seconds\n", add_time);

    // Time serial vector multiplication
    start = clock();
    vector_multiply(A, B, C);
    end = clock();
    double multiply_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Vector Multiplication C[%d] = %f\n", 9999999, C[9999999]);
    printf("Serial Vector Multiplication Time: %f seconds\n", multiply_time);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}