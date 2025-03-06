#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000  // Number of elements (10 million)

// Serial function for vector dot product
double vector_dot_product(double *A, double *B) {
    double dot_product = 0.0;
    for (int i = 0; i < N; i++) {
        dot_product += A[i] * B[i];
    }
    return dot_product;
}

int main() {
    double *A, *B;
    A = (double*)malloc(N * sizeof(double));
    B = (double*)malloc(N * sizeof(double));

    // Initialize the vectors with random values
    for (int i = 0; i < N; i++) {
        // A[i] = rand() % 1000 + 1;  // Random values between 1 and 1000
        // B[i] = rand() % 1000 + 1;
        A[i] = i + 1;
        B[i] = N - i;
    }

    // Measure the time taken by the serial dot product function
    clock_t start = clock();
    double result = vector_dot_product(A, B);
    clock_t end = clock();
    double serial_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Serial Dot Product: %f\n", result);
    printf("Serial Time: %f seconds\n", serial_time);

    // Cleanup
    free(A);
    free(B);

    return 0;
}