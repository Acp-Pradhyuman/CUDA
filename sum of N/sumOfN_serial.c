#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000  // Number of elements (10 million)

int main() {
    double *data = (double*)malloc(N * sizeof(double));
    double sum = 0.0;

    // // Seed the random number generator
    // srand(time(NULL));

    // Generate random double precision numbers and sum them
    for (int i = 0; i < N; i++) {
        // data[i] = ((double)rand() / RAND_MAX) * 1000.0;  // Random number between 0 and 1000
        data[i] = i + 1;
    }

    // Serial summing of numbers
    clock_t start_time = clock();
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    clock_t end_time = clock();

    // Print the sum and execution time
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Serial Sum: %f\n", sum);
    printf("Serial Time: %f seconds\n", time_taken);

    free(data);
    return 0;
}