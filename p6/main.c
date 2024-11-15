#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#define MAX(a,b) (((a)>(b))?(a):(b))



/**
 * @brief Set thread start index and how many numbers to sum
 *
 * This function computes the start and amount for thread to calculate sum in array
 *
 * @param threadNum Index of thread
 * @param numThreads Total number of executing threads
 * @param arrLen Length of array
 * @param range Pointer to start of range array
 */
void getThreadRange(int threadNum, int nThreads, int arrLen, int* range) {
    int sliceLen = arrLen / nThreads;
    range[0] = threadNum * sliceLen;
    range[1] = threadNum == nThreads - 1 ? arrLen - range[0] : (arrLen < range[0] ? 1 : sliceLen);
}

/**
 * @brief Get random array from time(NULL)
 *
 * This function computes a N sized array with random integers from min to max (exclusive)
 *
 * @param n Size of array
 * @param arr Array to put integers into
 * @param min Minimum integer (inclusive)
 * @param max Max integer (exclusive)
 */
void randomArray(unsigned int n, int *arr, int min, int max) {
    srand(time(NULL));

    if (n == 0 || min > max) {
        free(arr);
        abort();
    }

    #pragma omp target map(to: arr[0:n])
    {
        #pragma omp parallel for 
        for (int i=0; i < n; i++) {
            arr[i] = rand() % (max - min) + min;
        }
    }
}

/**
 * @brief Print integer array
 *
 * This function prints an integer array in a nice human readable format to stdout
 *
 * @param n Size of array
 * @param arr Array to put integers into
 */
void printIntArray(int n, int* arr) {
    printf("[ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("]\n");
}

/**
 * @brief Get max in array
 *
 * This function computes the max integer inside an array
 *
 * @param arrLen Length of array
 * @param arr Pointer to start of array
 *
 * @return Max integer in array
 */
int arrayMax(int arrLen, int *arr) {
    int max = INT_MIN;
    for (int i = 0; i < arrLen; i++) {
        max = MAX(max, arr[i]);
    }

    return max;
}

/**
 * @brief Get max in array with multiple threads
 *
 * This function computes the max integer inside an array using multiple threads
 *
 * @param arrLen Length of array
 * @param arr Pointer to start of array
 * @param nThreads Number of threads to use
 *
 * @return Max integer in array
 */
int arrayMaxMultithread(int arrLen, int* arr, int nThreads) {
    int max = INT_MIN;

    #pragma omp parallel num_threads(nThreads)
    {
        int range[2];
        getThreadRange(omp_get_thread_num(), nThreads, arrLen, range);
        int threadMax = arrayMax(range[1], arr + range[0]);
        #pragma omp critical
        {
            max = MAX(max, threadMax);
        }
    }
    return max;
}

int main(int argc, char* argv[]) {
    unsigned int arrLen = (unsigned int)atoi(argv[1]);

    int *arr = malloc(arrLen * sizeof(int));
    randomArray(arrLen, arr, -100, 100);
    printIntArray(arrLen, arr);

    int numThreads = INT_MIN;
    #pragma omp parallel
    {
        #pragma omp single
        {
            int threads = omp_get_num_threads();
            numThreads =  MAX(threads > arrLen / 2 ? arrLen / 2: threads, 1);
        }
    }

    double s = omp_get_wtime();
    int max = arrayMaxMultithread(arrLen, arr, numThreads);
    double ss = omp_get_wtime();

    printf("\nN threads used: %d\nMax: %d\nElapsed time: %.8f\n\n", numThreads, max, ss - s);

    s = omp_get_wtime();
    max = arrayMax(arrLen, arr);
    ss = omp_get_wtime();

    printf("N threads used: 1\nMax: %d\nElapsed time: %.8f\n\n", max, ss - s);
    free(arr);
    return 0;
}
