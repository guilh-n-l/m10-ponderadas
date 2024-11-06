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
void getThreadRange(int threadNum, int numThreads, int arrLen, int* range) {
    int sliceLen = arrLen / numThreads;
    range[0] = threadNum * sliceLen;
    range[1] = threadNum == numThreads - 1 ? arrLen - range[0] : (arrLen < range[0] ? 1 : sliceLen);
}


/**
 * @brief Sum all numbers inside an array
 *
 * This function computes the sum of every integer inside an array
 *
 * @param arrLen Length of array
 * @param arr Pointer to start of array
 *
 * @return Sum of integers in array
 */
int sumArray(unsigned int arrLen, int* arr) {
    if (arrLen == 0) {
        free(arr);
        abort();
    }

    int counter = 0;
    for (int i = 0; i < arrLen; i++) {
        counter += arr[i];
    }
    return counter;
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
 * @brief Adds all numbers inside an array with multiple threads
 *
 * This function computes the sum of every integer inside an array using multiple threads
 *
 * @param arrLen Length of array
 * @param arr Pointer to start of array
 *
 * @return Sum of integers in array
 */
int sumArrayMultithread(unsigned int arrLen, int *arr, unsigned int numThreads) {

    if (numThreads == 0 || arrLen == 0) {
        free(arr);
        abort();
    }

    int collector = 0;

    #pragma omp parallel num_threads(numThreads) reduction(+:collector)
    {
        int threadNum = omp_get_thread_num();

        int threadRangeArr[2];
        getThreadRange(threadNum, numThreads, arrLen, threadRangeArr);
        #pragma omp atomic
        collector += sumArray(threadRangeArr[1], &arr[threadRangeArr[0]]);
    }

    return collector;
}


/**
 * @brief Print an array of integers
 *
 * This function prints an array of integers to stdin
 *
 * @param arrLen Length of array
 * @param arr Pointer to start of array
 */
void printArrayString(unsigned int arrLen, int* arr) {
    printf("[ ");
    for (int i = 0; i < arrLen; i++) {
        printf("%d ", arr[i]);
    }
    printf("]\n");
}


int main(int argc, char* argv[]) {
    unsigned int arrLen = (unsigned int)atoi(argv[1]);

    int *arr = malloc(arrLen * sizeof(int));
    randomArray(arrLen, arr, -100, 100);
    printArrayString(arrLen, arr);

    int numThreads = -1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            int threads = omp_get_num_threads();
            numThreads =  MAX(threads > arrLen / 2 ? arrLen / 2: threads, 1);
        }
    }

    double s = omp_get_wtime();
    int collector = sumArrayMultithread(arrLen, arr, numThreads);
    double ss = omp_get_wtime();

    printf("\nN threads used: %d\nSum: %d\nElapsed time: %.8f\n\n", numThreads, collector, ss - s);

    s = omp_get_wtime();
    collector = sumArray(arrLen, arr);
    ss = omp_get_wtime();

    printf("N threads used: 1\nSum: %d\nElapsed time: %.8f\n", collector, omp_get_wtime() - s);
    free(arr);
    return 0;
}
