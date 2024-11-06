#include <string.h>
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

int sumArray(int arrLen, int* arr) {
    int counter = 0;
    for (int i = 0; i < arrLen; i++) {
        counter += arr[i];
    }
    return counter;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int arrLen = atoi(argv[1]);
    if (arrLen <= 0) {
        printf("0\n");
        return 0;
    }

    printf("Array: [ ");
    int *arr = malloc(arrLen * sizeof(int));

    for (int i=0; i < arrLen; i++) {
        arr[i] = rand() % 100;
        printf("%d ", arr[i]);
    }

    printf("]\n");

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
    int collector = 0;
    #pragma omp parallel num_threads(numThreads) reduction(+:collector)
    {
        int threadNum = omp_get_thread_num();

        int threadRangeArr[2];
        getThreadRange(threadNum, numThreads, arrLen, threadRangeArr);
        #pragma omp atomic
        collector += sumArray(threadRangeArr[1], &arr[threadRangeArr[0]]);
    }
    double ss = omp_get_wtime();

    printf("\nN threads used: %d\nSum: %d\nElapsed time: %.8f\n\n", numThreads, collector, ss - s);

    s = omp_get_wtime();
    collector = sumArray(arrLen, arr);
    ss = omp_get_wtime();

    printf("N threads used: 1\nSum: %d\nElapsed time: %.8f\n", collector, omp_get_wtime() - s);
    free(arr);
    return 0;
}
