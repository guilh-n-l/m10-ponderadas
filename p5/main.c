#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

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

    int  *threadsArr;
    #pragma omp parallel
    {
        if (numThreads == -1) {
            #pragma omp critical
            {
                int threads = omp_get_num_threads();
                numThreads =  threads > arrLen ? arrLen : threads;
                threadsArr = malloc(numThreads * sizeof(int));
            }
        }
    }
    double s = omp_get_wtime();
    #pragma omp parallel
    {

        int threadNum = omp_get_thread_num();

        if (threadNum + 1 <= arrLen) {
            int threadRangeArr[2];

            getThreadRange(threadNum, numThreads, arrLen, threadRangeArr);

            threadsArr[threadNum] = sumArray(threadRangeArr[1], &arr[threadRangeArr[0]]);
        }   
    }

    printf("\nN threads used: %d\nSum: %d\nElapsed time: %.8f\n\n", numThreads, sumArray(numThreads, threadsArr), omp_get_wtime() - s);
    free(threadsArr);

    s = omp_get_wtime();
    printf("N threads used: 1\nSum: %d\nElapsed time: %.8f\n", sumArray(arrLen, arr), omp_get_wtime() - s);
    free(arr);
    return 0;
}
