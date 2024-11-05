#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

void get_thread_range(int thread_num, int num_threads, int arr_len, int* range) {
    int slice_size = arr_len / num_threads;
    range[0] = thread_num * slice_size;
    if (thread_num == num_threads - 1) {
        range[1] = arr_len - 1;
    } else {
        range[1] = range[0] + (slice_size - 1);
    }
    if (range[1] < range[0]) {
        range[1] = range[0];
    }
}

int sum_thread_results(int num_threads, int arr_len, int* arr) {
    int collector = 0;
    int slice_size = arr_len / num_threads;
    for (int thread_num = 0; thread_num < num_threads; thread_num++) {
        collector += arr[thread_num * slice_size];
    }
    return collector;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int arr_len = atoi(argv[1]);
    if (arr_len <= 0) {
        printf("0\n");
        return 0;
    }
    printf("Array: [ ");
    int *arr = malloc(arr_len * sizeof(int));
    for (int i=0; i < arr_len; i++) {
        arr[i] = rand() % 100;
        printf("%d ", arr[i]);
    }
    printf("]\n");
    int num_threads = 1;
    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        if (thread_num + 1 <= arr_len) {
            if (omp_get_num_threads() > arr_len) {
                num_threads = arr_len;
            } else {
                num_threads = omp_get_num_threads();
            }
            int arr_thread[2];
            get_thread_range(thread_num, num_threads, arr_len, arr_thread);
            for (int i = arr_thread[0] + 1; i < arr_thread[1]; i++) {
                arr[arr_thread[0]] += i;
            }
        }   
    }
    printf("\nSum: %d\nN threads used: %d\n", sum_thread_results(num_threads, arr_len, arr), num_threads);
    free(arr);
    return 0;
}
