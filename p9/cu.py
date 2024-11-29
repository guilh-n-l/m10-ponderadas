import numpy as np
import cupy as cp
import threading as th
import multiprocessing as mp
from sys import argv
from random import randint
from time import time

sum_cupy = lambda arr: cp.array(arr).sum()


def partial_sum(arrs, idx):
    arrs[idx] = sum(arrs[idx])


def sum_th(arr, n_threads=8):
    arrs = np.array_split(arr, n_threads)
    thrs = [th.Thread(target=partial_sum, args=(arrs, i)) for i in range(n_threads)]
    for thr in thrs:
        thr.run()
    return sum(arrs)


def sum_mp(arr, n_procs=8):
    arrs = np.array_split(arr, n_procs)
    thrs = [mp.Process(target=partial_sum, args=(arrs, i)) for i in range(n_procs)]
    for thr in thrs:
        thr.run()
    return sum(arrs)


def main(n):
    arr = cp.random.randint(0, 100, n)
    start = time()
    res = sum_th(arr)
    elapsed = (time() - start) * 1e3
    print(f"Threading: Sum = {res}, Elapsed time = {elapsed}")
    start = time()
    res = sum_mp(arr)
    elapsed = (time() - start) * 1e3
    print(f"Multiprocessing: Sum = {res}, Elapsed time = {elapsed}")
    start = time()
    res = sum_cupy(arr)
    elapsed = (time() - start) * 1e3
    print(f"CuPy: Sum = {res}, Elapsed time = {elapsed}")


if __name__ == "__main__":
    if len(argv) > 1:
        main(int(argv[1]))
    else:
        raise ValueError("Must provide size of array in argv")
