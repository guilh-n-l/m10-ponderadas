from threading import Thread
from math import ceil
from random import randint
from sys import argv
from time import time, sleep

get_duration = lambda start_time: time() - start_time

get_duration.__doc__ = """Parameters:
    start_time: time() object to start duration counting from
Returns: Time duration between when function is called and start_time
"""

def split_list(list_, n):
    """
    Parameters:
        list_: List to split into n parts
        n: N° of parts to split list into
    Returns: List of slices
    """
    k, m = divmod(len(list_), n)
    return [list_[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def thread_func(i, list_, buf):
    """
    Parameters:
        i: index on buf
        list_: list_ with items to sum
        buf: buffer to write result in
    """
    buf[i] = sum(list_)


def sum_multithread_n_threads(list_, n_threads=2):
    """
    Parameters:
        list_: list_ with items to sum
        n_threads: n_threads to execute sum
    Returns: Sum of items in list
    """
    buf = [None] * n_threads
    slices = split_list(list_, n_threads)
    threads = [
        Thread(
            target=thread_func,
            args=(i, slices[i], buf),
        )
        for i in range(n_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(buf)


def sum_multithread_2_threads(list_):
    """
    Parameters:
        list_: list_ with items to sum
    Returns: Sum of items in list
    """
    buf = [0] * 2
    threads = [
        Thread(
            target=thread_func,
            args=(0, list_[: len(list_) // 2], buf),
        ),
        Thread(
            target=thread_func,
            args=(1, list_[len(list_) // 2 :], buf),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(buf)


def sum_singlethread(list_):
    """
    Parameters:
        list_: list_ with items to sum
    Returns: Sum of items in list
    """
    collector = 0
    for i in list_:
        collector += i
    return collector


# sleep_ = lambda: sleep(5)


# def sleep_multithread(n):
#     """
#     Runs sleep_ function in n threads
#     Parameters:
#         n: Number of thread to sleep
#     """
#     threads = [Thread(target=sleep_) for _ in range(n)]
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()


# def sleep_singlethread(n):
#     """
#     Runs sleep_ function n times in a single thread
#     Parameters:
#         n: Number of times to sleep
#     """
#     for _ in range(n):
#         sleep_()


def main(argv):
    if len(argv) < 2:
        raise Exception("Enter list size")
    list_ = [randint(1, 100) for _ in range(int(argv[1]))]
    print(list_)
    start = time()
    print(f"Sum using multithread: {sum_multithread_n_threads(list_)}")
    # sleep_multithread(5)
    print(f"Time using multithread: {get_duration(start)}u.t")
    start = time()
    print(f"Sum using singlethread: {sum_singlethread(list_)}")
    # sleep_singlethread(5)
    print(f"Time using singlethread: {get_duration(start)}u.t")


if __name__ == "__main__":
    main(argv)
