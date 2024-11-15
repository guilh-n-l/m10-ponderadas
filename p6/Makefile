CC      = gcc
CFLAGS  = -std=c99 -O2 -lm -fopenmp

default: run

run: main
	@OMP_NUM_THREADS=8 ./main $(ARR_SIZE) && rm -f main

main: main.c
	@$(CC) $(CFLAGS) -o main main.c

.PHONY: clean
clean:
	rm -f main
