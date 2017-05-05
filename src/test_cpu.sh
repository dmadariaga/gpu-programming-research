#!/bin/bash
gcc -std=gnu99 -o kmeans_cpu kmeans_cpu.c -fcilkplus -lcilkrts -lm

echo "unit,size,k,iter,t"

repetitions=5

for n in 1600 3200 # iterate over images
do
    for (( i=0; i<${repetitions}; i++)) # repetitions
    do
	for k in 3 5 10 20 # iterate over k
	do
	    for numIter in  100 1000 5000 10000 #iterate over numIter
	    do
	        CILK_NWORKERS=4 ./kmeans_cpu ../images/${n}.ppm ${k} ${numIter}
	    done
	done
    done
done

