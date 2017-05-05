#!/bin/bash

echo "size,k,iter,t"

repetitions=5

for n in 1600 3200 # iterate over cores
do
    for (( i=0; i<${repetitions}; i++)) # repetitions
    do
	for k in 3 5 10 20 # iterate over k
	do
	    for numIter in  100 1000 5000 #iterate over numIter
	    do
	        CILK_NWORKERS=4 ./template ${n}.ppm ${k} ${numIter}
	    done
	done
    done
done

