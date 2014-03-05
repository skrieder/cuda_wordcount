#! /bin/bash

nvcc -g -arch=sm_20 appendix_a/hashtable_gpu.cu -o hashtable_gpu