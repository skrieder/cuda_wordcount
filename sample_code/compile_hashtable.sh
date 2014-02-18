#! /bin/bash

nvcc -arch=sm_11 appendix_a/hashtable_gpu.cu -o hashtable_gpu