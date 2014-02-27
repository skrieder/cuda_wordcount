#! /bin/bash
src=$1
fileName=`echo $src| awk -F "." ' {print $1}'`
rm $fileName
nvcc -arch=sm_20 -g $src -o $fileName
