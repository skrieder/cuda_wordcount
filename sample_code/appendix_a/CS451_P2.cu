/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "book.h"

//#define LIMIT 64//default: 65536
#define ELEMENTS 64//Default 65536 for your dataset
#define SIZE ELEMENTS*sizeof(unsigned int)

__host__ __device__ void iterate(unsigned int* table);
__host__ __device__ unsigned int get(unsigned int* table, unsigned int key);

__device__ void put(unsigned int* table, unsigned int key){
        atomicAdd(&table[key], 1);
}

__host__ __device__ unsigned int get(unsigned int* table, unsigned int key){
  	unsigned int ret = table[key];//(unsigned long)location2->value;
  	return ret;
}

__global__ void add_to_table( unsigned int *keys, unsigned int* table ) {
  	// get the thread id
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int stride = blockDim.x * gridDim.x;// total num of threads.

  	while (tid < ELEMENTS) {//stripe
		unsigned int key = keys[tid]; 
		printf("add_to_table: key = %u, tid = %d, table[tid] = %u\n", key, tid, table[tid]);
    		put(table, key);
    		tid += stride;
  }
	__syncthreads();
}

// copy table back to host, verify elements are there
void verify_table( const unsigned int* dev_table ) {
	unsigned int* host_table;
	host_table = (unsigned int*)calloc(ELEMENTS, sizeof(unsigned int));
	HANDLE_ERROR( cudaMemcpy( host_table, dev_table, ELEMENTS * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
  	iterate(host_table);
  	printf("END VERIFY TABLE\n");
}

__host__ __device__ void iterate(unsigned int* table){
  	for(int i=0; i<ELEMENTS; i++){
    		printf("[%d]: {", i);
		unsigned key = i;
		printf("key = %u ",key);
		printf("value = %u}\n",table[key]);
  	}
}

int main( void ) {
  	printf("Starting main.\n");
  	printf("Elements = %u\n", ELEMENTS);

  	unsigned int *dev_keys;
	unsigned int *dev_table;

	unsigned int *buffer = (unsigned int*)calloc(1, SIZE);
	for (int i=0; i<ELEMENTS;i++){
                buffer[i]= ELEMENTS - 1;//i;
        }

  // allocate memory on the device for keys and copy to device
	HANDLE_ERROR( cudaMalloc( (void**)&dev_keys, SIZE ) );
  	HANDLE_ERROR( cudaMemcpy( dev_keys, buffer, SIZE, cudaMemcpyHostToDevice ) );

  // Initialize table on device
	HANDLE_ERROR( cudaMalloc( (void**)&dev_table, ELEMENTS * sizeof(unsigned int)) );
	HANDLE_ERROR( cudaMemset( dev_table, 0, ELEMENTS * sizeof(unsigned int) ) );

  	printf("Calling GPU func\n");
  // this launches 60 blocks with 256 threads each, each block is scheduled on a SM without any order guarantees

  	add_to_table<<<60,256>>>( dev_keys, dev_table);
  	cudaDeviceSynchronize();

  	verify_table( dev_table );

  	HANDLE_ERROR( cudaFree( dev_keys ) );
  	free( buffer );
  	return 0;
}
