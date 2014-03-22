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


#include "../common/book.h"
#include "lock.h"

#define LIMIT 64//default: 65536
#define ELEMENTS LIMIT
#define SIZE LIMIT*sizeof(unsigned int)

__host__ __device__ void iterate(unsigned int* table);
__host__ __device__ unsigned int get(unsigned int* table, unsigned int key);

__device__ void put(unsigned int* table, unsigned int key, Lock *lock, int tid){
	atomicAdd(&table[key], 1);
}


void initialize_table( unsigned int *table, int entries, int elements ) {
	HANDLE_ERROR( cudaMalloc( (void**)&table, elements * sizeof(unsigned int)) );
  	printf("In init table, elements = %d\n", elements);
}


__host__ __device__ unsigned int get(unsigned int* table, unsigned int key){
  	unsigned int ret = table[key];//(unsigned long)location2->value;
  	return ret;
}

__global__ void add_to_table( unsigned int *keys, void **values, unsigned int* table, Lock *lock ) {
  	// get the thread id
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int stride = blockDim.x * gridDim.x;// total num of threads.

  while (tid < ELEMENTS) {//stripe
	unsigned int key = keys[tid]; //Tony: should be here.
	printf("add_to_table: key = %u, tid = %d, table[tid] = %u\n", key, tid, table[tid]);
    	put(table, key, lock, tid);
    	tid += stride;
  }
	__syncthreads();
}

// copy table back to host, verify elements are there
void verify_table( const unsigned int* dev_table ) {
	unsigned int* host_table;
  	printf("Before copy table to host.\n");
	host_table = (unsigned int*)calloc(LIMIT, sizeof(unsigned int));
	HANDLE_ERROR( cudaMemcpy( host_table, dev_table, LIMIT * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
  	printf("After copy table to host.\n");
  	printf("ITERATE FROM VERIFY:\n");
  	iterate(host_table);
  	printf("END ITERATE FROM VERIFY:\n");
  	printf("END VERIFY TABLE\n");
}

__host__ __device__ void iterate(unsigned int* table){
  printf("Start iterate table\n");

  for(int i=0; i<LIMIT; i++){
    	printf("[%d]: {", i);
	unsigned key = i;
	printf("key = %u ",key);
	printf("value = %u}\n",table[key]);
  }
  printf("End iterate table\n");
}


int main( void ) {
  	printf("Starting main.\n");
  	printf("Elements = %u\n", ELEMENTS);

  	unsigned int *dev_keys;
	unsigned int *dev_table;
	unsigned int *host_table;
  	void **dev_values;

	unsigned int *buffer = (unsigned int*)calloc(1, SIZE);
	for (int i=0; i<ELEMENTS;i++){
                buffer[i]= ELEMENTS - 1;//i;
        }

  // allocate memory on the device
	HANDLE_ERROR( cudaMalloc( (void**)&dev_keys, SIZE ) );
  	HANDLE_ERROR( cudaMalloc( (void**)&dev_values, SIZE ) );
  	HANDLE_ERROR( cudaMemcpy( dev_keys, buffer, SIZE, cudaMemcpyHostToDevice ) );

  // copy the values to dev_values here
  // filled in by user of this code example
	HANDLE_ERROR( cudaMalloc( (void**)&dev_table, ELEMENTS * sizeof(unsigned int)) );
	HANDLE_ERROR( cudaMemset( dev_table, 0, ELEMENTS * sizeof(unsigned int) ) );
  	printf("Table initialized from host\n");

	Lock    lock[LIMIT];
  // create a device pointer for locks
  	Lock    *dev_lock;

  // allocate the device lock array
	HANDLE_ERROR( cudaMalloc( (void**)&dev_lock, LIMIT * sizeof( Lock ) ) );
  // move the lock array to the GPU
	HANDLE_ERROR( cudaMemcpy( dev_lock, lock, LIMIT * sizeof( Lock ), cudaMemcpyHostToDevice ) );
  // start a cuda event
  	cudaEvent_t     start, stop;

  	printf("Calling GPU func\n");
  // this launches 60 blocks with 256 threads each, each block is scheduled on a SM without any order guarantees

  	add_to_table<<<60,256>>>( dev_keys, dev_values, dev_table, dev_lock );
  	cudaDeviceSynchronize();
  	printf("GPU Call done\n");

  	float   elapsedTime;

  	verify_table( dev_table );
  	printf("After verify table\n");


  	printf("Before free table\n");  
  	printf("After free table\n");
  	HANDLE_ERROR( cudaFree( dev_lock ) );
  	HANDLE_ERROR( cudaFree( dev_keys ) );
  	HANDLE_ERROR( cudaFree( dev_values ) );
  	free( buffer );
  	return 0;
}
