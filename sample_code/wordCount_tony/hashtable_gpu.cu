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

#define SIZE    (100*1024*1024)
#define ELEMENTS    (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES     1024


struct Entry {
    unsigned int    key;
    void            *value;
    Entry           *next;
};

struct Table {
    size_t  count;
    Entry   **entries;
    Entry   *pool;
};

// TODO - This function needs to be modified to generate a hash based on a strong input
__device__ __host__ size_t hash( unsigned int key,
                                 size_t count ) {
    return key % count;
}

void initialize_table( Table &table, int entries,
                       int elements ) {
    table.count = entries;
    HANDLE_ERROR( cudaMalloc( (void**)&table.entries,
                              entries * sizeof(Entry*)) );
    HANDLE_ERROR( cudaMemset( table.entries, 0,
                              entries * sizeof(Entry*) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&table.pool,
                               elements * sizeof(Entry)) );
}

void copy_table_to_host( const Table &table, Table &hostTable) {
    hostTable.count = table.count;
    hostTable.entries = (Entry**)calloc( table.count,
                                         sizeof(Entry*) );
    hostTable.pool = (Entry*)malloc( ELEMENTS *
                                     sizeof( Entry ) );

    HANDLE_ERROR( cudaMemcpy( hostTable.entries, table.entries,
                              table.count * sizeof(Entry*),
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( hostTable.pool, table.pool,
                              ELEMENTS * sizeof( Entry ),
                              cudaMemcpyDeviceToHost ) );
    /*
    for (int i=0; i<table.count; i++) {
        if (hostTable.entries[i] != NULL)
            hostTable.entries[i] =
                (Entry*)((size_t)hostTable.entries[i] -
                (size_t)table.pool + (size_t)hostTable.pool);
    }
    for (int i=0; i<ELEMENTS; i++) {
        if (hostTable.pool[i].next != NULL)
            hostTable.pool[i].next =
                (Entry*)((size_t)hostTable.pool[i].next -
                (size_t)table.pool + (size_t)hostTable.pool);
    }
    */
}

void free_table( Table &table ) {
    HANDLE_ERROR( cudaFree( table.pool ) );
    HANDLE_ERROR( cudaFree( table.entries ) );
}

/* global add_to_table 
This function runs on the CUDA device. It takes a list of keys and void ** values along with a table and an array of locks. All of the threads in the current execution will stride across the array and insert relevant items into the table.*/

__global__ void add_to_table( unsigned int *keys, void **values, Table table, Lock *lock ) {

  // get the thread id for the current cuda thread context
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // set a stride based on cuda thread info
  int stride = blockDim.x * gridDim.x;

  // walk the data and hash and insert
  while (tid < ELEMENTS) {
    unsigned int key = keys[tid];
    size_t hashValue = hash( key, table.count );
    for (int i=0; i<32; i++) {
      if ((tid % 32) == i) {
	Entry *location = &(table.pool[tid]);
	location->key = key;
        
	// TODO - Rather than setting this to the value of a TID you would need to 
	// get the current value and add 1 for each occurrence of the hash
	location->value = values[tid];
	lock[hashValue].lock();
	location->next = table.entries[hashValue];
	table.entries[hashValue] = location;
	lock[hashValue].unlock();
      }
    }
    // thread id is increased by the size of the stride to get the next new chunk of data and avoid overwriting anything that is complete
    tid += stride;
  }
}

__host__ __device__ unsigned long get(Table table, unsigned int key){
	size_t hashValue = hash(key, table.count);
	Entry *location = &(table.pool[hashValue]);
//	location->key = key;
	unsigned long ret = location->value;
	return ret;
}


// copy table back to host, verify elements are there
void verify_table( const Table &dev_table ) {
    Table   table;
    
    // move table to host
    copy_table_to_host( dev_table, table );

    int count = 0;
    for (size_t i=0; i<table.count; i++) {
        Entry   *current = table.entries[i];
        while (current != NULL) {
            ++count;
            if (hash( current->key, table.count ) != i)
                printf( "%d hashed to %ld, but was located at %ld\n",
                        current->key,
                        hash(current->key, table.count), i );
            current = current->next;
        }
    }
    if (count != ELEMENTS)
        printf( "%d elements found in hash table.  Should be %ld\n",
                count, ELEMENTS );
    else
        printf( "All %d elements found in hash table.\n", count );

    free( table.pool );
    free( table.entries );
}


int main( void ) {
    
  // generates a large array of integers for the input data
  /* TODO - rather than generate a large block of int's you want to read from a text file and build an array of (char *)'s */

  unsigned int *buffer = (unsigned int*)big_random_block( SIZE );
  unsigned int *dev_keys;
  void         **dev_values;

  // allocate memory on the device
  HANDLE_ERROR( cudaMalloc( (void**)&dev_keys, SIZE ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_values, SIZE ) );

  // move the input data to the device
  HANDLE_ERROR( cudaMemcpy( dev_keys, buffer, SIZE, cudaMemcpyHostToDevice ) );

  // copy the values to dev_values here
  // filled in by user of this code example
  Table table;
  initialize_table( table, HASH_ENTRIES, ELEMENTS );

  // create a lock array for each entry
  Lock    lock[HASH_ENTRIES];
  // create a device pointer for locks
  Lock    *dev_lock;

  // allocate the device lock array
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lock, HASH_ENTRIES * sizeof( Lock ) ) );
  // move the lock array to the GPU
  HANDLE_ERROR( cudaMemcpy( dev_lock, lock, HASH_ENTRIES * sizeof( Lock ), cudaMemcpyHostToDevice ) );

  // start a cuda event
  cudaEvent_t     start, stop;
  HANDLE_ERROR( cudaEventCreate( &start ) );
  HANDLE_ERROR( cudaEventCreate( &stop ) );
  HANDLE_ERROR( cudaEventRecord( start, 0 ) );

  // call device function to parallel add to table
  // this launches 60 blocks with 256 threads each, each block is scheduled on a SM without any order guarantees
  add_to_table<<<60,256>>>( dev_keys, dev_values,table, dev_lock );

  // trigger event
  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );
  
  // print the timer
  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf( "Time to hash:  %3.1f ms\n", elapsedTime );

  // move table back and verify
  verify_table( table );

  // destroy CUDA event
  HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );
  
  // free memory
  free_table( table );
  HANDLE_ERROR( cudaFree( dev_lock ) );
  HANDLE_ERROR( cudaFree( dev_keys ) );
  HANDLE_ERROR( cudaFree( dev_values ) );
  free( buffer );
  return 0;
}
