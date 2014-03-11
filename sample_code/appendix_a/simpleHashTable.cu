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
  //    Entry           *next;
};

struct Table {
  size_t  count;
  Entry   **entries;
  Entry   *pool;

};

__device__ __host__ size_t hash( unsigned int key, size_t count ) {
  return key % count;
}

__host__ __device__ void iterate(Table table);
__device__ void put(Table table, unsigned int key, Lock *lock, int tid);
__host__ __device__ unsigned long get(Table table, unsigned int key);

__device__ void put(Table table, unsigned int key, Lock *lock, int tid){
  size_t hashValue = hash( key, table.count );
  if (tid ==0){
//    printf("HASH VALUE = %lu\n", hashValue);
  }
  for (int i=0; i<32; i++) {
    if ((tid % 32) == i) {
      Entry *location = &(table.pool[hashValue]);
      int temp_int;

      location->key = key;
      temp_int = get(table, key);
      lock[hashValue].lock();
      location->value = (void *)(temp_int + 1); 
      table.entries[hashValue] = location;
      lock[hashValue].unlock();
    }
  }
}

void initialize_table( Table &table, int entries, int elements ) {
  table.count = entries;
  printf("In init table, entries = %d\n", entries);
  printf("In init table, elements = %d\n", elements);
  // cuda malloc
  HANDLE_ERROR( cudaMalloc( (void**)&table.pool, elements * sizeof(Entry)) );
  HANDLE_ERROR( cudaMalloc( (void**)&table.entries, entries * sizeof(Entry*)) );
  // memset
  HANDLE_ERROR( cudaMemset( table.entries, 0, entries * sizeof(Entry*) ) );
}


void copy_table_to_host( const Table &table, Table &hostTable) {
  hostTable.count = table.count;
  hostTable.entries = (Entry**)calloc( table.count, sizeof(Entry*) );
  hostTable.pool = (Entry*)malloc( ELEMENTS * sizeof( Entry ) );
  HANDLE_ERROR( cudaMemcpy( hostTable.entries, table.entries, table.count * sizeof(Entry*), cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy( hostTable.pool, table.pool, ELEMENTS * sizeof( Entry ), cudaMemcpyDeviceToHost ) );
  for (int i=0; i<table.count; i++) {
    if (hostTable.entries[i] != NULL){
      hostTable.entries[i] =
        (Entry*)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool);
    }
  }
}

void free_table( Table &table ) {
  HANDLE_ERROR( cudaFree( table.pool ) );
  HANDLE_ERROR( cudaFree( table.entries ) );
}

__host__ __device__ unsigned long get(Table table, unsigned int key){
  size_t hashValue = hash(key, table.count);
  Entry *location2 = &(table.pool[hashValue]);
  unsigned long ret = (unsigned long)location2->value;
  return ret;
}

//Use this before on device, before you start to put 
__device__ void zero_out_values_in_table(Table table){
  printf("In zero out table\n");
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0){
    Entry *pool_entry = table.pool;
    memset ( (void *) pool_entry, 0, 1024*sizeof(Entry));
  }
}

__host__ __device__ void iterate(Table table){
  Entry *test_location;

  for(int i=0; i<HASH_ENTRIES; i++){
    test_location = &(table.pool[i]);
    printf("[%d]: {", i);
    printf("key = %d ", test_location->key);
    printf("value = %lu}\n", (unsigned long)test_location->value);
  }
  printf("End iterate table\n");
}
