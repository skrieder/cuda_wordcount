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
  void    *key;
  void    *value;
};

struct Table {
    size_t  count;
    Entry   **entries;
    Entry   *pool;
};

// TODO - This function needs to be modified to generate a hash based on a strong input
__device__ __host__ size_t hash( unsigned int key, size_t count ) {
  return key % count;
}

__host__ __device__ void iterate(Table table);
__device__ void put(Table table, unsigned int key, Lock *lock, int tid);
__host__ __device__ unsigned long get(Table table, void *key);

__host__ __device__ void new_iterate(Table table);


__device__ void put(Table table, void *key, Lock *lock, int tid){
  size_t hashValue = hash( (long)key, table.count );
  if (tid ==0){
    printf("HASH VALUE = %lu\n", hashValue);
  }
  for (int i=0; i<32; i++) {
    if ((tid % 32) == i) {
      Entry *location = &(table.pool[hashValue]);
      int temp_int;

      //char *temp_str = (char *)malloc(6);

            location->key = (void *)key;
      //      memcpy(location->key, temp_str, 6);

      temp_int = get(table, key);
      location->value = (void *)(temp_int + 1);
      lock[hashValue].lock();
      //      location->next = table.entries[hashValue];
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

    // 0 over 1014
    for (int i=0; i<table.count; i++) {
      if (hostTable.entries[i] != NULL){
	//	int x = (size_t)table.pool;
	//int y = (size_t)hostTable.pool;
      
	//	printf("[%d]: SIZE OF TABLE.POOL = %d, SIZE OF hostTABLE.pool = %d\n", i, x, y);

      hostTable.entries[i] = 
	  (Entry*)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool);
    }
}

    // 0 over 26M
    /*
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

__host__ __device__ unsigned long get(Table table, void *key){
  size_t hashValue = hash((long)key, table.count);
//	printf("In get: table.count= %lu\n", table.count);
//	printf("key = %d\n", key);
//	printf("hashValue = %lu\n", hashValue);
	
	Entry *location2 = &(table.pool[hashValue]);
//      location->key = key;
        unsigned long ret = (unsigned long)location2->value;

//        printf("In Get: ret = %lu\n", ret);
//	printf("In Get: location->value = %lu\n", (unsigned long) location2->value);
	return ret;
}

__device__ void zero_out_values_in_table(Table table){
  printf("In zero out table\n");
  //  int count = table.count;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0){
    //    Entry *temp_entry = table.entries[0];
    Entry *pool_entry = table.pool;
    memset ( (void *) pool_entry, 0, 1024*sizeof(Entry));
    
    //    for(int j=0; j<count; j++){
    //  temp_entry = &(table.pool[j]);
    //  memset ( (void *) temp_entry, 0, 24);
    //}
    //    printf("ITERATE IN ZERO OUT TABLE\n");
    //    iterate(table);
    //    printf("End zero out table\n");
  }
}

__global__ void add_to_table( unsigned int *keys, void **values, Table table, Lock *lock ) {
  // get the thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  //  unsigned int key = keys[tid];

  char *temp_str = "hello";
  
  void *key = (void *)malloc(6);

  memcpy(key, temp_str, 6);

  if(tid==0){
    
    printf("KEY ====== %s\n", (char *)key);
    printf("unsigned int %lu\n", sizeof(unsigned int));
    printf("TABLE COUNT = %lu\n", table.count);
    zero_out_values_in_table(table);
  }

  while (tid < ELEMENTS) {
    put(table, key, lock, tid);
    tid += stride;
  }
}

// copy table back to host, verify elements are there
void verify_table( const Table &dev_table ) {
    Table   table;

    printf("Before copy table to host.\n");
    // move table to host
    copy_table_to_host( dev_table, table );
    printf("After copy table to host.\n");

    // iterate table
    printf("ITERATE FROM VERIFY:\n");
    iterate(table);
    printf("END ITERATE FROM VERIFY:\n");
    /*
    int count = 0;
    for (size_t i=0; i<table.count; i++) {
        Entry   *current = table.entries[i];
        while (current != NULL) {
            ++count;
            if (hash( current->key, table.count ) != i)
	      printf( "%d hashed to %ld, but was located at %ld\n", current->key, hash(current->key, table.count), i );
            current = current->next;
        }
    }
    if (count != ELEMENTS)
        printf( "%d elements found in hash table.  Should be %ld\n",
                count, ELEMENTS );
    else
        printf( "All %d elements found in hash table.\n", count );
    */
    free( table.pool );
    free( table.entries );
    printf("END VERIFY TABLE\n");
}

__host__ __device__ void iterate(Table table){
  printf("Start iterate table\n");
  Entry *test_location;
 
  for(int i=0; i<HASH_ENTRIES; i++){
    printf("In for loop\n");
    // test_location = &(table.pool[i]);
    printf("test location set\n");
    printf("[%d]: {\n", i);
    //    char *temp_str = (char *)malloc(5);
    //  memcpy(temp_str, test_location->key, 5);
    //    printf("key = %s \n", (char *)test_location->key);
    printf("key = %s \n", (char *)(table.pool[i].key));
    
    //    printf("value = %lu}\n", (unsigned long)test_location->value);
    //    free(temp_str);
  }
  printf("End iterate table\n");
}

/*
__host__ __device__ void new_iterate(Table table){
  printf("Start iterate table\n");
  Entry *test_location = table.entries[0];
  printf("key = %l ,", (long)test_location->key);
  printf("value = %lu}\n", (unsigned long)test_location->value);


  for(int i=0; i<HASH_ENTRIES+1; i++){
    test_location = &(table.pool[i]);
    printf("[%d]: {", i);
    printf("key = %d ", test_location->key);
    printf("value = %lu}\n", (unsigned long)test_location->value);
  }

  printf("End iterate table\n");
}
*/

int main( void ) {
  printf("Starting main.\n");
  printf("Elements = %lu\n", ELEMENTS);
  // generates a large array of integers for the input data
  /* TODO - rather than generate a large block of int's you want to read from a text file and build an array of (char *)'s */

  //  void *str_buffer = (void *)

  void *buffer = (void *)big_random_block( SIZE );

  unsigned int *dev_keys;
  void **dev_values;

  // allocate memory on the device
  HANDLE_ERROR( cudaMalloc( (void**)&dev_keys, SIZE ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_values, SIZE ) );
  printf("Before memset\n");
  //  HANDLE_ERROR( cudaMemset( (void *)&dev_values, 0, SIZE) );
  printf("After memset\n");

  printf("On the host:\n");

  char * int_buff = (char *)buffer;

  printf("The buffer[0] = %c\n", int_buff[0]);
  printf("The buffer[1] = %c\n", int_buff[1]);
  // move the input data to the device
  HANDLE_ERROR( cudaMemcpy( dev_keys, buffer, SIZE, cudaMemcpyHostToDevice ) );

  // copy the values to dev_values here
  // filled in by user of this code example
  Table table;
  initialize_table( table, HASH_ENTRIES, ELEMENTS );
  printf("Table initialized from host\n");

  // create a host value

  // zero host value

  // copy host value into device value
  //  printf("ITERATE IN MAIN AFTER INIT:\n");
  //  iterate(table);

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

  printf("Calling GPU func\n");
  // call device function to parallel add to table
  // this launches 60 blocks with 256 threads each, each block is scheduled on a SM without any order guarantees
  add_to_table<<<60,256>>>( dev_keys, dev_values, table, dev_lock );
  cudaDeviceSynchronize();
  printf("GPU Call done\n");

  // trigger event
  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );
  
  // print the timer
  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf( "Time to hash:  %3.1f ms\n", elapsedTime );

  // move table back and verify
  verify_table( table );
  printf("After verify table\n");

  // destroy CUDA event
  HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );

  printf("Before free table\n");  
  // free memory
  free_table( table );
  printf("After free table\n");
  HANDLE_ERROR( cudaFree( dev_lock ) );
  HANDLE_ERROR( cudaFree( dev_keys ) );
  HANDLE_ERROR( cudaFree( dev_values ) );
  free( buffer );
  return 0;
}
