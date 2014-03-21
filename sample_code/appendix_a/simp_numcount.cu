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

#define HASH_ENTRIES     1024//1024 : how many hash slots in the table, ignore the collision problem.
#define SIZE    (4*1024)//100*1024*1024 //memory size, used for holding hash table keys
#define ELEMENTS  (SIZE / sizeof(unsigned int)) //default: no 4*. How many elements will be mapped into hashtable.
//#define HASH_ENTRIES     128//1024


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

// TODO - This function needs to be modified to generate a hash based on a strong input

__device__ __host__ size_t hash( unsigned int key, size_t count ) {

//	key = key*2654435761; //a hash fun.
  return key % count;
}

__host__ __device__ void iterate(Table table);
__device__ void put(Table table, unsigned int key, Lock *lock, int tid);
__host__ __device__ unsigned long get(Table table, unsigned int key);
__host__ __device__ void new_iterate(Table table);

int fileRead(FILE *fd, size_t size, void* buff){
        //buff = malloc(size);
        //memset(buff, '/0', size);
        //setvbuf(fd, NULL, _IONBF, size);
        fread(buff, size, 1, fd);
        return 0;
}


unsigned int lenStr(unsigned char *str){
        unsigned char *p=str;
        while(*p!='\0')
                p++;
        return(p-str);
}

//For CPU use only!!!
unsigned int tokenizeToLongArrayCPU(unsigned char* src, long * LongArray ){
        unsigned int i=0;
        unsigned char* pch;
        pch = (unsigned char*)strtok ((char*)src," ,.-");
        while (pch != NULL){
                long num = atol((char*)pch);
                LongArray[i] = num;
                pch =(unsigned char*) strtok (NULL, " ,.-");
                i++;
        }
        return i;
}


__device__ void put(Table table, unsigned int key, Lock *lock, int tid){
  size_t hashValue = hash( key, table.count );
  if (tid ==0){
    printf("HASH VALUE = %lu\n", hashValue);
  }
  for (int i=0; i<32; i++) {
    if ((tid % 32) == i) {
      Entry *location = &(table.pool[hashValue]);//hashValue
      int temp_int;

      location->key = key;
      temp_int = get(table, key);
	if(key!=0)
	//	printf("put(%u): get = %lu\n", key, temp_int);
      location->value = (void *)(temp_int + 1);
//      lock[hashValue].lock();
      //      location->next = table.entries[hashValue];
      table.entries[hashValue] = location;
//      lock[hashValue].unlock();

	printf("After put(%u): get(key) = %lu\n", key, get(table, key));

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
      //printf("[%d]: SIZE OF TABLE.POOL = %d, SIZE OF hostTABLE.pool = %d\n", i, x, y);

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
__device__ void zero_out_values_in_table(Table table){
  printf("In zero out table\n");
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0){
    Entry *pool_entry = table.pool;
    memset ( (void *) pool_entry, 0, 1024*sizeof(Entry));
    printf("ITERATE IN ZERO OUT TABLE\n");
    printf("End zero out table\n");
  }
}

__global__ void add_to_table( unsigned int *keys, void **values, Table table, Lock *lock ) {
  // get the thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  unsigned int key = keys[tid]; 

  if(tid==0){
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
  iterate(table);

  free( table.pool );
  free( table.entries );
  printf("END VERIFY TABLE\n");
}

__host__ __device__ void iterate(Table table){
  printf("Start iterate table\n");
  Entry *test_location;
 	int empty =0;
  for(int i=0; i<HASH_ENTRIES; i++){
    test_location = &(table.pool[i]);
    printf("[%d]: {", i);
    printf("key = %u ", test_location->key);
    printf("value = %lu}\n", (unsigned long)test_location->value);
	if(test_location->key==0 && (unsigned long)test_location->value == 0)
		empty++;
  }
  printf("End iterate table, empty slots found: %d\n", empty);
}

__host__ __device__ void new_iterate(Table table){
  printf("Start iterate table\n");
  Entry *test_location = table.entries[0];
  printf("key = %d ,", test_location->key);
  printf("value = %lu}\n", (unsigned long)test_location->value);

  /* 
  for(int i=0; i<HASH_ENTRIES+1; i++){
    test_location = &(table.pool[i]);
    printf("[%d]: {", i);
    printf("key = %d ", test_location->key);
    printf("value = %lu}\n", (unsigned long)test_location->value);
  }
  */
  printf("End iterate table\n");
}

int main( void ) {
  printf("Starting main.\n");
  printf("Elements = %d\n", ELEMENTS);
  // generates a large array of integers for the input data
  /* TODO - rather than generate a large block of int's you want to read from a text file and build an array of (char *)'s */

//  unsigned int *buffer = (unsigned int*)big_random_block( SIZE );

	unsigned int *buffer = (unsigned int*)calloc(1, ELEMENTS*sizeof(unsigned int));
//	unsigned int *buffer = (unsigned int*)big_random_block( SIZE );

	
	for (int i=0; i<ELEMENTS;i++){
		//printf("Old: buffer[%d]:%d\n", i, buffer[i]);
		buffer[i]= (unsigned int) i;//i;
	//	printf("New: buffer[%d]:%d\n", i, buffer[i]);
	}

	char* file = "numbers";
//	FILE* fd = fopen (file,"r");
//        unsigned char* buf;
      //  int size =1000000;
       // buf = (unsigned char*)malloc(SIZE);
      //  memset(buf, '\0', SIZE);
        //fileRead(fd, size, buf);
	//tokenizeToLongArrayCPU(buf,(long *) buffer); //buf: string, buffer: long array


  unsigned int *dev_keys;
  void **dev_values;

  // allocate memory on the device
  HANDLE_ERROR( cudaMalloc( (void**)&dev_keys, SIZE ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_values, SIZE ) );
  //printf("Before memset\n");
  //  HANDLE_ERROR( cudaMemset( (void *)&dev_values, 0, SIZE) );
  //printf("After memset\n");


/*
  printf("On the host:\n");
  printf("The buffer[0] = %d\n", buffer[0]);
  printf("The buffer[1] = %d\n", buffer[1]);

//Sample key for put:
	buffer[0]= 1;
	buffer[1]= 2;
	buffer[2]= 12345;
	buffer[3]= 2;
	buffer[4]= 3;
	buffer[5]= 12345;
	buffer[6]=3;
	buffer[7]=3;
buffer[8]=30;
buffer[9]=31;
buffer[10]=32;
buffer[11]=33;
buffer[12]=34;
buffer[13]=63;
buffer[14]=64;
buffer[11]=65; */
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
//60, 256
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
