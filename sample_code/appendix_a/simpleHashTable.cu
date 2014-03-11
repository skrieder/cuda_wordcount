// A simple GPGPU hashtable implementation, based on the code from "CUDA by Example."
// Scott J. Krieder - skrieder@iit.edu
// Tonglin Li - tli13@hawk.iit.edu
// Iman Sadooghi - isadoogh@iit.edu

#include "book.h"
#include "lock.h"

#define SIZE    (100*1024*1024)
#define ELEMENTS    (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES     1024

// an Entry contains a key and a value
struct Entry {
  unsigned int    key;
  void            *value;
};

// a table is a collection of Entry* which point to Entries in the pool
// the pool stores the actual data
struct Table {
  size_t  count;
  Entry   **entries;
  Entry   *pool;

};

// several header declerations for the newly written functions
__host__ __device__ void iterate(Table table);
__device__ void put(Table table, unsigned int key, Lock *lock, int tid);
__host__ __device__ unsigned long get(Table table, unsigned int key);


// a simple hashing function
__device__ __host__ size_t hash( unsigned int key, size_t count ) {
  return key % count;
}


// a simple put function, note that it take the Lock array as argument.
__device__ void put(Table table, unsigned int key, Lock *lock, int tid){
  size_t hashValue = hash( key, table.count );
  // on the gpu if you want to print something wrap it in a tid == 0
  // this will make only a single thread print instead of O(1000) prints
  if (tid ==0){
  }
  // 32 due to race conditions within warps, see book or Scott for more details
  for (int i=0; i<32; i++) {
    if ((tid % 32) == i) {
      // create a temp Entry *
      Entry *location = &(table.pool[hashValue]);
      int temp_int;
      // set the key in temp
      location->key = key;
      // get the value curently stored at that key, ie. first time through should be 0
      temp_int = get(table, key);
      // now grab the lock
      lock[hashValue].lock();
      // increase the stored value by one
      location->value = (void *)(temp_int + 1); 
      // update the actual table with your temp variable
      table.entries[hashValue] = location;
      // release the lock
      lock[hashValue].unlock();
    }
  }
}

// init the table, called from host
void initialize_table( Table &table, int entries, int elements ) {
  // how many entries are in the entire table
  table.count = entries;
  // cuda malloc
  HANDLE_ERROR( cudaMalloc( (void**)&table.pool, elements * sizeof(Entry)) );
  HANDLE_ERROR( cudaMalloc( (void**)&table.entries, entries * sizeof(Entry*)) );
  // memset and clear out the entry pointers to zero
  HANDLE_ERROR( cudaMemset( table.entries, 0, entries * sizeof(Entry*) ) );
}

// copy the table back to the host, called from host
void copy_table_to_host( const Table &table, Table &hostTable) {
  // set the count
  hostTable.count = table.count;
  // zero out the entries again
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

// free the tables
void free_table( Table &table ) {
  HANDLE_ERROR( cudaFree( table.pool ) );
  HANDLE_ERROR( cudaFree( table.entries ) );
}

// a simple get function
__host__ __device__ unsigned long get(Table table, unsigned int key){
  size_t hashValue = hash(key, table.count);
  Entry *location2 = &(table.pool[hashValue]);
  unsigned long ret = (unsigned long)location2->value;
  return ret;
}

// Use this before on device, before you start to put 
// This will zero out the table
__device__ void zero_out_values_in_table(Table table){
  printf("In zero out table\n");
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0){
    Entry *pool_entry = table.pool;
    memset ( (void *) pool_entry, 0, 1024*sizeof(Entry));
  }
}

// prints the table to the screen
__host__ __device__ void iterate(Table table){
  Entry *test_location;

  for(int i=0; i<HASH_ENTRIES; i++){
    test_location = &(table.pool[i]);
    printf("[%d]: {", i);
    printf("key = %d ", test_location->key);
    printf("value = %lu}\n", (unsigned long)test_location->value);
  }
}
