#include "../common/book.h"
#include "lock.h"
#include <iostream>
#include <stdio.h>
#include <string.h> //no use in GPU




#define SIZE    (100*1024*1024)
#define ELEMENTS    (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES     1024

unsigned long TABLE_SIZE = 0;

struct Entry {
	unsigned int key;
	void *value;
	unsigned long count;
	Entry *next;
};

struct Table {
	size_t count;
	Entry **entries;
	Entry *pool;
};

__device__ __host__ size_t hash(unsigned int key, size_t count) {
	return key % count;
}

void initialize_table(Table &table, int entries, int elements) {
	table.count = entries;
	HANDLE_ERROR(cudaMalloc((void** )&table.entries, entries * sizeof(Entry*)));
	HANDLE_ERROR(cudaMemset(table.entries, 0, entries * sizeof(Entry*)));
	HANDLE_ERROR(cudaMalloc((void** )&table.pool, elements * sizeof(Entry)));
}


void copy_table_to_host(const Table &table, Table &hostTable) {
	hostTable.count = table.count;
	hostTable.entries = (Entry**) calloc(table.count, sizeof(Entry*));
	hostTable.pool = (Entry*) malloc( ELEMENTS * sizeof(Entry));

	HANDLE_ERROR(cudaMemcpy(hostTable.entries, table.entries, table.count * sizeof(Entry*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy( hostTable.pool, table.pool, ELEMENTS * sizeof( Entry ), cudaMemcpyDeviceToHost));

	for (int i = 0; i < table.count; i++) {
		if (hostTable.entries[i] != NULL)
			hostTable.entries[i] = (Entry*) ((size_t) hostTable.entries[i] - (size_t) table.pool + (size_t) hostTable.pool);
	}

	for (int i = 0; i < ELEMENTS; i++) {
		if (hostTable.pool[i].next != NULL)
			hostTable.pool[i].next = (Entry*) ((size_t) hostTable.pool[i].next - (size_t) table.pool + (size_t) hostTable.pool);
	}
}

void free_table(Table &table) {
	HANDLE_ERROR(cudaFree(table.pool));
	HANDLE_ERROR(cudaFree(table.entries));
}

__device__ void put(char* key, Table table){
	
	
	//printf()

}

/*
__device__ unsigned long dev_hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c=0;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
        return hash % mod; 
}*/

__device__ __host__ unsigned long hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c=0;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
       	//printf("hash value before mod: %lu, and ", hash );
	//printf("mod: %lu\n", mod );
	return hash % mod;
}

__device__ void tokenize (const char* string, char* nextToken){
	//nextToken = strtok (string, ", .");
}

__global__ void hashTest(unsigned char* str, unsigned long mod, unsigned long* hashValue){
	*hashValue = hash_sdbm(str, mod);
}

__global__ void kernel( void ) {
	
}


int main ()
{
	unsigned long hash_size = 1024*1024;
	unsigned long hashValue =0;
  	unsigned char str[] ="This, a sample string.";
	unsigned long mod =1000000;
	TABLE_SIZE = mod;
	unsigned char* dev_str;
	unsigned long* dev_hashValue;
	//unsigned long* dev_mod;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_str, sizeof(str)));
	HANDLE_ERROR( cudaMalloc( (void**)&dev_hashValue, sizeof(unsigned long)));
	//HANDLE_ERROR( cudaMalloc( (void**)&dev_mod, sizeof(unsigned long)));	
	HANDLE_ERROR( cudaMemcpy( dev_str, &str, sizeof(str), cudaMemcpyHostToDevice));
	//HANDLE_ERROR( cudaMemcpy( dev_mod, &mod, sizeof(unsigned long), cudaMemcpyHostToDevice));
	hashTest<<<1,1>>>(dev_str, TABLE_SIZE, dev_hashValue);	

	HANDLE_ERROR( cudaMemcpy( &hashValue, dev_hashValue, sizeof(unsigned long), cudaMemcpyDeviceToHost ));	

	printf("The string is \"%s\", the CPU computed hash value is %lu, and the GPU computed hash value is %lu\n", str, hash_sdbm(str, TABLE_SIZE), hashValue);  	
	//kernel<<<1,1>>>();

  /*
  char * pch;
  printf ("Splitting string \"%s\" into tokens:\n",str);
  pch = strtok (str," ,.-");
  while (pch != NULL)
  {
    printf ("%s\n",pch);
    pch = strtok (NULL, " ,.-");
  }
  */


  return 0;
}
