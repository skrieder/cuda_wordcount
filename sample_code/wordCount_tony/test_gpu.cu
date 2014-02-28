#include "../common/book.h"
#include "lock.h"
#include <iostream>
#include <stdio.h>
#include <string.h> //no use in GPU




#define SIZE    (100*1024*1024)
#define ELEMENTS    (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES     1024

unsigned long TABLE_SIZE = 0;
struct Bucket{
    char key[10];
    unsigned long count;
	Lock lock;
};

struct Hashtable{
	unsigned long count;
	Bucket* table; // table has to be of length count
	//Lock lock[count];
};

__device__ __host__ unsigned long hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c=0;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
        //printf("hash value before mod: %lu, and ", hash );
        //printf("mod: %lu\n", mod );
        return hash % mod;
}

__device__ void put(unsigned char* key, Bucket* table, unsigned long mod){
	unsigned long index = hash_sdbm(key, mod);
	
	table[index].lock.lock();
	
	//table[index]->key = key;
	memcpy(table[index].key, key, sizeof(*key));
	table[index].count ++;
	
	table[index].lock.unlock();
	
	//printf()

}

__host__ __device__ void initTable(unsigned long size, Hashtable* i_table){
	i_table = (Hashtable*)malloc(sizeof(Hashtable));	
	i_table->count = size;
	i_table->table = (Bucket*)malloc(size * sizeof(Bucket));
}

void copy_table_to_host(const Hashtable &devTable, Hashtable &hostTable) {	
	hostTable.count = devTable.count;
	unsigned long count = devTable.count;
	hostTable.table = (Bucket*) malloc(count * sizeof(Bucket));

	HANDLE_ERROR(cudaMemcpy(hostTable.table, devTable.table, count* sizeof(Bucket), cudaMemcpyDeviceToHost) );
}
/*
void copy_table_to_host(const Table &table, Table &hostTable) {
	hostTable.count = table.count;
	hostTable.entries = (Entry**) calloc(table.count, sizeof(Entry*));
	hostTable.pool = (Entry*) malloc( ELEMENTS * sizeof(Entry));

	HANDLE_ERROR(
			cudaMemcpy(hostTable.entries, table.entries,
					table.count * sizeof(Entry*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(
			cudaMemcpy( hostTable.pool, table.pool, ELEMENTS * sizeof( Entry ), cudaMemcpyDeviceToHost ));
}
*/

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

  

  return 0;
}
