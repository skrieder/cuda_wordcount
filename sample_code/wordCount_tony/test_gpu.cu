#include "../common/book.h"
#include "lock.h"
#include <iostream>
#include <stdio.h>
#include <string.h> //no use in GPU




#define SIZE    (100*1024*1024)
#define ELEMENTS    (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES     1024

struct Entry {
	unsigned int key;
	void *value;
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

__device__ __host__ unsigned long hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
        return hash % mod; 
}

__global__ void tokenize (const char* string, char* nextToken){
	//nextToken = strtok (string, ", .");
}

__global__ void hashTest(unsigned char* str, unsigned long* hashValue){
	*hashValue = hash_sdbm(str, 1000000);
}

__global__ void kernel( void ) {
	
}


int main ()
{
	unsigned long hash_size = 1024*1024;
	unsigned long hashValue =0;
  	unsigned char str[] ="This, a sample string.";

	unsigned char* dev_str;
	unsigned long* dev_hashValue;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_str, sizeof(str)));
	HANDLE_ERROR( cudaMalloc( (void**)&dev_hashValue, sizeof(unsigned long)));
	HANDLE_ERROR( cudaMemcpy( dev_str, &str, sizeof(str), cudaMemcpyHostToDevice));
	
	hashTest<<<1,1>>>(dev_str, dev_hashValue);	

	HANDLE_ERROR( cudaMemcpy( &hashValue, dev_hashValue, sizeof(unsigned long), cudaMemcpyDeviceToHost ));	

	printf("The string is \"%s\", the CPU computed hash value is %lu, and the GPU computed hash value is %lu\n", str, hash_sdbm(str, hash_size), hashValue);  	
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
