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

unsigned int lenStr(unsigned char *str){
	unsigned char *p=str;
	while(*p!='\0')
		p++;
	return(p-str);
}

__device__ __host__ unsigned long hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c=0;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
        printf("hash value before mod: %lu, and ", hash );
        printf("mod: %lu\n", mod );
        return hash % mod;
}

__device__ __host__ void put(unsigned char* key, Bucket* table, unsigned long mod){
	unsigned long index = hash_sdbm(key, mod);
	
//	table[index].lock.lock();
	printf("put: pre-count =%ln\n", table[index].count); 
	printf("put: pre-key=%s\n", table[index].key);
	unsigned int l = lenStr(key);
	printf("key len=%d\n",l);
	memcpy(table[index].key, key, l+1 );
	printf("get key=%s\n", table[index].key);
	table[index].count ++;
	printf("put: post-count =%ln\n", table[index].count);
	printf("put: post-key=%s\n", table[index].key);
//	table[index].lock.unlock();
	

}

__host__ __device__ void initTable(unsigned long size, Hashtable* i_table){
	i_table = (Hashtable*)malloc(sizeof(Hashtable));	
	i_table->count = size;
	i_table->table = (Bucket*)malloc(size * sizeof(Bucket));
	for(unsigned long i=0; i <= size; i++){
		memset(i_table->table[i].key, '\0', 11 * sizeof(char));
		i_table->table[i].count = 0;
		
		printf("init: blank count = %lu\n", i_table->table[i].count);
	}
}

void copy_table_to_host(const Hashtable &devTable, Hashtable &hostTable) {	
	hostTable.count = devTable.count;
	unsigned long count = devTable.count;
	hostTable.table = (Bucket*) malloc(count * sizeof(Bucket));

	HANDLE_ERROR(cudaMemcpy(hostTable.table, devTable.table, count* sizeof(Bucket), cudaMemcpyDeviceToHost) );
}

__device__ void tokenize (const char* string, char* nextToken){
	//nextToken = strtok (string, ", .");
}

__global__ void hashTest(unsigned char* str, unsigned long mod, unsigned long* hashValue){
	*hashValue = hash_sdbm(str, mod);
}

__global__ void putTest(void){
	printf("Now in the putTest on device. \n");
	Hashtable * i_table;
	initTable(4, i_table);
        unsigned char* s1 = (unsigned char*) "abab5";
        unsigned char* s2 = (unsigned char*) "abababab10";
        unsigned char* s3 = (unsigned char*) "cdababab9";
        unsigned char* s4 = (unsigned char*) "cdababab10";	
	put(s1, i_table->table, 4);
	unsigned long index = hash_sdbm(s1, 4);
	printf("original key = %s, find key = %s, count = %lu\n", s1, i_table->table[index].key, i_table->table[index].count); 	

}

__global__ void kernel( void ) {
	
}

int main ()
{
	unsigned long hash_size = 1024*1024;
	unsigned long hashValue =0;
  	unsigned char str[] ="This, a sample string.";
	unsigned long mod = 4;
        unsigned char* s1 = (unsigned char*) "abab567";
        unsigned char* s2 = (unsigned char*) "abababab10";
        unsigned char* s3 = (unsigned char*) "cdababab9";
        unsigned char* s4 = (unsigned char*) "cdababa8";	
	printf("Before putTest\n");
//	putTest <<<1,1>>> ();
	printf("After putTest\n");
	Hashtable i_table;
	
	



	initTable(4, &i_table); //problem on this reference
	put(s1, (&i_table)->table, 4);
//	put(s2, i_table->table, 4);
//	put(s1, i_table->table, 4);
//	put(s4, i_table->table, 4);
        unsigned long index = hash_sdbm(s1, 4);
      //  printf("original key = %s, find key = %s, count = %lu\n", s1, i_table->table[index].key, i_table->table[index].count);
index = hash_sdbm(s2, 4);
      //  printf("original key = %s, find key = %s, count = %lu\n", s2, i_table->table[index].key, i_table->table[index].count);
index = hash_sdbm(s3, 4);
      //  printf("original key = %s, find key = %s, count = %lu\n", s3, i_table->table[index].key, i_table->table[index].count);
index = hash_sdbm(s4, 4);
      //  printf("original key = %s, find key = %s, count = %lu\n", s4, i_table->table[index].key, i_table->table[index].count);





/*

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
  */

  return 0;
}
