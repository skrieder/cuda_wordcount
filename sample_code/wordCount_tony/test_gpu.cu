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
	Bucket* next_collision;
};

struct Hashtable{
	unsigned long table_size;
	Bucket* table; // table has to be of length count
	//Lock lock[count];
};

unsigned int lenStr(unsigned char *str){
	unsigned char *p=str;
	while(*p!='\0')
		p++;
	return(p-str);
}

int strCmp(const char *temp1,const char *temp2){
    while(*temp1 && *temp2){
        if(*temp1==*temp2){
            temp1++;
            temp2++;
        }
        else{
            if(*temp1<*temp2){
                return -1;  
            }
            else{
                return 1;
            }
        }
    }
    return 0; //return 0 when strings are same
}


__device__ __host__ unsigned long hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c=0;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
//        printf("hash value before mod: %lu, and ", hash );
//        printf("mod: %lu\n", mod );
        return hash % mod;
}

__device__ __host__ void put(unsigned char* key, Bucket* table, unsigned long mod){
	unsigned long index = hash_sdbm(key, mod);
	
//	table[index].lock.lock();
//	memset((table)[index].key, '\0', 11 * sizeof(char));
//       (table)[index].count = 0;
//	printf("put: pre-count =%lu\n", (table)[index].count); 
//	printf("put: pre-key=%s\n",(table)[index].key);
	unsigned int l = lenStr(key);
//	printf("key len=%d\n",l);
	if(NULL == table[index].next_collision){
		if(1){
		}	
	}
	memcpy(table[index].key, key, l+1 );
//	printf("get key=%s\n", table[index].key);
	table[index].count ++;
//	printf("put: post-count =%lu\n", table[index].count);
//	printf("put: post-key=%s\n", table[index].key);
//	table[index].lock.unlock();
}

//put without collision handling
__device__ __host__ void put_nc(unsigned char* key, Bucket* table, unsigned long mod){
        unsigned long index = hash_sdbm(key, mod);

//      table[index].lock.lock();
        unsigned int l = lenStr(key);
        memcpy(table[index].key, key, l+1 );
        table[index].count ++;
//      table[index].lock.unlock();
}

//get without collision handling
__device__ __host__ unsigned long get_nc(unsigned char* key, Hashtable *hashTable, unsigned long mod){
	unsigned long index = hash_sdbm(key, mod);
	//printf("\n\n get count=%lu\n",(hashTable->table[index].count));
	return (hashTable->table[index].count);
}

__host__ __device__ void initTable(unsigned long size, Hashtable** i_table){
	*i_table = (Hashtable*)malloc(sizeof(Hashtable));	
	(*i_table)->table_size = size;
	(*i_table)->table = (Bucket*)malloc(size * sizeof(Bucket));
	for(unsigned long i=0; i <= size; i++){
		memset((*i_table)->table[i].key, '\0', 11 * sizeof(char));
		(*i_table)->table[i].count = 0;
		(*i_table)->table[i].next_collision = NULL;
//		printf("init: blank count = %lu\n", (*i_table)->table[i].count);
	}
}


__device__ __host__ Bucket* it_goto_entry(Hashtable *hashTable, unsigned long index){
	if(index <= (*hashTable).table_size){
		Bucket* ret = (Bucket*)malloc(sizeof(Bucket*));
		ret = &(hashTable->table[index]);
		
	}
	else
		return NULL;
}


void copy_table_to_host(const Hashtable &devTable, Hashtable &hostTable) {	
	hostTable.table_size = devTable.table_size;
	unsigned long count = devTable.table_size;
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
	initTable(4, &i_table);
        unsigned char* s1 = (unsigned char*) "abab5";
        unsigned char* s2 = (unsigned char*) "abababab10";
        unsigned char* s3 = (unsigned char*) "cdababab9";
        unsigned char* s4 = (unsigned char*) "cdababab10";	
//	put(s1, i_table->table, 4);
	unsigned long index = hash_sdbm(s1, 4);

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
//	printf("Before putTest\n");
//	putTest <<<1,1>>> ();
//	printf("After putTest\n");
	Hashtable *i_table; //&((i_table->table)[idx])
	initTable(4, &i_table); //problem on this reference
	printf("post-init key: %s\n",(*i_table).table[0].key);
	printf("post-init count: %lu\n",(*i_table).table[0].count);
	put_nc(s1, ((i_table)->table), 4);
	put_nc(s2, i_table->table, 4);
	put_nc(s1, i_table->table, 4);
	put_nc(s4, i_table->table, 4);
	printf("get s1, count= %lu\n",get_nc(s1, i_table, 4));
        unsigned long index = hash_sdbm(s1, 4);
	
	Bucket* item = (Bucket*)calloc(1, sizeof(Bucket));
	item = it_goto_entry(i_table, 5);
//	printf("iter key: %s\n",item->key);
 //       printf("inter count: %lu\n",item->count);
	
	if(NULL == item )
                printf("invalid index\n");
        else{
		printf("iter key: %s\n",item->key);	
		printf("inter count: %lu\n",item->count);
	}

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
