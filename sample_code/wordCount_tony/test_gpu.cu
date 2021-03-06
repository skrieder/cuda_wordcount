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
	unsigned char key[10];
	unsigned long count;
	Lock lock;
	Bucket* next_collision;
};

struct Hashtable{
	unsigned long table_size;
	Bucket** table; // Tony: change from * to **
  //Lock lock[count];
};

__host__ __device__ void put_nc(unsigned char* key, Bucket* table, unsigned long mod);

__device__ __host__ unsigned long get_nc(unsigned char* key, Hashtable *hashTable, unsigned long mod);

__host__ __device__ void initTable(unsigned long size, Hashtable** i_table);

//__global__ void parallel_insert_to_table(Hashtable d_master_hashtable, unsigned char **d_array, int num_threads){
__global__ void parallel_insert_to_table(Hashtable *d_master_hashtable, char *d_word, int num_threads){

  Hashtable *test_table;
  initTable(4, &test_table);

  printf("HEX: test_table=%x\n", test_table);

  d_master_hashtable = test_table;
  printf("HEX: d_master_table=%x\n", d_master_hashtable);

  // assert in func
  printf("Start GPU function: parallel_insert_to_table\n");

  unsigned char * temp_string = (unsigned char *)malloc(sizeof(char*)*4);
  //  printf("The temp string = %s\n", temp_string);

  memcpy(temp_string, d_word, sizeof(unsigned char *)*4);

  printf("The temp string = %s\n", temp_string);

  printf("calling put_nc\n");
  //  put_nc(temp_string, d_master_hashtable.table, 4);
  put_nc(temp_string, *(test_table->table), 4); //Tony
  put_nc(temp_string, *(test_table->table), 4);
  // put_nc(temp_string, d_master_hashtable->table, 4);
  printf("after call to put_nc\n");

  printf("Calling get_nc for string %s\n", temp_string);
  unsigned long temp_int = get_nc(temp_string, test_table, 4);
  //  unsigned long temp_int = get_nc(temp_string, &d_master_hashtable, 4);

  printf("The temp_int = %lu\n", temp_int);

  printf("After get_nc\n");

  // assert values in d_array
  //  printf("The word at d_array[0] = %s\n", d_array[0]);
  //  printf("The word at d_array[1] = %s\n", d_array[1]);
  //printf("The word at d_array[2] = %s\n", d_array[2]);
  //printf("The word at d_array[3] = %s\n", d_array[3]);


  // assert end func
  printf("End of parallel_insert_to_table\n");
}

__host__ int iterate(Hashtable *final_table){
  int i = 0;
  FILE *f = fopen("file.txt", "w");
  if (f == NULL)
    {
      printf("Error opening file!\n");
      exit(1);
    }
  long num_bucket = final_table->table_size;
  Bucket* buckets =  *(final_table->table);
  for (i = 0; i < num_bucket; i++)
    {
      printf("word: %s ", buckets[i].key);// print key                         
      printf ("%lu\n",buckets[i].count);// print count                         
      fprintf(f, "word: %s %lu\n",  buckets[i].key,  buckets[i].count);//write\
 to file                                                                                         
   }
  fclose(f);
  // buckets[0].key;                                                                     
  // buckets[0].count;                                                                   
  return 0;
}

__host__ __device__ unsigned int lenStr(unsigned char *str){
	unsigned char *p=str;
	while(*p!='\0')
		p++;
	return(p-str);
}

int strCmp(unsigned char *temp1,unsigned char *temp2){
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

void rand_str(unsigned char *dest, size_t length) {
	unsigned char charset[] = "0123456789"
                     "abcdefghijklmnopqrstuvwxyz"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    while (length-- > 0) {
        size_t index = (double) rand() / RAND_MAX * (sizeof charset - 1);
        *dest++ = charset[index];
    }
    *dest = '\0';
}

__device__ __host__ unsigned long hash_sdbm(unsigned char *str, unsigned long mod){
        unsigned long hash = 0; int c=0;
        while (c = *str++)
            hash = c + (hash << 6) + (hash << 16) - hash;
//        printf("hash value before mod: %lu, and ", hash );
//        printf("mod: %lu\n", mod );
        return hash % mod;
}
/*
__device__ __host__ void put(unsigned char* key, Bucket* table, unsigned long mod){
	unsigned long index = hash_sdbm(key, mod);
	
//	table[index].lock.lock();
//	memset((table)[index].key, '\0', 11 * sizeof(char));
//       (table)[index].count = 0;
//	printf("put: pre-count =%lu\n", (table)[index].count); 
//	printf("put: pre-key=%s\n",(table)[index].key);
	unsigned int l = lenStr(key);
//	printf("key len=%d\n",l);
	Bucket* p = &(table[index]);
	while(0 != strCmp(key, p->key) && NULL != p->next_collision){ //if find collision
		p = p->next_collision;
		
		
	}//key and p.key are same string, or need a new collision slot.
		
	if(0 == strCmp(key, p->key )){ //find existing token
		memcpy(p->key, key, l+1 );
		p->count ++;
	}else{ //need a new collision slot
		p->next_collision = (Bucket*)malloc(sizeof(Bucket));
		memcpy(p->next_collision->key, key, l+1);
		p->next_collision->count =1;
		p->next_collision->next_collision = NULL;
	}

	if(NULL == table[index].next_collision){
		if(1){
		}	
	}
	memcpy(table[index].key, key, l+1 );
	printf("put: key=%s, index=%lu \n", key, index);
	table[index].count ++;
//	printf("put: post-count =%lu\n", table[index].count);
//	printf("put: post-key=%s\n", table[index].key);
//	table[index].lock.unlock();
}
*/
//put without collision handling
__device__ __host__ void put_nc(unsigned char* key, Bucket* table, unsigned long mod){
//      table[index].lock.lock();

//  printf("In put_nc\n");
  unsigned long index = hash_sdbm(key, mod);
//  printf("hash is set\n");
  unsigned int l = lenStr(key)*sizeof(char);
//  printf("before memcpy in put_nc\n");
  
  // print the key from the params
//  printf("key from params %s\n", key);

  // print the value of size
//  printf("size to copy %d\n", l+1);

  printf("Put: Trying to set the key at table index %lu\n", index);

  printf("Put: Current count before put: %lu\n", table[index].count);

  // sometimes this copy fails...
  memcpy(table[index].key, key, l+1 );

  // print the key from the table
  printf("key from table %s\n", table[index].key);

//  printf("after memcpy in put_nc\n");
  table[index].count++;
//      table[index].lock.unlock();
//  printf("end of put_nc\n");
}

//get without collision handling
__device__ __host__ unsigned long get_nc(unsigned char* key, Hashtable *hashTable, unsigned long mod){
  printf("In get_nc\n");
  printf("The input key = %s\n", key);
  unsigned long index = hash_sdbm(key, mod);
  //printf("\n\n get count=%lu\n",(hashTable->table[index].count));
  printf("The key is = %s\n", hashTable->table[index]->key); //Tony
  return (hashTable->table[index]->count);//Tony
}

__host__ __device__ void initTable(unsigned long size, Hashtable** i_table){
  printf("Start of initTable\n");
	*i_table = (Hashtable*)malloc(sizeof(Hashtable));	
	(*i_table)->table_size = size;
	(*i_table)->table = (Bucket**)malloc(size * sizeof(Bucket));//Tony
	for(unsigned long i=0; i <= size; i++){
		memset((*i_table)->table[i]->key, '\0', 11 * sizeof(char));//Tony
		(*i_table)->table[i]->count = 0;//Tony
		(*i_table)->table[i]->next_collision = NULL;//Tony
//		printf("init: blank count = %lu\n", (*i_table)->table[i].count);
	}
	printf("End of initTable\n");
}

/*
__device__ __host__ Bucket* it_goto_entry(Hashtable *hashTable, unsigned long index){
	if(index <= (*hashTable).table_size){
		Bucket* ret = (Bucket*)malloc(sizeof(Bucket*));
		ret = &(hashTable->table[index]);
	}
	else
		return NULL;
}
*/

void copy_table_to_host(const Hashtable &devTable, Hashtable &hostTable) {	
	hostTable.table_size = devTable.table_size;
	unsigned long count = devTable.table_size;
	hostTable.table = (Bucket**) malloc(count * sizeof(Bucket));//Tony

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
	//        unsigned char* s2 = (unsigned char*) "abababab10";
        //unsigned char* s3 = (unsigned char*) "cdababab9";
        //unsigned char* s4 = (unsigned char*) "cdababab10";	
//	put(s1, i_table->table, 4);
	unsigned long index = hash_sdbm(s1, 4);

}

__global__ void kernel( void ) {
	
}

int main ()
{
  // catch errors yo
  cudaError_t err = cudaSuccess;

        // start of main
  //        unsigned long hash_size = 1024*1024;
  //        unsigned long hashValue =0;
  //unsigned char str[] ="Hello World Great Dayss"; // Tokenize this string TODO

	unsigned long mod = 1023; // auto gen this number TODO
	
        unsigned char* s1 = (unsigned char*) "Hello";
        unsigned char* s2 = (unsigned char*) "World";
        unsigned char* s3 = (unsigned char*) "Great";
        unsigned char* s4 = (unsigned char*) "Dayss";	

	// set the number of elements
	int num_elements = 4;

	// set the array size
	int array_size = sizeof(char*)*mod*6; //Tony: should this be char not char* ? No, tried and not the problem.

	// allocate the host array and table
	unsigned char** h_array = (unsigned char **)calloc(1, array_size);
	unsigned char** d_array;

	// init the table
	//	Hashtable d_master_hashtable;
	//initialize_table( d_master_hashtable, 4, 4);

      	Hashtable *d_master_hashtable;
      	Hashtable *h_master_hashtable;
//	initTable(4, &h_master_hashtable);

	// declare the device hashtable
	//Hashtable *d_master_hashtable;

	int size_of_hashtable = (sizeof(Bucket*)*num_elements);

	// allocate the device hashtable
	/*
	err = cudaMalloc((void **)&d_master_hashtable, (size_t)sizeof(Hashtable *));
	if (err != cudaSuccess){
	  fprintf(stderr, "Failed to copy the h_array to the d_array(error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}	
	*/
	//	printf("After allocate device hashtable\n");
	/*
	// allocate the buckets of the hashtable
	err = cudaMalloc((void **)&d_master_hashtable->table, (size_t)size_of_hashtable);
	if (err != cudaSuccess){
	  fprintf(stderr, "Failed to copy the h_array to the d_array(error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
	*/
	//	printf("After allocate hashtable entry array\n");
	/*
	// copy the host table into the device table
	printf("Before copy host table to d_table\n");
	err = cudaMemcpy((void **)d_master_hashtable, (void **)h_master_hashtable, (size_t)size_of_hashtable, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
	  fprintf(stderr, "Failed to copy the h_array to the d_array(error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}	
	printf("After cudaMemcpy\n");
	*/
	// hard code the arrays







/*
	h_array[0] = s1;
	h_array[1] = s2;
	h_array[2] = s3;
	h_array[3] = s4;

	// assert the arrays
	printf("The word at h_array[0] = %s\n", h_array[0]);
	printf("The word at h_array[1] = %s\n", h_array[1]);
	printf("The word at h_array[2] = %s\n", h_array[2]);
	printf("The word at h_array[3] = %s\n", h_array[3]);


	// allocate the d_array
	printf("Before cudaMalloc\n");
	err = cudaMalloc((void **)&d_array, array_size);
	if (err != cudaSuccess){
	  fprintf(stderr, "Failed to allocate the d_array(error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}	
	printf("After cudaMalloc of d_array\n");

	// copy h_array into d_array
	err = cudaMemcpy(d_array, h_array, array_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
	  fprintf(stderr, "Failed to copy the h_array to the d_array(error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}	
	printf("After cudaMemcpy\n");

	char *h_word = "Hello";
	char *d_word;
	cudaMalloc((void**)&d_word, (size_t)sizeof(char *)*6);
	cudaMemcpy(d_word, h_word, (sizeof(char *)*6), cudaMemcpyHostToDevice);
*/
	// launch GPU kernel
	int num_threads = 1;
	//	parallel_insert_to_table<<<1,num_threads>>>(d_master_hashtable, d_array, num_threads);
//	printf("before gpu call\n");
//	parallel_insert_to_table<<<1,num_threads>>>(d_master_hashtable, d_word, num_threads);
//	printf("after gpu call\n");
	// sync device
//	cudaDeviceSynchronize();

	// bring back hashtable
//	cudaMemcpy(&h_master_hashtable, &d_master_hashtable, size_of_hashtable, cudaMemcpyDeviceToHost);
	
//	printf("After copy hashtable back to host.\n");


	// iterate hash table
//	iterate(h_master_hashtable);

	// clean memory

	// return

	/* Everything past here is old code to be merged in.*/
	
	Hashtable* i_table;
	initTable(4, &i_table);	
//	printf("Before putTest\n");
//	putTest <<<1,1>>> ();
//	printf("After putTest\n");
//	printf("post-init key: %s\n",(*i_table).table[0].key);
//	printf("post-init count: %lu\n",(*i_table).table[0].count);
	put_nc(s1, (*((i_table)->table)), mod);//Tony
	put_nc(s2, *(i_table->table), mod);//Tony
	put_nc(s1, *(i_table->table), mod);//Tony
	put_nc(s4, *(i_table->table), mod);//Tony
	printf("=========================      get s1, count= %lu\n",get_nc(s1, i_table, mod));

	// send to GPU

	// insert sample strings to table
	/*
	int i=0;

	for(i=0;i<10;i++){
		unsigned char* str = (unsigned char*) malloc(10 * sizeof(char));
		rand_str(str, 10);
		printf("%s\n",str);
		put(str, ((i_table)->table), mod);
	}
	*/
	/* uncomment to test collision	
	for(i=0;i < mod;i++){
		Bucket* p = &((i_table)->table[i]);
		printf("index %d  -----------------:\n",i);
		printf("key: %s\,  ", p->key);
		printf("count: %lu\n",p->count);
		while(NULL != p->next_collision){
			printf("key: %s\,  ", p->key);
	                printf("count: %lu\n\n",p->count);
			p = p->next_collision;
		}
	}
	*/

/*	
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

*/



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
