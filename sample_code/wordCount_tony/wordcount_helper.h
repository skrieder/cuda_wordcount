#include <stdio.h>
#include <stdlib.h>

int fileRead(FILE *fd, size_t size, void* buff);
char *mystrtok(char* string,const char *delim);
unsigned int lenStr(unsigned char *str);

//Used by CPU only!! 
unsigned int tokenizeToArrayCPU(unsigned char* src,unsigned char** strArray); 

//Used for both CPU and GPU. See the c file for example.
unsigned int tokenizeToArrayGPU(unsigned char* src,unsigned char** strArray);

