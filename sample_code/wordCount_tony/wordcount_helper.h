#include <stdio.h>
#include <stdlib.h>

int fileRead(FILE *fd, size_t size, void* buff);
char *mystrtok(char* string,const char *delim);
unsigned int lenStr(unsigned char *str);
unsigned int tokenizeToArrayCPU(unsigned char* src,unsigned char** strArray);
unsigned int tokenizeToArrayGPU(unsigned char* src,unsigned char** strArray);
char *mystrtok(char* string,const char *delim);

