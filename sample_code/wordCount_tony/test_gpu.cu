#include "../common/book.h"
#include "lock.h"
#include <iostream>
#include <stdio.h>
#include <string.h>

__global__ void tokenize (const char* string, char* nextToken){
	nextToken = strtok (string, ", .");
}

__global__ void kernel( void ) {
}


int main ()
{
  	char str[] ="This, a sample string.";
  	
	kernel<<<1,1>>>();

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