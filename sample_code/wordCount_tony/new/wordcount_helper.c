#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "wordcount_helper.h"
char str[100];
enum {NOT_FOUND=0,FOUND};
static char *ptr;
const char *del;
char *mystrtok(char* string,const char *delim)
{
    int j,flag=NOT_FOUND;
    char *p;
    if(string != NULL)
    {
        ptr=string;
        p=string;
    }
    else
    {
        if(*ptr == '\0')
            return NULL;
 
        p=ptr;
    }
 
    while(*ptr != '\0')
    {
        del=delim;
        while(*del != '\0')
        {
            if(*ptr == *del)
            {
                if(ptr == p)
                {
                    p++;
                    ptr++;
                }
                else
                {
                    *ptr='\0';
                    ptr++;
 
                    return p;
                }
            }
            else
            {
                del++;
            }
        }
        ptr++;
    }
    return p;
}

int fileRead(FILE *fd, size_t size, void* buff){
	//buff = malloc(size);
	//memset(buff, '/0', size);
	//setvbuf(fd, NULL, _IONBF, size);
	fread(buff, size, 1, fd);
	return 0;
}




 
int main()
{
	char* file = "aaa";
	FILE* fd = fopen (file,"r");
	void* buff;
	int size =1000;
	buff = malloc(size);
        memset(buff, '\0', size);
	fileRead(fd, size, buff);
	printf("out:%s\n", buff);

}

