#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "wordcount_helper.h"

// char str[100];
enum {NOT_FOUND=0,FOUND};
static char *PTR;
const char *DELIM;
unsigned char *strTok(unsigned char* string, const char *delim)
{
    int j,flag=NOT_FOUND;
    unsigned char *p;
    if(string != NULL)
    {
        PTR=string;
        p=string;
    }
    else
    {
        if(*PTR == '\0')
            return NULL;

        p=PTR;
    }
 
    while(*PTR != '\0')
    {
        DELIM=delim;
        while(*DELIM != '\0')
        {
            if(*PTR == *DELIM)
            {
                if(PTR == p)
                {
                    p++;
                    PTR++;
                }
                else
                {
                    *PTR='\0';
                    PTR++;
 
                    return p;
                }
            }
            else
            {
                DELIM++;
            }
        }
        PTR++;
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


unsigned int lenStr(unsigned char *str){
        unsigned char *p=str;
        while(*p!='\0')
                p++;
        return(p-str);
}

//For CPU use only!!!
unsigned int tokenizeToArrayCPU(unsigned char* src,unsigned char** strArray ){
	unsigned int i=0;
	char* pch;
	pch = strtok (src," ,.-");
        while (pch != NULL){
                strArray[i] = pch;
                pch = strtok (NULL, " ,.-");
		i++;
        }
	return i;	
}

//For both CPU and GPU
unsigned int tokenizeToArrayGPU(unsigned char* src,unsigned char** strArray ){
	unsigned int i, j;
	unsigned char *p_str,*token;
	char* delim = " ";
	j=0;
	for (i = 1, p_str = src; ; i++, p_str = NULL){
	        token = mystrtok(p_str,delim);
        	if (token == NULL)
           		break;
		strArray[j] = token;
		j++;
	
    }
	return j;

}

char *mystrtok(char* string,const char *delim){
    static char *ptr;
    const char *del;
    int j;
    char *p;

    if(string != NULL){
        ptr=string;
        p=string;
    }
    else{
        if(*ptr == '\0')
            return NULL;
        p=ptr;
    }

    while(*ptr != '\0'){
        del=delim;
        while(*del != '\0'){
            if(*ptr == *del){
                if(ptr == p){
                    p++;
                    ptr++;
                }
                else{
                    *ptr='\0';
                    ptr++;
                    return p;
                }
            }
            else{
                del++;
            }
        }
        ptr++;
    }
    return p;
}
 
int main()
{
	char* file = "aaa_l";
	FILE* fd = fopen (file,"r");
	unsigned char* buff;
	int size =100000;
	buff = malloc(size);
        memset(buff, '\0', size);
	fileRead(fd, size, buff);
	unsigned long len = lenStr((unsigned char *)buff);
	unsigned char** strArray = (unsigned char**)calloc(1, (len + 1)*sizeof(char));
/*
	unsigned int i=0;
	unsigned char* delim, token, pch;
	delim = " ";

	pch = strtok (buff," ,.-");
	while (pch != NULL)
	{
	//strArray[i] = pch;
    //		printf ("%s\n",pch);
    		pch = strtok (NULL, " ,.-");
		i++;
  	}
*/

//	  char str[] ="- This, a sample string.";
//  	char * pch;
//  printf ("Splitting string \"%s\" into tokens:\n",str);
        //	printf("\n%d: %s",i,token);
	
	int n = tokenizeToArrayGPU(buff, strArray);
	printf("%d token\n", n);
	int i;
	for (i=0; i<n; i++){
		printf ("%s\n",strArray[i]);
	}
	
/*			
  	pch = strtok (buff," ,.-");
	int i=0;

  	while (pch != NULL){
		strArray[i] = pch;
	    	printf ("%s\n",strArray[i]);
    		pch = strtok (NULL, " ,.-");
  	}
*/



//	printf("out:%s\n", buff);

}

