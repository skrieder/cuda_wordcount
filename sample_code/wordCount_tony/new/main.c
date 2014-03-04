#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include "wordcount_helper.h"
char str[100];
enum {NOT_FOUND=0,FOUND};
static char *ptr;
const char *del;

int main()
{
    int i;
    char *p_str,*token;
    char delim[10];
 
    printf("\n Enter a string to tokenize: ");
    scanf("%[^\n]",str);
     
    getchar();
    printf("\n Enter a delimiter : ");
    scanf("%[^\n]",delim);
 
    for (i = 1, p_str = str; ; i++, p_str = NULL)
    {
        token = mystrtok(p_str,delim);
        if (token == NULL)
            break;
        printf("\n%d: %s",i,token);
    }
}
