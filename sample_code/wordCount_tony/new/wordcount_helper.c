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
/* 
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
*/
