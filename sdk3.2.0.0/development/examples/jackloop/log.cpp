#include "log.h"
#include <stdio.h>
#include <stdarg.h>

void Log::PrintLog( char * string, ... )
{
  va_list va;
  char text[1024];
 
  //Interate list
  va_start(va, string);
  vsprintf(text, string, va);
  va_end(va);

  printf( "%s", text );
  fflush(stdout); 
}
