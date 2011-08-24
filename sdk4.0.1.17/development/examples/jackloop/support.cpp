#include "support.h"
#include "defines.h"
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#endif

void* Support::MallocAligned( int size, int alignment, char**	orgptr ) 
{
  if( !size )    return NULL;
  
  if(alignment > 8) 
  {
    *orgptr = (char*)malloc(size + alignment);
    return (void*)(((uintptr)*orgptr + alignment - 1) & ~(alignment-1));
  }

  *orgptr = (char*)malloc(size);
  return *orgptr;
}


void Support::AbstractSleep( int mSec )
{
#ifdef WIN32
  Sleep(mSec);  //Sleep millisec
#else
  usleep(mSec * 1000); //Sleep microsec
#endif
}
