#ifndef SUPPORT_H
#define SUPPORT_H

#ifdef WIN32
  #include <windows.h>
  #include <stdio.h>

  typedef unsigned __int64 uint64;  
#else
  #include <string.h>
  #include <stdio.h>
  typedef unsigned long long uint64; 
#endif

class Support
{

public:
  static void AbstractSleep( int mSec );
  static void* MallocAligned( int size, int alignment, char**	orgptr);
};

#endif
