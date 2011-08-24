#ifndef BUFFER_H
#define BUFFER_H

#include "defines.h"

#define MAX 16

class BufferManager
{
public:
  BufferManager();
  ~BufferManager();

  int   AllocateBuffer( int size, int count, int dmaalign );
  char* GetBuffer( int jack );

private:
  void Init();
  void Free();

  int mCurrentRecordBuffer;
  int mCurrentDisplayBuffer;
  int mBufferCount;
  char *mpBuffer[MAX];
  char *mpBufferOrig[MAX];

  dvs_mutex mMutexRingBuffer;
};

#endif
