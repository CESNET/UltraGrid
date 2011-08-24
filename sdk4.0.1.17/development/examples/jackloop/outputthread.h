#ifndef OUTPUT_THREAD_H
#define OUTPUT_THREAD_H

#include "abstractthread.h"

class BufferManager;

class OutputThread : public AbstractThread
{
public:
  OutputThread( BufferManager *pBufferManager );
   
private:
  void run( void );
  void FillBlack();
};

#endif

