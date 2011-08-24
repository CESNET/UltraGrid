#ifndef INPUT_THREAD_H
#define INPUT_THREAD_H

#include "abstractthread.h"

class BufferManager;

class InputThread : public AbstractThread
{
public:
  InputThread( BufferManager *pBufferManager );
   
private:
  void run( void );
};

#endif

