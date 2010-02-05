#ifndef INPUT_THREAD_H
#define INPUT_THREAD_H

#include "abstractthread.h"

class InputThread : public AbstractThread
{
public:
  InputThread();
   
private:
  void run( void );
};

#endif

