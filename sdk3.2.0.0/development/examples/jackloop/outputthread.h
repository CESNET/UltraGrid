#ifndef OUTPUT_THREAD_H
#define OUTPUT_THREAD_H

#include "abstractthread.h"

class OutputThread : public AbstractThread
{
public:
  OutputThread();
   
private:
  void run( void );
  void FillBlack();
};

#endif

