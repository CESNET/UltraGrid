#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include "defines.h"
#include "log.h"

class InputThread;
class OutputThread;
class BufferManager;

class ThreadManager
{
public:
  ThreadManager();
  ~ThreadManager();

  int  StartThreads();
  void Stop();
   
private:
  void Init();
  void SetDmarect();
  void ConfigureBoardMemory();

  InputThread   *mpInputThread;
  OutputThread  *mpOutputThread;
  BufferManager *mpBufferManager;
};

#endif

