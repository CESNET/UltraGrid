#include <signal.h>

#include "log.h"
#include "threadmanager.h"


//Global ThreadManager
ThreadManager mThreadManager;


void signal_handler(int signum)
{
  mThreadManager . Stop();
}


int main(int argc, char* argv[])
{
  int result = true;
  
  //Connect signal handler
  signal(SIGTERM, signal_handler);
  signal(SIGINT,  signal_handler);

  //Do real work
  mThreadManager . StartThreads();
  
  //Disconnect signal handler
  signal(SIGTERM, NULL);
  signal(SIGINT, NULL);

  return result;
}
