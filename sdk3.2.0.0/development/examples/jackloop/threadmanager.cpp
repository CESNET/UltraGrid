#include "threadmanager.h"
#include "inputthread.h"
#include "outputthread.h"
#include "buffer.h"
#include "support.h"


ThreadManager::ThreadManager()
{
  //Set all member variables to the start value
  Init();

  //Build threads
  mpInputThread   = new InputThread();
  mpOutputThread  = new OutputThread();
  mpBufferManager = new BufferManager();
}


ThreadManager::~ThreadManager()
{

  if( mpInputThread ) {
    delete mpInputThread;
    mpInputThread = 0;
  }
  
  if( mpOutputThread ) {
    delete mpOutputThread;
    mpOutputThread = 0;
  }

  if( mpBufferManager ) {
    delete mpBufferManager;
    mpBufferManager = 0;
  }
}


void ThreadManager::Init()
{
  mpInputThread   = 0;
  mpOutputThread  = 0;
  mpBufferManager = 0;
}


int ThreadManager::StartThreads()
{
  int result = true;

  if( mpInputThread && mpOutputThread )
  {
    //Check current configuration
    mpInputThread  -> AnalyseJack();
    mpOutputThread -> AnalyseJack();

    //Log
    Log::PrintLog("Start real work:\n");
    Log::PrintLog("****************\n");

    //Check which jack has the biggest raster
    SetDmarect();

    //Start output thread
    mpInputThread -> StartThread( mpBufferManager );
       
    //Start rs422 thread
    mpOutputThread -> StartThread( mpBufferManager );
    
    while( mpInputThread->Running() || mpOutputThread->Running() )
    {
      //Do nothing until the worker threads are finally finished.
      Support::AbstractSleep( 200 );
    }
  }else{
    result = false;
    Log::PrintLog("Unable to create threads.\n");
  }

  return result;
}


void ThreadManager::Stop()
{
  if( mpInputThread ) {
    mpInputThread -> StopThread();
  }

  if( mpOutputThread ) {
    mpOutputThread -> StopThread();
  }
}


void ThreadManager::SetDmarect()
{
  //Check which jack has the biggest raster
  if( mpInputThread -> GetVideomode() == mpOutputThread -> GetVideomode() ) {
    mpOutputThread -> SetInputVideoMode( -1 );
  } else {
    mpOutputThread -> SetInputVideoMode( mpInputThread->GetVideomode(), mpInputThread->GetRasterXSize(), mpInputThread->GetRasterYSize(),mpInputThread->GetBufferSize());
  }
}
