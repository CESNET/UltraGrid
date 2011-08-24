#include "threadmanager.h"
#include "inputthread.h"
#include "outputthread.h"
#include "buffer.h"
#include "support.h"


ThreadManager::ThreadManager()
{
  //Set all member variables to their start value
  Init();

  //Build threads
  mpBufferManager = new BufferManager();
  mpInputThread   = new InputThread( mpBufferManager );
  mpOutputThread  = new OutputThread( mpBufferManager );
  
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

    //If the output storage buffersize is smaller than the input raster we have to set up the jack's memory by our own.
    //NOTE: That has to be done BEFORE opening a FIFO!
    ConfigureBoardMemory();

    //Check which jack has the biggest raster
    SetDmarect();

    //Start threads
    mpInputThread  -> StartThread();
    mpOutputThread -> StartThread();
    
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
    mpOutputThread -> SetInputVideoMode( mpInputThread->GetVideomode(), mpInputThread->GetRasterXSize(), mpInputThread->GetRasterYSize(), mpInputThread->GetVideoBufferSize() );
  }
}


void ThreadManager::ConfigureBoardMemory()
{
  int dvs_result = SV_OK;
  int jacks      = 0;
  sv_handle *pSV = 0;
  sv_jack_memoryinfo jackMemInfo[16];
  sv_jack_memoryinfo *pJackMemInfo[16];

  //Check if we really need to set up the DVS board's memory layout manually.
  if( mpInputThread  -> GetVideoBufferSize() > mpOutputThread -> GetVideoBufferSize() ) {

    // init structs
    for ( int jack = 0; jack < 16; jack++ ) {
      memset((void*)&jackMemInfo[jack], 0, sizeof(sv_jack_memoryinfo));
      pJackMemInfo[jack] = &jackMemInfo[jack];
    }

    // close open card handles of both threads
    mpInputThread  -> CloseDvsCard();
    mpOutputThread -> CloseDvsCard();

    // open a new card handle with full rights
    dvs_result = sv_openex(&pSV, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_DEFAULT, 0, 0);
    if( dvs_result != SV_OK ) {
      Log::PrintLog("ConfigureBoardMemory::sv_openex failed. %d.\n", dvs_result);
    }

    if( dvs_result == SV_OK ) {
      // input jack
      jackMemInfo[1].usage.percent     = 50;
      jackMemInfo[1].limit.override    = FALSE;

      // Set the output jack's buffer relevant values to the input jack's values.
      // reason for that:
      // If the output jack has a smaller storage buffer size than the input jack
      // there will be a problem with the size of the used buffer of the output jack.
      // To avoid these problems the size of the used buffer for the output jack has
      // to be set to the value of the bigger input jack buffer.
      // The driver will then use that value to calculate the correct value finally used for the output jack.
      jackMemInfo[0].usage.percent     = 50;
      jackMemInfo[0].limit.override    = TRUE;
      jackMemInfo[0].limit.xsize       = mpInputThread  -> GetRasterXSize();
      jackMemInfo[0].limit.ysize       = mpInputThread  -> GetRasterYSize();
      jackMemInfo[0].limit.storagemode = (mpInputThread -> GetVideomode() & ~SV_MODE_RASTERMASK);

      dvs_result = sv_jack_memorysetup( pSV, 0, pJackMemInfo, 2, &jacks, 0);
      if( dvs_result != SV_OK ) {
        Log::PrintLog("ConfigureBoardMemory::sv_jack_memorysetup failed. %d.\n", dvs_result);
      }
    }

    // close local card handle
    if( pSV ) {
      sv_close( pSV );
      pSV = 0;
    }

    // open card handles of both threads again
    mpInputThread  -> OpenDvsCard();
    mpOutputThread -> OpenDvsCard();
  }
}