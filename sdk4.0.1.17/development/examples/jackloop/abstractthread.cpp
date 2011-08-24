#include "abstractthread.h"
#include "support.h"
#include "buffer.h"

extern "C" {
#include "../common/dvs_support.h"
}

AbstractThread::AbstractThread( int jack, int opentype, char *pName, BufferManager *pBufferManager )
{ 
  //Init
  Init();

  //Save pointer
  mJack     = jack;
  mOpenType = opentype;
  strncpy( mName, pName, 16 );
  mpBufferManager = pBufferManager;
  
  //Init dvs specific values
  dvs_cond_init( &mWaitConditionRunning );
  dvs_mutex_init( &mMutexRunning );

  OpenDvsCard();
}


AbstractThread::~AbstractThread()
{
  //Clean dvs specific values
  dvs_cond_free( &mWaitConditionRunning );
  dvs_mutex_free( &mMutexRunning );

  //Close DVS card
  if( mpSV ) {
    sv_close( mpSV );
    mpSV = 0;
  }

  //Free BlackBuffer
  if( mpBlackBufferOrig ) {
    free( mpBlackBufferOrig );
  }
}


void AbstractThread::OpenDvsCard()
{
  int dvs_result = SV_OK;

  dvs_result = sv_openex( &mpSV, "", SV_OPENPROGRAM_DEMOPROGRAM, mOpenType, 0, 0);
  if( dvs_result != SV_OK ) {
    Log::PrintLog("OpenDvsCard::sv_openex failed. %d.\n", dvs_result);
  }
}


void AbstractThread::CloseDvsCard()
{
  if( mpSV ) {
    sv_close( mpSV );
    mpSV = 0;
  }
}


void AbstractThread::Init()
{
  memset( &mInputVideoMode, 0, sizeof( mInputVideoMode ) );
  memset( (void*)&mFifoInfo,   0, sizeof(sv_fifo_info) ); 
  memset( (void*)&mConfigInfo, 0, sizeof(sv_fifo_configinfo) );
 

  mRunning   = false;
  mVideomode = 0;
  mIomode    = 0;
  mXSize     = 0;
  mYSize     = 0;
  mJack      = 0;
  mOpenType  = 0;
  mVideoBufferSize  = 0;
  mpBlackBuffer     = 0;
  mpBlackBufferOrig = 0;
  mName[0]          = '\0';
  mGetbufferFlags   = 0;

  mpSV              = 0;
  mpFifo            = 0;
  mpBuffer          = 0;
  mpVideoBuffer     = 0;
  mpBufferManager   = 0;
}


void AbstractThread::StartThread()
{
  //Start dvs thread
  dvs_thread_create( &mThreadHandle, run_c, (void*) this, &mWaitConditionFinish);

  //Wait until thread is started
  dvs_cond_wait( &mWaitConditionRunning, &mMutexRunning, 0 );
}


void AbstractThread::StopThread()
{
  //Stop thread
  mRunning = 0;

  //Wait for real stop
  dvs_thread_exitcode( &mThreadHandle, &mWaitConditionFinish);
}


//C wrapper
void * AbstractThread::run_c( void *pThis )
{
  ((AbstractThread*)(pThis))->run();
  return 0;
}

/*
void AbstractThread::run()
{
  //Have to be implemended by spezialised class
}
*/

void AbstractThread::AnalyseJack()
{
  int dvs_result = SV_OK;
  sv_storageinfo storageInfo;

  //Init all sv_storageinfo members
  memset((void*)&storageInfo, 0, sizeof(sv_storageinfo));
 
  //Check sv_handle pointer
  if( mpSV == 0 ) {
    dvs_result = SV_ERROR_PARAMETER;
  }

  if( dvs_result == SV_OK )
  {
    //Get Videomode
    dvs_result = sv_jack_option_get(mpSV, mJack, SV_OPTION_VIDEOMODE, &mVideomode);
    if(dvs_result != SV_OK) {
      Log::PrintLog("AnalyseJack::sv_option_videomode failed. %d.\n", dvs_result);
    }

    //Get IoMode
    if( dvs_result == SV_OK ) {
      dvs_result = sv_jack_option_get(mpSV, mJack, SV_OPTION_IOMODE, &mIomode);
      if(dvs_result != SV_OK) {
        Log::PrintLog("AnalyseJack::sv_option_iomode failed. %d.\n", dvs_result);
      }
    }

    //Get XSize
    if( dvs_result == SV_OK ) {
      dvs_result = sv_jack_query(mpSV, mJack, SV_QUERY_RASTER_XSIZE, 0, &mXSize);
      if(dvs_result != SV_OK) {
        Log::PrintLog("AnalyseJack::SV_QUERY_RASTER_XSIZE failed. %d.\n", dvs_result);
      }
    }

    //Get YSize
    if( dvs_result == SV_OK ) {
      dvs_result = sv_jack_query(mpSV, mJack, SV_QUERY_RASTER_YSIZE, 0, &mYSize);
      if(dvs_result != SV_OK) {
        Log::PrintLog("AnalyseJack::SV_QUERY_RASTER_XSIZE failed. %d.\n", dvs_result);
      }
    }

    //Get the buffer size.
    if( dvs_result == SV_OK ) {
      dvs_result = sv_storage_status(mpSV, mJack, NULL, &storageInfo, sizeof(sv_storageinfo), SV_STORAGEINFO_COOKIEISJACK);
      if(dvs_result != SV_OK) {
        Log::PrintLog("AnalyseJack::sv_storage_status(jack: %d) failed. %d.\n", mJack, dvs_result);
      } else {
        //video buffer size calculation
        mVideoBufferSize = mXSize * mYSize * storageInfo.pixelsize_num / storageInfo.pixelsize_denom;
      }
    }

    //Print 
    if( dvs_result == SV_OK ) {
      printf("\nJack \"%s\" configuration:\n", mName);
      printf("***************************\n");
      printf("Video mode   : %s/%s/%s\n", sv_support_videomode2string(mVideomode), sv_support_colormode2string_mode(mVideomode), sv_support_bit2string_mode(mVideomode));
      printf("Video iomode : %s\n", sv_support_iomode2string_mode(mIomode) );
      printf("Resolution   : %dx%d\n\n", mXSize, mYSize );
    }
  }
}


void AbstractThread::SetInputVideoMode( int videomode, int xsize, int ysize, int buffersize )
{
  if(  videomode != -1 ) {
    mInputVideoMode.enabled   = true;
    mInputVideoMode.videomode = videomode;
    mInputVideoMode.xsize     = xsize;
    mInputVideoMode.ysize     = ysize;
    mInputVideoMode.size      = buffersize;
    mGetbufferFlags |= SV_FIFO_FLAG_STORAGEMODE;

    //Log
    Log::PrintLog("- Jack \"%s\" dynamic storagemode is enabled.\n", mName);
  } else {
    mGetbufferFlags = 0; 
  }
}


int AbstractThread::InitDvsStuff()
{
  int dvs_result = SV_OK;
  int fifo_flags = 0;
 
  //Check sv_handle pointer
  if( (mpSV == 0) || (mpBufferManager == 0) ) {
    dvs_result = SV_ERROR_PARAMETER;
  }

  //Init the fifo
  if( dvs_result == SV_OK ) {
    dvs_result = sv_fifo_init(mpSV, &mpFifo, mJack, FALSE, TRUE, fifo_flags, 0);
    if(dvs_result != SV_OK)  {
      Log::PrintLog("InitDvsStuff::sv_fifo_init failed. %d.\n", dvs_result);
    } 
  }

  //Get Buffersizes
  if(dvs_result == SV_OK) {
    dvs_result = sv_fifo_configstatus( mpSV, mpFifo, &mConfigInfo );
    if(dvs_result != SV_OK) {
      Log::PrintLog("InitDvsStuff::sv_fifo_configstatus failed. %d.\n", dvs_result);
    }
  }

  //Calculate buffersize
  if(dvs_result == SV_OK) {
    //Create RingBuffer
    if( mJack == 1 ) {
      mpBufferManager -> AllocateBuffer( mVideoBufferSize, 5, mConfigInfo.dmaalignment );
    }
  }

  //Create Blackbuffer
  if(dvs_result == SV_OK) {
    mpBlackBuffer = (char*)Support::MallocAligned( mConfigInfo.vbuffersize, mConfigInfo.dmaalignment, &mpBlackBufferOrig);
    if( mpBlackBuffer == 0 ) {
      dvs_result = SV_ERROR_PARAMETER;
      Log::PrintLog("InitDvsStuff::MallocAligned failed, unable to create black buffer. %d.\n", dvs_result);
    }else{
      memset( mpBlackBuffer, 0, mConfigInfo.vbuffersize );
    }
  }
  
  //Start the fifo
  if(dvs_result == SV_OK) {
    dvs_result = sv_fifo_start(mpSV, mpFifo);
    if(dvs_result != SV_OK)  {
      Log::PrintLog("InitDvsStuff::sv_fifo_start failed. %d.\n", dvs_result);
    }
  }
  
  return dvs_result;
}


int AbstractThread::CloseDvsStuff()
{
  //Close sv_fifo
  if( mpSV && mpFifo ) {
    sv_fifo_stop( mpSV, mpFifo, 0);
    sv_fifo_free( mpSV, mpFifo );
    mpFifo = 0;
  }

  return SV_OK;
}

