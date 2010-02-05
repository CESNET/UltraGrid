#ifndef ABSTRACT_THREAD_H
#define ABSTRACT_THREAD_H

#include "defines.h"
#include "log.h"

class BufferManager;

class AbstractThread
{
public:
  AbstractThread( int jack, int sv_rights, char* pName );
  ~AbstractThread();

  void StartThread( BufferManager *pBufferManager );
  void StopThread();
  int  Running(){ return mRunning; }

  void AnalyseJack();
  void SetInputVideoMode( int videomode, int xsize = 0, int ysize=0, int buffersize=0 );

  int  GetRasterXSize(){ return mXSize; }
  int  GetRasterYSize(){ return mYSize; }
  int  GetVideomode()  { return mVideomode; }
  int  GetBufferSize(){ return mBufferSize; }
  

protected:
  virtual void  run( void ) = 0;
  static  void* run_c(void*); //c_wrapper

  void Init();
  int  InitDvsStuff();
  int  CloseDvsStuff();
  
  int mRunning;
  int mVideomode;
  int mIomode;
  int mXSize;
  int mYSize;
  int mJack;
  int mSvRights;
  int mBufferSize;
  int mGetbufferFlags;
  int mNum;
  int mDenom;
  char mName[16];
  char *mpVideoBuffer;
  char *mpBlackBuffer;
  char *mpBlackBufferOrig;

  struct
  {
    int xoffset;
    int yoffset;
    int xsize;
    int ysize;
    int xdirection;
    int ydirection;
    int videomode;
    int size;
    int enabled;
  } mInputVideoMode;
 
  //Dvsthread stuff
  dvs_thread          mThreadHandle;
  dvs_cond            mWaitConditionFinish;
  dvs_cond            mWaitConditionRunning;
  dvs_mutex           mMutexRunning;
  sv_fifo             *mpFifo;
  sv_handle           *mpSV;
  sv_fifo_buffer*		  mpBuffer;			      // current fifo buffer
  sv_fifo_info			  mFifoInfo;		      // fifo status
  sv_fifo_configinfo  mConfigInfo;        // need for buffersize and dmaalignment

  BufferManager *mpBufferManager;
};

#endif

