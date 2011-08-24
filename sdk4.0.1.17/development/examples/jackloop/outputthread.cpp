#include "outputthread.h"
#include "buffer.h"

extern "C" {
#include "../common/dvs_support.h"
}


OutputThread::OutputThread
(
 BufferManager *pBufferManager
 ) : AbstractThread( 0, SV_OPENTYPE_OUTPUT, "OUTPUT", pBufferManager )
{
}


void OutputThread::run()
{
  int dvs_result = SV_OK;
  
  //Now I am running
  mRunning = 1;
  dvs_cond_broadcast( &mWaitConditionRunning, &mMutexRunning, 0 );
  
  //Init dvs stuff
  dvs_result = InitDvsStuff();
  
  if( dvs_result == SV_OK )
  {
    while( mRunning ) {

      //Get RingBufferBuffer
      mpVideoBuffer = mpBufferManager -> GetBuffer( mJack );

      if( mpVideoBuffer )
      {      
        dvs_result = sv_fifo_getbuffer(mpSV, mpFifo, &mpBuffer, NULL, mGetbufferFlags);
        if( dvs_result==SV_OK )
        {
          //Write dmaaddresses
          if(mpBuffer) {
            mpBuffer->dma.addr = mpVideoBuffer;
            mpBuffer->dma.size = mVideoBufferSize;
          }

          if( mInputVideoMode.enabled ){
            if( mpBuffer ) {
              mpBuffer->dma.size = mInputVideoMode.size;
            }
            mpBuffer->storage.storagemode = mInputVideoMode.videomode;
            mpBuffer->storage.xsize       = mInputVideoMode.xsize;
            mpBuffer->storage.ysize       = mInputVideoMode.ysize;
            mpBuffer->storage.xoffset     = 0;
            mpBuffer->storage.yoffset     = 0;
            mpBuffer->storage.dataoffset  = 0;
            mpBuffer->storage.lineoffset  = 0;

            //printf("OUTPUT: xsize:%d, ysize:%d, xoffset:%d, yoffset:%d\n", mDmaRect.xsize, mDmaRect.ysize, mpBuffer->storage.xoffset, mpBuffer->storage.yoffset);
          }

          if( mpBuffer ) {
            dvs_result = sv_fifo_putbuffer(mpSV, mpFifo, mpBuffer, 0);
            if( dvs_result!=SV_OK ) {
              mRunning = 0;
              printf("Outputfifo, sv_fifo_putbuffer failed. %d\n", dvs_result );
            }
          }

          sv_fifo_vsyncwait( mpSV, mpFifo );
        } else {
          mRunning = 0;
          printf("Outputfifo, sv_fifo_getbuffer failed. %d\n", dvs_result ); 
        }

      } else {
        sv_fifo_vsyncwait( mpSV, mpFifo );
      }
    }
  }
  
  //Close dvs stuff
  CloseDvsStuff();
  
  //Now I am finished
  mRunning  = 0;
  mExitCode = true;
  dvs_thread_exit(&mExitCode, &mWaitConditionFinish);
}

