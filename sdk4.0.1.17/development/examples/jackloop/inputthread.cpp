#include "inputthread.h"
#include "buffer.h"

extern "C" {
#include "../common/dvs_support.h"
}


InputThread::InputThread
(
 BufferManager *pBufferManager
 ) : AbstractThread( 1, SV_OPENTYPE_INPUT, "INPUT", pBufferManager )
{
}


void InputThread::run()
{
  int dvs_result = SV_OK;
    
  //Now I am running
  mRunning = 1;
  dvs_cond_broadcast( &mWaitConditionRunning, &mMutexRunning, 0 );
 
  //Init dvs stuff
  dvs_result = InitDvsStuff();

  if( dvs_result == SV_OK )
  {
    while( mRunning )
    {
      //Get RingBufferBuffer
      mpVideoBuffer = mpBufferManager -> GetBuffer( mJack );

      if( mpVideoBuffer ) {

       dvs_result = sv_fifo_getbuffer(mpSV, mpFifo, &mpBuffer, NULL, mGetbufferFlags);
       if( dvs_result == SV_OK ) {
          //Write dmaaddresses
          if(mpBuffer) {
            mpBuffer->dma.addr = mpVideoBuffer;
            mpBuffer->dma.size = mVideoBufferSize;
          }
          dvs_result = sv_fifo_putbuffer(mpSV, mpFifo, mpBuffer, NULL);
          if( dvs_result != SV_OK ) {
            printf("Inputfifo, sv_fifo_putbuffer failed. %d\n", dvs_result );
            mRunning = 0;
          }
       } else {
         printf("Inputfifo, sv_fifo_getbuffer failed. %d\n", dvs_result ); 
         mRunning = 0;
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
