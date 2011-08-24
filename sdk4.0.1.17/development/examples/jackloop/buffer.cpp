#include "buffer.h"
#include "support.h"
#include <stdlib.h>

BufferManager::BufferManager()
{
  Init();

  dvs_mutex_init( &mMutexRingBuffer );
}


BufferManager::~BufferManager()
{
  Free();

  dvs_mutex_free( &mMutexRingBuffer );
}


void BufferManager::Init()
{
  mCurrentRecordBuffer  = -1;
  mCurrentDisplayBuffer = -1;
  mBufferCount          = 0;
  memset( mpBuffer, 0, sizeof(mpBuffer) );
  memset( mpBufferOrig, 0, sizeof(mpBufferOrig));
}


void BufferManager::Free()
{
  for( int j = 0; j < MAX; j++ ){
    if( mpBufferOrig[j] ) {
      free(mpBufferOrig[j]);
    }
  }
  mCurrentRecordBuffer  = -1;
  mCurrentDisplayBuffer = -1;
  mBufferCount          = 0;
}


int BufferManager::AllocateBuffer( int size, int count, int dmaalign )
{
  int result = true;
  
  Free();

  if( count > MAX ){
    count = MAX;
  }
  mBufferCount = count;

  for( int i = 0; i < count; i++ ){
    mpBuffer[i] = (char*) Support::MallocAligned( size, dmaalign, &mpBufferOrig[i] );
    if( mpBuffer[i] == 0 ) {
      result = false;
    }
  }

  if( result == false ) {
    Free();
  } else {
    //Enable ring buffer
    mCurrentRecordBuffer = 0;
  }

  return result;
}


char* BufferManager::GetBuffer( int jack )
{
  char *pResultBuffer = 0;

  dvs_mutex_enter( &mMutexRingBuffer );

  if( jack == 1 ){
    if(  mCurrentRecordBuffer != -1 ) {
      //Save buffer
      pResultBuffer = mpBuffer[mCurrentRecordBuffer];
      //Calc new displaybuffer
      mCurrentDisplayBuffer = mCurrentRecordBuffer - 1;
      if( mCurrentDisplayBuffer < 0 ) {
        mCurrentDisplayBuffer = mBufferCount - 1;
      }
      //Calc new recordbuffer
      mCurrentRecordBuffer++;
      if( mCurrentRecordBuffer >= mBufferCount ) {
        mCurrentRecordBuffer = 0;
      }
    }
  }

  if( jack == 0 ){
    if(  mCurrentDisplayBuffer != -1 ) {
      //Save buffer
      pResultBuffer = mpBuffer[mCurrentDisplayBuffer];
    }
  }

  dvs_mutex_leave( &mMutexRingBuffer );

  return pResultBuffer;
}
