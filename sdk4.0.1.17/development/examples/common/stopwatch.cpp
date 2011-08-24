#ifndef EXCLUDE_FROM_DOCUMENTATION
/**
DVS internal only.
*/

#include "stopwatch.h"

StopWatch::StopWatch()
{
  mTimeCalibration = 0;
  mRunning         = false;
  mSaved = 0;

  //Time calibration
  dvs_time start, end;

#if defined(WIN32)
  //Get System Frequence
  QueryPerformanceFrequency( &mFrequence );
#endif

  ReadTimeFromSystem( &start );
  ReadTimeFromSystem( &start );
  ReadTimeFromSystem( &start );
  ReadTimeFromSystem( &end   );

  Delta(&start, &end, &mTimeCalibration);
}


StopWatch::~StopWatch()
{
}


void StopWatch::Start()
{
  ReadTimeFromSystem( &mTime );

  mRunning = true;
}


uint64 StopWatch::Elapsed()
{
  uint64 result = 0;

  dvs_time timeNow;
  ReadTimeFromSystem( &timeNow );

  Delta( &mTime, &timeNow, &result );

  return result;  
}


void StopWatch::ReadTimeFromSystem( dvs_time *pTime ) 
{
  //Use win or linux specific timer
  #if defined WIN32
    QueryPerformanceCounter( pTime );
  #else
    struct timezone tz;
    gettimeofday( pTime, &tz );
  #endif
}


void StopWatch::Delta( dvs_time *pStart, dvs_time *pEnd, uint64 * us )
{
#if defined(WIN32)
  if (pEnd->QuadPart < pStart->QuadPart) {
    *us = 0;
  } else if (mFrequence.QuadPart - mTimeCalibration) {
    *us = ((pEnd->QuadPart - pStart->QuadPart) * 1000000) / mFrequence.QuadPart - mTimeCalibration;
  } else {
    *us = 0;
  }
#else
  if ((pEnd->tv_sec < pStart->tv_sec) ||
    ((pEnd->tv_sec == pStart->tv_sec) && (pEnd->tv_usec < pStart->tv_usec))) {
    *us = 0;
  }else{
    *us = ((pEnd->tv_sec - pStart->tv_sec) * 1000000) + (pEnd->tv_usec - pStart->tv_usec) - mTimeCalibration;
  }
#endif
}

#endif //EXCLUDE_FROM_DOCUMENTATION
