#ifndef _STOPWATCH_H
#define _STOPWATCH_H

#ifndef EXCLUDE_FROM_DOCUMENTATION
/**
DVS internal only.
*/

#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <stdio.h>

#ifdef WIN32
  #include <windows.h>
  #include <winnt.h>
  typedef unsigned __int64 uint64;
  typedef LARGE_INTEGER dvs_time;
#else
  #include <sys/time.h>
  typedef long long unsigned int uint64;
  typedef struct timeval dvs_time;
#endif

class StopWatch
{

public:
  StopWatch();
  ~StopWatch();

  void Start();
  void Stop(){ mRunning = false; }
  uint64 Elapsed();
  bool Running(){ return mRunning; }

  void SaveElapsed() { mSaved = Elapsed(); }
  uint64 GetElapsed() { return mSaved; }

private:
  bool mRunning;
  uint64 mTimeCalibration;

  void ReadTimeFromSystem( dvs_time *pTime );
  void Delta( dvs_time *pStart, dvs_time *pEnd, uint64 * us );

  dvs_time mTime;
  uint64 mSaved;
  
#ifdef WIN32
  dvs_time mFrequence;
#endif

};

#endif

#endif //EXCLUDE_FROM_DOCUMENTATION

#endif /* !_STOPWATCH_H */
