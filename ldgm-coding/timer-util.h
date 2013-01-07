/*
 * =====================================================================================
 *
 *       Filename:  timer.h
 *
 *    Description:  Timer utils. 
 *
 *         Author:  Milan Kabat, kabat@ics.muni.cz
 *
 * =====================================================================================
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifndef WIN32

#ifndef  TIMER_INC
#define  TIMER_INC

#include <time.h>

/*! \class Timer_util
 *  \brief This class implements timer utils
 */
class Timer_util
{
    public:
	/** Constructor */
	Timer_util () {}
	/** Destructor */
	~Timer_util () {}           

	/** 
	 * Computes time elapsed between start and end in second (as a real value) 
	 * 
	 * @param start Point in time where timing started
	 * @param end Point in time where timing stopped
	 * @return Elapsed seconds as double value
	 */
	double
	    elapsed_time ( timespec start, timespec end) 
	    {
		timespec ts = dif(start, end);
		double temp = ts.tv_nsec/1000000;
		temp /= 1000;
		double clock = ts.tv_sec + temp;
		return clock;
	    }

	double
	    elapsed_time_us ( timespec start, timespec end) 
	    {
		timespec ts = dif(start, end);
		double temp = ts.tv_nsec/1000;
//		temp /= 1000;
		double clock = ts.tv_sec + temp;
		return clock;
	    }
	/**
	 * @param start Point in time where timing started
	 * @param end Point in time where timing stopped
	 * @return Elapsed time as timespec structure
	 */
	timespec
	    dif(timespec start, timespec end) {
		timespec temp;
		if ((end.tv_nsec-start.tv_nsec)<0) {
		    temp.tv_sec = end.tv_sec-start.tv_sec-1;
		    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
		} else {
		    temp.tv_sec = end.tv_sec-start.tv_sec;
		    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
		}
		return temp;
	    }


}; /*  -----  end of class Timer_util  ----- */

#endif   /* ----- #ifndef TIMER_INC  ----- */

#endif // WIN32

