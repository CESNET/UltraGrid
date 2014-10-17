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

#ifndef  TIMER_INC
#define  TIMER_INC

#include <chrono>

#if defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ == 6 // compat, GCC 4.6 didn't have steady_clock
#define steady_clock system_clock
#endif

/*! \class Timer_util
 *  \brief This class implements timer utils
 */
class Timer_util
{
    private:
         std::chrono::time_point<std::chrono::steady_clock> start_time;
         std::chrono::time_point<std::chrono::steady_clock> end_time;

    public:
         inline void start() {
             start_time = std::chrono::steady_clock::now();
         }

         inline void end() {
             end_time = std::chrono::steady_clock::now();
         }

	/** 
	 * Computes time elapsed between start and end in seconds (as a real value)
	 * 
	 * @return Elapsed seconds as double value
	 */
	inline double
	    elapsed_time () const
	    {
                return std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
	    }

	inline double
	    elapsed_time_ms () const
	    {
                return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time).count();
	    }

	inline long
	    elapsed_time_us () const
	    {
                return std::chrono::duration_cast<std::chrono::duration<long, std::micro>>(end_time - start_time).count();
	    }
}; /*  -----  end of class Timer_util  ----- */

#endif   /* ----- #ifndef TIMER_INC  ----- */

