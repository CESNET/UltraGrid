/*
 * AUTHOR:   N.Cihan Tas
 * MODIFIED: Ladan Gharai
 *           Colin Perkins
 * 
 * This program is the playout time calculation for each frame
 * received.
 *
 * Copyright (c) 2003 University of Southern California
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/ptime.h"

#define SKEW_THRESHOLD 1500
#define W              10       /* Window size for mapping to local timeline. */

#ifdef NDEF
/* Colin's book pg. 178 */
static uint32_t adjustment_due_to_skew(uint32_t ts, uint32_t curr_time)
{

        static int first_time = 1;
        static uint32_t delay_estimate;
        static uint32_t active_delay;
        int adjustment = 0;
        uint32_t d_n = ts - curr_time;

        if (first_time) {

                first_time = 0;
                delay_estimate = d_n;
                active_delay = d_n;
        } else {
                delay_estimate = (31 * delay_estimate + d_n) / 32.0;
        }

        if (active_delay - delay_estimate > SKEW_THRESHOLD) {

                adjustment = SKEW_THRESHOLD;
                active_delay = delay_estimate;
        }
        if ((double)active_delay - (double)delay_estimate < -SKEW_THRESHOLD) {
                adjustment = -SKEW_THRESHOLD;
                active_delay = delay_estimate;
        }

        return (uint32_t) ((double)active_delay + (double)adjustment);
}
#endif

#ifdef NDEF
static uint32_t map_local_timeline(uint32_t ts, uint32_t curr_time)
{

        static uint32_t d[W];
        static int w = 0;
        static int FULL = 0;
        uint32_t min;
        uint32_t new_difference = ts - curr_time;
        int i;

        d[w++] = new_difference;

        /* Do we have enough records? If so, begin to rewrite oldest first. */
        if (w >= W - 1) {

                FULL = 1;
                w = 0;

        }

        min = d[0];

        if (!FULL) {
                for (i = 1; i < w; i++)
                        if (d[i] < min)
                                //if (((d[i] - min) & (1<<15)) != 0)                                         
                                min = d[i];
        } else
                for (i = 1; i < W; i++)
                        if (d[i] < min)
                                //if (((d[i] - min) & (1<<15)) != 0)                                         
                                min = d[i];

        return (min);

}
#endif

//uint32_t
//calculate_playout_time(struct timeval atime_s, uint32_t timestamp){
//uint32_t atime = tv_timeval2uint(atime_s);
//return(map_local_timeline( atime , timestamp ) + adjustment_due_to_skew( atime , timestamp ));
//}
