/*
 * FILE:    ntp.h
 * AUTHOR:  O.Hodson
 * 
 * NTP utility functions to make rtp and rtp round time calculation
 * a little less painful.
 *
 * Copyright (c) 2000 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "compat/gettimeofday.h"

#include "ntp.h"

#define SECS_BETWEEN_1900_1970 2208988800u

void ntp64_time(uint32_t * ntp_sec, uint32_t * ntp_frac)
{
        struct timeval now;
        uint32_t tmp;           /* now.tv_usec is signed on many platforms; compensate */

        gettimeofday(&now, NULL);

        /* NB ntp_frac is in units of 1 / (2^32 - 1) secs. */
        *ntp_sec = now.tv_sec + SECS_BETWEEN_1900_1970;
        tmp = now.tv_usec;
        *ntp_frac = (tmp << 12) + (tmp << 8) - ((tmp * 3650) >> 6);
}
