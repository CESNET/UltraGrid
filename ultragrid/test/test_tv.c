/*
 * FILE:    test_tv.c
 * AUTHORS: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2003-2004 University of Glasgow
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "tv.h"
#include "test_tv.h"

int test_tv(void)
{
        struct timeval t1, t2;
        double d;
        double e = 0.0000001;   /* epsilon for double comparisons. this is evil. */
        uint32_t m1, m2, m3;

        printf
            ("Testing time conversion functions ........................................ ");

        /* Test 1: check that tv_add() works...                        */
        /*   1A) Normal operation                                      */
        t1.tv_sec = 1;
        t1.tv_usec = 0;
        tv_add(&t1, 0.8);
        if ((t1.tv_sec != 1) && (t1.tv_usec != 800000)) {
                printf("FAIL\n");
                printf("  tv_add() part A\n");
                return 1;
        }
        /*   1B) Wraparound of tv_usec during the addition             */
        tv_add(&t1, 0.5);
        if ((t1.tv_sec != 2) && (t1.tv_usec != 300000)) {
                printf("FAIL\n");
                printf("  tv_add() part B\n");
                return 1;
        }
        /*   1C) Wraparound when the offset is greater than one second */
        tv_add(&t1, 4.2);
        if ((t1.tv_sec != 6) && (t1.tv_usec != 500000)) {
                printf("FAIL\n");
                printf("  tv_add() part C\n");
                return 1;
        }

        /* Test 2: check that tv_gt() works...                         */
        /*   2A) Normal operation                                      */
        t1.tv_sec = 8143;
        t1.tv_usec = 500000;
        t2.tv_sec = 4294;
        t2.tv_usec = 345678;
        if (tv_gt(t2, t1) || !tv_gt(t1, t2)) {
                printf("FAIL\n");
                printf("  tv_gt() part A\n");
                return 1;
        }

        /*   2B) See if it works when tv_sec values are the same       */
        t1.tv_sec = 8147;
        t1.tv_usec = 500000;
        t2.tv_sec = 8147;
        t2.tv_usec = 345678;
        if (tv_gt(t2, t1) || !tv_gt(t1, t2)) {
                printf("FAIL\n");
                printf("  tv_gt() part B\n");
                return 1;
        }

        /* Test 3: check that tv_diff() works...                       */
        /*   3A) normal operation. comparing floats is tricky :(       */
        t1.tv_sec = 1065356371; /* Sunday afternoon, 5th October 2003 */
        t1.tv_usec = 234947;
        t2.tv_sec = 1065356528; /* ...a few minutes later, see how    */
        t2.tv_usec = 864316;    /* fast I type :)                     */
        d = tv_diff(t2, t1);
        if ((d <= 157.629369 - e) || (d >= 157.629369 + e)) {
                printf("FAIL\n");
                printf("  tv_diff: %f != 157.629369\n", d);
                return 1;
        }
        /*   3B) when prev_time is newer than curr_time?  Not what we  */
        /*       expect to work, but may as well make sure it gives a  */
        /*       sensible answer, rather than crashing out.            */
        d = tv_diff(t1, t2);
        if ((d <= -157.629369 - e) || (d >= -157.629369 + e)) {
                printf("FAIL\n");
                printf("  tv_diff: %f != -157.629369\n", d);
                return 1;
        }

        /* Test 4: check that tv_diff_usec() works...                  */
        /*   4A) normal operation                                      */
        t1.tv_sec = 1065356371;
        t1.tv_usec = 234947;
        t2.tv_sec = 1065356528;
        t2.tv_usec = 864316;
        if (tv_diff_usec(t2, t1) != 157629369) {
                printf("FAIL\n");
                printf("  tv_diff_usec: A\n");
                return 1;
        }
        /*   4B) see what happens if the tv_sec values are the same    */
        t1.tv_sec = 1065356371;
        t1.tv_usec = 234947;
        t2.tv_sec = 1065356371;
        t2.tv_usec = 864316;
        if (tv_diff_usec(t2, t1) != 629369) {
                printf("FAIL\n");
                printf("  tv_diff_usec: B\n");
                return 1;
        }
        /*   4C) close values on a second boundary                     */
        t1.tv_sec = 1065356371;
        t1.tv_usec = 999999;
        t2.tv_sec = 1065356372;
        t2.tv_usec = 000000;
        if (tv_diff_usec(t2, t1) != 1) {
                printf("FAIL\n");
                printf("  tv_diff_usec: B\n");
                return 1;
        }

        /* Test 5: check that get_local_mediatime() works...           */
        m1 = get_local_mediatime();
        usleep(250000);         /* Sleep for a minimum of 250ms */
        m2 = get_local_mediatime();
        if (m2 - m1 < 22500) {
                printf("FAIL\n");
                printf("  get_local_mediatime() runs fast\n");
                return 1;
        }
        m3 = get_local_mediatime();
        if (m3 < m2) {
                printf("FAIL\n");
                printf("  get_local_mediatime() time runs backwards\n");
                return 1;
        }

        printf("Ok\n");
        return 0;
}
