/*
 * FILE:    test_audio_hw.c
 * AUTHORS: Colin Perkins
 *
 * Test audio hardware probing routines
 *
 * Copyright (c) 2003-2004 University of Southern California
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
#include "debug.h"
#include "memory.h"
#include "audio_types.h"
#include "audio_hw.h"
#include "test_audio_hw.h"

int 
test_audio_hw(void)
{
	int	  		 	 devcnt, i, s;
	const audio_device_details_t	*dev;
	int				 rates[7]  = {8000, 11025, 16000, 22050, 24000, 44100, 48000};

	/*
	 * Step 1: Probe the hardware, and list the configuration...
	 */
	printf("Testing audio hardware detection ......................................... "); 
	if (!audio_init_interfaces()) {
		printf("FAIL\n");
		printf("    Cannot probe audio devices\n");
		return 1;
	}
	printf("Ok\n");

	devcnt = audio_get_device_count();
	for (i = 0; i < devcnt; i++) {
		dev = audio_get_device_details(i);
		printf("    \"%s\"\n", dev->name);

		printf("          mono:");
		for (s = 0; s < 7; s++) {
			if (audio_device_supports(dev->descriptor, rates[s], 1)) {
				printf(" %5dHz", rates[s]);
			} else {
				printf(" -----  ");
			}
		}
		printf("\n");

		printf("        stereo:");
		for (s = 0; s < 7; s++) {
			if (audio_device_supports(dev->descriptor, rates[s], 2)) {
				printf(" %5dHz", rates[s]);
			} else {
				printf(" -----  ");
			}
		}
		printf("\n");
	}

	audio_free_interfaces();
	return 0;
}

