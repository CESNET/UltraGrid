/*
 * FILE:    test_codec_vdvi.c
 * AUTHORS: Colin Perkins
 *
 * Test the VDVI codec
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
#include "bitstream.h"
#include "audio_codec/vdvi_impl.h"
#include "test_codec_vdvi.h"

#ifdef NDEF

#define NUM_TESTS 100000

static  u_char src[80], pad1[4], 
        dst[80], pad2[4], 
        coded[160], pad3[4], safe[80];

static void
check_padding()
{
        assert(pad1[0] == 0xff && pad2[0] == 0xff && pad3[0] == 0xff);
        assert(pad1[1] == 0xff && pad2[1] == 0xff && pad3[1] == 0xff);
        assert(pad1[2] == 0xff && pad2[2] == 0xff && pad3[2] == 0xff);
        assert(pad1[3] == 0xff && pad2[3] == 0xff && pad3[3] == 0xff);
}
#endif

int 
test_codec_vdvi(void)
{
	printf("Testing audio codec: VDVI ................................................ --\n"); 
#ifdef NDEF
        int i, n, coded_len, out_len, a, amp;

        memset(pad1, 0xff, 4); /* Memory overwrite test */
        memset(pad2, 0xff, 4);
        memset(pad3, 0xff, 4);

        srandom(123213);

        for(n = 0; n < NUM_TESTS; n++) {
                amp = (random() &0x0f);
                for(i = 0; i< 80; i++) {
                        a = (int)(amp * sin(M_PI * 2.0 * (float)i/16.0));
                        assert(abs(a) < 16);
                        src[i] = (a << 4) & 0xf0;
                        a = amp;
                        assert(abs(a) < 16);
                        src[i] |= (a & 0x0f);
                }

                memcpy(safe, src, 80);

                coded_len = vdvi_encode(src, 160, coded, 160);

                assert(!memcmp(src,safe,80));

                check_padding();
                out_len   = vdvi_decode(coded, 160, dst, 160);
                
                assert(!memcmp(src,safe,80));
                assert(!memcmp(dst,safe,80)); /* dst matches sources */

                assert(coded_len == out_len);

                check_padding();

                for(i = 0; i< 80; i++) {
                        assert(src[i] == dst[i]);
                }
        }
        printf("Ok\n");
#endif
        return 0;
}

