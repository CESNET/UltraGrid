/*
 * FILE:    test_bitstream.c
 * AUTHORS: Colin Perkins
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
#include "bitstream.h"
#include "test_bitstream.h"

#define BUFSIZE 10

int test_bitstream(void)
{
        bitstream_t *bs;
        u_char *buffer;
        int buflen;

        printf
            ("Testing bitstreams ....................................................... ");
        fflush(stdout);

        buffer = malloc(BUFSIZE);
        if (buffer == NULL) {
                printf("FAIL\n");
                return 1;
        }
        buflen = BUFSIZE;

        bs_create(&bs);
        bs_attach(bs, buffer, buflen);
        bs_put(bs, 0x0f, 4);
        bs_put(bs, 0x01, 1);
        bs_put(bs, 0x02, 3);
        bs_put(bs, 0xa8, 8);
        bs_put(bs, 0xff, 1);
        if (buffer[0] != 0xfa) {
                printf("FAIL\n");
                printf("  buffer[0] = 0x%02x\n", buffer[0]);
                return 1;
        }
        if (buffer[1] != 0xa8) {
                printf("FAIL\n");
                printf("  buffer[1] = 0x%02x\n", buffer[1]);
                return 1;
        }
        if (buffer[2] != 0x80) {
                printf("FAIL\n");
                printf("  buffer[2] = 0x%02x\n", buffer[2]);
                return 1;
        }
        bs_put(bs, 0x01, 7);
        if (buffer[2] != 0x81) {
                printf("FAIL\n");
                printf("  buffer[2] = 0x%02x\n", buffer[2]);
                return 1;
        }

        bs_destroy(&bs);

        printf("Ok\n");
        return 0;
}
