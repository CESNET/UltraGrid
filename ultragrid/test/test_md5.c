/*
 * FILE:    test_md5.c
 * AUTHORS: Colin Perkins
 *
 * Test vector for MD5, taken from RFC 1321
 *
 * Copyright (c) 2002-2004 University of Southern California
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
#include "crypto/md5.h"
#include "test_md5.h"

int test_md5(void)
{
        MD5_CTX context;
        unsigned char digest[16];
        unsigned char i1[] = "";
        unsigned char o1[] =
            { 0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09,
   0x98, 0xec, 0xf8, 0x42, 0x7e };
        unsigned char i2[] = "a";
        unsigned char o2[] =
            { 0x0c, 0xc1, 0x75, 0xb9, 0xc0, 0xf1, 0xb6, 0xa8, 0x31, 0xc3, 0x99,
   0xe2, 0x69, 0x77, 0x26, 0x61 };
        unsigned char i3[] = "abc";
        unsigned char o3[] =
            { 0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0, 0xd6, 0x96, 0x3f,
   0x7d, 0x28, 0xe1, 0x7f, 0x72 };
        unsigned char i4[] = "message digest";
        unsigned char o4[] =
            { 0xf9, 0x6b, 0x69, 0x7d, 0x7c, 0xb7, 0x93, 0x8d, 0x52, 0x5a, 0x2f,
   0x31, 0xaa, 0xf1, 0x61, 0xd0 };
        unsigned char i5[] = "abcdefghijklmnopqrstuvwxyz";
        unsigned char o5[] =
            { 0xc3, 0xfc, 0xd3, 0xd7, 0x61, 0x92, 0xe4, 0x00, 0x7d, 0xfb, 0x49,
   0x6c, 0xca, 0x67, 0xe1, 0x3b };
        unsigned char i6[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        unsigned char o6[] =
            { 0xd1, 0x74, 0xab, 0x98, 0xd2, 0x77, 0xd9, 0xf5, 0xa5, 0x61, 0x1c,
   0x2c, 0x9f, 0x41, 0x9d, 0x9f };
        unsigned char i7[] =
            "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
        unsigned char o7[] =
            { 0x57, 0xed, 0xf4, 0xa2, 0x2b, 0xe3, 0xc9, 0x55, 0xac, 0x49, 0xda,
   0x2e, 0x21, 0x07, 0xb6, 0x7a };

        printf
            ("Testing MD5 .............................................................. ");
        fflush(stdout);

        MD5Init(&context);
        MD5Update(&context, i1, strlen(i1));
        MD5Final(digest, &context);
        if (strncmp(digest, o1, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }

        MD5Init(&context);
        MD5Update(&context, i2, strlen(i2));
        MD5Final(digest, &context);
        if (strncmp(digest, o2, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }

        MD5Init(&context);
        MD5Update(&context, i3, strlen(i3));
        MD5Final(digest, &context);
        if (strncmp(digest, o3, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }

        MD5Init(&context);
        MD5Update(&context, i4, strlen(i4));
        MD5Final(digest, &context);
        if (strncmp(digest, o4, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }

        MD5Init(&context);
        MD5Update(&context, i5, strlen(i5));
        MD5Final(digest, &context);
        if (strncmp(digest, o5, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }

        MD5Init(&context);
        MD5Update(&context, i6, strlen(i6));
        MD5Final(digest, &context);
        if (strncmp(digest, o6, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }

        MD5Init(&context);
        MD5Update(&context, i7, strlen(i7));
        MD5Final(digest, &context);
        if (strncmp(digest, o7, 16) != 0) {
                printf("FAIL\n");
                return 1;
        }
        printf("Ok\n");
        return 0;
}
