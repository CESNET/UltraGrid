/*
 * FILE:    aes_encrypt.c
 * AUTHOR:  Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "crypto/crc.h"
#include "crypto/md5.h"
#include "crypto/openssl_encrypt.h"
#include "debug.h"

#include <string.h>
#ifdef HAVE_CRYPTO
#include <openssl/aes.h>
#include <openssl/rand.h>
#else
#define AES_BLOCK_SIZE 16
#endif

struct openssl_encrypt {
#ifdef HAVE_CRYPTO
        AES_KEY key;
#endif

        enum openssl_mode mode;

        unsigned char ivec[16];
        unsigned int num;
        unsigned char ecount[16];
};

int openssl_encrypt_init(struct openssl_encrypt **state, const char *passphrase,
                enum openssl_mode mode)
{
#ifndef HAVE_CRYPTO
        fprintf(stderr, "This " PACKAGE_NAME " version was build "
                        "without OpenSSL support!\n");
        return -1;
#endif
        struct openssl_encrypt *s = (struct openssl_encrypt *)
                calloc(1, sizeof(struct openssl_encrypt));

        MD5_CTX context;
        unsigned char hash[16];

        MD5Init(&context);
        MD5Update(&context, (const unsigned char *) passphrase,
                        strlen(passphrase));
        MD5Final(hash, &context);

#ifdef HAVE_CRYPTO
        AES_set_encrypt_key(hash, 128, &s->key);
        if (!RAND_bytes(s->ivec, 8)) {
                return -1;
        }
#endif
        s->mode = mode;
        assert(s->mode == MODE_AES128_CTR); // only functional by now

        *state = s;
        return 0;
}

static void openssl_encrypt_block(struct openssl_encrypt *s, unsigned char *plaintext,
                unsigned char *ciphertext, char *nonce_plus_counter, int len)
{
#ifndef HAVE_CRYPTO
        UNUSED(ciphertext);
        UNUSED(plaintext);
#endif
        if(nonce_plus_counter) {
                memcpy(nonce_plus_counter, (char *) s->ivec, 16);
                /* We do start a new block so we zero the byte counter
                 * Please NB that counter doesn't need to be incremented
                 * because the counter is incremented everytime s->num == 0,
                 * presumably before encryption, so setting it to 0 forces
                 * counter increment as well.
                 */
                if(s->num != 0) {
                        s->num = 0;
                }
        }

        switch(s->mode) {
                case MODE_AES128_CTR:
#ifdef HAVE_CRYPTO
                        AES_ctr128_encrypt(plaintext, ciphertext, len, &s->key, s->ivec,
                                        s->ecount, &s->num);
#endif
                        break;
                case MODE_AES128_ECB:
                        assert(len == AES_BLOCK_SIZE);
#ifdef HAVE_CRYPTO
                        AES_ecb_encrypt(plaintext, ciphertext,
                                        &s->key, AES_ENCRYPT);
#endif
                        break;
        }
}

void openssl_encrypt_destroy(struct openssl_encrypt *s)
{
        free(s);
}

int openssl_encrypt(struct openssl_encrypt *encryption,
                char *plaintext, int data_len, char *aad, int aad_len, char *ciphertext)
{
        uint32_t crc = 0xffffffff;
        memcpy(ciphertext, &data_len, sizeof(uint32_t));
        ciphertext += sizeof(uint32_t);
        char *nonce_and_counter = ciphertext;
        ciphertext += 16;

        if(aad_len > 0) {
                crc = crc32buf_with_oldcrc(aad, aad_len, crc);
        }

        for(int i = 0; i < data_len; i+=16) {
                int block_length = 16;
                if(data_len - i < 16) block_length = data_len - i;
                crc = crc32buf_with_oldcrc(plaintext + i, block_length, crc);
                openssl_encrypt_block(encryption,
                                (unsigned char *) plaintext + i,
                                (unsigned char *) ciphertext + i,
                                nonce_and_counter,
                                block_length);
                nonce_and_counter = NULL;
        }
        openssl_encrypt_block(encryption,
                        (unsigned char *) &crc,
                        (unsigned char *) ciphertext + data_len,
                        NULL,
                        sizeof(uint32_t));
        return data_len + sizeof(crc) + 16 + sizeof(uint32_t);
}

int openssl_get_overhead(struct openssl_encrypt *s)
{
        switch(s->mode) {
                case MODE_AES128_CTR:
                        return sizeof(uint32_t) /* data_len */ +
                                16 /* nonce + counter */ + sizeof(uint32_t) /* crc */;
                default:
                        abort();
        }
}

