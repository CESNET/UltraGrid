/*
 * FILE:    aes_decrypt.c
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
#endif // HAVE_CONFIG_H


#include "crypto/crc.h"
#include "crypto/md5.h"
#include "crypto/openssl_decrypt.h"
#include "debug.h"

#include <string.h>
#ifdef HAVE_CRYPTO
#include <openssl/aes.h>
#else
#define AES_BLOCK_SIZE 16
#endif

struct openssl_decrypt {
#ifdef HAVE_CRYPTO
        AES_KEY key;
#endif // HAVE_CRYPTO

        enum openssl_mode mode;

        unsigned char ivec[AES_BLOCK_SIZE];
        unsigned char ecount[AES_BLOCK_SIZE];
        unsigned int num;
};

int openssl_decrypt_init(struct openssl_decrypt **state,
                                const char *passphrase,
                                enum openssl_mode mode)
{
#ifndef HAVE_CRYPTO
        fprintf(stderr, "This " PACKAGE_NAME " version was build "
                        "without OpenSSL support!\n");
        return -1;
#endif // HAVE_CRYPTO

        struct openssl_decrypt *s =
                (struct openssl_decrypt *)
                calloc(1, sizeof(struct openssl_decrypt));

        MD5_CTX context;
        unsigned char hash[16];

        MD5Init(&context);
        MD5Update(&context, (const unsigned char *) passphrase,
                        strlen(passphrase));
        MD5Final(hash, &context);

        switch(mode) {
                case MODE_AES128_ECB:
#ifdef HAVE_CRYPTO
                        AES_set_decrypt_key(hash, 128, &s->key);
#endif
                        break;
                case MODE_AES128_CTR:
#ifdef HAVE_CRYPTO
                        AES_set_encrypt_key(hash, 128, &s->key);
#endif
                        break;
                default:
                        abort();
        }

        s->mode = mode;

        *state = s;
        return 0;
}

void openssl_decrypt_destroy(struct openssl_decrypt *s)
{
        if(!s)
                return;
        free(s);
}

static void openssl_decrypt_block(struct openssl_decrypt *s,
                const unsigned char *ciphertext, unsigned char *plaintext, const char *nonce_and_counter,
                int len)
{
#ifndef HAVE_CRYPTO
        UNUSED(ciphertext);
        UNUSED(plaintext);
#endif
        if(nonce_and_counter) {
                memcpy(s->ivec, nonce_and_counter, AES_BLOCK_SIZE);
                s->num = 0;
        }

        switch(s->mode) {
                case MODE_AES128_ECB:
                        assert(len == AES_BLOCK_SIZE);
#ifdef HAVE_CRYPTO
                        AES_ecb_encrypt(ciphertext, plaintext,
                                        &s->key, AES_DECRYPT);
#endif // HAVE_CRYPTO
                        break;
                case MODE_AES128_CTR:
#ifdef HAVE_CRYPTO
                        AES_ctr128_encrypt(ciphertext, plaintext, len, &s->key, s->ivec,
                                        s->ecount, &s->num);
#endif
                        break;
                default:
                        abort();
        }
}

int openssl_decrypt(struct openssl_decrypt *decrypt,
                const char *ciphertext, int ciphertext_len,
                const char *aad, int aad_len,
                char *plaintext)
{
        UNUSED(ciphertext_len);
        uint32_t data_len;
        memcpy(&data_len, ciphertext, sizeof(uint32_t));
        ciphertext += sizeof(uint32_t);

        const char *nonce_and_counter = ciphertext;
        ciphertext += 16;
        uint32_t expected_crc;
        uint32_t crc = 0xffffffff;
        if(aad_len > 0) {
                crc = crc32buf_with_oldcrc((const char *) aad, aad_len, crc);
        }
        for(unsigned int i = 0; i < data_len; i += 16) {
                int block_length = 16;
                if(data_len - i < 16) block_length = data_len - i;
                openssl_decrypt_block(decrypt,
                                (const unsigned char *) ciphertext + i,
                                (unsigned char *) plaintext + i,
                                nonce_and_counter, block_length);
                nonce_and_counter = NULL;
                crc = crc32buf_with_oldcrc((char *) plaintext + i, block_length, crc);
        }
        openssl_decrypt_block(decrypt,
                        (const unsigned char *) ciphertext + data_len,
                        (unsigned char *) &expected_crc,
                        0, sizeof(uint32_t));
        if(crc != expected_crc) {
                return 0;
        }
        return data_len;
}

