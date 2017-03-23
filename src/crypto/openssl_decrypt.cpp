/**
 * @file   crypto/openssl_decrypt.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
#include "lib_common.h"

#include <string.h>
#include <openssl/aes.h>

struct openssl_decrypt {
        AES_KEY key;

        unsigned char ivec[AES_BLOCK_SIZE];
        unsigned char ecount[AES_BLOCK_SIZE];
        unsigned int num;
};

static int openssl_decrypt_init(struct openssl_decrypt **state,
                                const char *passphrase)
{
        struct openssl_decrypt *s =
                (struct openssl_decrypt *)
                calloc(1, sizeof(struct openssl_decrypt));

        MD5_CTX context;
        unsigned char hash[16];

        MD5Init(&context);
        MD5Update(&context, (const unsigned char *) passphrase,
                        strlen(passphrase));
        MD5Final(hash, &context);

        AES_set_encrypt_key(hash, 128, &s->key);
        // for ECB it should be AES_set_decrypt_key(hash, 128, &s->key);

        *state = s;
        return 0;
}

static void openssl_decrypt_destroy(struct openssl_decrypt *s)
{
        if(!s)
                return;
        free(s);
}

static void openssl_decrypt_block(struct openssl_decrypt *s,
                const unsigned char *ciphertext, unsigned char *plaintext, const char *ivec_or_nonce_and_counter,
                int len, enum openssl_mode mode)
{
        if (ivec_or_nonce_and_counter) {
                memcpy(s->ivec, ivec_or_nonce_and_counter, AES_BLOCK_SIZE);
                s->num = 0;
        }

        switch (mode) {
                case MODE_AES128_NONE:
                        abort();
                case MODE_AES128_ECB:
                        assert(len == AES_BLOCK_SIZE);
                        AES_ecb_encrypt(ciphertext, plaintext,
                                        &s->key, AES_DECRYPT);
                        break;
                case MODE_AES128_CTR:
#ifdef HAVE_AES_CTR128_ENCRYPT
                        AES_ctr128_encrypt(ciphertext, plaintext, len, &s->key, s->ivec,
                                        s->ecount, &s->num);
#else
                        log_msg(LOG_LEVEL_ERROR, "AES CTR not compiled in!\n");
#endif
                        break;
                case MODE_AES128_CFB:
                        {
                                int inum = s->num;
                                AES_cfb128_encrypt(ciphertext, plaintext, len, &s->key, s->ivec,
                                                &inum, AES_DECRYPT);
                                s->num = inum;
                        }
                        break;
                default:
                        abort();
        }
}

static int openssl_decrypt(struct openssl_decrypt *decrypt,
                const char *ciphertext, int ciphertext_len,
                const char *aad, int aad_len,
                char *plaintext, enum openssl_mode mode)
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
                                nonce_and_counter, block_length, mode);
                nonce_and_counter = NULL;
                crc = crc32buf_with_oldcrc((char *) plaintext + i, block_length, crc);
        }
        openssl_decrypt_block(decrypt,
                        (const unsigned char *) ciphertext + data_len,
                        (unsigned char *) &expected_crc,
                        0, sizeof(uint32_t), mode);
        if(crc != expected_crc) {
                return 0;
        }
        return data_len;
}

static const struct openssl_decrypt_info functions = {
        openssl_decrypt_init,
        openssl_decrypt_destroy,
        openssl_decrypt,
};

REGISTER_MODULE(openssl_decrypt, &functions, LIBRARY_CLASS_UNDEFINED, OPENSSL_DECRYPT_ABI_VERSION);

