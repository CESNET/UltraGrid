/**
 * @file   crypto/openssl_encrypt.cpp
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
/**
 * @file
 * Encryption is done on per-packet basis, for every packet, new IV is
 * generated, which is send on the beginning, then length of data (uint32_t),
 * and actual encrypted data with:
 *
 * 1. 32-bit CRC (encryptied) for non-GCM
 * 2. GCM tag after encrypted data
 *
 * Encryption algorithm is set in transmit.cpp, detected on receiver. Required
 * algorightms are currently GCM (default) and CBC.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <string.h>
#ifdef HAVE_WOLFSSL
#define OPENSSL_EXTRA
#define WC_NO_HARDEN
#include <wolfssl/options.h>
#include <wolfssl/openssl/aes.h>
#include <wolfssl/openssl/err.h>
#include <wolfssl/openssl/evp.h>
#include <wolfssl/openssl/rand.h>
#else
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

#include "crypto/crc.h"
#include "crypto/md5.h"
#include "crypto/openssl_encrypt.h"
#include "debug.h"
#include "lib_common.h"

#define GCM_TAG_LEN 16
#define MOD_NAME "[encrypt] "

struct openssl_encrypt {
        EVP_CIPHER_CTX *ctx;
        const EVP_CIPHER *cipher;
        enum openssl_mode mode;
        unsigned char key_hash[16];
};

const void *get_cipher(enum openssl_mode mode) {
        switch (mode) {
               case MODE_AES128_NONE:
                       abort();
               case MODE_AES128_CFB:
#if defined HAVE_EVP_AES_128_CFB128
                       return EVP_aes_128_cfb128();
#endif
                       break;
               case MODE_AES128_CTR:
#if defined HAVE_EVP_AES_128_CTR
                       return EVP_aes_128_ctr();
#endif
                       break;
               case MODE_AES128_ECB:
#if defined HAVE_EVP_AES_128_ECB
                       return EVP_aes_128_ecb();
#endif
                       break;
               case MODE_AES128_CBC:
                       return EVP_aes_128_cbc();
                       break;
               case MODE_AES128_GCM:
                       return EVP_aes_128_gcm();
                       break;
        }
        return NULL;
}

static int openssl_encrypt_init(struct openssl_encrypt **state, const char *passphrase,
                enum openssl_mode mode)
{
        struct openssl_encrypt *s = (struct openssl_encrypt *)
                calloc(1, sizeof(struct openssl_encrypt));

        MD5CTX context;

        MD5Init(&context);
        MD5Update(&context, (const unsigned char *) passphrase,
                        strlen(passphrase));
        MD5Final(s->key_hash, &context);

        s->cipher = get_cipher(mode);
        if (s->cipher == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cipher %d not available!\n", (int) mode);
                return -1;
        }

        s->ctx = EVP_CIPHER_CTX_new();
        s->mode = mode;
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Encryption set to mode %d\n", (int) mode);

        *state = s;
        return 0;
}

static void openssl_encrypt_destroy(struct openssl_encrypt *s)
{
        EVP_CIPHER_CTX_free(s->ctx);
        free(s);
}

#define CHECK(action, errmsg) do { int rc = action; if (rc != 1) { log_msg(LOG_LEVEL_ERROR, MOD_NAME errmsg ": %s\n", ERR_error_string(ERR_get_error(), NULL)); return 0; } } while(0)

static int openssl_encrypt(struct openssl_encrypt *encryption,
                char *plaintext, int data_len, char *aad, int aad_len, char *ciphertext)
{
        memcpy(ciphertext, &data_len, sizeof(uint32_t));
        int total_len = sizeof(uint32_t);

        unsigned char ivec[16];
        if (RAND_bytes(ivec, 8) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot generate random bytes!\n");
                return 0;
        }
        memcpy(ciphertext + total_len, ivec, sizeof ivec);
        total_len += sizeof ivec;

        CHECK(EVP_CipherInit(encryption->ctx, encryption->cipher, encryption->key_hash, ivec, 1), "Cannot initialize cipher");
        /* Set IV length if default 12 bytes (96 bits) is not appropriate */
        CHECK(EVP_CIPHER_CTX_ctrl(encryption->ctx, EVP_CTRL_GCM_SET_IVLEN, sizeof ivec, NULL), "set IV len");
        int out_len = 0;
        if (encryption->mode == MODE_AES128_GCM) {
                if (aad_len > 0) {
                        EVP_EncryptUpdate(encryption->ctx, NULL, &out_len, (unsigned char *) aad, aad_len);
                }
        }
        CHECK(EVP_CipherUpdate(encryption->ctx, (unsigned char *) ciphertext + total_len, &out_len, (unsigned char *) plaintext, data_len), "EVP_CipherUpdate");
        total_len += out_len;
        if (encryption->mode != MODE_AES128_GCM) {
                uint32_t crc = crc32buf(aad, aad_len);
                crc = crc32buf_with_oldcrc(plaintext, data_len, crc);
                CHECK(EVP_CipherUpdate(encryption->ctx, (unsigned char *) ciphertext + total_len, &out_len, (unsigned char *) &crc, sizeof crc), "EVP_CipherUpdate CRC");
                total_len += out_len;
        }
        CHECK(EVP_CipherFinal(encryption->ctx, (unsigned char *) ciphertext + total_len, &out_len), "EVP_CipherFinal");
        total_len += out_len;
        if (encryption->mode == MODE_AES128_GCM) {
                CHECK(EVP_CIPHER_CTX_ctrl(encryption->ctx, EVP_CTRL_GCM_GET_TAG, GCM_TAG_LEN, ciphertext + total_len), "GCM get tag");
                total_len += GCM_TAG_LEN;
        }

        return total_len;
}

static int openssl_get_overhead(struct openssl_encrypt *s)
{
        return sizeof(uint32_t) /* data_len */ +
                16 /* nonce + counter */ + (s->mode == MODE_AES128_GCM ? GCM_TAG_LEN : sizeof(uint32_t) /* crc */)
                + (s->mode == MODE_AES128_ECB ? 15 : 0 /* padding */);
}

static const struct openssl_encrypt_info functions = {
        openssl_encrypt_init,
        openssl_encrypt_destroy,
        openssl_encrypt,
        openssl_get_overhead,
};

REGISTER_MODULE(openssl_encrypt, &functions, LIBRARY_CLASS_UNDEFINED, OPENSSL_ENCRYPT_ABI_VERSION);

