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


#include <string.h>
#ifdef HAVE_WOLFSSL
#define OPENSSL_EXTRA
#define WC_NO_HARDEN
#include <wolfssl/options.h>
#include <wolfssl/openssl/aes.h>
#include <wolfssl/openssl/err.h>
#include <wolfssl/openssl/evp.h>
#else
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#endif

#include "crypto/crc.h"
#include "crypto/md5.h"
#include "crypto/openssl_decrypt.h"
#include "crypto/openssl_encrypt.h" // get_cipher
#include "debug.h"
#include "lib_common.h"

#define GCM_TAG_LEN 16
#define MOD_NAME "[decrypt] "

struct openssl_decrypt {
        EVP_CIPHER_CTX *ctx;
        unsigned char key_hash[16];

        unsigned char ivec[AES_BLOCK_SIZE];
        unsigned char ecount[AES_BLOCK_SIZE];
        unsigned int num;
};

static int openssl_decrypt_init(struct openssl_decrypt **state,
                                const char *passphrase)
{
        struct openssl_decrypt *s =
                calloc(1, sizeof(struct openssl_decrypt));

        MD5CTX context;

        MD5Init(&context);
        MD5Update(&context, (const unsigned char *) passphrase,
                        strlen(passphrase));
        MD5Final(s->key_hash, &context);

        s->ctx = EVP_CIPHER_CTX_new();
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Enabled stream decryption.\n");

        *state = s;
        return 0;
}

static void openssl_decrypt_destroy(struct openssl_decrypt *s)
{
        if(!s) {
                return;
        }
        EVP_CIPHER_CTX_free(s->ctx);
        free(s);
}

#define CHECK(action, errmsg) do { int rc = action; if (rc != 1) { log_msg(LOG_LEVEL_ERROR, MOD_NAME errmsg ": %s\n", ERR_error_string(ERR_get_error(), NULL)); return 0; } } while(0)
#pragma GCC diagnostic ignored "-Wcast-qual"
static int openssl_decrypt(struct openssl_decrypt *decrypt,
                const char *ciphertext, int ciphertext_len,
                const char *aad, int aad_len,
                char *plaintext, enum openssl_mode mode)
{
        const EVP_CIPHER *cipher = get_cipher(mode);
        if (cipher == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cipher %d not available!\n", (int) mode);
                return 0;
        }
        uint32_t data_len;
        memcpy(&data_len, ciphertext, sizeof(uint32_t));
        assert ((size_t) ciphertext_len >= data_len + sizeof(uint32_t) + 16 + sizeof(uint32_t));
        ciphertext += sizeof(uint32_t);
        const unsigned char *iv = (const unsigned char *) ciphertext;
        ciphertext += 16;
        ciphertext_len -= 20;

        CHECK(EVP_CipherInit(decrypt->ctx, cipher, decrypt->key_hash, iv, 0), "Unable to initialize cipher");
        CHECK(EVP_CIPHER_CTX_ctrl(decrypt->ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL), "set IV len"); // default IV len is presumably 12 bytes

        int out_len = 0;
        if (mode == MODE_AES128_GCM) {
                ciphertext_len -= GCM_TAG_LEN;
                if (aad && aad_len > 0) {
                        if (!EVP_DecryptUpdate(decrypt->ctx, NULL, &out_len, (void *) aad, aad_len)) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "AAD processing: %s\n", ERR_error_string(ERR_get_error(), NULL));
                        }
                }
        }
        CHECK(EVP_CipherUpdate(decrypt->ctx, (unsigned char *) plaintext, &out_len, (const unsigned char *) ciphertext, ciphertext_len), "EVP_CipherUpdate");
        int total_len = out_len;
        if (mode == MODE_AES128_GCM) {
                CHECK(EVP_CIPHER_CTX_ctrl(decrypt->ctx, EVP_CTRL_GCM_SET_TAG, GCM_TAG_LEN, (void *) (ciphertext + ciphertext_len)), "GCM set tag");
        }
        CHECK(EVP_CipherFinal(decrypt->ctx, (unsigned char *) plaintext + out_len, &out_len), "EVP_CipherFinal");
        total_len += out_len;

        if (mode != MODE_AES128_GCM) {
                uint32_t expected_crc = 0;
                assert((size_t) total_len >= data_len + sizeof expected_crc);
                memcpy(&expected_crc, plaintext + data_len, sizeof expected_crc);
                uint32_t crc = crc32buf(aad, aad_len);
                crc = crc32buf_with_oldcrc(plaintext, data_len, crc);
                if (crc != expected_crc) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: Packet dropped AES - wrong CRC!\n");
                        return 0;
                }
        }

        return data_len;
}

static const struct openssl_decrypt_info functions = {
        openssl_decrypt_init,
        openssl_decrypt_destroy,
        openssl_decrypt,
};

REGISTER_MODULE(openssl_decrypt, &functions, LIBRARY_CLASS_UNDEFINED, OPENSSL_DECRYPT_ABI_VERSION);

