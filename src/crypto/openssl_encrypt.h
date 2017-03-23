/**
 * @file   crypto/openssl_encrypt.h
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

#ifndef OPENSSL_ENCRYPT_H_
#define OPENSSL_ENCRYPT_H_

#ifdef __cplusplus

struct openssl_encrypt;

enum openssl_mode {
        MODE_AES128_NONE = 0,
        MODE_AES128_CTR = 1, // no autenticity, only integrity (CRC)
        MODE_AES128_CFB = 2,
        MODE_AES128_MAX = MODE_AES128_CFB,
        MODE_AES128_ECB = -1, // do not use
};


#define MAX_CRYPTO_EXTRA_DATA 24 // == maximal overhead of available encryptions
#define MAX_CRYPTO_PAD 0 // CTR does not need padding
#define MAX_CRYPTO_EXCEED (MAX_CRYPTO_EXTRA_DATA + MAX_CRYPTO_PAD)

#define OPENSSL_ENCRYPT_ABI_VERSION 1

struct openssl_encrypt_info {
        /**
         * Initializes encryption
         * @param[out] state      created state
         * @param[in]  passphrase key material (NULL-terminated)
         * @param[in]  mode
         * @retval      0         success
         * @retval     <0         failure
         * @retval     >0         state not created
         */
        int (*init)(struct openssl_encrypt **state,
                        const char *passphrase, enum openssl_mode mode);
        /**
         * Destroys state
         */
        void (*destroy)(struct openssl_encrypt *state);
        /**
         * Encrypts a block of data
         *
         * @param[in] encryption    state
         * @param[in] plaintext     plain text
         * @param[in] plaintext_len length of plain text
         * @param[in] aad           Additional Authenticated Data
         *                          this won't be encrypted but passed in plaintext along ciphertext.
         *                          These data are autheticated only if working in some AE mode
         * @param[in] aad_len       length of AAD text
         * @param[out] ciphertext   resulting ciphertext, can be up to (plaintext_len + MAX_CRYPTO_EXCEED) length
         * @returns   size of writen ciphertext
         */
        int (*encrypt)(struct openssl_encrypt *encryption,
                        char *plaintext, int plaintext_len, char *aad, int aad_len, char *ciphertext);
        /**
         * Returns maximal number of bytest that the ciphertext length may exceed plaintext for selected
         * encryption.
         *
         * @param[in] encryption    state
         * @returns max overhead (must be <= MAX_CRYPTO_EXCEED)
         */
        int (*get_overhead)(struct openssl_encrypt *encryption);
};

#endif // __cplusplus

#endif // OPENSSL_ENCRYPT_H_

