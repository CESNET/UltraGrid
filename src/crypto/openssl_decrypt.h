/*
 * FILE:   openssl_encrypt.h
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2002 University of Southern California
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

#ifndef OPENSSL_DECRYPT_H_
#define OPENSSL_DECRYPT_H_

#include "crypto/openssl_encrypt.h"

struct openssl_decrypt;

/**
 * Creates decryption state
 *
 * @param[out] state     state pointer
 * @param[in] passphrase key material (NULL-terminated)
 * @param[in] mode       mode
 * @retval    0          ok
 * @retval   <0          failure
 * @retval   >0          state was not created
 */
int openssl_decrypt_init(struct openssl_decrypt **state,
                const char *passphrase, enum openssl_mode mode);
/**
 * Destroys decryption state
 */
void openssl_decrypt_destroy(struct openssl_decrypt *state);
/**
 * Decrypts block of data
 *
 * @param[in] decrypt decrypt state
 * @param[in] ciphertext encrypted text
 * @param[in] ciphertext_len lenght of encrypted text
 * @param[in] aad Aditional Authenticated Data (see openssl_encrypt documentation)
 * @param[in] aad_len length of aad block
 * @param[out] plaintext otput plaintext
 * @retval 0 if checksum doesn't match
 * @retval >0 length of output plaintext
 */
int openssl_decrypt(struct openssl_decrypt *decrypt,
                const char *ciphertext, int ciphertext_len,
                const char *aad, int aad_len,
                char *plaintext);

#endif //  OPENSSL_DECRYPT_H_

