/**
 * @file   utils/hash_table.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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

#ifndef DICTIONARY_H_A228804B_D47D_4AE3_85A4_D479101FDAA8
#define DICTIONARY_H_A228804B_D47D_4AE3_85A4_D479101FDAA8

#include <stddef.h>

struct dictionary *dictionary_init(void);
void dictionary_insert(struct dictionary *dictionary, const char *key,
                       const char *val);
void dictionary_insert2(struct dictionary *dictionary, const char *key_val);
const char *dictionary_lookup(struct dictionary *dictionary, const char *key);
const char *dictionary_first(struct dictionary *dictionary, const char **key);
const char *dictionary_next(struct dictionary *dictionary, const char **key);
void        dictionary_destroy(struct dictionary *dictionary);

#define DICTIONARY_ITERATE(dict, key_var, val_var) for (const char *key_var = NULL, \
             *val_var = dictionary_first(dict, &(key_var)); \
             (val_var) != NULL; \
             (val_var) = dictionary_next(dict, &(key_var)))

#endif // defined DICTIONARY_H_A228804B_D47D_4AE3_85A4_D479101FDAA8
