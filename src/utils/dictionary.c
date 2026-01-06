/**
 * @file   utils/dictionary.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * this file implementents simple key/value dictionary in string
 *
 * Currenttly simple linked listi is used with linear lookup/insertion.
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

#include "dictionary.h"

#include <assert.h> // for assert
#include <stdlib.h> // for calloc, free
#include <string.h> // for strdup, strchr
#include <stdint.h> // for uint32_t

#include "utils//macros.h"  // for to_fourcc

#define MAGIC to_fourcc('d', 'i', 'c', 't')

struct item {
        char        *key;
        char        *val;
        struct item *next;
};

struct dictionary {
        uint32_t magic;
        struct item *items;

        struct item *iterator;
};

struct dictionary *
dictionary_init()
{
        struct dictionary *dictionary = calloc(1, sizeof *dictionary);
        dictionary->magic = MAGIC;
        return dictionary;
}

/**
 * insert specified element
 */
void
dictionary_insert(struct dictionary *dictionary, const char *key,
                  const char *val)
{
        struct item **cur_p = &dictionary->items;
        while (*cur_p != NULL) {
                struct item *cur = *cur_p;
                if (strcmp(cur->key, key) == 0) { // replace
                        free(cur->val);
                        cur->val = strdup(val);
                        return;
                }
                if (strcmp(cur->key, key) > 0) { // prepend
                        struct item *new = calloc(1, sizeof(struct item));
                        new->key         = strdup(key);
                        new->val         = strdup(val);
                        new->next        = cur;
                        *cur_p           = new;
                        return;
                }
                cur_p = &(*cur_p)->next;
        }
        // append as last item
        struct item *new = calloc(1, sizeof(struct item));
        new->key         = strdup(key);
        new->val         = strdup(val);
        *cur_p           = new;
}

/**
 * insert an element in format key=vale
 */
void
dictionary_insert2(struct dictionary *dictionary, const char *key_val)
{
        assert(strchr(key_val, '=') != NULL);
        char *key = strdup(key_val);
        char *val = strchr(key, '=');
        *val      = '\0';
        val += 1;
        dictionary_insert(dictionary, key, val);
        free(key);
}

const char *
dictionary_lookup(struct dictionary *dictionary, const char *key)
{
        struct item *cur = dictionary->items;
        while (cur != NULL) {
                if (strcmp(cur->key, key) == 0) {
                        return cur->key;
                }
                if (strcmp(cur->key, key) > 0) {
                        return NULL;
                }
        };
        return NULL;
}

/**
 * create an iterator and return key of first element
 *
 * @note
 * As the iterator state is stored internally, creating the
 * iterator (calling this fn) invalidates any previous.
 */
const char *
dictionary_first(struct dictionary *dictionary, const char **key)
{
        if (dictionary->items == NULL) {
                return NULL;
        }
        dictionary->iterator = dictionary->items;
        *key                 = dictionary->iterator->key;
        return dictionary->iterator->val;
}

/**
 * get next element
 *
 * this must be called after dictionary_first()
 *
 * When dictionary_next() returns nullptr, no more elements are
 * available and the internal iterator is invalidated.
 */
const char *
dictionary_next(struct dictionary *dictionary, const char **key)
{
        if (dictionary->iterator->next == NULL) {
                return NULL;
        }
        dictionary->iterator = dictionary->iterator->next;
        *key                 = dictionary->iterator->key;
        return dictionary->iterator->val;
}

void
dictionary_destroy(struct dictionary *dictionary)
{
        if (dictionary == NULL) {
                return;
        }
        assert(dictionary->magic == MAGIC);
        struct item *cur = dictionary->items;
        while (cur != NULL) {
                struct item *next = cur->next;
                free(cur->key);
                free(cur->val);
                free(cur);
                cur = next;
        };
        free(dictionary);
}
