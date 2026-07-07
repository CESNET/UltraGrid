/**
 * @file   utils/list.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2026 CESNET, zájmové sdružení právnických osob
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

#include "utils/list.h"

#include <assert.h> // for assert
#include <stdint.h> // for uint32_t
#include <stdlib.h> // for free, calloc, abort, size_t

#include "compat/c23.h"   // IWYU pragma: keep
#include "utils/macros.h" // for to_fourcc

#define MAGIC to_fourcc('U', 'T', 'l', 'i')

typedef struct simple_linked_list_item {
        void                           *data;
        struct simple_linked_list_item *next;
} list_item;

struct simple_linked_list {
        uint32_t   magic;
        size_t     count;
        list_item *first;
        list_item *last;
};

struct simple_linked_list *
simple_linked_list_init(void)
{
        struct simple_linked_list *ret = calloc(1, sizeof *ret);
        ret->magic                     = MAGIC;
        return ret;
}

void
simple_linked_list_destroy(struct simple_linked_list *l)
{
        if (l == nullptr) {
                return;
        }
        assert(l->magic == MAGIC);

        list_item *item = l->first;
        while (item != nullptr) {
                list_item *tmp = item;
                item           = item->next;
                free(tmp);
        }
        free(l);
}

void
simple_linked_list_prepend(struct simple_linked_list *l, void *data)
{
        list_item *new_item = calloc(1, sizeof *new_item);
        new_item->data      = data;
        new_item->next      = l->first;
        l->first            = new_item;

        l->count += 1;
        if (l->count == 1) {
                l->first = l->last = new_item;
        }
}

void
simple_linked_list_append(struct simple_linked_list *l, void *data)
{
        list_item *new_item = calloc(1, sizeof *new_item);
        new_item->data      = data;

        if (l->last != nullptr) {
                l->last->next = new_item;
        }
        l->last = new_item;

        l->count += 1;
        if (l->count == 1) {
                l->first = l->last = new_item;
        }
}

bool
simple_linked_list_append_if_less(struct simple_linked_list *l, void *data,
                                  int max_size)
{
        if (l->count >= (unsigned) max_size) {
                return false;
        }
        simple_linked_list_append(l, data);
        return true;
}

void *
simple_linked_list_pop(struct simple_linked_list *l)
{
        if (l->count == 0) {
                return nullptr;
        }
        list_item *ret_item = l->first;
        l->first            = ret_item->next;

        void *ret = ret_item->data;
        free(ret_item);

        l->count -= 1;
        if (l->count == 0) {
                l->first = l->last = nullptr;
        }

        return ret;
}

int
simple_linked_list_size(struct simple_linked_list *l)
{
        return l->count;
}

void *
simple_linked_list_first(struct simple_linked_list *l)
{
        if (l->count == 0) {
                return nullptr;
        }
        return l->first->data;
}

void *
simple_linked_list_last(struct simple_linked_list *l)
{
        return l->last->data;
}

list_it
simple_linked_list_it_init(struct simple_linked_list *l)
{
        if (l->count == 0) {
                return nullptr;
        }
        return l->first;
}

void *
simple_linked_list_it_next(list_it *it)
{
        list_item *item = *it;

        void *ret = item->data;

        item = item->next;
        *it = item;
        return ret;
}

void *
simple_linked_list_it_peek_next(list_it const *it)
{
        list_item *item = *it;
        return item->data;
}

bool
simple_linked_list_remove(struct simple_linked_list *l, void *item)
{
        list_item  *it        = l->first;
        list_item **prev_next = &l->first;

        while (it != nullptr) {
                if (it->data == item) {
                        (*prev_next) = it->next;
                        free(it);

                        l->count -= 1;
                        if (l->count == 0) {
                                l->first = l->last = nullptr;
                        }

                        return true;
                }
                prev_next = &it->next;
                it        = it->next;
        }

        return false;
}

void *
simple_linked_list_remove_index(struct simple_linked_list *l, int index)
{
        if (index >= (int) l->count) {
                return nullptr;
        }

        list_item  *it        = l->first;
        list_item **prev_next = &l->first;
        int         i         = 0;

        while (it != nullptr) {
                if (i == index) {
                        (*prev_next) = it->next;
                        void *ret    = it->data;
                        free(it);

                        l->count -= 1;
                        if (l->count == 0) {
                                l->first = l->last = nullptr;
                        }

                        return ret;
                }
                prev_next = &it->next;
                it        = it->next;
                i++;
        }
        abort(); // handled by the cond at the beginning
}
