/**
 * @file   utils/list.cpp
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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

#include <list>

#include "utils/list.h"

using std::list;

struct simple_linked_list {
        list <void *> l;
};

struct sll_it {
        list <void *>::iterator end;
        list <void *>::iterator it;
};

struct simple_linked_list *simple_linked_list_init(void)
{
        return new simple_linked_list();
}

void simple_linked_list_destroy(struct simple_linked_list *l)
{
        delete l;
}

void simple_linked_list_prepend(struct simple_linked_list *l, void *data)
{
        l->l.push_front(data);
}

void simple_linked_list_append(struct simple_linked_list *l, void *data)
{
        l->l.push_back(data);
}

bool simple_linked_list_append_if_less(struct simple_linked_list *l, void *data, int max_size)
{
        if (l->l.size() >= (unsigned) max_size) {
                return false;
        }
        l->l.push_back(data);
        return true;
}

void *simple_linked_list_pop(struct simple_linked_list *l)
{
        if (simple_linked_list_size(l) == 0) {
                return NULL;
        }
        void *ret = l->l.front();
        l->l.pop_front();

        return ret;
}

int simple_linked_list_size(struct simple_linked_list *l)
{
        return l->l.size();
}

void *simple_linked_list_first(struct simple_linked_list *l)
{
        return l->l.front();
}

void *simple_linked_list_last(struct simple_linked_list *l)
{
        return l->l.back();
}

void *simple_linked_list_it_init(struct simple_linked_list *l)
{
        if (l->l.size() == 0)
                return NULL;
        auto ret = new sll_it();
        ret->it = l->l.begin();
        ret->end = l->l.end();
        return ret;
}

void *simple_linked_list_it_next(void **i)
{
        auto sit = (sll_it *) *i;

        void *ret = *sit->it;
        ++sit->it;
        if (sit->it == sit->end) {
                delete sit;
                *i = NULL;
        }
        return ret;
}

void *simple_linked_list_it_peek_next(const void *it)
{
        const auto *sit = static_cast<const sll_it *>(it);

        return *sit->it;
}

void simple_linked_list_it_destroy(void *i)
{
        delete (sll_it *) i;
}

bool simple_linked_list_remove(struct simple_linked_list *l, void *item)
{
        for (auto it = l->l.begin(); it != l->l.end(); ++it) {
                if (*it == item) {
                        l->l.erase(it);
                        return true;
                }
        }
        return false;
}

void *simple_linked_list_remove_index(struct simple_linked_list *l, int index)
{
        int i = 0;
        for (auto it = l->l.begin(); it != l->l.end(); ++it) {
                if (i++ == index) {
                        void *ret = *it;
                        l->l.erase(it);
                        return ret;
                }
        }
        return NULL;
}

