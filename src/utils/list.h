/**
 * @file   utils/list.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2016 CESNET, z. s. p. o.
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

#ifndef SIMPLE_LINKED_LIST_H_
#define SIMPLE_LINKED_LIST_H_

#ifdef __cplusplus
extern "C" {
#endif

struct simple_linked_list;
struct simple_linked_list *simple_linked_list_init(void);
void simple_linked_list_destroy(struct simple_linked_list *);
void simple_linked_list_prepend(struct simple_linked_list *, void *data);
void simple_linked_list_append(struct simple_linked_list *, void *data);
bool simple_linked_list_append_if_less(struct simple_linked_list *, void *data, int max_size);
void *simple_linked_list_pop(struct simple_linked_list *);
int simple_linked_list_size(struct simple_linked_list *);
void *simple_linked_list_first(struct simple_linked_list *); ///< returns first element of list, UB if empty
void *simple_linked_list_last(struct simple_linked_list *); ///< returns last element of list, UB if empty

/** iterator
 *
 * usage:
 * for(void *it = simple_linked_list_it_init(list); it != NULL; ) {
 *          o-type *inst = simple_linked_list_it_next(&it);
 *          process(inst);
 *          if (something_why_to_leave_the_loop) {
 *                  simple_linked_list_it_destroy(it);
 *                  break;
 *          }
 * }
 */
void *simple_linked_list_it_init(struct simple_linked_list *);
void *simple_linked_list_it_next(void **it);
void *simple_linked_list_it_peek_next(const void *it); ///< return next item without actually incrementing iterator, UB if item is last (it == NULL)
                                                       ///< @returns item
void simple_linked_list_it_destroy(void *it); ///< Should be used when it != NULL, eg. when leaving the loop before the end

/**
 * @retval TRUE if removed
 * @retval FALSE if not found
 */
int simple_linked_list_remove(struct simple_linked_list *, void *);

/**
 * @retval pointer pointer to removed value
 * @retval NULL if not found
 */
void *simple_linked_list_remove_index(struct simple_linked_list *, int index);

#ifdef __cplusplus
}
#endif

#endif// SIMPLE_LINKED_LIST_H_
