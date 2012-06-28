/*
 * AUTHOR:   Ladan Gharai/Colin Perkins
 * 
 * Copyright (c) 2003-2004 University of Southern California
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
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 */

#include "ll.h"

#include <stdlib.h>

using namespace std;

struct linked_list  *ll_create()
{
        return (struct linked_list *) calloc(1, sizeof(struct linked_list));
}

void ll_insert(struct linked_list *ll, int val, int len) {
        struct node *cur;
        struct node **ref;
        if(!ll->head) {
                ll->head = (struct node *) malloc(sizeof(struct node));
                ll->head->val = val;
                ll->head->len = len;
                ll->head->next = NULL;
                return;
        }
        ref = &ll->head;
        cur = ll->head;
        while (cur != NULL) {
                if (val == cur->val) return;
                if (val < cur->val) {
                        struct node *new_node = (struct node *) malloc(sizeof(struct node));
                        (*ref) = new_node; 
                        new_node->val = val;
                        new_node->len = len;
                        new_node->next = cur;
                        return;
                }
                ref = &cur->next;
                cur = cur->next;
        }
        struct node *new_node = (struct node *) malloc(sizeof(struct node));
        (*ref) = new_node; 
        new_node->val = val;
        new_node->len = len;
        new_node->next = NULL;
}

void ll_destroy(struct linked_list *ll) {
        struct node *cur = ll->head;
        struct node *tmp;

        while (cur != NULL) {
                tmp = cur->next;
                free(cur);
                cur = tmp;
        }
        free(ll);
}

unsigned int ll_count (struct linked_list *ll) {
        unsigned int ret = 0u;
        struct node *cur = ll->head;
        while(cur != NULL) {
                ++ret;
                cur = cur->next;
        }
        return ret;
}

unsigned int ll_count_bytes (struct linked_list *ll) {
        unsigned int ret = 0u;
        struct node *cur = ll->head;
        while(cur != NULL) {
                ret += cur->len;
                cur = cur->next;
        }
        return ret;
}

std::map<int, int> ll_to_map(struct linked_list *ll)
{
        map<int, int> res;

        struct node *cur = ll->head;
        while(cur != NULL) {
                res.insert(pair<int, int>(cur->val, cur->len));
                cur = cur->next;
        }

        return res;
}

