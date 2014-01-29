/**
 * @file   rtp/pc.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2013 CESNET z.s.p.o.
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

#include "pc.h"

#include <stdlib.h>

using namespace std;

struct node {
        struct node *next;
        int val;
        int len;
};

struct packet_counter {
        struct node *head;
};

/**
 * @brief Creates packet counter.
 */
struct packet_counter  *pc_create()
{
        return (struct packet_counter *) calloc(1, sizeof(struct packet_counter));
}

/**
 * @brief Adds packet to the counter.
 * @param pc  packet counter instance
 * @param val packet offset in buffer
 * @param len packet length
 */
void pc_insert(struct packet_counter *pc, int val, int len) {
        struct node *cur;
        struct node **ref;
        if(!pc->head) {
                pc->head = (struct node *) malloc(sizeof(struct node));
                pc->head->val = val;
                pc->head->len = len;
                pc->head->next = NULL;
                return;
        }
        ref = &pc->head;
        cur = pc->head;
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

/**
 * @brief Destroys packet counter created with pc_create()
 */
void pc_destroy(struct packet_counter *pc) {
        struct node *cur = pc->head;
        struct node *tmp;

        while (cur != NULL) {
                tmp = cur->next;
                free(cur);
                cur = tmp;
        }
        free(pc);
}

/**
 * @brief Retuns total number of packets in buffere.
 * @param pc state
 * @return total number of packets in buffer
 */
unsigned int pc_count (struct packet_counter *pc) {
        unsigned int ret = 0u;
        struct node *cur = pc->head;
        while(cur != NULL) {
                ++ret;
                cur = cur->next;
        }
        return ret;
}

/**
 * @brief Retuns total of bytes in buffer.
 * @param pc state
 * @return total number of bytes in buffer
 */
unsigned int pc_count_bytes (struct packet_counter *pc) {
        unsigned int ret = 0u;
        struct node *cur = pc->head;
        while(cur != NULL) {
                ret += cur->len;
                cur = cur->next;
        }
        return ret;
}

/**
 * @brief Converts packet counter to map representation
 * @param pc state
 * @return resulting map representation
 */
std::map<int, int> pc_to_map(struct packet_counter *pc)
{
        map<int, int> res;

        struct node *cur = pc->head;
        while(cur != NULL) {
                res.insert(pair<int, int>(cur->val, cur->len));
                cur = cur->next;
        }

        return res;
}

