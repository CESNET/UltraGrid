/*
 * FILE:   pdb.c - Participant database
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *         Orion Hodson
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c)      2004 University of Glasgow
 * Copyright (c) 2002-2003 University of Southern California
 * Copyright (c) 1999-2000 University College London
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 *
 * Largely based on common/src/btree.c revision 1.7 from the UCL 
 * Robust-Audio Tool v4.2.25. Code is based on the algorithm in:
 *  
 *   Introduction to Algorithms by Corman, Leisserson, and Rivest,
 *   MIT Press / McGraw Hill, 1990.
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
 * $Revision: 1.4 $
 * $Date: 2009/12/11 15:29:39 $
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/rtp.h"            /* Needed by pbuf.h */
#include "rtp/pbuf.h"
#include "tfrc.h"
#include "pdb.h"

#define PDB_MAGIC	0x10101010
#define PDB_NODE_MAGIC	0x01010101

typedef struct s_pdb_node {
        uint32_t key;
        void *data;
        struct s_pdb_node *parent;
        struct s_pdb_node *left;
        struct s_pdb_node *right;
        uint32_t magic;
} pdb_node_t;

struct pdb {
        pdb_node_t *root;
        uint32_t magic;
        int count;
};

/*****************************************************************************/
/* Debugging functions...                                                    */
/*****************************************************************************/

#define BTREE_MAGIC      0x10101010
#define BTREE_NODE_MAGIC 0x01010101

#ifdef DEBUG
static
#ifndef HAVE_MACOSX
__thread
#endif
int pdb_count;

static void pdb_validate_node(pdb_node_t * node, pdb_node_t * parent)
{
        assert(node->magic == BTREE_NODE_MAGIC);
        assert(node->parent == parent);
        pdb_count++;
        if (node->left != NULL) {
                pdb_validate_node(node->left, node);
        }
        if (node->right != NULL) {
                pdb_validate_node(node->right, node);
        }
}
#endif

static void pdb_validate(struct pdb *t)
{
        assert(t->magic == BTREE_MAGIC);
#ifdef DEBUG
        pdb_count = 0;
        if (t->root != NULL) {
                pdb_validate_node(t->root, NULL);
        }
        assert(pdb_count == t->count);
#endif
}

/*****************************************************************************/
/* Utility functions                                                         */
/*****************************************************************************/

static pdb_node_t *pdb_min(pdb_node_t * x)
{
        if (x == NULL) {
                return NULL;
        }
        while (x->left) {
                x = x->left;
        }
        return x;
}

static pdb_node_t *pdb_successor(pdb_node_t * x)
{
        pdb_node_t *y;

        if (x->right != NULL) {
                return pdb_min(x->right);
        }

        y = x->parent;
        while (y != NULL && x == y->right) {
                x = y;
                y = y->parent;
        }

        return y;
}

static pdb_node_t *pdb_search(pdb_node_t * x, uint32_t key)
{
        while (x != NULL && key != x->key) {
                if (key < x->key) {
                        x = x->left;
                } else {
                        x = x->right;
                }
        }
        return x;
}

static void pdb_insert_node(struct pdb *tree, pdb_node_t * z)
{
        pdb_node_t *x, *y;

        pdb_validate(tree);
        y = NULL;
        x = tree->root;
        while (x != NULL) {
                y = x;
                assert(z->key != x->key);
                if (z->key < x->key) {
                        x = x->left;
                } else {
                        x = x->right;
                }
        }

        z->parent = y;
        if (y == NULL) {
                tree->root = z;
        } else if (z->key < y->key) {
                y->left = z;
        } else {
                y->right = z;
        }
        tree->count++;
        pdb_validate(tree);
}

static pdb_node_t *pdb_delete_node(struct pdb *tree, pdb_node_t * z)
{
        pdb_node_t *x, *y;

        pdb_validate(tree);
        if (z->left == NULL || z->right == NULL) {
                y = z;
        } else {
                y = pdb_successor(z);
        }

        if (y->left != NULL) {
                x = y->left;
        } else {
                x = y->right;
        }

        if (x != NULL) {
                x->parent = y->parent;
        }

        if (y->parent == NULL) {
                tree->root = x;
        } else if (y == y->parent->left) {
                y->parent->left = x;
        } else {
                y->parent->right = x;
        }

        z->key = y->key;
        z->data = y->data;

        tree->count--;

        pdb_validate(tree);
        return y;
}

/*****************************************************************************/

struct pdb *pdb_init(void)
{
        struct pdb *db = malloc(sizeof(struct pdb));
        if (db != NULL) {
                db->magic = PDB_MAGIC;
                db->count = 0;
                db->root = NULL;
        }
        return db;
}

void pdb_destroy(struct pdb **db_p)
{
        struct pdb *db = *db_p;

        pdb_validate(db);
        if (db->root != NULL) {
                printf
                    ("WARNING: participant database not empty - cannot destroy\n");
                // TODO: participants should be removed using pdb_remove() 
        }

        free(db);
        *db_p = NULL;
}

static struct pdb_e *pdb_create_item(uint32_t ssrc)
{
        struct pdb_e *p = malloc(sizeof(struct pdb_e));
        if (p != NULL) {
                gettimeofday(&(p->creation_time), NULL);
                p->ssrc = ssrc;
                p->sdes_cname = NULL;
                p->sdes_name = NULL;
                p->sdes_email = NULL;
                p->sdes_phone = NULL;
                p->sdes_loc = NULL;
                p->sdes_tool = NULL;
                p->sdes_note = NULL;
                p->video_decoder_state = NULL;
                p->pt = 255;
                p->playout_buffer = pbuf_init();
                p->tfrc_state = tfrc_init(p->creation_time);
        }
        return p;
}

int pdb_add(struct pdb *db, uint32_t ssrc)
{
        /* Add an item to the participant database, indexed by ssrc. */
        /* Returns 0 on success, 1 if the participant is already in  */
        /* the database, 2 for other failures.                       */
        pdb_node_t *x;
        struct pdb_e *i;

        pdb_validate(db);
        x = pdb_search(db->root, ssrc);
        if (x != NULL) {
                debug_msg("Item already exists - ssrc %x\n", ssrc);
                return 1;
        }

        i = pdb_create_item(ssrc);
        if (i == NULL) {
                debug_msg("Unable to create database entry - ssrc %x\n", ssrc);
                return 2;
        }

        x = (pdb_node_t *) malloc(sizeof(pdb_node_t));
        x->key = ssrc;
        x->data = i;
        x->parent = NULL;
        x->left = NULL;
        x->right = NULL;
        x->magic = BTREE_NODE_MAGIC;
        pdb_insert_node(db, x);
        debug_msg("Added participant %x\n", ssrc);
        return 0;
}

struct pdb_e *pdb_get(struct pdb *db, uint32_t ssrc)
{
        /* Return a pointer to the item indexed by ssrc, or NULL if   */
        /* the item is not present in the database.                   */
        pdb_node_t *x;

        pdb_validate(db);
        x = pdb_search(db->root, ssrc);
        if (x != NULL) {
                return x->data;
        }
        return NULL;
}

int pdb_remove(struct pdb *db, uint32_t ssrc, struct pdb_e **item)
{
        /* Remove the item indexed by ssrc. Return zero on success.   */
        pdb_node_t *x;

        pdb_validate(db);
        x = pdb_search(db->root, ssrc);
        if (x == NULL) {
                debug_msg("Item not on tree - ssrc %ul\n", ssrc);
                *item = NULL;
                return 1;
        }

        /* Note value that gets freed is not necessarily the the same as node
         * that gets removed from tree since there is an optimization to avoid
         * pointer updates in tree which means sometimes we just copy key and
         * data from one node to another.  
         */
        *item = x->data;
        x = pdb_delete_node(db, x);
        free(x);
        return 0;
}

/* 
 * Iterator functions 
 */

struct pdb_e *pdb_iter_init(struct pdb *db, pdb_iter_t *it)
{
        if (db->root == NULL) {
                return NULL;    /* The database is empty */
        }
        *it = (pdb_node_t *) pdb_min(db->root);
        return ((pdb_node_t *) *it)->data;
}

struct pdb_e *pdb_iter_next(pdb_iter_t *it)
{
        assert(*it != NULL);
        *it = (pdb_node_t *)pdb_successor((pdb_node_t *)*it);
        if (*it == NULL) {
                return NULL;
        }
        return ((pdb_node_t *) *it)->data;
}

void pdb_iter_done(pdb_iter_t *it)
{
        *it = NULL;
}

