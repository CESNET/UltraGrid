#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "utils/list.h"

struct node;

struct node {
        void *val;
        struct node *next;
};

struct simple_linked_list {
        struct node *head;
        struct node *tail;
        int size;
};

struct simple_linked_list *simple_linked_list_init(void)
{
        return calloc(1, sizeof(struct simple_linked_list));
}

void simple_linked_list_destroy(struct simple_linked_list *l)
{
        struct node *n = l->head;
        while(n != NULL) {
                struct node *tmp = n;
                n = n->next;
                free(tmp);
        }
        free(l);
}

void simple_linked_list_append(struct simple_linked_list *l, void *data)
{
        struct node *new_node = calloc(1, sizeof(struct node));
        new_node->val = data;
        if(l->tail) {
                l->tail->next = new_node;
                l->tail = l->tail->next;
        } else {
                l->head = l->tail = new_node;
        }
        l->size += 1;
}

void *simple_linked_list_pop(struct simple_linked_list *l)
{
        assert(l->head != NULL);

        struct node *n = l->head;
        void *ret;
        l->head = l->head->next;

        if(!l->head)
                l->tail = NULL;

        l->size--;

        ret = n->val;
        free(n);

        return ret;
}

int simple_linked_list_size(struct simple_linked_list *l)
{
        return l->size;
}

void *simple_linked_list_it_init(struct simple_linked_list *l)
{
        return l->head;
}

void *simple_linked_list_it_next(void **it)
{
        struct node *n = *it;
        assert(n);
        *it = n->next;
        return n->val;
}

