#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "utils/list.h"

#include <list>

using namespace std;

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

void simple_linked_list_append(struct simple_linked_list *l, void *data)
{
        l->l.push_back(data);
}

void *simple_linked_list_pop(struct simple_linked_list *l)
{
        void *ret = l->l.front();
        l->l.pop_front();

        return ret;
}

int simple_linked_list_size(struct simple_linked_list *l)
{
        return l->l.size();
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

void simple_linked_list_it_destroy(void *i)
{
        delete (sll_it *) i;
}

int simple_linked_list_remove(struct simple_linked_list *l, void *item)
{
        for (auto it = l->l.begin(); it != l->l.end(); ++it) {
                if (*it == item) {
                        l->l.erase(it);
                        return TRUE;
                }
        }
        return FALSE;
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

