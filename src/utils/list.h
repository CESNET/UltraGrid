#ifndef SIMPLE_LINKED_LIST_H_
#define SIMPLE_LINKED_LIST_H_

#ifdef __cplusplus
extern "C" {
#endif

struct simple_linked_list;
struct simple_linked_list *simple_linked_list_init(void);
void simple_linked_list_destroy(struct simple_linked_list *);
void simple_linked_list_append(struct simple_linked_list *, void *data);
void *simple_linked_list_pop(struct simple_linked_list *);
int simple_linked_list_size(struct simple_linked_list *);


#ifdef __cplusplus
}
#endif

#endif// SIMPLE_LINKED_LIST_H_
