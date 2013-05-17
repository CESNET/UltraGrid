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

/** iterator
 *
 * usage:
 * for(void *it = simple_linked_list_it_init(list); it != NULL; ) {
 *          o-type *inst = simple_linked_list_it_next(&it);
 *          process(inst);
 * }
 */
void *simple_linked_list_it_init(struct simple_linked_list *);
void *simple_linked_list_it_next(void **it);

#ifdef __cplusplus
}
#endif

#endif// SIMPLE_LINKED_LIST_H_
