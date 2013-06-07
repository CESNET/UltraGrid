#ifndef STATS_H_
#define STATS_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct control_state;
struct stats;

struct stats *stats_new_statistics(struct control_state *control, char * name);
void stats_update_int(struct stats *, int64_t);
void stats_format(struct stats *s, char *buffer, int buffer_len);
void stats_destroy(struct stats *);

#ifdef __cplusplus
}
#endif

#endif// STATS_H_

