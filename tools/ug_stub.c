/**
 * @file
 * replacement for UltraGrid functions/global objects that are not linked-in,
 * usually the ones located in host.cpp
 */

#include <stddef.h>

const char *
get_commandline_param(const char *key)
{
        (void) key;
        return NULL;
}

void
register_param(const char *param, const char *doc)
{
        (void) param;
        (void) doc;
}
