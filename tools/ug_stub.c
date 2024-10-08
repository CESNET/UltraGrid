/**
 * @file
 * replacement for UltraGrid functions/global objects that are not linked-in,
 * usually the ones located in host.cpp
 */

#include <stdbool.h>
#include <stddef.h>

char *uv_argv[] = { "ug_stub", NULL };

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

bool
tok_in_argv(char **argv, const char *tok)
{
        (void) argv, (void) tok;
        return false;
}
