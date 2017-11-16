/**
 * @file   video_capture_params.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @ingroup vidcap
 */
/*
 * Copyright (c) 2013-2017 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <string>

#include "utils/config_file.h"
#include "video_capture_params.h"

using namespace std;

/**
 * Defines parameters passed to video capture driver.
  */
struct vidcap_params {
        char  *driver; ///< driver name
        char  *fmt;    ///< driver options
        unsigned int flags;  ///< one of @ref vidcap_flags

        char *requested_capture_filter;
        char  *name;   ///< input name (capture alias in config file or complete config if not alias)
        struct vidcap_params *next; /**< Pointer to next vidcap params. Used by aggregate capture drivers.
                                     *   Last device in list has @ref driver set to NULL. */
        struct module *parent;
};

/**
 * This function does 2 things:
 * * checks whether @ref vidcap_params::name is not an alias
 * * tries to find capture filter for @ref vidcap_params::name if not given
 * @retval true  if alias dispatched successfully
 * @retval false otherwise
 */
static bool vidcap_dispatch_alias(struct vidcap_params *params)
{
        bool ret;
        char buf[1024];
        string real_capture;
        struct config_file *conf =
                config_file_open(default_config_file(buf,
                                        sizeof(buf)));
        if (conf == NULL)
                return false;
        real_capture = config_file_get_alias(conf, "capture", params->name);
        if (real_capture.empty()) {
                ret = false;
        } else {
                params->driver = strdup(real_capture.c_str());
                if (strchr(params->driver, ':')) {
                        char *delim = strchr(params->driver, ':');
                        params->fmt = strdup(delim + 1);
                        *delim = '\0';
                } else {
                        params->fmt = strdup("");
                }
                ret = true;
        }


        if (params->requested_capture_filter == NULL) {
                string matched_cap_filter = config_file_get_capture_filter_for_alias(conf,
                                        params->name);
                if (!matched_cap_filter.empty())
                        params->requested_capture_filter = strdup(matched_cap_filter.c_str());
        }

        config_file_close(conf);

        return ret;
}


/**
 * @brier Allocates blank @ref vidcap_params structure.
 */
struct vidcap_params *vidcap_params_allocate(void)
{
        return (struct vidcap_params *) calloc(1, sizeof(struct vidcap_params));
}

/**
 * @brier Allocates blank @ref vidcap_params structure.
 *
 * Follows curr struct in the virtual list.
 * @param curr structure to be appended after
 * @returns pointer to newly created structure
 */
struct vidcap_params *vidcap_params_allocate_next(struct vidcap_params *curr)
{
        curr->next = vidcap_params_allocate();
        return curr->next;
}

/**
 * @brier Returns next item in virtual @ref vidcap_params list.
 */
struct vidcap_params *vidcap_params_get_next(const struct vidcap_params *curr)
{
        return curr->next;
}

/**
 * @brier Returns n-th item in @ref vidcap_params list.
 */
struct vidcap_params *vidcap_params_get_nth(struct vidcap_params *curr, int index)
{
        struct vidcap_params *ret = curr;
        for (int i = 0; i < index; i++) {
                ret = ret->next;
                if (ret == NULL) {
                        return NULL;
                }
        }
        if (ret->driver == NULL) {
                // this is just the stopper of the list...
                return NULL;
        }
        return ret;
}

/**
 * Fills the structure with device config string in format either driver[:params] or
 * alias.
 */
void vidcap_params_set_device(struct vidcap_params *params, const char *config)
{
        free(params->name);
        free(params->driver);
        free(params->fmt);

        params->name = strdup(config);

        if (!vidcap_dispatch_alias(params)) {
                params->driver = strdup(config);
                if (strchr(params->driver, ':')) {
                        char *delim = strchr(params->driver, ':');
                        *delim = '\0';
                        params->fmt = strdup(delim + 1);
                } else {
                        params->fmt = strdup("");
                }
        }
}

void vidcap_params_set_capture_filter(struct vidcap_params *params,
                const char *req_capture_filter)
{
        params->requested_capture_filter = strdup(req_capture_filter);
}

void vidcap_params_set_flags(struct vidcap_params *params, unsigned int flags)
{
        params->flags = flags;
}

const char *vidcap_params_get_driver(const struct vidcap_params *params)
{
        return params->driver;
}

const char *vidcap_params_get_fmt(const struct vidcap_params *params)
{
        return params->fmt;
}

unsigned int vidcap_params_get_flags(const struct vidcap_params *params)
{
        return params->flags;
}

const char *vidcap_params_get_name(const struct vidcap_params *params)
{
        return params->name;
}

struct module *vidcap_params_get_parent(const struct vidcap_params *params)
{
        return params->parent;
}

/**
 * Creates deep copy of @ref vidcap_params structure.
 */
struct vidcap_params *vidcap_params_copy(const struct vidcap_params *params)
{
        if (!params)
                return NULL;

        struct vidcap_params *ret = (struct vidcap_params *) calloc(1, sizeof(struct vidcap_params));

        if (params->driver)
                ret->driver = strdup(params->driver);
        if (params->fmt)
                ret->fmt = strdup(params->fmt);
        if (params->requested_capture_filter)
                ret->requested_capture_filter =
                        strdup(params->requested_capture_filter);
        if (params->name)
                ret->name = strdup(params->name);
        if (params->flags)
                ret->flags = params->flags;

        ret->next = NULL; // there is high probability that the pointer will be invalid

        return ret;
}

/**
 * Frees all members of the given structure as well as its members.
 *
 * @param[in] buf structure to be feed
 */
void vidcap_params_free_struct(struct vidcap_params *buf)
{
        if (!buf)
                return;

        free(buf->driver);
        free(buf->fmt);
        free(buf->requested_capture_filter);
        free(buf->name);

        free(buf);
}

