/**
 * @file   jack_common.h
 * @author Martin Piatka     <piatka@cesnet.cz>
 *         Martin Pulec      <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2021 CESNET, z. s. p. o.
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

#ifndef JACK_COMMON_H
#define JACK_COMMON_H

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <jack/jack.h>
#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "lib_common.h"

#include "types.h"

typedef int (*jack_activate_t)(jack_client_t *client) JACK_OPTIONAL_WEAK_EXPORT;
typedef int (*jack_client_close_t)(jack_client_t *client) JACK_OPTIONAL_WEAK_EXPORT;
typedef jack_client_t *(*jack_client_open_t)(const char *client_name,
                jack_options_t options,
                jack_status_t *status, ...) JACK_OPTIONAL_WEAK_EXPORT;
typedef int (*jack_connect_t)(jack_client_t *client,
                const char *source_port,
                const char *destination_port) JACK_OPTIONAL_WEAK_EXPORT;
typedef int (*jack_deactivate_t)(jack_client_t *client) JACK_OPTIONAL_WEAK_EXPORT;
typedef int (*jack_disconnect_t)(jack_client_t *client,
                     const char *source_port,
                     const char *destination_port) JACK_OPTIONAL_WEAK_EXPORT;
typedef void (*jack_free_t)(void* ptr) JACK_OPTIONAL_WEAK_EXPORT;
typedef const char **(*jack_get_ports_t)(jack_client_t *client,
                const char *port_name_pattern,
                const char *type_name_pattern,
                unsigned long flags) JACK_OPTIONAL_WEAK_EXPORT;
typedef jack_nframes_t (*jack_get_sample_rate_t)(jack_client_t *) JACK_OPTIONAL_WEAK_EXPORT;
typedef void * (*jack_port_get_buffer_t)(jack_port_t *port, jack_nframes_t) JACK_OPTIONAL_WEAK_EXPORT;
typedef const char * (*jack_port_name_t)(const jack_port_t *port) JACK_OPTIONAL_WEAK_EXPORT;
typedef jack_port_t * (*jack_port_register_t)(jack_client_t *client,
                const char *port_name,
                const char *port_type,
                unsigned long flags,
                unsigned long buffer_size) JACK_OPTIONAL_WEAK_EXPORT;
typedef int (*jack_set_process_callback_t)(jack_client_t *client,
                JackProcessCallback process_callback,
                void *arg) JACK_OPTIONAL_WEAK_EXPORT;
typedef int (*jack_set_sample_rate_callback_t)(jack_client_t *client,
                JackSampleRateCallback srate_callback,
                void *arg) JACK_OPTIONAL_WEAK_EXPORT;

struct libjack_connection {
        LIB_HANDLE  libjack; ///< lib connection

        jack_activate_t                 activate;
        jack_client_close_t             client_close;
        jack_client_open_t              client_open;
        jack_connect_t                  connect;
        jack_deactivate_t               deactivate;
        jack_disconnect_t               disconnect;
        jack_free_t                     free;
        jack_get_ports_t                get_ports;
        jack_get_sample_rate_t          get_sample_rate;
        jack_port_get_buffer_t          port_get_buffer;
        jack_port_name_t                port_name;
        jack_port_register_t            port_register;
        jack_set_process_callback_t     set_process_callback;
        jack_set_sample_rate_callback_t set_sample_rate_callback;
};

static void close_libjack(struct libjack_connection *s)
{
        if (s == NULL) {
                return;
        }
        dlclose(s->libjack);
        free(s);
}

#define JACK_DLSYM(sym) s->sym = (void *) dlsym(s->libjack, "jack_" #sym); if (s->sym == NULL) { log_msg(LOG_LEVEL_ERROR, "JACK symbol %s not found: %s\n", "jack_" #sym, dlerror()); close_libjack(s); return NULL; }

static struct libjack_connection *open_libjack(void)
{
        struct libjack_connection *s = calloc(1, sizeof(struct libjack_connection));
        const char *shlib =
#ifdef _WIN32
                "C:/Windows/libjack64.dll";
#elif defined (__APPLE__)
                "libjack.dylib";
#elif defined (__linux__)
                "libjack.so";
#else
                "";
#endif
        s->libjack = dlopen(shlib, RTLD_NOW);
        if (s->libjack == NULL) {
                log_msg(LOG_LEVEL_ERROR, "JACK library \"%s\" opening failed: %s\n", shlib, dlerror());
                free(s);
                return NULL;
        }
        JACK_DLSYM(activate);
        JACK_DLSYM(client_close)
        JACK_DLSYM(client_open)
        JACK_DLSYM(connect);
        JACK_DLSYM(deactivate);
        JACK_DLSYM(disconnect);
        JACK_DLSYM(free)
        JACK_DLSYM(get_ports)
        JACK_DLSYM(get_sample_rate);
        JACK_DLSYM(set_sample_rate_callback);
        JACK_DLSYM(port_get_buffer);
        JACK_DLSYM(port_name);
        JACK_DLSYM(port_register);
        JACK_DLSYM(set_process_callback);
        return s;
}

static inline struct device_info *audio_jack_probe(const char *client_name,
                                                   unsigned long port_flags,
                                                   int *count)
{
        jack_client_t *client;
        jack_status_t status;
        char *last_name = NULL;
        int i;
        const char **ports;
        int port_count = 0; 
        struct libjack_connection *libjack = open_libjack();
        if (!libjack) {
                return NULL;
        }

        *count = 0;
        client = libjack->client_open(client_name, JackNullOption, &status);
        if(status & JackFailure) {
                log_msg(LOG_LEVEL_ERROR, "Opening JACK client failed.\n");
                close_libjack(libjack);
                return NULL;
        }

        ports = libjack->get_ports(client, NULL, NULL, port_flags);
        if(ports == NULL) {
                log_msg(LOG_LEVEL_ERROR, "Unable to enumerate JACK ports.\n");
                close_libjack(libjack);
                return NULL;
        }

        for(port_count = 0; ports[port_count] != NULL; port_count++);

        struct device_info *available_devices = port_count > 0 ? calloc(port_count, sizeof(struct device_info)) : NULL;

        int channel_count = 0;
        for(i = 0; ports[i] != NULL; i++) {
                char *item = strdup(ports[i]);
                assert(item != NULL);
                char *save_ptr = NULL;
                char *name;

                ++channel_count;
                name = strtok_r(item, "_", &save_ptr);
                if (name == NULL) { // shouldn't happen
                        log_msg(LOG_LEVEL_ERROR, "Incorrect JACK name: %s!\n", ports[i]);
                        free(item);
                        continue;
                }
                if(last_name && strcmp(last_name, name) != 0) {
                        sprintf(available_devices[*count].name, "jack:%s (%d channels)", last_name, channel_count);
                        sprintf(available_devices[*count].dev, ":\"%s\"", last_name);
                        channel_count = 0;
                        (*count)++;
                }
                free(last_name);
                last_name = strdup(name);
                free(item);
        }
        if(last_name) {
                sprintf(available_devices[*count].name, "jack:%s (%d channels)", last_name, channel_count);
                sprintf(available_devices[*count].dev, ":\"%s\"", last_name);
                (*count)++;
        }
        free(last_name);
        libjack->free(ports);
        libjack->client_close(client);
        close_libjack(libjack);

        return available_devices;
}


#endif //JACK_COMMON_H

