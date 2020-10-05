/**
 * @file   jack_common.h
 * @author Martin Piatka     <piatka@cesnet.cz>
 *         Martin Pulec      <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2020 CESNET, z. s. p. o.
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

#include <stdio.h>
#include <jack/jack.h>
#include "debug.h"
#include "types.h"

static inline struct device_info *audio_jack_probe(const char *client_name,
                                                   unsigned long port_flags,
                                                   int *count)
{
        jack_client_t *client;
        jack_status_t status;
        char *last_name = NULL;
        int i;
        int channel_count;
        const char **ports;
        int port_count = 0; 
        struct device_info *available_devices = NULL;

        *count = 0;
        client = jack_client_open(client_name, JackNullOption, &status);
        if(status & JackFailure) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Opening JACK client failed.\n");
                return NULL;
        }

        ports = jack_get_ports(client, NULL, NULL, port_flags);
        if(ports == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to enumerate ports.\n");
                return NULL;
        }

        for(port_count = 0; ports[port_count] != NULL; port_count++);

        available_devices = calloc(port_count, sizeof(struct device_info));

        channel_count = 0;
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
                        sprintf(available_devices[*count].id, "jack:%s", last_name);
                        channel_count = 0;
                        (*count)++;
                }
                free(last_name);
                last_name = strdup(name);
                free(item);
        }
        if(last_name) {
                sprintf(available_devices[*count].name, "jack:%s (%d channels)", last_name, channel_count);
                sprintf(available_devices[*count].id, "jack:%s", last_name);
                (*count)++;
        }
        free(last_name);
        jack_free(ports);
        jack_client_close(client);

        return available_devices;
}


#endif //JACK_COMMON_H

