/**
 * @file   utils/parallel_conv.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#include "utils/parallel_conv.h"
#include "utils/worker.h"

struct parallel_pix_conv_data {
        decoder_t decode;
        int height;
        unsigned char *out_data;
        int out_linesize;
        const unsigned char *in_data;
        int in_linesize;
};

static void *parallel_pix_conv_task(void *arg) {
        struct parallel_pix_conv_data *data = arg;
        for (int y = 0; y < data->height; ++y) {
                data->decode(data->out_data, data->in_data, data->out_linesize, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                data->out_data += data->out_linesize;
                data->in_data += data->in_linesize;
        }
        return NULL;
}

void parallel_pix_conv(int height, char *out, int out_linesize, const char *in, int in_linesize, decoder_t decode, int threads)
{
        struct parallel_pix_conv_data data[threads];

        for (int i = 0; i < threads; ++i) {
                data[i].decode = decode;
                data[i].height = height / threads;
                data[i].out_data = (unsigned char *) out + i * data[i].height * out_linesize;
                data[i].out_linesize = out_linesize;
                data[i].in_data = (const unsigned char *) in + i * data[i].height * in_linesize;
                data[i].in_linesize = in_linesize;
                if (i == threads - 1) {
                        data[i].height = height - (threads - 1) * (height / threads);
                }
        }

        task_run_parallel(parallel_pix_conv_task, threads, data, sizeof data[0], NULL);
}

