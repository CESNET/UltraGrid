/*
 * FILE:    video_compress.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#include "host.h"
#include "config.h"
#include "debug.h"
#include "fastdxt.h"
#include <pthread.h>
#include "compat/platform_semaphore.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <assert.h>
#ifdef HAVE_MACOSX
#include <malloc/malloc.h>
#else                           /* HAVE_MACOSX */
#include <malloc.h>
#endif                          /* HAVE_MACOSX */
#include <string.h>
#include <unistd.h>
#include "video_compress.h"
#include "libdxt.h"

extern int should_exit;
volatile static int fastdxt_should_exit = FALSE;

#ifndef HAVE_MACOSX
#define uint64_t 	unsigned long
#endif                          /* HAVE_MACOSX */

/* NOTE: These threads busy wait, so at *most* set this to one less than the
 * total number of cores on your system (Also 3 threads will work)! Also, if
 * you see tearing in the rendered image try increasing the number of threads
 * by 1 (For a dual dual-core Opteron 285 3 threads work great). 
 * -- iwsmith <iwsmith@cct.lsu.ed> 9 August 2007
 */
#define MAX_THREADS 16
#define NUM_THREADS_DEFAULT 3

/* Ok, we are going to decompose the problem into 2^n pieces (generally
 * 2, 4, or 8). We will most likely need to do the following:
 * 1. Convert 10-bit -> 8bit
 * 2. Convert YUV 4:2:2 -> 8 bit RGB
 * 3. Compress 8 bit RGB -> DXT
 */

struct video_compress {
        int num_threads;
        unsigned char *buffer[MAX_THREADS];
        unsigned char *output_data;
        pthread_mutex_t lock;
        volatile int thread_count;
        pthread_t thread_ids[MAX_THREADS];
        int tx_aux;
        codec_t tx_color_spec;
        sem_t thread_compress[MAX_THREADS];
        sem_t threads_done;
        
        decoder_t decoder;

        struct video_frame *frame;
        struct tile *tile;
};

static void compress_thread(void *args);
void reconfigure_compress(struct video_compress *compress, int width, int height, codec_t codec, int aux, double fps);

void reconfigure_compress(struct video_compress *compress, int width, int height, codec_t codec, int aux, double fps)
{
        int x;

        fprintf(stderr, "Compression reinitialized for %ux%u video.\n", 
                        width, height);
        /* Store original attributes to allow format change detection */
        compress->tx_aux = aux;
        compress->tx_color_spec = codec;

        compress->tile->width = width;
        compress->tile->height = height;
        compress->frame->color_spec = codec;
        compress->frame->aux = aux;
        compress->frame->fps = fps;

        switch (codec) {
                case RGB:
                        compress->decoder = (decoder_t) vc_copylineRGBtoRGBA;
                        compress->frame->aux |= AUX_RGB;
                        break;
                case RGBA:
                        compress->decoder = (decoder_t) memcpy;
                        compress->frame->aux |= AUX_RGB;
                        break;
                case R10k:
                        compress->decoder = (decoder_t) vc_copyliner10k;
                        compress->frame->aux |= AUX_RGB;
                        break;
                case UYVY:
                case Vuy2:
                case DVS8:
                        compress->decoder = (decoder_t) memcpy;
                        compress->frame->aux |= AUX_YUV;
                        break;
                case v210:
                        compress->decoder = (decoder_t) vc_copylinev210;
                        compress->frame->aux |= AUX_YUV;
                        break;
                case DVS10:
                        compress->decoder = (decoder_t) vc_copylineDVS10;
                        compress->frame->aux |= AUX_YUV;
                        break;
                default:
                        error_with_code_msg(128, "Unknown codec %d!", codec);
        }
        
        int h_align = 0;
        int i;
        for(i = 0; codec_info[i].name != NULL; ++i) {
                if(codec == codec_info[i].codec) {
                        h_align = codec_info[i].h_align;
                }
        }
        assert(h_align != 0);
        compress->tile->linesize = tile_get(compress->frame, 0, 0)->width * 
                (compress->frame->aux & AUX_RGB ? 4 /*RGBA*/: 2/*YUV 422*/);
        
        if(compress->frame->aux & AUX_RGB) {
                compress->frame->color_spec = DXT1;
        } else {
                compress->frame->color_spec = DXT1_YUV;
        }

        /* We will deinterlace the output frame */
        compress->frame->aux &= ~AUX_INTERLACED;

        for (x = 0; x < compress->num_threads; x++) {
                int my_height = (height / compress->num_threads) / 4 * 4;
                if(x == compress->num_threads - 1) {
                        my_height = compress->tile->height - my_height /* "their height" */ * x;
                }
                compress->buffer[x] =
                    (unsigned char *)malloc(width * my_height * 4);
        }
#ifdef HAVE_MACOSX
        compress->output_data = (unsigned char *)malloc(width * height * 4);
        compress->tile->data = (char *)malloc(width * height * 4);
#else
        /*
         *  memalign doesn't exist on Mac OS. malloc always returns 16  
         *  bytes aligned memory
         *
         *  see: http://www.mythtv.org/pipermail/mythtv-dev/2006-January/044309.html
         */
        compress->output_data = (unsigned char *)memalign(16, width * height * 4);
        compress->tile->data = (char *)memalign(16, width * height * 4);
#endif                          /* HAVE_MACOSX */
        memset(compress->output_data, 0, width * height * 4);
        memset(compress->tile->data, 0, width * height * 4 / 8);
}

void *fastdxt_init(const char *num_threads_str)
{
        /* This function does the following:
         * 1. Allocate memory for buffers 
         * 2. Spawn compressor threads
         */
        int x;
        struct video_compress *compress;

        if(num_threads_str && strcmp(num_threads_str, "help") == 0) {
                printf("FastDXT usage:\n");
                printf("\t-FastDXT[:<num_threads>]\n");
                printf("\t\t<num_threads> - count of compress threads (default %d)\n", NUM_THREADS_DEFAULT);
                return NULL;
        }

        compress = calloc(1, sizeof(struct video_compress));
        /* initial values */
        compress->num_threads = 0;
        if(num_threads_str == NULL)
                compress->num_threads = NUM_THREADS_DEFAULT;
        else
                compress->num_threads = atoi(num_threads_str);
        assert (compress->num_threads >= 1 && compress->num_threads <= MAX_THREADS);

        compress->frame = vf_alloc(1, 1);
        compress->tile = tile_get(compress->frame, 0, 0);
        
        compress->tile->width = 0;
        compress->tile->height = 0;

        compress->thread_count = 0;
        if (pthread_mutex_init(&(compress->lock), NULL)) {
        perror("Error initializing mutex!");
                exit(128);
        }

        platform_sem_init(&compress->threads_done, 0, 0);
        for (x = 0; x < compress->num_threads; x++) {
                platform_sem_init(&compress->thread_compress[x], 0, 0);
        }

        pthread_mutex_lock(&(compress->lock));

        for (x = 0; x < compress->num_threads; x++) {
                if (pthread_create
                    (&(compress->thread_ids[x]), NULL, (void *)compress_thread,
                     (void *)compress)) {
                        perror("Unable to create compressor thread!");
                        exit(x);
                }
        }
        pthread_mutex_unlock(&(compress->lock));
        return compress;
}

struct video_frame * fastdxt_compress(void *args, struct video_frame *tx)
{
        /* This thread will be called from main.c and handle the compress_threads */
        struct video_compress *compress = (struct video_compress *)args;
        unsigned int x;
        unsigned char *line1, *line2;

        assert(tx->grid_height == 1 && tx->grid_width == 1);
        assert(tile_get(tx, 0, 0)->width % 4 == 0);
        assert(tile_get(tx, 0, 0)->height % 4 == 0);
        
        pthread_mutex_lock(&(compress->lock));

        if(tile_get(tx, 0, 0)->width != compress->tile->width ||
                        tile_get(tx, 0, 0)->height != compress->tile->height ||
                        tx->aux != compress->tx_aux ||
                        tx->color_spec != compress->tx_color_spec)
        {
                reconfigure_compress(compress, tile_get(tx, 0, 0)->width, tile_get(tx, 0, 0)->height, tx->color_spec, tx->aux, tx->fps);
        }

        line1 = (unsigned char *)tx->tiles[0].data;
        line2 = compress->output_data;

        for (x = 0; x < compress->tile->height; ++x) {
                int src_linesize = vc_get_linesize(compress->tile->width, compress->tx_color_spec);
                compress->decoder(line2, line1, compress->tile->linesize,
                                0, 8, 16);
                line1 += src_linesize;
                line2 += compress->tile->linesize;
        }

        if(tx->aux & AUX_INTERLACED)
                vc_deinterlace(compress->output_data, compress->tile->linesize,
                                compress->tile->height);


        for (x = 0; x < compress->num_threads; x++) {
                platform_sem_post(&compress->thread_compress[x]);
        }

        for (x = 0; x < compress->num_threads; x++) {
                platform_sem_wait(&compress->threads_done);
        }

        compress->tile->data_len = compress->tile->width * compress->tile->height / 2;
        
        pthread_mutex_unlock(&(compress->lock));

        return compress->frame;
}

static void compress_thread(void *args)
{
        struct video_compress *compress = (struct video_compress *)args;
        int myId, range, my_range, x;
        int my_height;
        unsigned char *retv;

        pthread_mutex_lock(&(compress->lock));
        myId = compress->thread_count;
        compress->thread_count++;
        pthread_mutex_unlock(&(compress->lock));

        fprintf(stderr, "Compress thread %d online.\n", myId);

        while (1) {
                platform_sem_wait(&compress->thread_compress[myId]);
                if(fastdxt_should_exit) break;

                my_height = (compress->tile->height / compress->num_threads) / 4 * 4;
                range = my_height * compress->tile->width; /* for all threads except the last */

                if(myId == compress->num_threads - 1) {
                        my_height = compress->tile->height - my_height /* "their height" */ * myId;
                }
                my_range = my_height * compress->tile->width;

                if(compress->frame->aux & AUX_YUV)
                {
                        unsigned char *input;
                        input = (compress->output_data) + myId
                                * range * 2;
                        retv = compress->buffer[myId];
                        /* Repack the data to YUV 4:4:4 Format */
                        for (x = 0; x < my_range; x += 2) {
                                retv[4 * x] = input[2 * x + 1]; //Y1
                                retv[4 * x + 1] = input[2 * x]; //U1
                                retv[4 * x + 2] = input[2 * x + 2];     //V1
                                retv[4 * x + 3] = 255;  //Alpha

                                retv[4 * x + 4] = input[2 * x + 3];     //Y2
                                retv[4 * x + 5] = input[2 * x]; //U1
                                retv[4 * x + 6] = input[2 * x + 2];     //V1
                                retv[4 * x + 7] = 255;  //Alpha
                        }
                } else {
                        retv = (compress->output_data) + myId * range * 4;
                }

                DirectDXT1(retv,
                               ((unsigned char *) compress->tile->data) + myId * range / 2,
                               compress->tile->width, my_height);

                platform_sem_post(&compress->threads_done);
        }
        
        platform_sem_post(&compress->threads_done);
}

void fastdxt_done(void *args)
{
        struct video_compress *compress = (struct video_compress *)args;
        int x;
        
        pthread_mutex_lock(&(compress->lock)); /* wait for fastdxt_compress if running */
        fastdxt_should_exit = TRUE;
        
        for (x = 0; x < compress->num_threads; x++) {
                platform_sem_post(&compress->thread_compress[x]);
        }

        for (x = 0; x < compress->num_threads; x++) {
                platform_sem_wait(&compress->threads_done);
        }

        pthread_mutex_unlock(&(compress->lock));
        
        pthread_mutex_destroy(&(compress->lock));
        
        for (x = 0; x < compress->num_threads; ++x)
                free(compress->buffer[x]);
        free(compress);
                
        
}
