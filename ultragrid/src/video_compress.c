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
#include <pthread.h>
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

#ifndef HAVE_MACOSX
#define uint64_t 	unsigned long
#endif                          /* HAVE_MACOSX */

/* NOTE: These threads busy wait, so at *most* set this to one less than the
 * total number of cores on your system (Also 3 threads will work)! Also, if
 * you see tearing in the rendered image try increasing the number of threads
 * by 1 (For a dual dual-core Opteron 285 3 threads work great). 
 * -- iwsmith <iwsmith@cct.lsu.ed> 9 August 2007
 */
#define NUM_THREADS 3
#define HD_HEIGHT 1080
#define HD_WIDTH 1920

/* Ok, we are going to decompose the problem into 2^n pieces (generally
 * 2, 4, or 8). We will most likely need to do the following:
 * 1. Convert 10-bit -> 8bit
 * 2. Convert YUV 4:2:2 -> 8 bit RGB
 * 3. Compress 8 bit RGB -> DXT
 */

struct video_compress {
        unsigned char *buffer[NUM_THREADS];
        unsigned char *output_data;
        pthread_mutex_t lock;
        volatile int thread_count, len[NUM_THREADS], go[NUM_THREADS];
        pthread_t thread_ids[NUM_THREADS];
        int tx_aux;
        codec_t tx_color_spec;

        struct video_frame frame;
};

static void compress_thread(void *args);
void reconfigure_compress(struct video_compress *compress, int width, int height, codec_t codec, int aux);

void reconfigure_compress(struct video_compress *compress, int width, int height, codec_t codec, int aux)
{
        int x;

        fprintf(stderr, "Compression reinitialized for %ux%u video.\n", 
                        width, height);
        /* Store original attributes to allow format change detection */
        compress->tx_aux = aux;
        compress->tx_color_spec = codec;

        compress->frame.width = width;
        compress->frame.height = height;
        compress->frame.color_spec = codec;
        compress->frame.src_bpp = get_bpp(codec);
        compress->frame.aux = aux;

        switch (codec) {
                case RGBA:
                        compress->frame.decoder = (decoder_t) memcpy;
                        compress->frame.aux |= AUX_RGB;
                        break;
                case R10k:
                        compress->frame.decoder = (decoder_t) vc_copyliner10k;
                        compress->frame.rshift = 0;
                        compress->frame.gshift = 8;
                        compress->frame.bshift = 16;
                        compress->frame.aux |= AUX_RGB;
                        break;
                case UYVY:
                case Vuy2:
                case DVS8:
                        compress->frame.decoder = (decoder_t) memcpy;
                        compress->frame.aux |= AUX_YUV;
                        break;
                case v210:
                        compress->frame.decoder = (decoder_t) vc_copylinev210;
                        compress->frame.aux |= AUX_YUV;
                        break;
                case DVS10:
                        compress->frame.decoder = (decoder_t) vc_copylineDVS10;
                        compress->frame.aux |= AUX_YUV;
                        break;
                case DXT1:
                        fprintf(stderr, "Input frame is already comperssed!");
                        exit(128);
        }
        compress->frame.src_linesize = compress->frame.width * compress->frame.src_bpp;
        compress->frame.dst_linesize = compress->frame.width * 
                (compress->frame.aux == AUX_RGB ? 4 /*RGBA*/: 2/*YUV 422*/);
        compress->frame.color_spec = DXT1;

        /* We will deinterlace the output frame */
        compress->frame.aux &= ~AUX_INTERLACED;

        for (x = 0; x < NUM_THREADS; x++) {
                compress->buffer[x] =
                    (unsigned char *)malloc(width * height * 4 / NUM_THREADS);
        }
#ifdef HAVE_MACOSX
        compress->output_data = (unsigned char *)malloc(width * height * 4);
        compress->frame.data = (char *)malloc(width * height * 4);
#else
        /*
         *  memalign doesn't exist on Mac OS. malloc always returns 16  
         *  bytes aligned memory
         *
         *  see: http://www.mythtv.org/pipermail/mythtv-dev/2006-January/044309.html
         */
        compress->output_data = (unsigned char *)memalign(16, width * height * 4);
        compress->frame.data = (char *)memalign(16, width * height * 4);
#endif                          /* HAVE_MACOSX */
        memset(compress->output_data, 0, width * height * 4);
        memset(compress->frame.data, 0, width * height * 4 / 8);
}

struct video_compress *initialize_video_compression(void)
{
        /* This function does the following:
         * 1. Allocate memory for buffers 
         * 2. Spawn compressor threads
         */
        int x;
        struct video_compress *compress;

        compress = calloc(1, sizeof(struct video_compress));
        /* initial values */
        compress->frame.width = 0;
        compress->frame.height = 0;

        compress->thread_count = 0;
        if (pthread_mutex_init(&(compress->lock), NULL)) {
        perror("Error initializing mutex!");
                exit(128);
        }

        pthread_mutex_lock(&(compress->lock));

        for (x = 0; x < NUM_THREADS; x++) {
                if (pthread_create
                    (&(compress->thread_ids[x]), NULL, (void *)compress_thread,
                     (void *)compress)) {
                        perror("Unable to create compressor thread!");
                        exit(x);
                }

                compress->go[x] = 0;
                compress->len[x] = 0;
        }
        pthread_mutex_unlock(&(compress->lock));
        return compress;
}

struct video_frame * compress_data(void *args, struct video_frame *tx)
{
        /* This thread will be called from main.c and handle the compress_threads */
        struct video_compress *compress = (struct video_compress *)args;
        unsigned int x, total = 0;
        unsigned char *line1, *line2;

        if(tx->width != compress->frame.width ||
                        tx->height != compress->frame.height ||
                        tx->aux != compress->tx_aux ||
                        tx->color_spec != compress->tx_color_spec)
        {
                reconfigure_compress(compress, tx->width, tx->height, tx->color_spec, tx->aux);
        }

        line1 = (unsigned char *)tx->data;
        line2 = compress->output_data;

        for (x = 0; x < compress->frame.height; ++x) {
                compress->frame.decoder(line2, line1, compress->frame.src_linesize,
                                compress->frame.rshift, compress->frame.gshift, compress->frame.bshift);
                line1 += compress->frame.src_linesize;
                line2 += compress->frame.dst_linesize;
        }


        if(tx->aux & AUX_INTERLACED)
                vc_deinterlace(compress->output_data, compress->frame.src_linesize,
                                compress->frame.height);


        for (x = 0; x < NUM_THREADS; x++) {
                compress->go[x] = 1;
        }

        while (total != compress->frame.width * compress->frame.height / 2) {
                //This is just getting silly...
                total = 0;
                for (x = 0; x < NUM_THREADS; x++) {
                        total += compress->len[x];
                }
        }

        compress->frame.data_len = total;

        return &compress->frame;
}

static void compress_thread(void *args)
{
        struct video_compress *compress = (struct video_compress *)args;
        int myId, myEnd, myStart, range, x;
        unsigned char *retv;

        pthread_mutex_lock(&(compress->lock));
        myId = compress->thread_count;
        compress->thread_count++;
        pthread_mutex_unlock(&(compress->lock));

        while (compress->go[myId] == 0) {
                //Busywait
        }

        range = compress->frame.width * compress->frame.height / NUM_THREADS;
        myStart = myId * range;
        myEnd = (myId + 1) * range;
        fprintf(stderr, "Thread %d online, handling elements %d - %d\n", myId,
                myStart, myEnd - 1);

        while (1) {
                while (compress->go[myId] == 0) {
                        //Busywait
                }
                if(compress->frame.aux & AUX_YUV)
                {
                        unsigned char *input;
                        input = (compress->output_data) +
                            (myId * compress->frame.width * compress->frame.height * 2 / NUM_THREADS);
                        retv = compress->buffer[myId];
                        /* Repack the data to YUV 4:4:4 Format */
                        for (x = 0; x < range; x += 2) {
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
                        retv = (compress->output_data) +
                            (myId * compress->frame.width * compress->frame.height * 4 / NUM_THREADS);
                }

                compress->len[myId] =
                    DirectDXT1(retv,
                               ((unsigned char *) compress->frame.data) + myId * compress->frame.width * compress->frame.height / 2 / (NUM_THREADS),
                               compress->frame.width, compress->frame.height / NUM_THREADS);
                compress->go[myId] = 0;

        }
}

void compress_exit(void *args)
{
        struct video_compress *compress = (struct video_compress *)args;
        int x;

        for (x = 0; x < compress->thread_count; x++) {
                pthread_kill(compress->thread_ids[x], SIGKILL);
        }

        for (x = 0; x < NUM_THREADS; ++x)
                free(compress->buffer[x]);
}
