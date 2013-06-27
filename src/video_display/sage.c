/*
 * FILE:    video_display/sage.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "video_codec.h"
#include "video_display.h"
#include "video_display/sage.h"

#include <GL/gl.h>

#include <X11/Xlib.h>
#include <sys/time.h>
#include <assert.h>

#include <semaphore.h>
#include <signal.h>
#include <pthread.h>

#include "video_display/sage_wrapper.h"
#include <host.h>
#include <tv.h>

#include <video_codec.h>

#define MAGIC_SAGE	DISPLAY_SAGE_ID

struct state_sage {
        struct video_frame *frame;
        struct tile *tile;
        codec_t requestedDisplayCodec;

        /* Thread related information follows... */
        pthread_t thread_id;
        sem_t semaphore;

        volatile unsigned int buffer_writable:1;
        volatile unsigned int grab_waiting:1;
        pthread_cond_t buffer_writable_cond;
        pthread_mutex_t buffer_writable_lock;

        /* For debugging... */
        uint32_t magic;
        int appID, nodeID;
        
        void *sage_state;

        const char             *confName;
        const char             *fsIP;

        int                     frames;
        struct timeval          t, t0;
};

static volatile bool should_exit = false;

/** Prototyping */
int display_sage_handle_events(void)
{
        return 0;
}

void display_sage_run(void *arg)
{
        struct state_sage *s = (struct state_sage *)arg;
        s->magic = MAGIC_SAGE;

        while (!should_exit) {
                //display_sage_handle_events();

                sem_wait(&s->semaphore);
                if (should_exit)
                        break;

                sage_swapBuffer(s->sage_state);
                s->tile->data = (char *) sage_getBuffer(s->sage_state);

                pthread_mutex_lock(&s->buffer_writable_lock);
                s->buffer_writable = 1;
                if(s->grab_waiting) {
                        pthread_cond_broadcast(&s->buffer_writable_cond);
                }
                pthread_mutex_unlock(&s->buffer_writable_lock);

                s->frames++;

                gettimeofday(&s->t, NULL);
                double seconds = tv_diff(s->t, s->t0);
                if (seconds >= 5) {
                        float fps = s->frames / seconds;
                        fprintf(stderr, "[SAGE] %d frames in %g seconds = %g FPS\n",
                                s->frames, seconds, fps);
                        s->t0 = s->t;
                        s->frames = 0;
                }
        }
}

void *display_sage_init(char *fmt, unsigned int flags)
{
        UNUSED(fmt);
        UNUSED(flags);
        struct state_sage *s;

        s = (struct state_sage *)malloc(sizeof(struct state_sage));
        assert(s != NULL);

        s->confName = NULL;
        s->fsIP = sage_network_device; // NULL unless in SAGE TX mode
        s->requestedDisplayCodec = (codec_t) -1;

        if(fmt) {
                if(strcmp(fmt, "help") == 0) {
                        printf("SAGE usage:\n");
                        printf("\tuv -t sage[:config=<config_file>|:fs=<fsIP>][:codec=<fcc>]\n");
                        printf("\t                      <config_file> - SAGE app config file, default \"ultragrid.conf\"\n");
                        printf("\t                      <fsIP> - FS manager IP address\n");
                        printf("\t                      <fcc> - FourCC of codec that will be used to transmit to SAGE\n");
                        printf("\t                              Supported options are UYVY, RGBA, RGB or DXT1\n");
                        return &display_init_noerr;
                } else {
                        char *save_ptr = NULL;
                        char *item;

                        while((item = strtok_r(fmt, ":", &save_ptr))) {
                                fmt = NULL;
                                if(strncmp(item, "config=", strlen("config=")) == 0) {
                                        s->confName = item + strlen("config=");
                                } else if(strncmp(item, "codec=", strlen("codec=")) == 0) {
                                         strlen("codec=");
                                         uint32_t fourcc;
                                         if(strlen(item + strlen("codec=")) != sizeof(fourcc)) {
                                                 fprintf(stderr, "Malformed FourCC code (wrong length).\n");
                                                 free(s); return NULL;
                                         }
                                         memcpy((void *) &fourcc, item + strlen("codec="), sizeof(fourcc));
                                         s->requestedDisplayCodec = get_codec_from_fcc(fourcc);
                                         if(s->requestedDisplayCodec == (codec_t) -1) {
                                                 fprintf(stderr, "Codec not found according to FourCC.\n");
                                                 free(s); return NULL;
                                         }
                                         if(s->requestedDisplayCodec != UYVY &&
                                                         s->requestedDisplayCodec != RGBA &&
                                                         s->requestedDisplayCodec != RGB &&
                                                         s->requestedDisplayCodec != DXT1
#ifdef SAGE_NATIVE_DXT5YCOCG
                                                         && s->requestedDisplayCodec != DXT5
#endif // SAGE_NATIVE_DXT5YCOCG
                                                         ) {
                                                 fprintf(stderr, "Entered codec is not nativelly supported by SAGE.\n");
                                                 free(s); return NULL;
                                         }
                                } else if(strncmp(item, "fs=", strlen("fs=")) == 0) {
                                        s->fsIP = item + strlen("fs=");
                                } else {
                                        fprintf(stderr, "[SAGE] unrecognized configuration: %s\n",
                                                        item);
                                        free(s);
                                        return NULL;
                                }
                        }
                }
        }

        struct stat sb;
        if(s->confName) {
                if(stat(s->confName, &sb)) {
                        perror("Unable to use SAGE config file");
                        return NULL;
                }
        } else if(stat("ultragrid.conf", &sb) == 0) {
                s->confName = "ultragrid.conf";
        }
        if(s->confName) {
                printf("[SAGE] Using config file %s.\n", s->confName);
        }
        if(s->confName == NULL && s->fsIP == NULL) {
                fprintf(stderr, "[SAGE] Unable to locate FS manager address. "
                                "Set either in config file or from command line.\n");
                return NULL;
        }

        s->magic = MAGIC_SAGE;

        gettimeofday(&s->t0, NULL);

        s->frames = 0;
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        
        /* sage init */
        //FIXME sem se musi propasovat ty spravne parametry argc argv
        s->appID = 0;
        s->nodeID = 1;

        s->sage_state = NULL;

        /* thread init */
        sem_init(&s->semaphore, 0, 0);

        s->buffer_writable = 1;
        s->grab_waiting = 1;
        pthread_mutex_init(&s->buffer_writable_lock, 0);
        pthread_cond_init(&s->buffer_writable_cond, NULL);

        /*if (pthread_create
            (&(s->thread_id), NULL, display_thread_sage, (void *)s) != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/

        debug_msg("Window initialized %p\n", s);

        return (void *)s;
}

void display_sage_finish(void *state)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);

        should_exit = true;

        // there was already issued should_exit...
        display_sage_putf(s, NULL, PUTF_BLOCKING);
        // .. so thread should exit after this call
}

void display_sage_done(void *state)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);

        sem_destroy(&s->semaphore);
        pthread_cond_destroy(&s->buffer_writable_cond);
        pthread_mutex_destroy(&s->buffer_writable_lock);
        vf_free(s->frame);
        sage_shutdown(s->sage_state);
        //sage_delete(s->sage_state);
}

struct video_frame *display_sage_getf(void *state)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);

        pthread_mutex_lock(&s->buffer_writable_lock);
        if(should_exit) {
                pthread_mutex_unlock(&s->buffer_writable_lock);
                return NULL;
        }
        while (!s->buffer_writable) {
                s->grab_waiting = TRUE;
                pthread_cond_wait(&s->buffer_writable_cond,
                                &s->buffer_writable_lock);
                s->grab_waiting = FALSE;
        }
        pthread_mutex_unlock(&s->buffer_writable_lock);

        return s->frame;
}

int display_sage_putf(void *state, struct video_frame *frame, int nonblock)
{
        int tmp;
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);
        UNUSED(frame);
        UNUSED(nonblock);

        /* ...and signal the worker */
        pthread_mutex_lock(&s->buffer_writable_lock);
        s->buffer_writable = 0;
        pthread_mutex_unlock(&s->buffer_writable_lock);

        sem_post(&s->semaphore);
#ifndef HAVE_MACOSX
        sem_getvalue(&s->semaphore, &tmp);
        if (tmp > 1)
                printf("frame drop!\n");
#endif
        return 0;
}

int display_sage_reconfigure(void *state, struct video_desc desc)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);
        assert(desc.color_spec == RGBA || desc.color_spec == RGB || desc.color_spec == UYVY ||
                        desc.color_spec == DXT1
#ifdef SAGE_NATIVE_DXT5YCOCG
                        || desc.color_spec == DXT5
#endif // SAGE_NATIVE_DXT5YCOCG
                        );
        
        s->tile->width = desc.width;
        s->tile->height = desc.height;
        s->frame->fps = desc.fps;
        s->frame->interlacing = desc.interlacing;
        s->frame->color_spec = desc.color_spec;

        // SAGE fix - SAGE threads apparently do not process signals correctly so we temporarily
        // block all signals while creating SAGE
        sigset_t mask, old_mask;
        sigemptyset(&mask);
        sigaddset(&mask, SIGINT);
        sigaddset(&mask, SIGTERM);
        sigaddset(&mask, SIGHUP);
        pthread_sigmask(SIG_BLOCK, &mask, &old_mask);

        if(s->sage_state) {
                sage_shutdown(s->sage_state);
        }

        s->sage_state = initSage(s->confName, s->fsIP, s->appID, s->nodeID,
                        s->tile->width, s->tile->height, desc.color_spec);

        // calling thread should be able to process signals afterwards
        pthread_sigmask(SIG_UNBLOCK, &old_mask, NULL);

        s->tile->data = (char *) sage_getBuffer(s->sage_state);
        s->tile->data_len = vc_get_linesize(s->tile->width, desc.color_spec) * s->tile->height;

        return TRUE;
}

display_type_t *display_sage_probe(void)
{
        display_type_t *dt;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id = DISPLAY_SAGE_ID;
                dt->name = "sage";
                dt->description = "SAGE";
        }
        return dt;
}

int display_sage_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_sage *s = (struct state_sage *)state;
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB, DXT1
#ifdef SAGE_NATIVE_DXT5YCOCG
                , DXT5
#endif // SAGE_NATIVE_DXT5YCOCG
        };
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(s->requestedDisplayCodec != (codec_t) -1) {
                                if(sizeof(codec_t) <= *len) {
                                        memcpy(val, &s->requestedDisplayCodec, sizeof(codec_t));
                                        *len = sizeof(codec_t);
                                } else {
                                        return FALSE;
                                }
                        } else {
                                if(sizeof(codecs) <= *len) {
                                        memcpy(val, codecs, sizeof(codecs));
                                        *len = sizeof(codecs);
                                } else {
                                        return FALSE;
                                }
                        }
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

void display_sage_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

int display_sage_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

