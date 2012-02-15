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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
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

#include <video_codec.h>

#define MAGIC_SAGE	DISPLAY_SAGE_ID

struct state_sage {
        struct video_frame *frame;
        struct tile *tile;

        /* Thread related information follows... */
        pthread_t thread_id;
        sem_t semaphore;

        volatile int buffer_writable;
        pthread_cond_t buffer_writable_cond;
        pthread_mutex_t buffer_writable_lock;

        /* For debugging... */
        uint32_t magic;
        int appID, nodeID;
        
        void *sage_state;

        int                     frames;
        struct timeval          t, t0;
};

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
                pthread_cond_broadcast(&s->buffer_writable_cond);
                pthread_mutex_unlock(&s->buffer_writable_lock);

                double seconds = tv_diff(t, t0);
                if (seconds >= 5) {
                        float fps = frames / seconds;
                        fprintf(stderr, "[SAGE] %d frames in %g seconds = %g FPS\n",
                                frames, seconds, fps);
                        t0 = t;
                        frames = 0;
                }
        }
}

void *display_sage_init(char *fmt, unsigned int flags)
{
        UNUSED(fmt);
        UNUSED(flags);
        struct state_sage *s;

        if(fmt && strcmp(fmt, "help") == 0) {
                printf("No configuration needed for SAGE\n");
                return NULL;
        }
        
        s = (struct state_sage *)malloc(sizeof(struct state_sage));
        s->magic = MAGIC_SAGE;

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
        display_sage_putf(s, NULL);
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
        while (!s->buffer_writable)
                pthread_cond_wait(&s->buffer_writable_cond,
                                &s->buffer_writable_lock);
        pthread_mutex_unlock(&s->buffer_writable_lock);

        return s->frame;
}

int display_sage_putf(void *state, char *frame)
{
        int tmp;
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);
        UNUSED(frame);

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
                        desc.color_spec == DXT1);
        
        s->tile->width = desc.width;
        s->tile->height = desc.height;
        s->frame->fps = desc.fps;
        s->frame->interlacing = desc.interlacing;
        s->frame->color_spec = desc.color_spec;

        if(s->sage_state) {
                sage_shutdown(s->sage_state);
        }

        s->sage_state = initSage(s->appID, s->nodeID, s->tile->width, s->tile->height, desc.color_spec);

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
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB, DXT1};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
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

struct audio_frame * display_sage_get_audio_frame(void *state)
{
        UNUSED(state);
        return NULL;
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

