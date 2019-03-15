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
#include "lib_common.h"
#include "video.h"
#include "video_display.h"

#include <GL/gl.h>

#include <X11/Xlib.h>
#include <sys/time.h>
#include <assert.h>

#include <semaphore.h>
#include <signal.h>
#include <pthread.h>

#include <sail.h>
#include <misc.h>

#include <host.h>
#include <tv.h>

#include <video_codec.h>

#define MAGIC_SAGE	0x3e960c47

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
        
        sail *sage_state;

        const char             *confName;
        const char             *fsIP;

        int                     frames;
        struct timeval          t, t0;

        volatile bool           should_exit;
        bool                    is_tx;
        bool                    deinterlace;
};

/** Prototyping */
static int display_sage_handle_events(void)
{
        return 0;
}

static void display_sage_run(void *arg)
{
        struct state_sage *s = (struct state_sage *)arg;
        s->magic = MAGIC_SAGE;

        while (!s->should_exit) {
                display_sage_handle_events();

                sem_wait(&s->semaphore);
                if (s->should_exit)
                        break;

                if (s->deinterlace) {
                        vc_deinterlace((unsigned char *) s->tile->data, vc_get_linesize(s->tile->width,
                                                s->frame->color_spec), s->tile->height);
                }

                s->sage_state->swapBuffer(SAGE_NON_BLOCKING);
                sageMessage msg;
                if (s->sage_state->checkMsg(msg, false) > 0) {
                        switch (msg.getCode()) {
                        case APP_QUIT:
                                sage::printLog("Ultragrid: QUIT message");
                                exit_uv(1);
                                break;
                        }
                }

                s->tile->data = (char *) s->sage_state->getBuffer();

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
                        log_msg(LOG_LEVEL_INFO, "[SAGE] %d frames in %g seconds = %g FPS\n",
                                s->frames, seconds, fps);
                        s->t0 = s->t;
                        s->frames = 0;
                }
        }
}

static void *display_sage_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(fmt);
        UNUSED(flags);
        UNUSED(parent);
        struct state_sage *s;

        s = (struct state_sage *) calloc(1, sizeof(struct state_sage));
        assert(s != NULL);

        s->confName = NULL;

        if(fmt) {
                if(strcmp(fmt, "help") == 0) {
                        printf("SAGE usage:\n");
                        printf("\tuv -t sage[:config=<config_file>|:fs=<fsIP>][:codec=<fcc>][:d]\n");
                        printf("\t                      <config_file> - SAGE app config file, default \"ultragrid.conf\"\n");
                        printf("\t                      <fsIP> - FS manager IP address\n");
                        printf("\t                      <fcc> - FourCC of codec that will be used to transmit to SAGE\n");
                        printf("\t                              Supported options are UYVY, RGBA, RGB or DXT1\n");
                        printf("\t                      d - deinterlace output\n");
                        return &display_init_noerr;
                } else {
                        char *tmp, *parse_str;
                        tmp = parse_str = strdup(fmt);
                        char *save_ptr = NULL;
                        char *item;

                        while((item = strtok_r(parse_str, ":", &save_ptr))) {
                                parse_str = NULL;
                                if(strncmp(item, "config=", strlen("config=")) == 0) {
                                        s->confName = item + strlen("config=");
                                } else if(strncmp(item, "codec=", strlen("codec=")) == 0) {
                                         uint32_t fourcc;
                                         if(strlen(item + strlen("codec=")) != sizeof(fourcc)) {
                                                 fprintf(stderr, "Malformed FourCC code (wrong length).\n");
                                                 free(s); return NULL;
                                         }
                                         memcpy((void *) &fourcc, item + strlen("codec="), sizeof(fourcc));
                                         s->requestedDisplayCodec = get_codec_from_fcc(fourcc);
                                         if(s->requestedDisplayCodec == VIDEO_CODEC_NONE) {
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
                                } else if(strcmp(item, "tx") == 0) {
                                        s->is_tx = true;
                                } else if(strncmp(item, "fs=", strlen("fs=")) == 0) {
                                        s->fsIP = item + strlen("fs=");
                                } else if(strcmp(item, "d") == 0) {
                                        s->deinterlace = true;
                                } else {
                                        fprintf(stderr, "[SAGE] unrecognized configuration: %s\n",
                                                        item);
                                        free(s);
                                        return NULL;
                                }
                        }
                        free(tmp);
                }
        }

        if (!s->is_tx) {
                // read config file only if we are in dispaly mode (not sender mode)
                struct stat sb;
                if(s->confName) {
                        if(stat(s->confName, &sb)) {
                                perror("Unable to use SAGE config file");
                                free(s);
                                return NULL;
                        }
                } else if(stat("ultragrid.conf", &sb) == 0) {
                        s->confName = "ultragrid.conf";
                }
                if(s->confName) {
                        printf("[SAGE] Using config file %s.\n", s->confName);
                }
        }

        if(s->confName == NULL && s->fsIP == NULL) {
                fprintf(stderr, "[SAGE] Unable to locate FS manager address. "
                                "Set either in config file or from command line.\n");
                free(s);
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

static void display_sage_done(void *state)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);

        sem_destroy(&s->semaphore);
        pthread_cond_destroy(&s->buffer_writable_cond);
        pthread_mutex_destroy(&s->buffer_writable_lock);
        vf_free(s->frame);
        if (s->sage_state)
                s->sage_state->shutdown();
        //delete s->sage_state;
        free(s);
}

static struct video_frame *display_sage_getf(void *state)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);

        pthread_mutex_lock(&s->buffer_writable_lock);
        while (!s->buffer_writable) {
                s->grab_waiting = TRUE;
                pthread_cond_wait(&s->buffer_writable_cond,
                                &s->buffer_writable_lock);
                s->grab_waiting = FALSE;
        }
        pthread_mutex_unlock(&s->buffer_writable_lock);

        return s->frame;
}

static int display_sage_putf(void *state, struct video_frame *frame, int nonblock)
{
        int tmp;
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);
        UNUSED(frame);
        UNUSED(nonblock);

        if(!frame) {
                s->should_exit = true;
        }

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

/*
 * Either confName or fsIP should be NULL
 */
static sail *initSage(const char *confName, const char *fsIP, int appID, int nodeID, int width,
                int height, codec_t codec)
{
        sail *sageInf; // sage sail object

        sageInf = new sail;
        sailConfig sailCfg;

        // default values
        if(fsIP) {
                strncpy(sailCfg.fsIP, fsIP, SAGE_IP_LEN - 1);
        }
        sailCfg.fsPort = 20002;
        strncpy(sailCfg.masterIP, "127.0.0.1", SAGE_IP_LEN - 1);
        sailCfg.nwID = 1;
        sailCfg.msgPort = 23010;
        sailCfg.syncPort = 13010;
        sailCfg.blockSize = 64;
        sailCfg.winX = sailCfg.winY = 0;
        sailCfg.winWidth = width;
        sailCfg.winHeight = height;
        sailCfg.streamType = SAGE_BLOCK_NO_SYNC;
        sailCfg.protocol = SAGE_UDP;
        sailCfg.asyncUpdate = false;

        if(confName) {
                sailCfg.init((char *) confName);
        }
        char appName[] = "ultragrid";
        sailCfg.setAppName(appName);
        sailCfg.rank = nodeID;
        sailCfg.appID = appID;
        sailCfg.resX = width;
        sailCfg.resY = height;

        sageRect renderImageMap;
        renderImageMap.left = 0.0;
        renderImageMap.right = 1.0;
        renderImageMap.bottom = 0.0;
        renderImageMap.top = 1.0;

        sailCfg.imageMap = renderImageMap;
        switch (codec) {
                case DXT1:
                        sailCfg.pixFmt = PIXFMT_DXT;
                        break;
#ifdef SAGE_NATIVE_DXT5YCOCG
                case DXT5:
                        sailCfg.pixFmt = PIXFMT_DXT5YCOCG
                        break;
#endif // SAGE_NATIVE_DXT5YCOCG
                case RGBA:
                        sailCfg.pixFmt = PIXFMT_8888;
                        break;
                case UYVY:
                        sailCfg.pixFmt = PIXFMT_YUV;
                        break;
                case RGB:
                        sailCfg.pixFmt = PIXFMT_888;
                        break;
                default:
                        abort();
        }

        if(codec == DXT1) {
                sailCfg.rowOrd = BOTTOM_TO_TOP;
#ifdef SAGE_NATIVE_DXT5YCOCG
        } else if(codec == DXT5) {
                sailCfg.rowOrd = BOTTOM_TO_TOP;
#endif // SAGE_NATIVE_DXT5YCOCG
        } else {
                sailCfg.rowOrd = TOP_TO_BOTTOM;
        }
        sailCfg.master = true;

        sageInf->init(sailCfg);

        return sageInf;
}

static int display_sage_reconfigure(void *state, struct video_desc desc)
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
                s->sage_state->shutdown();
                //delete s->sage_state; // this used to cause crashes
        }

        s->sage_state = initSage(s->confName, s->fsIP, s->appID, s->nodeID,
                        s->tile->width, s->tile->height, desc.color_spec);

        // calling thread should be able to process signals afterwards
        pthread_sigmask(SIG_UNBLOCK, &old_mask, NULL);

        s->tile->data = (char *) s->sage_state->getBuffer();
        s->tile->data_len = vc_get_linesize(s->tile->width, desc.color_spec) * s->tile->height;

        return TRUE;
}

static int display_sage_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_sage *s = (struct state_sage *)state;
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB, DXT1
#ifdef SAGE_NATIVE_DXT5YCOCG
                , DXT5
#endif // SAGE_NATIVE_DXT5YCOCG
        };
        int rgb_shift[] = {0, 8, 16};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(s->requestedDisplayCodec != VIDEO_CODEC_NONE) {
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
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
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

static void display_sage_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_sage_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_sage_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].id, "SAGE");
                strcpy((*available_cards)[0].name, "SAGE display wall");
                (*available_cards)[0].repeatable = true;
        },
        display_sage_init,
        display_sage_run,
        display_sage_done,
        display_sage_getf,
        display_sage_putf,
        display_sage_reconfigure,
        display_sage_get_property,
        display_sage_put_audio_frame,
        display_sage_reconfigure_audio,
};

REGISTER_MODULE(sage, &display_sage_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

