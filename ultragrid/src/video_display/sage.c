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
#include "video_display.h"
#include "video_display/sage.h"

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

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
        struct video_frame frame;
        struct video_frame *tiles;
        int put_frame:1;

        /* Thread related information follows... */
        pthread_t thread_id;
        sem_t semaphore;

        volatile int buffer_writable;
        pthread_cond_t buffer_writable_cond;
        pthread_mutex_t buffer_writable_lock;

        /* For debugging... */
        uint32_t magic;
        int appID, nodeID;
        int sage_initialized;
};

/** Prototyping */
void sage_reconfigure_screen(void *s, unsigned int width, unsigned int height,
                codec_t codec, double fps, int aux, struct tile_info tile_info);
static struct video_frame * get_tile_buffer(void *s, struct tile_info tile_info);
static void update_tile_data(struct state_sage *s);

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

                sage_swapBuffer();
                s->frame.data = sage_getBuffer();
                if (s->frame.aux & AUX_TILED)
                        update_tile_data(s);

                pthread_mutex_lock(&s->buffer_writable_lock);
                s->buffer_writable = 1;
                pthread_cond_broadcast(&s->buffer_writable_cond);
                pthread_mutex_unlock(&s->buffer_writable_lock);
        }
}

void *display_sage_init(void)
{
        struct state_sage *s;

        s = (struct state_sage *)malloc(sizeof(struct state_sage));
        s->magic = MAGIC_SAGE;

        /* sage init */
        //FIXME sem se musi propasovat ty spravne parametry argc argv
        s->appID = 0;
        s->nodeID = 1;

        s->sage_initialized = 0;
        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t)sage_reconfigure_screen;
        s->frame.get_tile_buffer = (get_tile_buffer_t) get_tile_buffer;

        s->frame.rshift = 0;
        s->frame.gshift = 8;
        s->frame.bshift = 16;


        s->frame.aux &= ~AUX_TILED; /* do not expect tiled video by default */
        s->tiles = NULL;

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

void display_sage_done(void *state)
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);
        sem_destroy(&s->semaphore);
        pthread_cond_destroy(&s->buffer_writable_cond);
        pthread_mutex_destroy(&s->buffer_writable_lock);
        if(s->tiles == NULL)
                free(s->tiles);
        sage_shutdown();
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

        return &s->frame;
}

int display_sage_putf(void *state, char *frame)
{
        int tmp;
        struct state_sage *s = (struct state_sage *)state;

        assert(s->magic == MAGIC_SAGE);
        UNUSED(frame);

        /* ...and signal the worker */

        if(s->put_frame) {
                pthread_mutex_lock(&s->buffer_writable_lock);
                s->buffer_writable = 0;
                pthread_mutex_unlock(&s->buffer_writable_lock);

                sem_post(&s->semaphore);
                sem_getvalue(&s->semaphore, &tmp);
                if (tmp > 1)
                        printf("frame drop!\n");
        }

        return 0;
}

void sage_reconfigure_screen(void *arg, unsigned int width, unsigned int height,
                codec_t codec, double fps, int aux, struct tile_info tile_info)
{
        struct state_sage *s = (struct state_sage *)arg;

        int yuv;

        assert(s->magic == MAGIC_SAGE);
        if (aux & AUX_TILED) {
                s->frame.width = width * tile_info.x_count;
                s->frame.height = height * tile_info.y_count;
                s->put_frame = FALSE; /* put only if we have last tile */
        } else {
                s->frame.width = width;
                s->frame.height = height;
                s->put_frame = TRUE; /* this is not tiled so put every frame */
        }

        switch (codec) {
                case R10k:
                        s->frame.decoder = (decoder_t)vc_copyliner10k;
                        s->frame.dst_bpp = get_bpp(RGBA);
                        yuv = 0;
                        break;
                case RGBA:
                        s->frame.decoder = (decoder_t)memcpy; /* or vc_copylineRGBA?
                                                                 but we have default
                                                                 {r,g,b}shift */
                        
                        s->frame.dst_bpp = get_bpp(RGBA);
                        yuv = 0;
                        break;
                case v210:
                        s->frame.decoder = (decoder_t)vc_copylinev210;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        yuv = 1;
                        break;
                case DVS10:
                        s->frame.decoder = (decoder_t)vc_copylineDVS10;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        yuv = 1;
                        break;
                case Vuy2:
                case DVS8:
                case UYVY:
                        s->frame.decoder = (decoder_t)memcpy;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        yuv = 1;
                        break;
        }

        s->frame.fps = fps;
        s->frame.aux = aux;
        s->frame.src_bpp = get_bpp(codec);
        s->frame.color_spec = codec; // src (!)
        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.dst_pitch = s->frame.dst_linesize;
        s->frame.data_len = s->frame.dst_linesize * s->frame.height;
        s->frame.dst_x_offset = 0;

        if(s->sage_initialized)
                sage_shutdown();
        // warning s->frame.{width,height} !!!!
        initSage(s->appID, s->nodeID, s->frame.width, s->frame.height, yuv);
        s->sage_initialized = 1;
        s->frame.data = sage_getBuffer();

        if (s->tiles != NULL) {
                free(s->tiles);
                s->tiles = NULL;
        }
        if (aux & AUX_TILED) {
                int x, y;
                const int x_cnt = tile_info.x_count;

                s->frame.tile_info = tile_info;
                s->tiles = (struct video_frame *)
                        malloc(s->frame.tile_info.x_count *
                                        s->frame.tile_info.y_count *
                                        sizeof(struct video_frame));
                for (y = 0; y < s->frame.tile_info.y_count; ++y)
                        for(x = 0; x < s->frame.tile_info.x_count; ++x) {
                                memcpy(&s->tiles[y*x_cnt + x], &s->frame,
                                                sizeof(struct video_frame));
                                s->tiles[y*x_cnt + x].width = width;
                                s->tiles[y*x_cnt + x].height = height;
                                s->tiles[y*x_cnt + x].tile_info.pos_x = x;
                                s->tiles[y*x_cnt + x].tile_info.pos_y = y;
                                s->tiles[y*x_cnt + x].dst_x_offset +=
                                        x * width * s->frame.dst_bpp;
                                s->tiles[y*x_cnt + x].data +=
                                        y * height *  s->frame.dst_pitch;
                                s->tiles[y*x_cnt + x].src_linesize =
                                        vc_getsrc_linesize(width, codec);
                                s->tiles[y*x_cnt + x].dst_linesize =
                                        vc_getsrc_linesize((x + 1) * width, codec);

                        }
        }
        

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

static struct video_frame * get_tile_buffer(void *state, struct tile_info tile_info) 
{
        struct state_sage *s = (struct state_sage *)state;

        assert(s->tiles != NULL); /* basic sanity test... */
        if(tile_info.pos_x + tile_info.pos_y * tile_info.x_count ==
                                        s->frame.tile_info.x_count * 
                                        s->frame.tile_info.y_count - 1)
                s->put_frame = TRUE; /* this is the last tile */
        else
                s->put_frame = FALSE; /* this is not the last tile */

        return &s->tiles[tile_info.pos_x + tile_info.pos_y * tile_info.x_count];
}

static void update_tile_data(struct state_sage *s) 
{
        int x, y;
        int x_cnt;

        x_cnt = s->frame.tile_info.x_count;
        for (y = 0; y < s->frame.tile_info.y_count; ++y)
                for(x = 0; x < s->frame.tile_info.x_count; ++x)
                        s->tiles[y*x_cnt + x].data =
                                s->frame.data + 
                                y * s->tiles[y*x_cnt + x].height *
                                s->frame.dst_pitch;
}
