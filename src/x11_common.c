/*
 * FILE:    x11_common.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "x11_common.h"

#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "debug.h"
#include "utils/resource_manager.h"

#define resource_symbol "X11-state"

struct x11_state {
        pthread_once_t XInitThreadsHasRun;
        volatile int threads_init;
        Display *display;
        pthread_mutex_t lock;
        int display_opened_here; /* indicates wheather we opened the display
                                          in such a case, we count references and if 0,
                                          we close it */
        int ref_num;
        bool initialized;
};

struct x11_state *get_state(void);

struct x11_state *get_state() {
        struct x11_state *state;
        rm_lock();
        state = (struct x11_state *) rm_get_shm(resource_symbol, sizeof(struct x11_state));
        if(!state->initialized) {
                state->XInitThreadsHasRun = PTHREAD_ONCE_INIT;
                state->threads_init = FALSE;
                state->display = NULL;
                pthread_mutex_init(&state->lock, NULL);
                state->display_opened_here = TRUE;
                state->ref_num = 0;
                state->initialized = true;
        }
        rm_unlock();

        return state;
}

void x11_enter_thread(void)
{
        struct x11_state *s = get_state();
        pthread_mutex_lock(&s->lock);
        pthread_once(&s->XInitThreadsHasRun, (void ((*)(void)))XInitThreads);
        s->threads_init = TRUE;
        pthread_mutex_unlock(&s->lock);
}

void x11_set_display(void *disp)
{
        struct x11_state *s = get_state();
        Display *d = disp;
        if (d == NULL)
                return;
        pthread_mutex_lock(&s->lock);
        if(s->display != NULL) {
                fprintf(stderr, __FILE__ ": Fatal error: Display already set.\n");
                abort();
        }
        if(s->threads_init == FALSE) {
                fprintf(stderr, __FILE__ ": WARNING: Doesn't entered threads. Please report a bug.\n");
        }
        s->display = d;
        s->display_opened_here = FALSE;
        pthread_mutex_unlock(&s->lock);
}

void * x11_acquire_display(void)
{
        struct x11_state *s = get_state();
        if(!s->display) {
                s->display = XOpenDisplay(0);
                s->display_opened_here = TRUE;
        }
        
        if ( !s->display )
        {
                fprintf(stderr, "Failed to open X display\n" );
                return NULL;
        }
        
        s->ref_num++;
        return s->display;
}

void * x11_get_display(void)
{
        struct x11_state *s = get_state();
        return s->display;
}

void x11_release_display() {
        struct x11_state *s = get_state();
        s->ref_num--;
        
        if(s->ref_num < 0) {
                fprintf(stderr, __FILE__ ": WARNING: Unpaired glx_free call.");
        }
        
        if(s->display_opened_here && s->ref_num == 0) {
                fprintf(stderr, "Display closed (last client disconnected)\n");
                XCloseDisplay( s->display );
                s->display = NULL;
        }
}

void x11_lock(void)
{
        struct x11_state *s = get_state();
        pthread_mutex_lock(&s->lock);
}

void x11_unlock(void)
{
        struct x11_state *s = get_state();
        pthread_mutex_unlock(&s->lock);
}

