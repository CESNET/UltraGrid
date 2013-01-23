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

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "x11_common.h"
#include <pthread.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

static pthread_once_t XInitThreadsHasRun = PTHREAD_ONCE_INIT;
static volatile int threads_init = FALSE;
static Display *display = NULL;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int display_opened_here = TRUE; /* indicates wheather we opened the display
                                          in such a case, we count references and if 0,
                                          we close it */
static int ref_num = 0;

static void _x11_set_display(void *disp);
static void _x11_enter_thread(void);
void _x11_unused(void);
static void * _x11_acquire_display(void);
static void * _x11_get_display(void);
static void _x11_release_display(void);
static void _x11_lock(void);
static void _x11_unlock(void);

static void _x11_enter_thread(void)
{
        pthread_mutex_lock(&lock);
        pthread_once(&XInitThreadsHasRun, (void ((*)(void)))XInitThreads);
        threads_init = TRUE;
        pthread_mutex_unlock(&lock);
}
void (*x11_enter_thread)(void) = _x11_enter_thread;

static void _x11_set_display(void *disp)
{
        Display *d = disp;
        if (d == NULL)
                return;
        pthread_mutex_lock(&lock);
        if(display != NULL) {
                fprintf(stderr, __FILE__ ": Fatal error: Display already set.\n");
                abort();
        }
        if(threads_init == FALSE) {
                fprintf(stderr, __FILE__ ": WARNING: Doesn't entered threads. Please report a bug.\n");
        }
        display = d;
        display_opened_here = FALSE;
        pthread_mutex_unlock(&lock);
}

static void * _x11_acquire_display(void)
{
        if(!display) {
                display = XOpenDisplay(0);
                display_opened_here = TRUE;
        }
        
        if ( !display )
        {
                fprintf(stderr, "Failed to open X display\n" );
                return NULL;
        }
        
        ref_num++;
        return display;
}
void * (*x11_acquire_display)(void) = _x11_acquire_display;

static void * _x11_get_display(void)
{
        return display;
}
void * (*x11_get_display)(void) = _x11_get_display;

static void _x11_release_display() {
        ref_num--;
        
        if(ref_num < 0) {
                fprintf(stderr, __FILE__ ": WARNING: Unpaired glx_free call.");
        }
        
        if(display_opened_here && ref_num == 0) {
                fprintf(stderr, "Display closed (last client disconnected)\n");
                XCloseDisplay( display );
                display = NULL;
        }
}
void (*x11_release_display)(void) = _x11_release_display;

void (*x11_set_display)(void *) = _x11_set_display;

static void _x11_lock(void)
{
        pthread_mutex_lock(&lock);
}

void (*x11_lock)(void) = _x11_lock;

static void _x11_unlock(void)
{
        pthread_mutex_unlock(&lock);
}

void (*x11_unlock)(void) = _x11_unlock;

/* used only to force compilator to export symbols */
void _x11_unused()
{
        x11_enter_thread();
        x11_set_display(0);
        x11_lock();
        x11_unlock();
        x11_acquire_display();
        x11_release_display();
        x11_get_display();
}

