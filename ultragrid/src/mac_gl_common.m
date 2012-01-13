/*
 * FILE:    mac_gl_common.c
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

#include "mac_gl_common.h"

#import <Cocoa/Cocoa.h>


#define MAC_GL_MAGIC 0xa23f4f28u

@interface UltraGridOpenGLView : NSOpenGLView
{
        NSWindow *window;
}

-(void) initialize;
@end

struct state_mac_gl
{
        uint32_t magic;
        UltraGridOpenGLView *view;
        NSAutoreleasePool *autoreleasePool;
};

void *mac_gl_init(void)
{
        struct state_mac_gl *s;

        s = (struct state_mac_gl *) malloc(sizeof(struct state_mac_gl));
        s->magic = MAC_GL_MAGIC;

        NSApplicationLoad();

        NSApp = [NSApplication sharedApplication];

        s->autoreleasePool = [[NSAutoreleasePool alloc] init];

        s->view = [[UltraGridOpenGLView alloc] initWithFrame:NSMakeRect(0, 0, 100, 100) pixelFormat: [UltraGridOpenGLView defaultPixelFormat]];
        [ s->view display];
        [ s->view initialize];

        return s;
}

void mac_gl_free(void * state)
{
        struct state_mac_gl *s = (struct state_mac_gl *) state;

        [ s->view release ];
        [ s->autoreleasePool release ];
        free(s);
}

@implementation UltraGridOpenGLView
-(void) initialize
{
        NSOpenGLContext *context;

        window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 100, 100)
                                             styleMask:NSBorderlessWindowMask
                                               backing:NSBackingStoreBuffered
                                                 defer:NO];
        [window autorelease];
        //[window setTitle:@"UltraGrid OpenGL view"];

        context = [[NSOpenGLContext alloc] initWithFormat: [NSOpenGLView defaultPixelFormat]
                                             shareContext:nil];
        [self setOpenGLContext:context];
        [context makeCurrentContext];
        [context release];

        [window setContentSize:NSMakeSize(100, 100)];
}
@end

