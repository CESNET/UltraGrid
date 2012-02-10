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
#include <OpenGL/gl.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <string.h>
#include <stdlib.h>


#if OS_VERSION_MAJOR < 11
#warning "You are compling on pre-10.7 Mac OS X version. Core OpenGL 3.2 profile won't work"
#warning "in case you'll try to use this binary with Lion or higher (only legacy OpenGL)."
#endif

#define MAC_GL_MAGIC 0xa23f4f28u

struct state_mac_gl;

int get_mac_kernel_version_major(void);

int get_mac_kernel_version_major()
{
        int mib[2];
        size_t len;
        char *kernelVersion;

        // Get the kernel's version as a string called "kernelVersion":
        mib[0] = CTL_KERN;
        mib[1] = KERN_OSRELEASE;
        sysctl(mib, 2, NULL, &len, NULL, 0);
        kernelVersion = malloc(len * sizeof(char));
        sysctl(mib, 2, kernelVersion, &len, NULL, 0);

        return atoi(kernelVersion);
}

@interface UltraGridOpenGLView : NSOpenGLView
{
        NSWindow *window;
}

-(void) initialize: (struct state_mac_gl *) s;
@end

struct state_mac_gl
{
        uint32_t magic;
        UltraGridOpenGLView *view;
        NSAutoreleasePool *autoreleasePool;
        NSOpenGLPixelFormat * pixFmt;
};

void *mac_gl_init(mac_opengl_version_t ogl_version)
{
        struct state_mac_gl *s;
        NSOpenGLPixelFormatAttribute attrs[3];
        int mac_version_major;

        mac_version_major = get_mac_kernel_version_major();

        if(ogl_version == MAC_GL_PROFILE_3_2) { /* Lion or later */
#if OS_VERSION_MAJOR >= 11
                if(mac_version_major < 11) {
                        fprintf(stderr, "[Mac OpenGL] Unable to activate OpenGL 3.2 for pre-Lion Macs.\n");
                        return NULL;
                }
                attrs[0] = kCGLPFAOpenGLProfile;
                attrs[1] = kCGLOGLPVersion_3_2_Core; // kCGLOGLPVersion_Legacy;
                attrs[2] = 0;
#else
                fprintf(stderr, "[Mac OpenGL] OpenGL 3.2 support was not compiled in! Did you compile with an older Mac?\n");
                return NULL;
#endif
        } else if(ogl_version == MAC_GL_PROFILE_LEGACY) {
#if OS_VERSION_MAJOR >= 11
                if(mac_version_major >= 11) {
                        attrs[0] = kCGLPFAOpenGLProfile;
                        attrs[1] = kCGLOGLPVersion_Legacy; // kCGLOGLPVersion_Legacy;
                        attrs[2] = 0;
                }
#else
                attrs[0] = 0;
#endif
        }

        s = (struct state_mac_gl *) malloc(sizeof(struct state_mac_gl));
        s->magic = MAC_GL_MAGIC;

        NSApplicationLoad();

        NSApp = [NSApplication sharedApplication];

        s->autoreleasePool = [[NSAutoreleasePool alloc] init];

        if(mac_version_major >= 11) {
                s->pixFmt = [[NSOpenGLPixelFormat alloc] initWithAttributes: attrs];
        } else {
                s->pixFmt = [NSOpenGLView defaultPixelFormat];
        }

        if(!s->pixFmt) {
                fprintf(stderr, "[Mac OpenGL] Failed to acquire pixel format.\n");
                return NULL;
        }

        s->view = [[UltraGridOpenGLView alloc] initWithFrame:NSMakeRect(0, 0, 100, 100) pixelFormat: s->pixFmt];

        [ s->view display];
        [ s->view initialize: s];

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
-(void) initialize: (struct state_mac_gl *) s
{
        NSOpenGLContext *context;

        window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 100, 100)
                                             styleMask:NSBorderlessWindowMask
                                               backing:NSBackingStoreBuffered
                                                 defer:NO];
        [window autorelease];
        //[window setTitle:@"UltraGrid OpenGL view"];

        context = [[NSOpenGLContext alloc] initWithFormat: s->pixFmt
                                             shareContext:nil];
        [self setOpenGLContext:context];
        [context makeCurrentContext];
        [context release];
        printf("OpenGL renderer: %s %s (GLSL: %s)\n", glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

        [window setContentSize:NSMakeSize(100, 100)];
}
@end

