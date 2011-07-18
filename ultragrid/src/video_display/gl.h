/*
 * FILE:    video_display/gl.h
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
 * 3. All rertising materials mentioning features or use of this software
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

#include <video_codec.h>

#define DISPLAY_GL_ID  0xba370a2a

display_type_t          *display_gl_probe(void);
void                    *display_gl_init(char *fmt);
void                     display_gl_run(void *state);
void                     display_gl_done(void *state);
struct video_frame      *display_gl_getf(void *state);
int                      display_gl_putf(void *state, char *frame);

int                      display_gl_handle_events(void *arg);


// source code for a shader unit (xsedmik)
static char * glsl = "\
uniform sampler2D Ytex;\
uniform sampler2D Utex,Vtex;\
\
void main(void) {\
float nx,ny,r,g,b,y,u,v;\
vec4 txl,ux,vx;\
nx=gl_TexCoord[0].x;\
ny=gl_TexCoord[0].y;\
y=texture2D(Ytex,vec2(nx,ny)).r;\
u=texture2D(Utex,vec2(nx,ny)).r;\
v=texture2D(Vtex,vec2(nx,ny)).r;\
y=1.1643*(y-0.0625);\
u=u-0.5;\
v=v-0.5;\
r=y+1.5958*v;\
g=y-0.39173*u-0.81290*v;\
b=y+2.017*u;\
gl_FragColor=vec4(r,g,b,1.0);\
}";

/* DXT related -- there shoud be only one kernel 
 * since both do basically the same thing */
static char * frag = "\
uniform sampler2D yuvtex;\
\
void main(void) {\
vec4 col = texture2D(yuvtex, gl_TexCoord[0].st);\
\
float Y = col[0];\
float U = col[1]-0.5;\
float V = col[2]-0.5;\
Y=1.1643*(Y-0.0625);\
\
float G = Y-0.39173*U-0.81290*V;\
float B = Y+2.017*U;\
float R = Y+1.5958*V;\
\
gl_FragColor=vec4(R,G,B,1.0);}";

static char * vert = "\
void main() {\
gl_TexCoord[0] = gl_MultiTexCoord0;\
gl_Position = ftransform();}";

