/*
 * FILE:    video_codec.h
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
#ifndef __video_codec_h

#define __video_codec_h

typedef enum {
        RGBA,
        UYVY,
        Vuy2,
        DVS8,
        R10k,
        v210,
        DVS10,
} codec_t;


struct video_frame {
        codec_t              color_spec;
        unsigned int         width;
        unsigned int         height;
        char                 *data; /* this is not beginning of the frame buffer actually but beginning of displayed data,
                                     * it is the case display is centered in larger window, 
                                     * i.e., data = pixmap start + x_start + y_start*linesize
                                     */
        unsigned int         data_len; /* relative to data pos, not framebuffer size! */      
        unsigned int         dst_linesize; /* framebuffer pitch */
        unsigned int         dst_pitch; /* framebuffer pitch - it can be larger if SDL resolution is larger than data */
        unsigned int         src_linesize; /* display data pitch */
        unsigned int         dst_x_offset; /* X offset in frame buffer */
        double               src_bpp;
        double               dst_bpp;
        int                  rshift;
        int                  gshift;
        int                  bshift;
        void                 (*decoder)(unsigned char *dst, unsigned char *src, int len, int rshift, int gshift, int bshift);
        void                 (*reconfigure)(void *state, int width, int height, codec_t color_spec);
        void                 *state;
};


struct codec_info_t {
        codec_t codec;
        const char *name;
        unsigned int fcc;
        int h_align;
        double bpp;
        unsigned rgb:1;
} codec_info_t;

extern const struct codec_info_t codec_info[];

void show_codec_help(void);
double get_bpp(codec_t codec);

int vc_getsrc_linesize(unsigned int width, codec_t codec);

void vc_deinterlace(unsigned char *src, long src_linesize, int lines);
void vc_copylineDVS10(unsigned char *dst, unsigned char *src, int src_len);
void vc_copylinev210(unsigned char *dst, unsigned char *src, int dst_len);
void vc_copyliner10k(unsigned char *dst, unsigned char *src, int len, int rshift, int gshift, int bshift);
void vc_copylineRGBA(unsigned char *dst, unsigned char *src, int len, int rshift, int gshift, int bshift);

#endif
