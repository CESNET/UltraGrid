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
#ifndef __video_decompress_h

#define __video_decompress_h
#include "video_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

struct state_decompress;

#define DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME  1          /* int */

/**
 * initializes decompression and returns internal state
 */
typedef  void *(*decompress_init_t)();
/**
 * Recompresses decompression for specified video description
 */
typedef  int (*decompress_reconfigure_t)(void * state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec);
/**
 * Decompresses data from buffer of src_len into dst
 */
typedef int (*decompress_decompress_t)(void *state, unsigned char *dst,
                unsigned char *buffer, unsigned int src_len, int frame_seq);

/**
 * @param state decoder state
 * @param property  ID of queried property
 * @param val return value
 * @param len  IN - max bytes that may be written to val
 *             OUT - number of bytes actually written
 */
typedef  int (*decompress_get_property_t)(void *state, int property, void *val, size_t *len);

/**
 * Cleanup function
 */
typedef  void (*decompress_done_t)(void *);


struct decode_from_to {
        codec_t from;
        codec_t to;

        uint32_t decompress_index;
        /* priority to select this decoder if there are multiple matches
         * range [0..100], lower is better
         */
        int priority;
};
extern struct decode_from_to decoders_for_codec[];
extern const int decoders_for_codec_count;

/**
 * must be called before initalization of decoders
 */
void initialize_video_decompress(void);

/**
 * Checks wheather there is decompressor with given magic present and thus can
 * be initialized with decompress_init
 *
 * @see decompress_init
 * @retval TRUE if decoder is present and can be initialized
 * @retval FALSE if decoder could not be initialized (not found)
 */
int decompress_is_available(unsigned int decoder_index);

/**
 * Initializes decompressor or the given magic
 *
 * @retval NULL if initialization failed
 * @retval not-NULL state of new decompressor
 */
struct state_decompress *decompress_init(unsigned int magic);
int decompress_reconfigure(struct state_decompress *, struct video_desc, int rshift, int gshift, int bshift, int pitch, codec_t out_codec);
/**
 * @param frame_seq sequential number of frame
 * @retval TRUE if decompressed successfully
 * @retval FALSE if decompressing failed
 */
int decompress_frame(struct state_decompress *, unsigned char *dst,
                unsigned char *src, unsigned int src_len, int frame_seq);
/**
 * For description see above - decompress_get_property_t
 */
int decompress_get_property(struct state_decompress *state, int property, void *val, size_t *len);
void decompress_done(struct state_decompress *);

#ifdef __cplusplus
}
#endif

#endif /* __video_decompress_h */
