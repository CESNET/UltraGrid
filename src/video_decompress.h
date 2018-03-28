/**
 * @file video_decompress.h
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @ingroup video_decompress
 * @brief API for video decompress drivers
 */
/*
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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

#define VIDEO_DECOMPRESS_ABI_VERSION 4

/**
 * @defgroup video_decompress Video Decompress
 * @{
 */
#ifndef __video_decompress_h
#define __video_decompress_h

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct state_decompress;

/**
 * This property tells that even broken frame (with missing data)
 * can be passed to decompressor. Otherwise, broken frame is discarded.
 */
#define DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME  1          /* int */

/**
 * initializes decompression and returns internal state
 */
typedef  void *(*decompress_init_t)();
/**
 * Recompresses decompression for specified video description
 * @param[in] desc      video description
 * @param[in] rshift    requested output red shift (if output is RGB(A))
 * @param[in] gshift    requested output green shift
 * @param[in] bshift    requested output blue shift
 * @param[in] pitch     requested output pitch
 * @param[in] out_codec requested output pixelformat
 * @retval FALSE        if reconfiguration failed
 * @retval TRUE         if reconfiguration succeeded
 */
typedef  int (*decompress_reconfigure_t)(void * state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec);

typedef enum {
        DECODER_NO_FRAME = 0, //Frame not decoded yet 
        DECODER_GOT_FRAME,    //Frame decoded and written to destination
        DECODER_CANT_DECODE   //Decoder can't decode to selected out_codec
} decompress_status;

/**
 * @brief Decompresses video frame
 * @param[in]  state         decompress state
 * @param[out] dst           buffer where uncompressed frame will be written
 * @note
 * Length of the result isn't returned because is known from the informations
 * passed with decompress_reconfigure.
 * @param[in] buffer         buffer where uncompressed frame will be written
 * @param[in] src_len        length of the compressed buffer
 * @param[in] frame_seq      sequential number of frame. Subsequent frames
 *                           has sequential number +1. The point is to signalize
 *                           decompressor when one or more frames got lost (interframe compress).
 * @note
 * Currently used perhaps only for VP8, H.264 uses Periodic Intra Refresh.
 * @retval    DECODER_GOT_FRAME        if decompressed successfully
 * @retval    DECODER_NO_FRAME         if the frame wasn't decoded yet
 * @retval    DECODER_CANT_DECODE      if decoding to selected out_codec failed
 */
typedef decompress_status (*decompress_decompress_t)(void *state, unsigned char *dst,
                unsigned char *buffer, unsigned int src_len, int frame_seq);

/**
 * @param state decoder state
 * @param property  ID of queried property
 * @param val return value
 * @param[in] len max bytes that may be written to val
 * @param[out] len number of bytes actually written
 */
typedef  int (*decompress_get_property_t)(void *state, int property, void *val, size_t *len);

/**
 * Cleanup function
 */
typedef  void (*decompress_done_t)(void *);

/**
 * Struct of this type defines decompressor for codec to specified output codec
 */
struct decode_from_to {
        codec_t from; ///< input (compressed) codec
        codec_t to;   ///< output pixelformat

        /** Priority to select this decoder if there are multiple matches for
         * specified compress/pixelformat pair.
         * Range is [0..1000], lower is better.
         */
        int priority;
};

typedef const struct decode_from_to *(*decompress_get_available_decoders_t)(void);

struct video_decompress_info {
        decompress_init_t init;
        decompress_reconfigure_t reconfigure;
        decompress_decompress_t decompress;
        decompress_get_property_t get_property;
        decompress_done_t done;
        decompress_get_available_decoders_t get_available_decoders;
};

bool decompress_init_multi(codec_t from, codec_t to, struct state_decompress **out, int count);
int decompress_reconfigure(struct state_decompress *, struct video_desc, int rshift, int gshift, int bshift, int pitch, codec_t out_codec);
decompress_status decompress_frame(struct state_decompress *, unsigned char *dst,
                unsigned char *src, unsigned int src_len, int frame_seq);
int decompress_get_property(struct state_decompress *state, int property, void *val, size_t *len);
void decompress_done(struct state_decompress *);

#ifdef __cplusplus
}
#endif

#endif /* __video_decompress_h */
/** @} */ // end of video_decompress

