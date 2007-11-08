/*
 * FILE:     audio_codec/l8.c
 * AUTHORS:  Orion Hodson
 * MODIFIED: Colin Perkins
 *
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_codec_types.h"
#include "audio_codec/l8.h"

/* Note payload numbers are dynamic and selected so:
 * (a) we always have one codec that can be used at each sample rate and freq
 * (b) to backwards match earlier releases.
 */

static acodec_format_t cs[] = {
        {"Linear-8", "L8-8K-Mono",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 160, {DEV_S16,  8000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"Linear-8", "L8-8K-Stereo",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 320, {DEV_S16,  8000, 16, 2, 2 * 160 * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"Linear-8", "L8-16K-Mono",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 160, {DEV_S16,  16000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 10 ms */
        {"Linear-8", "L8-16K-Stereo",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 320, {DEV_S16,  16000, 16, 2, 2 * 160 * BYTES_PER_SAMPLE}}, /* 10 ms */
        {"Linear-8", "L8-32K-Mono",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 160, {DEV_S16,  32000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 5 ms */
        {"Linear-8", "L8-32K-Stereo",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 320, {DEV_S16,  32000, 16, 2, 2 * 160 * BYTES_PER_SAMPLE}}, /* 5 ms */
        {"Linear-8", "L8-44K-Mono",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 160, {DEV_S16,  44100, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 3.6 ms */
        {"Linear-8", "L8-44K-Stereo",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 320, {DEV_S16,  44100, 16, 2, 2 * 160 * BYTES_PER_SAMPLE}}, /* 3.6 ms */
        {"Linear-8", "L8-48K-Mono",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 160, {DEV_S16,  48000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 3.3 ms */
        {"Linear-8", "L8-48K-Stereo",  
         "Linear 8 uncompressed audio.", 
         ACODEC_PAYLOAD_DYNAMIC, 0, 320, {DEV_S16,  48000, 16, 2, 2 * 160 * BYTES_PER_SAMPLE}} /* 3.3 ms */
};

#define L8_NUM_FORMATS sizeof(cs)/sizeof(acodec_format_t)

uint16_t
l8_get_formats_count()
{
        return (uint16_t)L8_NUM_FORMATS;
}

const acodec_format_t *
l8_get_format(uint16_t idx)
{
        assert(idx < L8_NUM_FORMATS);
        return &cs[idx];
}

/*
 * "From draft-ietf-avt-profile-09.txt
 *
 * 4.5.12 L8
 *
 * L8 denotes linear audio data samples, using 8-bits of precision with
 * an offset of 128, that is, the most negative signal is encoded as
 * zero."
 */

int
l8_encode(uint16_t idx, u_char *state, sample *src, coded_unit *out)
{
	uint8_t		*dst;
	uint32_t	i, samples;

        assert(idx < L8_NUM_FORMATS);
        UNUSED(state);

        out->state     = NULL;
        out->state_len = 0;
        out->data_len  = cs[idx].mean_coded_frame_size;
        out->data      = (u_char*) malloc(out->data_len);

        samples	= out->data_len;
	dst	= (uint8_t*)out->data;
	for (i = 0; i < samples; i++) {
		dst[i] = (uint8_t)((src[i] >> 8) + 128);
	}
        return samples;
}

int
l8_decode(uint16_t idx, u_char *state, coded_unit *in, sample *dst)
{
        int samples, i;
        uint8_t *src;

        assert(idx < L8_NUM_FORMATS);
        UNUSED(state);

        samples	= in->data_len;
	src	= (uint8_t*)in->data;
	for (i = 0; i < samples; i++) {
		dst[i] = (((sample)src[i]) - 128) << 8;
	}
        return samples;
}


