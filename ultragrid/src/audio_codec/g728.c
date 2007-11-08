/*
 * FILE:    codec_g728.c
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 2000-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_codec_types.h"
#include "audio_codec/g728.h"
#include "bitstream.h"

#include "g728lib.h"

/* G728's frame size is 5 samples, RTP payload requires 4 G728 to make
 * RTP unit.  */

#define G728_SAMPLES_PER_RTPFRAME	20
#define G728_RTPFRAME_SIZE      	5
#define G728_SAMPLES_PER_FRAME  	5

/* ------------------------------------------------------------------------- */
/* Capability advertising */

static codec_format_t cs[] = {
        {	
		"G728", "G728-8K-Mono",  
		"ITU G.728 LD-CELP codec.", 
		15, 		/* Payload type */
		0, 
		G728_RTPFRAME_SIZE, 
		{
			DEV_S16,  8000, 16, 1, 
			G728_SAMPLES_PER_RTPFRAME * 
			BYTES_PER_SAMPLE
		}
	}	
};

#define G728_NUM_FORMATS sizeof(cs)/sizeof(codec_format_t)

uint16_t
g728_get_formats_count()
{
        return (uint16_t)G728_NUM_FORMATS;
}

const codec_format_t *
g728_get_format(uint16_t idx)
{
        assert(idx < G728_NUM_FORMATS);
        return &cs[idx];
}

/* ------------------------------------------------------------------------- */
/* Encoder specifics */

int 
g728_encoder_create(uint16_t idx, u_char **s)
{
        assert(idx < G728_NUM_FORMATS);
	*s = NULL;
	g728_encoder_init();
        return TRUE;
}

void
g728_encoder_destroy(uint16_t idx, u_char **s)
{
        assert(idx < G728_NUM_FORMATS);
	assert(*s == NULL);
        UNUSED(idx);
}

int
g728_encoder_do(uint16_t idx, u_char *encoder_state, sample *inbuf, coded_unit *c)
{
	bitstream_t	*bs;
	int16_t		cw;
	int		i;

        assert(encoder_state == NULL);
        assert(inbuf);
        assert(idx < G728_NUM_FORMATS);

        c->state     = NULL;
        c->state_len = 0;
        c->data      = (u_char*)block_alloc(cs[idx].mean_coded_frame_size);
        c->data_len  = cs[idx].mean_coded_frame_size;

        memset(c->data, 0, c->data_len);

	bs_create(&bs);
	bs_attach(bs, c->data, c->data_len);

	for (i = 0; i < G728_SAMPLES_PER_RTPFRAME; i += G728_SAMPLES_PER_FRAME) {
		g728_encode(&cw, inbuf + i , G728_SAMPLES_PER_FRAME);
		bs_put(bs, (u_char)(cw >> 8),   2);
		bs_put(bs, (u_char)(cw & 0xff), 8);
	}
	bs_destroy(&bs);

        return c->data_len;
}

/* ------------------------------------------------------------------------- */
/* Decoder specifics */

int 
g728_decoder_create(uint16_t idx, u_char **s)
{
        assert(idx < G728_NUM_FORMATS);
	*s = NULL;
	g728_decoder_init();
        return TRUE;
}

void
g728_decoder_destroy(uint16_t idx, u_char **s)
{
        assert(idx < G728_NUM_FORMATS);
	assert(*s == NULL);
        UNUSED(idx);
}

int
g728_decoder_do(uint16_t idx, u_char *decoder_state, coded_unit *c, sample *dst)
{
	bitstream_t	*bs;
	int16_t		cw;
        int 		i;
	
        assert(decoder_state == NULL);
        assert(c != NULL);
        assert(dst != NULL);
        assert(idx < G728_NUM_FORMATS);
        assert(c->state_len == 0);
        assert(c->data_len == cs[idx].mean_coded_frame_size);

	bs_create(&bs);
	bs_attach(bs, c->data, c->data_len);
	for(i = 0; i < G728_SAMPLES_PER_RTPFRAME / G728_SAMPLES_PER_FRAME; i++) {
		cw = (bs_get(bs, 2) << 8) | bs_get(bs, 8);
		g728_decode(dst + i * G728_SAMPLES_PER_FRAME, G728_SAMPLES_PER_FRAME, &cw);
	}	
	bs_destroy(&bs);

	return c->data_len;
}




