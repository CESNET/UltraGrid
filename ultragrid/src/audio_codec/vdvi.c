/*
 * FILE:    codec_vdvi.c
 * AUTHORS: Orion Hodson
 *          Colin Perkins
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
#include "bitstream.h"
#include "audio_types.h"
#include "audio_codec_types.h"
#include "audio_codec/vdvi.h"
#include "audio_codec/vdvi_impl.h"
#include "audio_codec/dvi_impl.h"

static acodec_format_t cs[] = {
        {"VDVI", "VDVI-8K-Mono",  
         "Variable Rate IMA ADPCM codec.", 
         77, 	/* Payload type */
	 4, 
	 80, 
         {DEV_S16,  8000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"VDVI", "VDVI-16K-Mono",  
         "Variable Rate IMA ADPCM codec.", 
         78, 	/* Payload type */
	 4, 
	 80, 
         {DEV_S16, 16000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 10  ms */
        {"VDVI", "VDVI-32K-Mono",  
         "Variable Rate IMA ADPCM codec.", 
         79, 	/* Payload type */
	 4, 
	 80, 
         {DEV_S16, 32000, 16, 1, 160 * BYTES_PER_SAMPLE}}, /* 5   ms */
        {"VDVI", "VDVI-48K-Mono",  
         "Variable Rate IMA ADPCM codec.", 
         80, 	/* Payload type */
	 4, 
	 80, 
         {DEV_S16, 48000, 16, 1, 160 * BYTES_PER_SAMPLE}}  /* 3.3 ms */
};

#define VDVI_NUM_FORMATS sizeof(cs)/sizeof(acodec_format_t)

uint16_t
vdvi_get_formats_count()
{
        return (uint16_t)VDVI_NUM_FORMATS;
}

const acodec_format_t *
vdvi_get_format(uint16_t idx)
{
        assert(idx < VDVI_NUM_FORMATS);
        return &cs[idx];
}

typedef struct {
        struct adpcm_state *as;
        bitstream_t        *bs;
} vdvi_state_t;

int 
vdvi_state_create(uint16_t idx, u_char **s)
{
        vdvi_state_t *v;

        if (idx < VDVI_NUM_FORMATS) {
                v = (vdvi_state_t*) malloc(sizeof(vdvi_state_t));
                if (v == NULL) {
                        return FALSE;
                }
                v->as = (struct adpcm_state*) malloc(sizeof(struct adpcm_state));
                if (v->as == NULL) {
                        free(v);
                        return FALSE;
                }
                memset(v->as, 0, sizeof(struct adpcm_state));
                if (bs_create(&v->bs) == FALSE) {
                        free(v->as);
                        free(v);
                        return FALSE;
                }
                *s = (unsigned char*)v;
                return TRUE;
        }
        return 0;
}

void
vdvi_state_destroy(uint16_t idx, u_char **s)
{
        vdvi_state_t *v;

        v = (vdvi_state_t*)*s;
        free(v->as);
        bs_destroy(&v->bs);
        free(v);
        *s = (u_char*)NULL;
        UNUSED(idx);
}

/* Buffer of maximum length of vdvi coded data - never know how big
 * it needs to be
 */

int
vdvi_encoder(uint16_t idx, u_char *encoder_state, sample *inbuf, coded_unit *c)
{
        int samples, len;

        u_char dvi_buf[80];
        u_char vdvi_buf[160];
        vdvi_state_t *v;

        assert(encoder_state);
        assert(inbuf);
        assert(idx < VDVI_NUM_FORMATS);
        UNUSED(idx);

        v = (vdvi_state_t*)encoder_state;
        
        /* Transfer state and fix ordering */
        c->state     = (u_char*) malloc(sizeof(struct adpcm_state));
        c->state_len = sizeof(struct adpcm_state);
        memcpy(c->state, v->as, sizeof(struct adpcm_state));

        /* Fix coded state for byte ordering */
	((struct adpcm_state*)c->state)->valprev = htons(((struct adpcm_state*)c->state)->valprev);
        
        samples = cs[idx].format.bytes_per_block * 8 / cs[idx].format.bits_per_sample;
        
        assert(samples == 160);

        adpcm_coder(inbuf, dvi_buf, samples, v->as);

        bs_attach(v->bs, vdvi_buf, sizeof(vdvi_buf)/sizeof(vdvi_buf[0]));
        memset(vdvi_buf, 0, sizeof(vdvi_buf)/sizeof(vdvi_buf[0]));
        len = vdvi_encode(dvi_buf, 160, v->bs);
        c->data     = (u_char*) malloc(len); 
        c->data_len = len;
        memcpy(c->data, vdvi_buf, len);

        return len;
}

int
vdvi_decoder(uint16_t idx, u_char *decoder_state, coded_unit *c, sample *data)
{
        int samples, len; 
        u_char dvi_buf[80];
        vdvi_state_t *v;

        assert(decoder_state);
        assert(c);
        assert(data);
        assert(idx < VDVI_NUM_FORMATS);

        v = (vdvi_state_t*)decoder_state;

	if (c->state_len > 0) {
		assert(c->state_len == sizeof(struct adpcm_state));
		memcpy(v->as, c->state, sizeof(struct adpcm_state));
		v->as->valprev = ntohs(v->as->valprev);
	}

        bs_attach(v->bs, c->data, c->data_len);
        len = vdvi_decode(v->bs, dvi_buf, 160);

        samples = cs[idx].format.bytes_per_block / sizeof(sample);
	adpcm_decoder(dvi_buf, data, samples, v->as);

        return samples;
}

int
vdvi_peek_frame_size(uint16_t idx, u_char *data, int data_len)
{
        bitstream_t *bs;
        u_char       dvi_buf[80];
        int          len;

        UNUSED(idx);

        bs_create(&bs);
        bs_attach(bs, data, data_len);
        len = vdvi_decode(bs, dvi_buf, 160);
        bs_destroy(&bs);
        assert(len <= data_len);
        return len;
}

