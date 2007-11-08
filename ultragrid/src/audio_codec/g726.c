/*
 * FILE:    codec_g726.c
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
 
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_codec_types.h"
#include "audio_codec/g726.h"
#include "audio_codec/g726_impl.h"
#include "bitstream.h"

#define G726_SAMPLES_PER_FRAME 160

static acodec_format_t cs[] = {
        /* G726-40 **********************************************/
        {"G726-40", "G726-40-8K-Mono",  
         "ITU G.726-40 ADPCM codec. Sun Microsystems public implementation.", 
         107, 	/* Payload type */
	 0, 100, 
         {DEV_S16,  8000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"G726-40", "G726-40-16K-Mono",  
         "ITU G.726-40 ADPCM codec. Sun Microsystems public implementation.", 
         108, 	/* Payload type */
	 0, 100, 
         {DEV_S16, 16000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 10  ms */
        {"G726-40", "G726-40-32K-Mono",  
         "ITU G.726-40 ADPCM codec. Sun Microsystems public implementation.", 
         110, 	/* Payload type */
	 0, 100, 
         {DEV_S16, 32000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 5  ms */
        {"G726-40", "G726-40-48K-Mono",  
         "ITU G.726-40 ADPCM codec. Sun Microsystems public implementation.", 
         120, 	/* Payload type */
	 0, 100, 
         {DEV_S16, 48000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 3.3 ms */
        /* G726-32 ***********************************************/
        {"G726-32", "G726-32-8K-Mono",  
         "ITU G.726-32 ADPCM codec. Sun Microsystems public implementation.", 
         2,   	/* Payload type */
	 0, 80, 
         {DEV_S16,  8000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"G726-32", "G726-32-16K-Mono",  
         "ITU G.726-32 ADPCM codec. Sun Microsystems public implementation.", 
         104, 	/* Payload type */
	 0, 80, 
         {DEV_S16, 16000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 10  ms */
        {"G726-32", "G726-32-32K-Mono",  
         "ITU G.726-32 ADPCM codec. Sun Microsystems public implementation.", 
         105, 	/* Payload type */
	 0, 80, 
         {DEV_S16, 32000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 5  ms */
        {"G726-32", "G726-32-48K-Mono",  
         "ITU G.726-32 ADPCM codec. Sun Microsystems public implementation.", 
         106, 	/* Payload type */
	 0, 80, 
         {DEV_S16, 48000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 3.3 ms */
        /* Entries 0-3 G726-24 ***********************************************/
        {"G726-24", "G726-24-8K-Mono",  
         "ITU G.726-24 ADPCM codec. Sun Microsystems public implementation.", 
         100, 	/* Payload type */
	 0, 60, 
         {DEV_S16,  8000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"G726-24", "G726-24-16K-Mono",  
         "ITU G.726-24 ADPCM codec. Sun Microsystems public implementation.", 
         101, 	/* Payload type */
	 0, 60, 
         {DEV_S16, 16000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 10  ms */
        {"G726-24", "G726-24-32K-Mono",  
         "ITU G.726-24 ADPCM codec. Sun Microsystems public implementation.", 
         102, 	/* Payload type */
	 0, 60, 
         {DEV_S16, 32000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 5  ms */
        {"G726-24", "G726-24-48K-Mono",  
         "ITU G.726-24 ADPCM codec. Sun Microsystems public implementation.", 
         103, 	/* Payload type */
	 0, 60, 
         {DEV_S16, 48000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 3.3 ms */
        /* G726-16 ***********************************************/
        {"G726-16", "G726-16-8K-Mono",  
         "ITU G.726-16 ADPCM codec. Marc Randolph modified Sun Microsystems public implementation.", 
         96, 	/* Payload type */
	 0, 40, 
         {DEV_S16,  8000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 20  ms */
        {"G726-16", "G726-16-16K-Mono",  
         "ITU G.726-16 ADPCM codec. Marc Randolph modified Sun Microsystems public implementation.", 
         97, 	/* Payload type */
	 0, 40, 
         {DEV_S16, 16000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}}, /* 10  ms */
        {"G726-16", "G726-16-32K-Mono",  
         "ITU G.726-16 ADPCM codec. Marc Randolph modified Sun Microsystems public implementation.", 
         98, 	/* Payload type */
	 0, 40, 
         {DEV_S16, 32000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 5  ms */
        {"G726-16", "G726-16-48K-Mono",  
         "ITU G.726-16 ADPCM codec. Marc Randolph modified Sun Microsystems public implementation.", 
         99, 	/* Payload type */
	 0, 40, 
         {DEV_S16, 48000, 16, 1, G726_SAMPLES_PER_FRAME * BYTES_PER_SAMPLE}},  /* 3.3 ms */
};

#define G726_16        3
#define G726_24        2
#define G726_32        1
#define G726_40        0

#define G726_NUM_FORMATS sizeof(cs)/sizeof(acodec_format_t)

/* In G726_NUM_RATES, 4 one for 16, 24, 32, 48 */
#define G726_NUM_RATES   (G726_NUM_FORMATS / 4)

typedef struct {
        struct g726_state  *gs;
} g726_t;

uint16_t
g726_get_formats_count()
{
        return (uint16_t)G726_NUM_FORMATS;
}

const acodec_format_t *
g726_get_format(uint16_t idx)
{
        assert(idx < G726_NUM_FORMATS);
        return &cs[idx];
}

int 
g726_state_create(uint16_t idx, u_char **s)
{
        g726_t *g;

        if (idx >=  G726_NUM_FORMATS) {
                return FALSE;
        }

        g = (g726_t*) malloc(sizeof(g726_t));
        if (g == NULL) {
                return FALSE;
        }

        g->gs = (struct g726_state*) malloc(sizeof(struct g726_state));
        if (g->gs == NULL) {
                        free(g);
                        return FALSE;
        }
        g726_init_state(g->gs);

        *s = (u_char*)g;

        return TRUE;
}

void
g726_state_destroy(uint16_t idx, u_char **s)
{
        g726_t *g;

        assert(idx < G726_NUM_FORMATS);

        g = (g726_t*)*s;
        free(g->gs);
        free(g);
        *s = (u_char*)NULL;

        UNUSED(idx);
}

/* G726 packing is little endian (i.e. gratuitously painful on modern
 * machines) */

static int
g726_pack(u_char *buf, u_char *cw, u_char num_cw, int bps)
{
	int i, bits = 0, x = 0;

	for (i = 0; i < num_cw; i++) {
		buf[x] |= cw[i] << bits;
		bits += bps;
		assert((bits != 8) || (i == num_cw - 1));
		if (bits > 8) {
			bits &= 0x07;
			x++;
			buf[x] |= cw[i] >> (bps - bits);
		}
	}
	return (num_cw * bps / 8);
}

static int
g726_unpack(u_char *cw, u_char *buf, u_char num_cw, int bps) 
{
	int i = 0, bits = 0, x = 0;
	u_char mask = 0;

	while (i < bps) {
		mask |= 1 << i;
		i++;
	}

	for(i = 0; i < num_cw; i++) {
		cw[i] = (buf[x] >> bits) & mask;
		bits += bps;
		assert((bits != 8) || (i == num_cw - 1));
		if (bits > 8) {
			bits &= 0x07;
			x++;
			cw[i] |= buf[x] << (bps - bits);
			cw[i] &= mask;
		}
	}
	return (num_cw * bps / 8);
}

int
g726_encode(uint16_t idx, u_char *encoder_state, sample *inbuf, coded_unit *c)
{
        register sample *s;
        g726_t *g;
        int     i;
        u_char  cw[8]; /* Maximum of 8 codewords in octet aligned packing */
	u_char *out;

        assert(encoder_state);
        assert(inbuf);
        assert(idx < G726_NUM_FORMATS);

        s = inbuf;
        g = (g726_t*)encoder_state;

        c->state     = NULL;
        c->state_len = 0;
        c->data      = (u_char*) malloc(cs[idx].mean_coded_frame_size);
        c->data_len  = cs[idx].mean_coded_frame_size;

        memset(c->data, 0, c->data_len);
	out = c->data;

        idx = idx / G726_NUM_RATES;
        switch(idx) {
        case G726_16:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 4) {
			cw[0] = g726_16_encoder(s[i], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[1] = g726_16_encoder(s[i + 1], AUDIO_ENCODING_LINEAR, g->gs);
			cw[2] = g726_16_encoder(s[i + 2], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[3] = g726_16_encoder(s[i + 3], AUDIO_ENCODING_LINEAR, g->gs);
			out += g726_pack(out, cw, 4, 2);
                }
                break;
        case G726_24:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 8) {
			cw[0] = g726_24_encoder(s[i], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[1] = g726_24_encoder(s[i + 1], AUDIO_ENCODING_LINEAR, g->gs);
			cw[2] = g726_24_encoder(s[i + 2], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[3] = g726_24_encoder(s[i + 3], AUDIO_ENCODING_LINEAR, g->gs);
			cw[4] = g726_24_encoder(s[i + 4], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[5] = g726_24_encoder(s[i + 5], AUDIO_ENCODING_LINEAR, g->gs);
			cw[6] = g726_24_encoder(s[i + 6], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[7] = g726_24_encoder(s[i + 7], AUDIO_ENCODING_LINEAR, g->gs);
			out += g726_pack(out, cw, 8, 3);
                }
                break;
        case G726_32:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 2) {
			cw[0] = g726_32_encoder(s[i], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[1] = g726_32_encoder(s[i + 1], AUDIO_ENCODING_LINEAR, g->gs);
			out += g726_pack(out, cw, 2, 4);
                }
                break;
        case G726_40:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 8) {
			cw[0] = g726_40_encoder(s[i], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[1] = g726_40_encoder(s[i + 1], AUDIO_ENCODING_LINEAR, g->gs);
			cw[2] = g726_40_encoder(s[i + 2], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[3] = g726_40_encoder(s[i + 3], AUDIO_ENCODING_LINEAR, g->gs);
			cw[4] = g726_40_encoder(s[i + 4], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[5] = g726_40_encoder(s[i + 5], AUDIO_ENCODING_LINEAR, g->gs);
			cw[6] = g726_40_encoder(s[i + 6], AUDIO_ENCODING_LINEAR, g->gs);
                        cw[7] = g726_40_encoder(s[i + 7], AUDIO_ENCODING_LINEAR, g->gs);
			out += g726_pack(out, cw, 8, 5);
                }
                break;
        }

        return c->data_len;
}

int
g726_decode(uint16_t idx, u_char *decoder_state, coded_unit *c, sample *dst)
{
	u_char cw[8], *in;
        int i;
	
        g726_t *g; 

        /* paranoia! */
        assert(decoder_state != NULL);
        assert(c != NULL);
        assert(dst != NULL);
        assert(idx < G726_NUM_FORMATS);
        assert(c->state_len == 0);
        assert(c->data_len == cs[idx].mean_coded_frame_size);

        g = (g726_t*)decoder_state;

	in = c->data;

        idx = idx / G726_NUM_RATES;

        switch(idx) {
        case G726_16:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 4) {
			in += g726_unpack(cw, in, 4, 2);
			dst[i + 0] = (sample)g726_16_decoder(cw[0], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 1] = (sample)g726_16_decoder(cw[1], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 2] = (sample)g726_16_decoder(cw[2], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 3] = (sample)g726_16_decoder(cw[3], AUDIO_ENCODING_LINEAR, g->gs);
                }	
                break;	
        case G726_24:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 8) {
			in += g726_unpack(cw, in, 8, 3);
			dst[i + 0] = (sample)g726_24_decoder(cw[0], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 1] = (sample)g726_24_decoder(cw[1], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 2] = (sample)g726_24_decoder(cw[2], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 3] = (sample)g726_24_decoder(cw[3], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 4] = (sample)g726_24_decoder(cw[4], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 5] = (sample)g726_24_decoder(cw[5], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 6] = (sample)g726_24_decoder(cw[6], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 7] = (sample)g726_24_decoder(cw[7], AUDIO_ENCODING_LINEAR, g->gs);
                }
                break;
        case G726_32:
                for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 2) {
			in += g726_unpack(cw, in, 2, 4);
			dst[i + 0] = (sample)g726_32_decoder(cw[0], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 1] = (sample)g726_32_decoder(cw[1], AUDIO_ENCODING_LINEAR, g->gs);
		}
		break;
	case G726_40:
		for(i = 0; i < G726_SAMPLES_PER_FRAME; i += 8) {
			in += g726_unpack(cw, in, 8, 5);
			dst[i + 0] = (sample)g726_40_decoder(cw[0], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 1] = (sample)g726_40_decoder(cw[1], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 2] = (sample)g726_40_decoder(cw[2], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 3] = (sample)g726_40_decoder(cw[3], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 4] = (sample)g726_40_decoder(cw[4], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 5] = (sample)g726_40_decoder(cw[5], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 6] = (sample)g726_40_decoder(cw[6], AUDIO_ENCODING_LINEAR, g->gs);
			dst[i + 7] = (sample)g726_40_decoder(cw[7], AUDIO_ENCODING_LINEAR, g->gs);
		}
		break;
	}

	return c->data_len;
}




