/*
 * FILE:    codec_gsm.h
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
#include "audio_codec/gsm.h"
#include "audio_codec/gsm_impl.h"

static acodec_format_t cs[] = {
        {"GSM", "GSM-8K-Mono", 
         "Full rate GSM speech codec. (c) 1992 J. Degener and C. Bormann, Technische Universitaet Berlin.",
         3, 0, GSM_FRAMESIZE,
         {DEV_S16, 8000, 16, 1, 160 * BYTES_PER_SAMPLE}},
        {"GSM", "GSM-16K-Mono", 
         "Full rate GSM speech codec. (c) 1992 J. Degener and C. Bormann, Technische Universitaet Berlin.",
         118, 0, GSM_FRAMESIZE,
         {DEV_S16, 16000, 16, 1, 160 * BYTES_PER_SAMPLE}},
        {"GSM", "GSM-32K-Mono", 
         "Full rate GSM speech codec. (c) 1992 J. Degener and C. Bormann, Technische Universitaet Berlin.",
         119, 0, GSM_FRAMESIZE,
         {DEV_S16, 32000, 16, 1, 160 * BYTES_PER_SAMPLE}},
        {"GSM", "GSM-48K-Mono", 
         "Full rate GSM speech codec. (c) 1992 J. Degener and C. Bormann, Technische Universitaet Berlin.",
         123, 0, GSM_FRAMESIZE,
         {DEV_S16, 48000, 16, 1, 160 * BYTES_PER_SAMPLE}}
};

#define GSM_NUM_FORMATS (sizeof(cs)/sizeof(acodec_format_t))

uint16_t
gsm_get_formats_count()
{
        return GSM_NUM_FORMATS;
}

const acodec_format_t*
gsm_get_format(uint16_t idx)
{
        assert(idx < GSM_NUM_FORMATS);
        return &cs[idx];
}

int
gsm_state_create(uint16_t idx, u_char **state)
{
        assert(idx < GSM_NUM_FORMATS);
        UNUSED(idx);
        *state = (u_char*) gsm_create();
        return GSM_FRAMESIZE;
}

void
gsm_state_destroy(uint16_t idx, u_char **state)
{
        assert(idx < GSM_NUM_FORMATS);
        UNUSED(idx);
        
        gsm_destroy((gsm)*state);
        *state = (u_char*)NULL;
}

int
gsm_encoder  (uint16_t idx, u_char *state, sample *in, coded_unit *out)
{
        assert(idx < GSM_NUM_FORMATS);
        assert(state);
        assert(in);
        assert(out);
        UNUSED(idx);

        out->state     = NULL;
        out->state_len = 0;
        out->data      = (u_char*) malloc(GSM_FRAMESIZE);
        out->data_len  = GSM_FRAMESIZE;

        gsm_encode((gsm)state, in, (gsm_byte*)out->data);
        return out->data_len;
}

int
gsm_decoder (uint16_t idx, u_char *state, coded_unit *in, sample *out)
{
        assert(idx < GSM_NUM_FORMATS);
        assert(state);
        assert(in && in->data);
        assert(out);

        UNUSED(idx);
        gsm_decode((gsm)state, (gsm_byte*)in->data, (gsm_signal*)out);
        return cs[idx].format.bytes_per_block / BYTES_PER_SAMPLE;
}

int  
gsm_repair (uint16_t idx, u_char *state, uint16_t consec_lost,
            coded_unit *prev, coded_unit *missing, coded_unit *next)
{
	/* GSM 06.11 repair mechanism */
	int		i;
	gsm_byte	*rep = NULL;
 	char		xmaxc;

        assert(prev);
        assert(missing);

        if (missing->data) {
                debug_msg("lpc_repair: missing unit had data!\n");
                free(missing->data);
        }
        
        missing->data     = (u_char*) malloc(GSM_FRAMESIZE);
        missing->data_len = GSM_FRAMESIZE;

        rep = (gsm_byte*)missing->data;
	memcpy(rep, prev->data, GSM_FRAMESIZE);

        if (consec_lost > 0) {
                /* If not first loss start fading */
		for(i=6;i<28;i+=7) {
			xmaxc  = (rep[i] & 0x1f) << 1;
			xmaxc |= (rep[i+1] >> 7) & 0x01;
			if (xmaxc > 4) { 
				xmaxc -= 4;
			} else { 
				xmaxc = 0;
			}
			rep[i]   = (rep[i] & 0xe0) | (xmaxc >> 1);
			rep[i+1] = (rep[i+1] & 0x7f) | ((xmaxc & 0x01) << 7);
		}
        }

        UNUSED(idx);
        UNUSED(state);
        UNUSED(next);
        
        return TRUE;
}
