/*
 * FILE:     audio_codec/lpc.c
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
#include "audio_codec/lpc.h"
#include "audio_codec/lpc_impl.h"

static acodec_format_t cs[] = {
        {"LPC", "LPC-8K-Mono", 
         "Pitch excited linear prediction codec (C) R. Zuckerman. Contributed by R. Frederick.",
         7, 0, LPCTXSIZE,
         {DEV_S16, 8000, 16, 1, 160 * BYTES_PER_SAMPLE}}
};

#define LPC_NUM_FORMATS (sizeof(cs)/sizeof(acodec_format_t))

uint16_t
lpc_get_formats_count()
{
        return LPC_NUM_FORMATS;
}

const acodec_format_t*
lpc_get_format(uint16_t idx)
{
        assert(idx < LPC_NUM_FORMATS);
        return &cs[idx];
}

void
lpc_setup(void)
{
        lpc_init();
}

int
lpc_encoder_state_create(uint16_t idx, u_char **state)
{
        assert(idx < LPC_NUM_FORMATS);
        UNUSED(idx);
        *state = (u_char*) malloc(sizeof(lpc_encstate_t));
        lpc_enc_init((lpc_encstate_t*) *state);
        return sizeof(lpc_encstate_t);
}

void
lpc_encoder_state_destroy(uint16_t idx, u_char **state)
{
        assert(idx < LPC_NUM_FORMATS);
        UNUSED(idx);
        
        free(*state);
        *state = (u_char*)NULL;
}

int
lpc_decoder_state_create(uint16_t idx, u_char **state)
{
        assert(idx < LPC_NUM_FORMATS);
        UNUSED(idx);
        *state = (u_char*) malloc(sizeof(lpc_intstate_t));
        lpc_dec_init((lpc_intstate_t*) *state);
        return sizeof(lpc_intstate_t);
}

void
lpc_decoder_state_destroy(uint16_t idx, u_char **state)
{
        assert(idx < LPC_NUM_FORMATS);
        UNUSED(idx);
        
        free(*state);
        *state = (u_char*)NULL;
}

int
lpc_encoder  (uint16_t idx, u_char *state, sample *in, coded_unit *out)
{
        assert(idx < LPC_NUM_FORMATS);
        assert(in);
        assert(out);
        UNUSED(idx);
        UNUSED(state);

        out->state     = NULL;
        out->state_len = 0;
        out->data      = (u_char*) malloc(LPCTXSIZE);
        out->data_len  = LPCTXSIZE;

        lpc_analyze((const short*)in, 
                    (lpc_encstate_t*)state, 
                    (lpc_txstate_t*)out->data);
        return out->data_len;
}

int
lpc_decoder (uint16_t idx, u_char *state, coded_unit *in, sample *out)
{
        assert(idx < LPC_NUM_FORMATS);
        assert(state);
        assert(in && in->data);
        assert(out);

        UNUSED(idx);
        lpc_synthesize((short*)out,  
                       (lpc_txstate_t*)in->data, 
                       (lpc_intstate_t*)state);
        return cs[idx].format.bytes_per_block / BYTES_PER_SAMPLE;
}

int  
lpc_repair (uint16_t idx, u_char *state, uint16_t consec_lost,
            coded_unit *prev, coded_unit *missing, coded_unit *next)
{
        lpc_txstate_t *lps;

        assert(prev);
        assert(missing);

        if (missing->data) {
                debug_msg("lpc_repair: missing unit had data!\n");
                free(missing->data);
        }
        
        missing->data     = (u_char*) malloc(LPCTXSIZE);
        missing->data_len = LPCTXSIZE;
        
        assert(prev->data);
        assert(prev->data_len == LPCTXSIZE);
        memcpy(missing->data, prev->data, LPCTXSIZE);       
        
        lps = (lpc_txstate_t*)missing->data;
        lps->gain = (u_char)((float)lps->gain * 0.8f);

        UNUSED(next);
        UNUSED(consec_lost);
        UNUSED(state);
        UNUSED(idx);

        return TRUE;
}
