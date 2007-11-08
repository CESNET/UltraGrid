/*
 * FILE:     audio_hw/solaris_util.c
 * PROGRAM:  RAT/UltraGrid
 * AUTHOR:   Orion Hodson
 * MODIFIED: Colin Perkins
 *
 * Copyright (c)      2004 University of Glasgow
 * Copyright (c)      2003 University of Southern California
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/solaris_util.h"

void 
af2apri(audio_format *fmt, audio_prinfo_t *ap)
{
        assert(fmt);
        assert(ap);
        ap->sample_rate = fmt->sample_rate;
        ap->channels    = fmt->channels;
        ap->precision   = fmt->bits_per_sample;
	ap->buffer_size   = 160 * fmt->channels * (fmt->sample_rate / 8000) * (fmt->bits_per_sample / 8);

        switch(fmt->encoding) {
        case DEV_PCMU: ap->encoding = AUDIO_ENCODING_ULAW;   assert(ap->precision == 8);  break;
        case DEV_S8:   ap->encoding = AUDIO_ENCODING_LINEAR; assert(ap->precision == 8);  break;
        case DEV_S16:  ap->encoding = AUDIO_ENCODING_LINEAR; assert(ap->precision == 16); break;
        default: debug_msg("Format not recognized\n"); assert(0); 
        }
}

