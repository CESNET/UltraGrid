/*
 * FILE:    codec_dvi.h
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: dvi.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

/* Just wrote the RAT interface, see codec_dvi.c for coder copyright [oth] */

#ifndef _CODEC_DVI_H_
#define _CODEC_DVI_H_

uint16_t               dvi_get_formats_count (void);
const acodec_format_t* dvi_get_format        (uint16_t idx);
int                   dvi_state_create      (uint16_t idx, u_char **state);
void                  dvi_state_destroy     (uint16_t idx, u_char **state);
int                   dvi_encode            (uint16_t idx, u_char *state, sample     *in, coded_unit *out);
int                   dvi_decode            (uint16_t idx, u_char *state, coded_unit *in, sample     *out);

#endif /* _CODEC_DVI_H_ */
