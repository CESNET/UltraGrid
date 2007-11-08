/*
 * FILE:    codec_g726.h
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: g726.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

/* Just wrote the RAT interface, see codec_dvi.c for coder copyright [oth] */

#ifndef _CODEC_G726_H_
#define _CODEC_G726_H_

uint16_t               g726_get_formats_count (void);
const acodec_format_t* g726_get_format        (uint16_t idx);
int                   g726_state_create      (uint16_t idx, u_char **state);
void                  g726_state_destroy     (uint16_t idx, u_char **state);
int                   g726_encode            (uint16_t idx, u_char *state, sample     *in, coded_unit *out);
int                   g726_decode            (uint16_t idx, u_char *state, coded_unit *in, sample     *out);

#endif /* _CODEC_G726_H_ */
