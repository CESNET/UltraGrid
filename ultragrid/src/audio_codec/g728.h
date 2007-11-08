/*
 * FILE:    codec_g728.h
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: g728.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _CODEC_G728_H_
#define _CODEC_G728_H_

#ifdef HAVE_G728

uint16_t              g728_get_formats_count (void);
const codec_format_t* g728_get_format        (uint16_t idx);
int                   g728_encoder_create    (uint16_t idx, u_char **state);
void                  g728_encoder_destroy   (uint16_t idx, u_char **state);
int                   g728_encoder_do        (uint16_t idx, u_char *state, sample     *in, coded_unit *out);
int                   g728_decoder_create    (uint16_t idx, u_char **state);
void                  g728_decoder_destroy   (uint16_t idx, u_char **state);
int                   g728_decoder_do         (uint16_t idx, u_char *state, coded_unit *in, sample     *out);

#endif /* HAVE_G728 */

#endif /* _CODEC_G728_H_ */




