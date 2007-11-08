/*
 * FILE:    codec_vdvi.h
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: vdvi.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

/* Just wrote the RAT interface, see codec_dvi.c for coder copyright [oth] */

#ifndef _CODEC_VDVI_H_
#define _CODEC_VDVI_H_

uint16_t               vdvi_get_formats_count (void);
const acodec_format_t* vdvi_get_format        (uint16_t idx);
int                   vdvi_state_create      (uint16_t idx, u_char **state);
void                  vdvi_state_destroy     (uint16_t idx, u_char **state);
int                   vdvi_encoder           (uint16_t idx, u_char *state, sample     *in, coded_unit *out);
int                   vdvi_decoder           (uint16_t idx, u_char *state, coded_unit *in, sample     *out);
int                   vdvi_peek_frame_size   (uint16_t idx, u_char *data, int data_len);

#endif /* _CODEC_VDVI_H_ */



