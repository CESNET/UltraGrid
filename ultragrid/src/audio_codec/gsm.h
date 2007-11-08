/*
 * FILE:    codec_gsm.h
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: gsm.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _CODEC_GSM_H_
#define _CODEC_GSM_H_

uint16_t                      gsm_get_formats_count (void);
const struct s_codec_format* gsm_get_format(uint16_t idx);

int  gsm_state_create  (uint16_t idx, u_char **state);
void gsm_state_destroy (uint16_t idx, u_char **state);
int  gsm_encoder       (uint16_t idx, u_char *state, sample *in, coded_unit *out);
int  gsm_decoder       (uint16_t idx, u_char *state, coded_unit *in, sample *out);

int  gsm_repair        (uint16_t idx, u_char *state, uint16_t consec_lost,
                        coded_unit *prev, coded_unit *missing, coded_unit *next);

#endif /* _CODEC_GSM_H_ */
