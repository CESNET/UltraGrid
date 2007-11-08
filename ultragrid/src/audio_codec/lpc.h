/*
 * FILE:    codec_lpc.h
 * AUTHORS: Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: lpc.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _CODEC_LPC_H_
#define _CODEC_LPC_H_

uint16_t                      lpc_get_formats_count (void);
const struct s_codec_format* lpc_get_format(uint16_t idx);

void lpc_setup(void);

int  lpc_encoder_state_create  (uint16_t idx, u_char **state);
void lpc_encoder_state_destroy (uint16_t idx, u_char **state);
int  lpc_encoder (uint16_t idx, u_char *state, sample *in, coded_unit *out);

int  lpc_decoder_state_create  (uint16_t idx, u_char **state);
void lpc_decoder_state_destroy (uint16_t idx, u_char **state);
int  lpc_decoder               (uint16_t idx, u_char *state, coded_unit *in, sample *out);

int  lpc_repair  (uint16_t idx, u_char *state, uint16_t consec_lost,
                  coded_unit *prev, coded_unit *missing, coded_unit *next);

#endif /* _CODEC_LPC_H_ */
