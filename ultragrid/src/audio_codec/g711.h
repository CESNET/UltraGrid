/*
 * FILE:    codec_g711.h
 * PROGRAM: RAT
 * AUTHOR:  Orion Hodson
 *
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Id: g711.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef CODEC_G711_H
#define CODEC_G711_H

extern short    mulawtolin[256];
extern unsigned char lintomulaw[65536];

extern short    alawtolin[256];
extern unsigned char lintoalaw[8192]; 

#define s2u(x)	lintomulaw[((unsigned short)(x))]
#define u2s(x)	mulawtolin[((unsigned char)(x))]
#define s2a(x)  lintoalaw[((unsigned short)(x))>>3]
#define a2s(x)  alawtolin[((unsigned char)(x))]

struct s_coded_unit;

void g711_init(void);

uint16_t                      g711_get_formats_count (void);
const struct s_codec_format* g711_get_format (uint16_t idx);
int                          g711_encode     (uint16_t idx, 
                                              u_char *state, 
                                              sample  *in, 
                                              struct s_coded_unit *out);
int                          g711_decode     (uint16_t idx, 
                                              u_char *state, 
                                              struct s_coded_unit *in, 
                                              sample     *out);

#endif
