/*
 * FILE:    bitstream.h
 * PROGRAM: RAT
 * AUTHOR:  Orion Hodson
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifndef RAT_BITSTREAM_H
#define RAT_BITSTREAM_H

typedef struct s_bitstream bitstream_t;

int    bs_create     (bitstream_t **b);
int    bs_destroy    (bitstream_t **b);
int    bs_attach     (bitstream_t *b, u_char *buf, int blen);
int    bs_put        (bitstream_t *b, u_char  bits, uint8_t nbits); 
u_char bs_get        (bitstream_t *b, uint8_t nbits);
int    bs_bytes_used (bitstream_t *b);

#endif /* RAT_BITSTREAM_H */

