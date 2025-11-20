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

#ifndef RAT_BITSTREAM_H
#define RAT_BITSTREAM_H

#include "stdint.h"

typedef struct s_bitstream bitstream_t;

int    bs_create     (bitstream_t **b);
int    bs_destroy    (bitstream_t **b);
int    bs_attach     (bitstream_t *b, unsigned char *buf, int blen);
int    bs_put        (bitstream_t *b, unsigned char  bits, uint8_t nbits);
unsigned char bs_get        (bitstream_t *b, uint8_t nbits);
int    bs_bytes_used (bitstream_t *b);

#endif /* RAT_BITSTREAM_H */

