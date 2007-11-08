/*
 * These are for the RAT project:
 *
 * $Id: dvi_impl.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

/***********************************************************
Copyright 1992 by Stichting Mathematisch Centrum, Amsterdam, The
Netherlands.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its 
documentation for any purpose and without fee is hereby granted, 
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in 
supporting documentation, and that the names of Stichting Mathematisch
Centrum or CWI not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior permission.

STICHTING MATHEMATISCH CENTRUM DISCLAIMS ALL WARRANTIES WITH REGARD TO
THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS, IN NO EVENT SHALL STICHTING MATHEMATISCH CENTRUM BE LIABLE
FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

******************************************************************/

#ifndef _DVI_H_
#define _DVI_H_

struct adpcm_state {
    short valprev;		/* Previous output value */
    unsigned char index;	/* Index into stepsize table */
    unsigned char pad;
};

void
adpcm_coder   (const short *inp, unsigned char *outp, int len, struct adpcm_state *state);
void 
adpcm_decoder (const unsigned char *inp, short *outp, int len, struct adpcm_state *state);

#endif /* _DVI_H_ */
