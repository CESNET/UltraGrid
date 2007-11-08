/*
 * FILE:    codec_acm.h
 * PROGRAM: RAT
 * AUTHOR:  O.Hodson
 *
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Id: acm.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#if  !defined(_CODEC_ACM_H_)
#define _CODEC_ACM_H_

#ifdef WIN32
struct s_acm_state;

void acmStartup(void);
void acmShutdown(void);

struct s_acm_state* 
acmEncoderCreate(struct s_codec *cp);

void
acmEncode(struct s_acm_state *s, sample *src, struct s_coded_unit *dst);

void 
acmEncoderDestroy(struct s_acm_state *s);

#endif /* WIN32 */

#endif /* _CODEC_ACM_H_ */
