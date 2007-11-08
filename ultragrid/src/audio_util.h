/*
 * FILE:    audio.h
 * PROGRAM: RAT
 * AUTHOR:  Isidor Kouvelas / Orion Hodson / Colin Perkins
 *
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Id: audio_util.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef __AUDIO_UTIL_H__
#define __AUDIO_UTIL_H__

void	audio_zero   (sample *buf, int len, deve_e type);

void    audio_mix    (sample *dst, sample *in, int len);

#ifdef WIN32
BOOL    mmx_present();
void    audio_mix_mmx(sample *dst, sample *in, int len);
#endif

void    audio_scale_buffer(sample *buf, int len, double scale); 

/* Energy calculation */
uint16_t audio_avg_energy (sample *buf, uint32_t dur, uint32_t channels);

sample   audio_abs_max(sample *buf, uint32_t samples);

void     audio_blend(sample *from, sample *to, sample *dst, int samples, int channels);

/* Biasing operations */

struct s_bias_ctl;

struct s_bias_ctl*
        bias_ctl_create(int channels, int freq);

void    bias_ctl_destroy(struct s_bias_ctl *bc);

void    bias_remove (struct s_bias_ctl *bc, sample *buf, int len);

#endif /* __AUDIO_UTIL_H__ */
