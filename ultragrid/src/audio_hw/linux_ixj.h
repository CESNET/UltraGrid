/*
 * FILE:     auddev_oss.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: linux_ixj.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _AUDDEV_IXJ_H_
#define _AUDDEV_IXJ_H_

int  ixj_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void ixj_audio_close      (audio_desc_t ad);
void ixj_audio_drain      (audio_desc_t ad);
int  ixj_audio_duplex     (audio_desc_t ad);
void ixj_audio_set_igain   (audio_desc_t ad, int gain);
int  ixj_audio_get_igain   (audio_desc_t ad);
void ixj_audio_set_ogain (audio_desc_t ad, int vol);
int  ixj_audio_get_ogain (audio_desc_t ad);
void ixj_audio_loopback   (audio_desc_t ad, int gain);
int  ixj_audio_read       (audio_desc_t ad, u_char *buf, int buf_bytes);
int  ixj_audio_write      (audio_desc_t ad, u_char *buf, int buf_bytes);
void ixj_audio_non_block  (audio_desc_t ad);
void ixj_audio_block      (audio_desc_t ad);

void         ixj_audio_oport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t ixj_audio_oport_get   (audio_desc_t ad);
int          ixj_audio_oport_count (audio_desc_t ad);
const audio_port_details_t*
             ixj_audio_oport_details (audio_desc_t ad, int idx);

void         ixj_audio_iport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t ixj_audio_iport_get   (audio_desc_t ad);
int          ixj_audio_iport_count (audio_desc_t ad);
const audio_port_details_t*
             ixj_audio_iport_details (audio_desc_t ad, int idx);

int  ixj_audio_is_ready  (audio_desc_t ad);
void ixj_audio_wait_for  (audio_desc_t ad, int delay_ms);
int  ixj_audio_supports  (audio_desc_t ad, audio_format *fmt);

/* Functions to get names of oss devices */
int         ixj_audio_init      (void);             /* This fn works out what we have           */
int         ixj_get_device_count(void);             /* Then this one tells us the number of 'em */
char       *ixj_get_device_name (audio_desc_t idx); /* Then this one tells us the name          */

#endif /* _AUDDEV_IXJ_H_ */
