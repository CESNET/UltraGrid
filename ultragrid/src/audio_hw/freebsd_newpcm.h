/*
 * FILE:     auddev_newpcm.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: freebsd_newpcm.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _AUDDEV_NEWPCM_H_
#define _AUDDEV_NEWPCM_H_

int  newpcm_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void newpcm_audio_close      (audio_desc_t ad);
void newpcm_audio_drain      (audio_desc_t ad);
int  newpcm_audio_duplex     (audio_desc_t ad);

void newpcm_audio_set_igain  (audio_desc_t ad, int gain);
int  newpcm_audio_get_igain  (audio_desc_t ad);
void newpcm_audio_set_ogain  (audio_desc_t ad, int vol);
int  newpcm_audio_get_ogain  (audio_desc_t ad);
void newpcm_audio_loopback   (audio_desc_t ad, int gain);

int  newpcm_audio_read       (audio_desc_t ad, u_char *buf, int buf_len);
int  newpcm_audio_write      (audio_desc_t ad, u_char *buf, int buf_len);
void newpcm_audio_non_block  (audio_desc_t ad);
void newpcm_audio_block      (audio_desc_t ad);

void         newpcm_audio_oport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t newpcm_audio_oport_get   (audio_desc_t ad);
int          newpcm_audio_oport_count (audio_desc_t ad);
const audio_port_details_t*
     newpcm_audio_oport_details       (audio_desc_t ad, int idx);

void         newpcm_audio_iport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t newpcm_audio_iport_get   (audio_desc_t ad);
int          newpcm_audio_iport_count (audio_desc_t ad);
const audio_port_details_t*
     newpcm_audio_iport_details       (audio_desc_t ad, int idx);

int  newpcm_audio_is_ready  (audio_desc_t ad);
void newpcm_audio_wait_for  (audio_desc_t ad, int delay_ms);
int  newpcm_audio_supports  (audio_desc_t ad, audio_format *f);

/* Functions to get names of devices */
int         newpcm_audio_query_devices (void);
int         newpcm_get_device_count    (void);
char       *newpcm_get_device_name     (audio_desc_t ad);

#endif /* _AUDDEV_NEWPCM_H_ */
