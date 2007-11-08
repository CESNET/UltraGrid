/*
 * FILE:     auddev_sparc.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: solaris.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _AUDDEV_SPARC_H_
#define _AUDDEV_SPARC_H_
int  sparc_audio_device_count (void);
char *
     sparc_audio_device_name(audio_desc_t ad);
int  sparc_audio_supports(audio_desc_t ad, audio_format *fmt);

int  sparc_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void sparc_audio_close      (audio_desc_t ad);
void sparc_audio_drain      (audio_desc_t ad);
int  sparc_audio_duplex     (audio_desc_t ad);
void sparc_audio_set_igain   (audio_desc_t ad, int gain);
int  sparc_audio_get_igain   (audio_desc_t ad);
void sparc_audio_set_ogain (audio_desc_t ad, int vol);
int  sparc_audio_get_ogain (audio_desc_t ad);
void sparc_audio_loopback   (audio_desc_t ad, int gain);
int  sparc_audio_read       (audio_desc_t ad, u_char *buf, int in_bytes);
int  sparc_audio_write      (audio_desc_t ad, u_char *buf, int out_bytes);
void sparc_audio_non_block  (audio_desc_t ad);
void sparc_audio_block      (audio_desc_t ad);

void          sparc_audio_oport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t  sparc_audio_oport_get   (audio_desc_t ad);
int           sparc_audio_oport_count (audio_desc_t ad);
const audio_port_details_t*
              sparc_audio_oport_details (audio_desc_t ad, int idx);

void          sparc_audio_iport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t  sparc_audio_iport_get   (audio_desc_t ad);
int           sparc_audio_iport_count (audio_desc_t ad);
const audio_port_details_t*
              sparc_audio_iport_details (audio_desc_t ad, int idx);

int  sparc_audio_is_ready  (audio_desc_t ad);
void sparc_audio_wait_for  (audio_desc_t ad, int delay_ms);
#endif /* _AUDDEV_SPARC_H_ */
