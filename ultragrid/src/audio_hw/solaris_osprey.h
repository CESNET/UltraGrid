/*
 * FILE:     auddev_osprey.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: solaris_osprey.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _AUDDEV_OSPREY_H_
#define _AUDDEV_OSPREY_H_

int osprey_audio_init(void);

int  osprey_audio_device_count(void);
char*  
     osprey_audio_device_name(audio_desc_t ad);

int  osprey_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void osprey_audio_close      (audio_desc_t ad);
void osprey_audio_drain      (audio_desc_t ad);
int  osprey_audio_duplex     (audio_desc_t ad);
void osprey_audio_set_igain   (audio_desc_t ad, int gain);
int  osprey_audio_get_igain   (audio_desc_t ad);
void osprey_audio_set_ogain (audio_desc_t ad, int vol);
int  osprey_audio_get_ogain (audio_desc_t ad);
void osprey_audio_loopback   (audio_desc_t ad, int gain);
int  osprey_audio_read       (audio_desc_t ad, u_char *buf, int in_bytes);
int  osprey_audio_write      (audio_desc_t ad, u_char *buf, int out_bytes);
void osprey_audio_non_block  (audio_desc_t ad);
void osprey_audio_block      (audio_desc_t ad);

void          osprey_audio_oport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t  osprey_audio_oport_get   (audio_desc_t ad);
int           osprey_audio_oport_count (audio_desc_t ad);
const audio_port_details_t*
              osprey_audio_oport_details (audio_desc_t ad, int idx);

void          osprey_audio_iport_set     (audio_desc_t ad, audio_port_t port);
audio_port_t  osprey_audio_iport_get     (audio_desc_t ad);
int           osprey_audio_iport_count   (audio_desc_t ad);
const audio_port_details_t*
              osprey_audio_iport_details (audio_desc_t ad, int idx);

int  osprey_audio_next_iport (audio_desc_t ad);
int  osprey_audio_is_ready  (audio_desc_t ad);
void osprey_audio_wait_for  (audio_desc_t ad, int delay_ms);

#endif /* _AUDDEV_OSPREY_H_ */
