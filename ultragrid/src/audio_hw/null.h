/*
 * File:     auddev_null.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: null.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _AUDDEV_NULL_H_
#define _AUDDEV_NULL_H_

int  null_audio_device_count(void);
char*  null_audio_device_name(audio_desc_t ad);
int  null_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void null_audio_close      (audio_desc_t ad);
void null_audio_drain      (audio_desc_t ad);
int  null_audio_duplex     (audio_desc_t ad);
void null_audio_set_igain  (audio_desc_t ad, int gain);
int  null_audio_get_igain  (audio_desc_t ad);
void null_audio_set_ogain  (audio_desc_t ad, int vol);
int  null_audio_get_ogain  (audio_desc_t ad);
void null_audio_loopback   (audio_desc_t ad, int gain);
int  null_audio_read       (audio_desc_t ad, u_char *buf, int buf_len);
int  null_audio_write      (audio_desc_t ad, u_char *buf, int buf_len);
void null_audio_non_block  (audio_desc_t ad);
void null_audio_block      (audio_desc_t ad);

void         null_audio_oport_set     (audio_desc_t ad, audio_port_t port);
audio_port_t null_audio_oport_get     (audio_desc_t ad);
int          null_audio_oport_count   (audio_desc_t ad);
const audio_port_details_t*
             null_audio_oport_details (audio_desc_t ad, int idx);

void         null_audio_iport_set     (audio_desc_t ad, audio_port_t port);
audio_port_t null_audio_iport_get     (audio_desc_t ad);
int          null_audio_iport_count   (audio_desc_t ad);
const audio_port_details_t*
             null_audio_iport_details (audio_desc_t ad, int idx);


int  null_audio_is_ready  (audio_desc_t ad);
void null_audio_wait_for  (audio_desc_t ad, int delay_ms);
int  null_audio_supports  (audio_desc_t ad, audio_format *fmt);
#endif /* _AUDDEV_NULL_H_ */
