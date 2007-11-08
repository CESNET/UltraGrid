/*
 * FILE:     auddev_win32.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: win32.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _AUDDEV_W32SDK_H_
#define _AUDDEV_W32SDK_H_

int  w32sdk_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void w32sdk_audio_close      (audio_desc_t ad);
void w32sdk_audio_drain      (audio_desc_t ad);
int  w32sdk_audio_duplex     (audio_desc_t ad);
void w32sdk_audio_set_igain   (audio_desc_t ad, int gain);
int  w32sdk_audio_get_igain   (audio_desc_t ad);
void w32sdk_audio_set_ogain (audio_desc_t ad, int vol);
int  w32sdk_audio_get_ogain (audio_desc_t ad);
void w32sdk_audio_loopback   (audio_desc_t ad, int gain);
int  w32sdk_audio_read       (audio_desc_t ad, u_char *buf, int buf_bytes);
int  w32sdk_audio_write      (audio_desc_t ad, u_char *buf, int buf_bytes);
void w32sdk_audio_non_block  (audio_desc_t ad);
void w32sdk_audio_block      (audio_desc_t ad);

void         w32sdk_audio_oport_set     (audio_desc_t ad, audio_port_t port);
audio_port_t w32sdk_audio_oport_get     (audio_desc_t ad);
int          w32sdk_audio_oport_count   (audio_desc_t ad);
const audio_port_details_t*
             w32sdk_audio_oport_details (audio_desc_t ad, int idx);

void         w32sdk_audio_iport_set     (audio_desc_t ad, audio_port_t port);
audio_port_t w32sdk_audio_iport_get     (audio_desc_t ad);
int          w32sdk_audio_iport_count   (audio_desc_t ad);
const audio_port_details_t*
             w32sdk_audio_iport_details (audio_desc_t ad, int idx);

int  w32sdk_audio_is_ready  (audio_desc_t ad);
void w32sdk_audio_wait_for  (audio_desc_t ad, int delay_ms);
int  w32sdk_audio_supports  (audio_desc_t ad, audio_format *paf);

/* Functions to get names of win32 devices */
int   w32sdk_audio_init(void);		/* Startup initialization                   */
int   w32sdk_audio_free(void);		/* Free the device... what a concept!       */
int   w32sdk_get_device_count(void);	/* Then this one tells us the number of 'em */
char *w32sdk_get_device_name(int idx);	/* Then this one tells us the name          */

#endif /* _AUDDEV_W32SDK_H_ */
