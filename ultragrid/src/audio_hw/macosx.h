/*
 * FILE:     auddev_maxosx.h
 * PROGRAM:  RAT 4
 * AUTHOR:   Juraj Sucik
 *
 */

#ifndef _AUDDEV_MACOSX_H_
#define _AUDDEV_MACOSX_H_

int  macosx_audio_open				(audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void macosx_audio_close				(audio_desc_t ad);
void macosx_audio_drain				(audio_desc_t ad);
int  macosx_audio_duplex			(audio_desc_t ad);
void macosx_audio_set_igain			(audio_desc_t ad,int gain);
int  macosx_audio_get_igain			(audio_desc_t ad);
void macosx_audio_set_ogain			(audio_desc_t ad,int vol);
int  macosx_audio_get_ogain			(audio_desc_t ad);
void macosx_audio_loopback			(audio_desc_t ad, int gain);
int  macosx_audio_read			(audio_desc_t ad, u_char *buf,int read_bytes);
int macosx_audio_write			(audio_desc_t ad,u_char* data,int write_bytes);
void macosx_audio_non_block			(audio_desc_t ad);
void macosx_audio_block			(audio_desc_t ad);
void macosx_audio_oport_set			(audio_desc_t ad, audio_port_t port);
audio_port_t macosx_audio_oport_get	(audio_desc_t ad);
int  macosx_audio_oport_count		(audio_desc_t ad);
const audio_port_details_t*
	 macosx_audio_oport_details		(audio_desc_t ad, int idx);
void macosx_audio_iport_set			(audio_desc_t ad, audio_port_t port);
audio_port_t macosx_audio_iport_get		(audio_desc_t ad);
int  macosx_audio_iport_count			(audio_desc_t ad);
const audio_port_details_t*
	 macosx_audio_iport_details		(audio_desc_t ad, int idx);
int  macosx_audio_is_ready			(audio_desc_t ad);
void macosx_audio_wait_for			(audio_desc_t ad, int delay_ms);
int  macosx_audio_supports			(audio_desc_t ad, audio_format *fmt);
int	 macosx_audio_init			(void);
int	 macosx_audio_device_count		(void);
char *macosx_audio_device_name			(audio_desc_t idx);

#endif /* _AUDDEV_MACOSX_H_ */

