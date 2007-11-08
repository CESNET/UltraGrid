/*
 * FILE:    audio_hw.h
 * AUTHOR:  Orion Hodson
 *
 * Note: Original audio interface by Isidor Kouvelas, Colin Perkins, 
 * and Orion Hodson.  Orion Hodson has gone through and modularised 
 * this code so that RAT can detect and use multiple audio devices.
 *
 * Copyright (c) 2002-2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:58 $
 */

#ifndef _AUDDEV_H_
#define _AUDDEV_H_

#include "audio_types.h"

/****************************************************************************/
/* Audio interface fn's for dealing with multiple devices/device interfaces */
int      audio_init_interfaces    (void);
int      audio_free_interfaces    (void);
uint32_t audio_get_device_count   (void);
const audio_device_details_t*
         audio_get_device_details (uint32_t idx);
int      audio_get_null_device    (void); /* gets null dev interface */
int      audio_device_supports    (audio_desc_t ad, uint16_t rate, uint16_t channels);
int      audio_device_is_open     (audio_desc_t ad);

/****************************************************************************/
/* Audio functions implemented by device interfaces                         */
int     audio_open     (audio_desc_t ad, audio_format *in_format, audio_format *out_format);
void	audio_close    (audio_desc_t ad);
void	audio_drain    (audio_desc_t ad);
void	audio_set_igain(audio_desc_t ad, int gain);
int     audio_duplex   (audio_desc_t ad);
int	audio_get_igain(audio_desc_t ad);
void	audio_set_ogain(audio_desc_t ad, int vol);
int	audio_get_ogain(audio_desc_t ad);
void    audio_loopback (audio_desc_t ad, int gain);
int	audio_read     (audio_desc_t ad, sample *buf, int samples);
int	audio_write    (audio_desc_t ad, sample *buf, int samples);
void	audio_non_block(audio_desc_t ad);
void	audio_block    (audio_desc_t ad);

void	                    audio_set_oport         (audio_desc_t ad, audio_port_t);
audio_port_t                audio_get_oport         (audio_desc_t ad);
int	                    audio_get_oport_count   (audio_desc_t ad);
const audio_port_details_t* audio_get_oport_details (audio_desc_t ad, int port_idx);

void	                    audio_set_iport         (audio_desc_t ad, audio_port_t);
audio_port_t                audio_get_iport         (audio_desc_t ad);
int	                    audio_get_iport_count   (audio_desc_t ad);
const audio_port_details_t* audio_get_iport_details (audio_desc_t ad, int port_idx);

int     audio_is_ready      (audio_desc_t ad);
void    audio_wait_for      (audio_desc_t ad, int granularity_ms);

const audio_format* audio_get_ifmt (audio_desc_t ad);
const audio_format* audio_get_ofmt (audio_desc_t ad);

/* audio_get_samples_{read,written} return the actual number of
 * sample instances that have been read/written.  audio_get_device_time
 * returns the number of samples read rounded down to the nearest
 * number of whole sample blocks.
 */

uint32_t audio_get_device_time     (audio_desc_t ad);
uint32_t audio_get_samples_written (audio_desc_t ad);
uint32_t audio_get_samples_read    (audio_desc_t ad);

#endif /* _AUDDEV_H_ */
