/*
 * FILE:    auddev_alsa.c
 * PROGRAM: RAT ALSA 0.9+/final audio driver.
 * AUTHOR:  Steve Smith
 *
 * Copyright (c) 2003 University of Sydney
 * Distributed under the same terms as RAT itself.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#include "config_unix.h"
#include "debug.h"
#include <alsa/asoundlib.h>
#include "audio_types.h"


/* Define some tuneables: */

/* Buffer length, in fractions of a second.  This value is used to
 * divide the sample-rate to define the buffer-size. */
#define RAT_ALSA_BUFFER_DIVISOR 5

// External prototypes
int  alsa_audio_open       (audio_desc_t ad, audio_format* ifmt, audio_format *ofmt);
void alsa_audio_close      (audio_desc_t ad);
void alsa_audio_drain      (audio_desc_t ad);
int  alsa_audio_duplex     (audio_desc_t ad);
void alsa_audio_set_igain   (audio_desc_t ad, int gain);
int  alsa_audio_get_igain   (audio_desc_t ad);
void alsa_audio_set_ogain (audio_desc_t ad, int vol);
int  alsa_audio_get_ogain (audio_desc_t ad);
int alsa_audio_read (audio_desc_t ad, u_char *buf, int bytes);
int alsa_audio_write (audio_desc_t ad, u_char *buf, int bytes);
void alsa_audio_non_block  (audio_desc_t ad);
void alsa_audio_block      (audio_desc_t ad);

void         alsa_audio_oport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t alsa_audio_oport_get   (audio_desc_t ad);
int          alsa_audio_oport_count (audio_desc_t ad);
const audio_port_details_t* alsa_audio_oport_details (audio_desc_t ad, int idx);

void         alsa_audio_iport_set   (audio_desc_t ad, audio_port_t port);
audio_port_t alsa_audio_iport_get   (audio_desc_t ad);
int          alsa_audio_iport_count (audio_desc_t ad);
const audio_port_details_t* alsa_audio_iport_details (audio_desc_t ad, int idx);

int  alsa_audio_is_ready  (audio_desc_t ad);
void alsa_audio_wait_for  (audio_desc_t ad, int delay_ms);
int  alsa_audio_supports  (audio_desc_t ad, audio_format *fmt);

int alsa_audio_init (void);
int alsa_get_device_count (void);
char *alsa_get_device_name (audio_desc_t idx);
