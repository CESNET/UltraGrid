/*
 * FILE:     audio_hw/solaris.c
 * PROGRAM:  RAT
 * AUTHOR:   Isidor Kouvelas
 * MODIFIED: Colin Perkins / Orion Hodson
 *
 * Copyright (c)      2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config_unix.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/solaris.h"
#include "audio_hw/solaris_util.h"
#include "codec_g711.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/audioio.h>
#ifdef NDEF	/* Only needed on pre-solaris 2.6 machines <frederic.vecoven@sun.com> */
#include <multimedia/audio_encode.h>
#include <multimedia/audio_hdr.h>
#endif

#ifndef AUDIO_CD
#define AUDIO_CD 4
#endif

static audio_info_t	dev_info;

static int audio_fd = -1; 

#define bat_to_device(x)	((x) * AUDIO_MAX_GAIN / MAX_AMP)
#define device_to_bat(x)	((x) * MAX_AMP / AUDIO_MAX_GAIN)

int 
sparc_audio_device_count()
{
        return 1;
}

char*
sparc_audio_device_name(audio_desc_t ad)
{
        UNUSED(ad);
        return "Sun Audio Device";
}

int
sparc_audio_supports(audio_desc_t ad, audio_format *fmt)
{
        UNUSED(ad);
        if ((!(fmt->sample_rate % 8000) || !(fmt->sample_rate % 11025)) && 
            (fmt->channels == 1 || fmt->channels == 2)) {
                return TRUE;
        }
        return FALSE;
}

/* Try to open the audio device.                        */
/* Returns TRUE if ok, 0 otherwise. */
int
sparc_audio_open(audio_desc_t ad, audio_format* ifmt, audio_format* ofmt)
{
	audio_info_t	tmp_info;
	char            audiodev[256], *str;

        if (audio_fd != -1) {
                debug_msg("Device already open!");
                sparc_audio_close(ad);
                return FALSE;
        }

	if ((str = getenv("AUDIODEV")) != NULL) {
		strncpy(audiodev, str, 252);	/* 252 to allow for the strcat later... */
	} else {
		strcpy(audiodev, "/dev/audio");
	}
	audio_fd = open(audiodev, O_RDWR | O_NDELAY);

	if (audio_fd > 0) {
		AUDIO_INITINFO(&dev_info);
		dev_info.monitor_gain       = 0;
		dev_info.output_muted       = 0; /* 0==not muted */
                af2apri(ifmt, &dev_info.record);
                af2apri(ofmt, &dev_info.play);
		dev_info.play.gain          = (AUDIO_MAX_GAIN - AUDIO_MIN_GAIN) * 0.75;
		dev_info.record.gain        = (AUDIO_MAX_GAIN - AUDIO_MIN_GAIN) * 0.75;
		dev_info.play.port          = AUDIO_HEADPHONE;
		dev_info.record.port        = AUDIO_MICROPHONE;
		dev_info.play.balance       = AUDIO_MID_BALANCE;
		dev_info.record.balance     = AUDIO_MID_BALANCE;

		memcpy(&tmp_info, &dev_info, sizeof(audio_info_t));
		if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&tmp_info) < 0) {
			if (ifmt->encoding == DEV_S16) {
				debug_msg("Old hardware detected: can't do 16 bit audio, trying 8 bit...\n");
                                audio_format_change_encoding(ifmt, DEV_PCMU);
                                audio_format_change_encoding(ofmt, DEV_PCMU);
                                af2apri(ifmt, &dev_info.record);
                                af2apri(ofmt, &dev_info.play);
				if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0) {
					perror("Setting MULAW audio parameters");
                                        sparc_audio_close(audio_fd);
					return FALSE;
				}
			} else {
				perror("Setting audio parameters");
				return FALSE;
			}
		}

                /* XXX driver issue - on Ultra II's if you don't drain
                 * the device before reading commences then the device
                 * reads in blocks of 500ms irrespective of the
                 * blocksize set.  After a minute or so it flips into the
                 * correct mode, but obviously this is too late to be 
                 * useful for most apps. grrr.
                 */

                sparc_audio_drain(ad);

		return audio_fd;
	} else {
		/* Because we opened the device with O_NDELAY
		 * the waiting flag was not updated so update
		 * it manually using the audioctl device...  */
		strcat(audiodev, "ctl");
		audio_fd = open(audiodev, O_RDWR);
		AUDIO_INITINFO(&dev_info);
		dev_info.play.waiting = 1;
		if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0) {
#ifdef DEBUG
			perror("Setting requests");
#endif
		}
                if (audio_fd > 0) {
                        sparc_audio_close(audio_fd);
                }
		return FALSE;
	}
}

/* Close the audio device */
void
sparc_audio_close(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);
	if (audio_fd <= 0) {
                debug_msg("Invalid desc");
		return;
        }

	close(audio_fd);
	audio_fd = -1;
}

/* Flush input buffer */
void
sparc_audio_drain(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	ioctl(audio_fd, I_FLUSH, (caddr_t)FLUSHR);
}

/* Gain and volume values are in the range 0 - MAX_AMP */

void
sparc_audio_set_igain(audio_desc_t ad, int gain)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);
	dev_info.record.gain = bat_to_device(gain);
	if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0)
		perror("Setting gain");
}

int
sparc_audio_get_igain(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);
	if (ioctl(audio_fd, AUDIO_GETINFO, (caddr_t)&dev_info) < 0)
		perror("Getting gain");
	return (device_to_bat(dev_info.record.gain));
}

void
sparc_audio_set_ogain(audio_desc_t ad, int vol)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);
	dev_info.play.gain = bat_to_device(vol);
	if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0)
		perror("Setting volume");
}

int
sparc_audio_get_ogain(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);
	if (ioctl(audio_fd, AUDIO_GETINFO, (caddr_t)&dev_info) < 0)
		perror("Getting gain");
	return (device_to_bat(dev_info.play.gain));
}

void
sparc_audio_loopback(audio_desc_t ad, int gain)
{
        UNUSED(ad); assert(audio_fd > 0);

        /* Nasty bug on Ultra 30's a loopback gain of anything above
         * 90 on my machine is unstable.  On the earlier Ultra's
         * anything below 90 produces no perceptible loopback. Could
         * use sysinfo() with SI_PLATFORM but then we might have to
         * maintain a list of hardware with this problem.
         */
        gain = gain * 85/100;

        AUDIO_INITINFO(&dev_info);
	dev_info.monitor_gain = bat_to_device(gain);
	if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0)
		perror("Setting loopback");
}

int
sparc_audio_read(audio_desc_t ad, u_char *buf, int buf_bytes)
{
	int len;

        UNUSED(ad); assert(audio_fd > 0);

        if ((len = read(audio_fd, (char *)buf, buf_bytes)) < 0) {
                len = 0;
        } 

        return len;
}

int
sparc_audio_write(audio_desc_t ad, u_char *buf, int buf_bytes)
{
	int done, this_write;

        UNUSED(ad); assert(audio_fd > 0);

        done = 0;
        while(done != buf_bytes) {
                this_write = write(audio_fd, buf, buf_bytes - done);
		if (errno != EINTR)
			return (buf_bytes - done);
		done += this_write;
		buf  += this_write;
	}

	return done;
}

/* Set ops on audio device to be non-blocking */
void
sparc_audio_non_block(audio_desc_t ad)
{
	int	on = 1;

        UNUSED(ad); assert(audio_fd > 0);

	if (ioctl(audio_fd, FIONBIO, (char *)&on) < 0)
		fprintf(stderr, "Failed to set non blocking mode on audio device!\n");
}

/* Set ops on audio device to block */
void
sparc_audio_block(audio_desc_t ad)
{
	int	on = 0;

        UNUSED(ad); assert(audio_fd > 0);

	if (ioctl(audio_fd, FIONBIO, (char *)&on) < 0)
		fprintf(stderr, "Failed to set blocking mode on audio device!\n");
}

static const audio_port_details_t out_ports[] = {
        { AUDIO_SPEAKER,   AUDIO_PORT_SPEAKER},
        { AUDIO_HEADPHONE, AUDIO_PORT_HEADPHONE},
        { AUDIO_LINE_OUT,  AUDIO_PORT_LINE_OUT }
};

#define NUM_OUT_PORTS (sizeof(out_ports)/sizeof(out_ports[0]))

void
sparc_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);

        if (port != AUDIO_SPEAKER && port != AUDIO_HEADPHONE && port != AUDIO_LINE_OUT) {
                debug_msg("Port not recognized\n");
                port = AUDIO_SPEAKER;
        }
        dev_info.play.port = port;
        if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0) {
                perror("Setting port");
        }
}

audio_port_t
sparc_audio_oport_get(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);
	if (ioctl(audio_fd, AUDIO_GETINFO, (caddr_t)&dev_info) < 0)
		perror("Getting port");
	return (dev_info.play.port);
}

int
sparc_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_OUT_PORTS;
}

const audio_port_details_t*
sparc_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        if (idx >= 0 && idx < (int)NUM_OUT_PORTS) {
                return &out_ports[idx];
        }
        return NULL;
}

static const audio_port_details_t in_ports[] = {
        { AUDIO_MICROPHONE, AUDIO_PORT_MICROPHONE},
        { AUDIO_LINE_IN,    AUDIO_PORT_LINE_IN},
        { AUDIO_CD,         AUDIO_PORT_CD}
};

#define NUM_IN_PORTS (sizeof(out_ports)/sizeof(out_ports[0]))

void
sparc_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
        int old_port, cur_port;
        UNUSED(ad); assert(audio_fd > 0);

        if (port != AUDIO_MICROPHONE && port != AUDIO_LINE_IN && port != AUDIO_CD) {
                port = AUDIO_MICROPHONE;
        }

        old_port = sparc_audio_iport_get(ad);

	AUDIO_INITINFO(&dev_info);
	dev_info.record.port = port;
	if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0) {
		perror("Setting port");
                /* If no CD-rom present then setting port fails silently... 
                 * not actually tested this code on a machine with a CD-rom
                 * since some kind folks stole it from g11... Fallbacks...
                 */
        }

        cur_port = sparc_audio_iport_get(ad);
        if (cur_port == 0 && port == AUDIO_CD && old_port == AUDIO_MICROPHONE) {
                debug_msg("CD failed trying line\n");
                sparc_audio_iport_set(ad, AUDIO_LINE_IN);
        } else if (cur_port == 0 && port == AUDIO_CD && old_port == AUDIO_LINE_IN) {
                debug_msg("CD failed trying mic\n");
                sparc_audio_iport_set(ad, AUDIO_MICROPHONE);
        }
}

audio_port_t
sparc_audio_iport_get(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	AUDIO_INITINFO(&dev_info);
	if (ioctl(audio_fd, AUDIO_GETINFO, (caddr_t)&dev_info) < 0)
		perror("Getting port");

	return (dev_info.record.port);
}

int
sparc_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_IN_PORTS;
}

const audio_port_details_t*
sparc_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        if (idx >= 0 && idx < (int)NUM_IN_PORTS) {
                return &in_ports[idx];
        }
        return NULL;
}

int
sparc_audio_duplex(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

        return 1;
}

static int
sparc_audio_select(audio_desc_t ad, int delay_us)
{
        fd_set rfds;
        struct timeval tv, s1, s2;

        UNUSED(ad); assert(audio_fd > 0);

        tv.tv_sec = 0;
        tv.tv_usec = delay_us;

        FD_ZERO(&rfds);
        FD_SET(audio_fd, &rfds);

        gettimeofday (&s1, 0);
        select(audio_fd+1, &rfds, NULL, NULL, &tv);
        gettimeofday (&s2, 0);

        s2.tv_usec -= s1.tv_usec;
        s2.tv_sec  -= s1.tv_sec;
        
        if (s2.tv_usec < 0) {
                s2.tv_usec += 1000000;
                s2.tv_sec  -= 1;
        }

/*        printf("delay %d pause %ld\n", delay_us, s2.tv_usec / 1000); */

        return FD_ISSET(audio_fd, &rfds);
}

void
sparc_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        UNUSED(ad); assert(audio_fd > 0);
        sparc_audio_select(ad, delay_ms * 1000);
}

int 
sparc_audio_is_ready(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);
        return sparc_audio_select(ad, 0);
}

