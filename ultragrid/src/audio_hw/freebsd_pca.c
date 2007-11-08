/*
 * FILE:     audio_hw/freebsd_pca.c
 * PROGRAM:  RAT
 * AUTHOR:   Jim Lowe (james@cs.uwm.edu) 
 * MODIFIED: Orion Hodson
 *
 * Copyright (c) 2002 University of Southern California
 * Copyright (c) 1996-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 * PCA speaker support for FreeBSD.
 * This is an output only device so we pretend we read audio...
 */

#include <machine/pcaudioio.h>
#include "config_unix.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/freebsd_pca.h"
#include "audio_codec/g711.h"
#include "memory.h"
#include "debug.h"

static audio_info_t   dev_info;			/* For PCA device */
static int            audio_fd;
static struct timeval last_read_time;
static int            bytes_per_block;
static int            present;

static int avail_bytes; /* Number of bytes available because of under read */

#define pca_bat_to_device(x)	((x) * AUDIO_MAX_GAIN / MAX_AMP)
#define pca_device_to_bat(x)	((x) * MAX_AMP / AUDIO_MAX_GAIN)

int
pca_audio_device_count()
{
        return present;
}

char*
pca_audio_device_name(audio_desc_t ad)
{
        UNUSED(ad);
        return "PCA Audio Device";
}

int
pca_audio_init()
{
        int audio_fd;
        if ((audio_fd = open("/dev/pcaudio", O_WRONLY | O_NDELAY)) != -1) {
                close(audio_fd);
                present = 1;
                return TRUE;
        }
        return FALSE;
}

/*
 * Try to open the audio device.
 * Return: valid file descriptor if ok, -1 otherwise.
 */

int
pca_audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
	audio_info_t tmp_info;

        UNUSED(ofmt);
        assert(audio_format_match(ifmt, ofmt));

        if (ifmt->sample_rate != 8000 || ifmt->channels != 1) {
                return FALSE;
        }

        avail_bytes = 0;

	audio_fd = open("/dev/pcaudio", O_WRONLY | O_NDELAY );

	if (audio_fd > 0) {
		AUDIO_INITINFO(&dev_info);
		dev_info.monitor_gain     = 0;
		dev_info.play.sample_rate = ifmt->sample_rate;
		dev_info.play.channels    = ifmt->channels;
		dev_info.play.gain	      = (AUDIO_MAX_GAIN - AUDIO_MIN_GAIN) * 0.75;
		dev_info.play.port	      = 0;

                if (ifmt->encoding != DEV_PCMU) {
                        audio_format_change_encoding(ifmt, DEV_PCMU);
                }

                if (ofmt->encoding != DEV_PCMU) {
                        audio_format_change_encoding(ofmt, DEV_PCMU);
                }

                assert(ifmt->bits_per_sample == 8);
                dev_info.play.encoding  = AUDIO_ENCODING_ULAW;
                dev_info.play.precision   = 8;

                bytes_per_block = ofmt->bytes_per_block;

		memcpy(&tmp_info, &dev_info, sizeof(audio_info_t));

		if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&tmp_info) < 0) {
			perror("pca_audio_info: setting parameters");
                        pca_audio_close(ad);
			return FALSE;
		}

                return TRUE;
	} else {
		/* 
		 * Because we opened the device with O_NDELAY, the wait
		 * flag was not updaed so update it manually.
		 */
		audio_fd = open("/dev/pcaudioctl", O_WRONLY);
		if (audio_fd < 0) {
			AUDIO_INITINFO(&dev_info);
			dev_info.play.waiting = 1;
			(void)ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info);
			close(audio_fd);
		}
		audio_fd = -1;
	}
	return FALSE;
}

/*
 * Shutdown.
 */
void
pca_audio_close(audio_desc_t ad)
{
        UNUSED(ad);
	if(audio_fd > 0)
		(void)close(audio_fd);
	audio_fd = -1;
	return;
}

/*
 * Flush input buffer.
 */
void
pca_audio_drain(audio_desc_t ad)
{
        UNUSED(ad);
        avail_bytes = 0;
	return;
}

/*
 * Set record gain.
 */
void
pca_audio_set_igain(audio_desc_t ad, int gain)
{
        UNUSED(ad);
        UNUSED(gain);
	return;
}

/*
 * Get record gain.
 */
int
pca_audio_get_igain(audio_desc_t ad)
{
        UNUSED(ad);
	return 0;
}

int
pca_audio_duplex(audio_desc_t ad)
{
        UNUSED(ad);
        /* LIE! LIE! LIE! LIE! 
         * But we really only support full duplex devices
         */
        return TRUE;
}

/*
 * Set play gain.
 */
void
pca_audio_set_ogain(audio_desc_t ad, int vol)
{
        UNUSED(ad);
        AUDIO_INITINFO(&dev_info);
        dev_info.play.gain = pca_bat_to_device(vol);
        if (ioctl(audio_fd, AUDIO_SETINFO, (caddr_t)&dev_info) < 0) 
                perror("pca_audio_set_ogain");

	return;
}

/*
 * Get play gain.
 */
int
pca_audio_get_ogain(audio_desc_t ad)
{
        UNUSED(ad);
	AUDIO_INITINFO(&dev_info);
	if (ioctl(audio_fd, AUDIO_GETINFO, (caddr_t)&dev_info) < 0)
		perror("pca_audio_get_ogain");
	return pca_device_to_bat(dev_info.play.gain);
}

/*
 * Record audio data.
 */
int
pca_audio_read(audio_desc_t ad, u_char *buf, int buf_bytes)
{
	/*
	 * Reading data from internal PC speaker is a little difficult,
	 * so just return the time (in audio samples) since the last time called.
	 */
	int	                read_bytes;
	struct timeval          curr_time;
	static int              virgin = TRUE;

        UNUSED(ad);

	if (virgin) {
		gettimeofday(&last_read_time, NULL);
		virgin = FALSE;
	}

	gettimeofday(&curr_time, NULL);
	read_bytes = (curr_time.tv_sec  - last_read_time.tv_sec) * 1000 + (curr_time.tv_usec - last_read_time.tv_usec) / 1000;
        /* diff from ms to samples */
        read_bytes *= dev_info.play.sample_rate / 1000 * dev_info.play.precision / 8 * dev_info.play.channels;

        if (read_bytes + avail_bytes < bytes_per_block) {
                return 0;
        }

        if (buf_bytes > read_bytes + avail_bytes) {
                /* Have requested more bytes than we read this time and
                 * are available in reserve.
                 */
                read_bytes += avail_bytes;
                avail_bytes = 0;
        } else {
                avail_bytes += read_bytes - buf_bytes;
                read_bytes   = buf_bytes;
        }
        assert(avail_bytes >= 0);

        memcpy(&last_read_time, &curr_time, sizeof(struct timeval));
        memset(buf, 0, read_bytes);
        return read_bytes;
}

/*
 * Playback audio data.
 */
int
pca_audio_write(audio_desc_t ad, u_char *buf, int write_bytes)
{
	int	 nbytes;

        UNUSED(ad);

        if ((nbytes = write(audio_fd, buf, write_bytes)) != write_bytes) {
		if (errno == EWOULDBLOCK) {	/* XXX */
                        perror("pca_audio_write");
			return 0;
		}
		if (errno != EINTR) {
			perror("pca_audio_write");
			return (write_bytes - nbytes);
		}
	} 
    
	return write_bytes;
}

/*
 * Set options on audio device to be non-blocking.
 */
void
pca_audio_non_block(audio_desc_t ad)
{
	int on = 1;

        UNUSED(ad);

	if (ioctl(audio_fd, FIONBIO, (char *)&on) < 0)
		perror("pca_audio_non_block");
 
	return;
}

/*
 * Set options on audio device to be blocking.
 */
void
pca_audio_block(audio_desc_t ad)
{
	int on = 0;

        UNUSED(ad);

	if (ioctl(audio_fd, FIONBIO, (char *)&on) < 0)
		perror("pca_audio_block");
	return;
}

#define PCA_SPEAKER    0x0101
#define PCA_MICROPHONE 0x0201

static audio_port_details_t in_ports[] = {
        { PCA_MICROPHONE, AUDIO_PORT_MICROPHONE}
};

static audio_port_details_t out_ports[] = {
        { PCA_SPEAKER,    AUDIO_PORT_SPEAKER}
};

/*
 * Set output port.
 */
void
pca_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
	/* There is only one port... */
        UNUSED(ad); UNUSED(port);

        assert(port == PCA_SPEAKER);

	return;
}

/*
 * Get output port.
 */
audio_port_t
pca_audio_oport_get(audio_desc_t ad)
{
	/* There is only one port... */
        UNUSED(ad);

	return out_ports[0].port;
}

int
pca_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return 1;
}

const audio_port_details_t*
pca_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        UNUSED(idx);
        assert(idx == 0);
        return &out_ports[0];
}

/*
 * Set input port.
 */
void
pca_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	/* Hmmm.... */
        UNUSED(ad);
        UNUSED(port);
	return;
}

/*
 * Get input port.
 */
audio_port_t
pca_audio_iport_get(audio_desc_t ad)
{
	/* Hmm...hack attack */
        UNUSED(ad);
	return in_ports[0].port;
}

int
pca_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return 1;
}

const audio_port_details_t*
pca_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        UNUSED(idx);
        assert(idx == 0);
        return &in_ports[0];
}

/*
 * Enable hardware loopback
 */
void 
pca_audio_loopback(audio_desc_t ad, int gain)
{
        UNUSED(ad);
        UNUSED(gain);
        /* Nothing doing... */
}

/*
 * For external purposes this function returns non-zero
 * if audio is ready.
 */
int
pca_audio_is_ready(audio_desc_t ad)
{
        struct timeval now;
        uint32_t read_bytes;

        UNUSED(ad);

        gettimeofday(&now,NULL);
	read_bytes = (now.tv_sec  - last_read_time.tv_sec) * 1000 + (now.tv_usec - last_read_time.tv_usec)/1000;
        read_bytes *= dev_info.play.sample_rate / 1000 * dev_info.play.precision / 8 * dev_info.play.channels;

        if (read_bytes + avail_bytes > (unsigned)bytes_per_block) return TRUE;
        return FALSE;
}

void
pca_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        if (pca_audio_is_ready(ad)) {
                return;
        } else {
                usleep(delay_ms * 1000);
        }
}

int
pca_audio_supports(audio_desc_t ad, audio_format *fmt)
{
        UNUSED(ad);
        if (fmt->channels == 1 && fmt->sample_rate == 8000) return TRUE;
        return FALSE;
}
