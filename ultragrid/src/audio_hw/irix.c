/*
 * FILE:     audio_hw/irix.c
 * PROGRAM:  RAT
 * AUTHORS:  Isidor Kouvelas + Colin Perkins + Orion Hodson
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include <audio.h>
#include "config_unix.h"
#include "config_win32.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/irix.h"
#include "debug.h"

#define QSIZE		16000		/* Two seconds for now... */
#define AVG_SIZE	20000
#define SAMSIG		7500.0
#define BASEOFF         0x0a7f

#define SGI_OPORT_SPEAKER    0x0101
#define SGI_IPORT_MICROPHONE 0x0203
#define SGI_IPORT_LINE_IN    0x0202
#define SGI_IPORT_APANEL     0x0201

#define RAT_TO_SGI_DEVICE(x)	((x) * 255 / MAX_AMP)
#define SGI_DEVICE_TO_RAT(x)	((x) * MAX_AMP / 255)

static int      audio_fd = -1;
static ALport	rp = NULL, wp = NULL;	/* Read and write ports */
static int      bytes_per_block;
static audio_port_t iport = SGI_IPORT_APANEL;

int
sgi_audio_device_count()
{
        return 1;
}

char*
sgi_audio_device_name(audio_desc_t ad)
{
        UNUSED(ad);
        return "SGI Audio Device";
}

/*
 * Try to open the audio device.
 * Return TRUE if successfull FALSE otherwise.
 */

int
sgi_audio_open(audio_desc_t ad, audio_format* ifmt, audio_format *ofmt)
{
	ALconfig	c;
	int		cmd[8];

        if (audio_fd != -1) {
                sgi_audio_close(ad);
        }

        if (ifmt->encoding != DEV_S16) return FALSE;

	if ((c = ALnewconfig()) == NULL) {
		fprintf(stderr, "ALnewconfig error\n");
		exit(1);
	}

        switch(ifmt->channels) {
        case 1:
                ALsetchannels(c, AL_MONO); break;
        case 2:
                ALsetchannels(c, AL_STEREO); break;
        default:
                sgi_audio_close(ad);
        }

	ALsetwidth(c, AL_SAMPLE_16);
	ALsetqueuesize(c, QSIZE);
	ALsetsampfmt(c, AL_SAMPFMT_TWOSCOMP);

	if ((wp = ALopenport("RAT write", "w", c)) == NULL) {
		fprintf(stderr, "ALopenport (write) error\n");
                sgi_audio_close(ad);
                return FALSE;
        }

	if ((rp = ALopenport("RAT read", "r", c)) == NULL) {
		fprintf(stderr, "ALopenport (read) error\n");
                sgi_audio_close(ad);
                return FALSE;
        }

	cmd[0] = AL_OUTPUT_RATE;
	cmd[1] = ofmt->sample_rate;
	cmd[2] = AL_INPUT_RATE;
	cmd[3] = ifmt->sample_rate;
	cmd[4] = AL_MONITOR_CTL;
	cmd[5] = AL_MONITOR_OFF;
	/*cmd[6] = AL_INPUT_SOURCE;*/
	/*cmd[7] = AL_INPUT_MIC;*/

	if (ALsetparams(AL_DEFAULT_DEVICE, cmd, 6L/*was 8L*/) == -1) {
		fprintf(stderr, "audio_open/ALsetparams error\n");
                sgi_audio_close(ad);
        }

	/* Get the file descriptor to use in select */
	audio_fd = ALgetfd(rp);

	if (ALsetfillpoint(rp, ifmt->bytes_per_block) < 0) {
                debug_msg("ALsetfillpoint failed (%d samples)\n", ifmt->bytes_per_block);
        }
        bytes_per_block = ifmt->bytes_per_block;
        
	/* We probably should free the config here... */
        
	return TRUE;
}

/* Close the audio device */
void
sgi_audio_close(audio_desc_t ad)
{
        UNUSED(ad);
	ALcloseport(rp);
	ALcloseport(wp);
        audio_fd = -1;
}

/* Flush input buffer */
void
sgi_audio_drain(audio_desc_t ad)
{
	u_char buf[QSIZE];

        UNUSED(ad); assert(audio_fd > 0);

	while(sgi_audio_read(audio_fd, buf, QSIZE) == QSIZE);
}

/* Gain and volume values are in the range 0 - MAX_AMP */

void
sgi_audio_set_igain(audio_desc_t ad, int gain)
{
	int	cmd[4];

        UNUSED(ad); assert(audio_fd > 0);

	cmd[0] = AL_LEFT_INPUT_ATTEN;
	cmd[1] = 255 - RAT_TO_SGI_DEVICE(gain);
	cmd[2] = AL_RIGHT_INPUT_ATTEN;
	cmd[3] = cmd[1];
	ALsetparams(AL_DEFAULT_DEVICE, cmd, 4L);
}

int
sgi_audio_get_igain(audio_desc_t ad)
{
	int	cmd[2];

        UNUSED(ad); assert(audio_fd > 0);

	cmd[0] = AL_LEFT_INPUT_ATTEN;
	ALgetparams(AL_DEFAULT_DEVICE, cmd, 2L);
	return (MAX_AMP - SGI_DEVICE_TO_RAT(cmd[1]));
}

void
sgi_audio_set_ogain(audio_desc_t ad, int vol)
{
	int	cmd[4];

        UNUSED(ad); assert(audio_fd > 0);

	cmd[0] = AL_LEFT_SPEAKER_GAIN;
	cmd[1] = RAT_TO_SGI_DEVICE(vol);
	cmd[2] = AL_RIGHT_SPEAKER_GAIN;
	cmd[3] = cmd[1];
	ALsetparams(AL_DEFAULT_DEVICE, cmd, 4L);
}

int
sgi_audio_get_ogain(audio_desc_t ad)
{
	int	cmd[2];

        UNUSED(ad); assert(audio_fd > 0);

	cmd[0] = AL_LEFT_SPEAKER_GAIN;
	ALgetparams(AL_DEFAULT_DEVICE, cmd, 2L);
	return (SGI_DEVICE_TO_RAT(cmd[1]));
}

static int non_block = 1;	/* Initialise to non blocking */

int
sgi_audio_read(audio_desc_t ad, u_char *buf, int buf_bytes)
{
	int   		len;

        UNUSED(ad); assert(audio_fd > 0);
        
	if (non_block) {
		if ((len = ALgetfilled(rp)) < bytes_per_block )
			return (0);
                len = min(len, buf_bytes);
	} else {
		len = (int)buf_bytes;
        }

        len /= BYTES_PER_SAMPLE; /* We only open device in 16 bit mode */

	if (len > QSIZE) {
		fprintf(stderr, "audio_read: too big!\n");
		len = QSIZE;
	}

	ALreadsamps(rp, buf, len);

	return ((int)len * BYTES_PER_SAMPLE);
}

int
sgi_audio_write(audio_desc_t ad, u_char *buf, int buf_bytes)
{
        int samples;
        UNUSED(ad); assert(audio_fd > 0);

        samples = buf_bytes / BYTES_PER_SAMPLE;
	if (samples > QSIZE) {
		fprintf(stderr, "audio_write: too big!\n");
		samples = QSIZE;
	}

	/* Will block */

	ALwritesamps(wp, buf, samples);
	return buf_bytes;
}

/* Set ops on audio device to be non-blocking */
void
sgi_audio_non_block(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	non_block = 1;
}

/* Set ops on audio device to block */
void
sgi_audio_block(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);

	non_block = 0;
}

static audio_port_details_t out_ports[] = {
        { SGI_OPORT_SPEAKER, AUDIO_PORT_SPEAKER}
};

#define SGI_NUM_OPORTS (sizeof(out_ports)/sizeof(out_ports[0]))

static audio_port_details_t in_ports[] = {
	{ SGI_IPORT_APANEL,	"APanel" },
        { SGI_IPORT_MICROPHONE, AUDIO_PORT_MICROPHONE },
        { SGI_IPORT_LINE_IN,    AUDIO_PORT_LINE_IN }
};

#define SGI_NUM_IPORTS (sizeof(in_ports)/sizeof(in_ports[0]))

void
sgi_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
        UNUSED(ad); 
        assert(audio_fd > 0);
        assert(port == SGI_OPORT_SPEAKER);
        UNUSED(port);
}

audio_port_t
sgi_audio_oport_get(audio_desc_t ad)
{
        UNUSED(ad); 
        assert(audio_fd > 0);

	return (SGI_OPORT_SPEAKER);
}

int 
sgi_audio_oport_count(audio_desc_t ad) 
{
        UNUSED(ad);
        return (int)SGI_NUM_OPORTS;
}

const audio_port_details_t*
sgi_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert((unsigned)idx < SGI_NUM_OPORTS && idx >= 0);
        return &out_ports[idx];
}

void
sgi_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	int pvbuf[2];

        UNUSED(ad); assert(audio_fd > 0);

        switch(port) {
        case SGI_IPORT_LINE_IN: 
                pvbuf[0] = AL_INPUT_SOURCE;
                pvbuf[1] = AL_INPUT_LINE;
                ALsetparams(AL_DEFAULT_DEVICE, pvbuf, 2);
                break;
        case SGI_IPORT_MICROPHONE: 
        default:
                pvbuf[0] = AL_INPUT_SOURCE;
                pvbuf[1] = AL_INPUT_MIC;
                ALsetparams(AL_DEFAULT_DEVICE, pvbuf, 2);
                break;
	case SGI_IPORT_APANEL:
		break;
        }
        iport = port;
}

audio_port_t
sgi_audio_iport_get(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);
	return iport;
}

int
sgi_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);
        return (int)SGI_NUM_IPORTS;
}

const audio_port_details_t*
sgi_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert(idx < (int)SGI_NUM_IPORTS && idx >= 0);
        return &in_ports[idx];
}

void 
sgi_audio_loopback(audio_desc_t ad, int gain)
{
        int pvbuf[4];
        int  pvcnt;

        UNUSED(ad); assert(audio_fd > 0);

        pvcnt = 2;
        pvbuf[0] = AL_MONITOR_CTL;
        pvbuf[1] = AL_MONITOR_OFF;

        if (gain) {
                pvcnt = 6;
                pvbuf[1] = AL_MONITOR_ON;
                pvbuf[2] = AL_LEFT_MONITOR_ATTEN;
                pvbuf[3] = 255 - RAT_TO_SGI_DEVICE(gain);
                pvbuf[4] = AL_RIGHT_MONITOR_ATTEN;
                pvbuf[5] = pvbuf[3];
        }
        
        if (ALsetparams(AL_DEFAULT_DEVICE, pvbuf, pvcnt) != 0) {
                debug_msg("loopback failed\n");
        }
}

int
sgi_audio_duplex(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);
        return 1;
}

int  
sgi_audio_is_ready(audio_desc_t ad)
{
        UNUSED(ad); assert(audio_fd > 0);
        
        if (ALgetfilled(rp) >= ALgetfillpoint(rp)) {
                return TRUE;
        } else {
                return FALSE;
        }
}

void 
sgi_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        struct timeval tv;
        fd_set rfds;
        UNUSED(ad); assert(audio_fd > 0);

        tv.tv_sec  = 0;
        tv.tv_usec = delay_ms * 1000;

        FD_ZERO(&rfds);
        FD_SET(audio_fd, &rfds);
        
        select(audio_fd + 1, &rfds, NULL, NULL, &tv);
}

