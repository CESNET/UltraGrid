/*
 * FILE:    audio_hw/null.c
 * PROGRAM: RAT
 * AUTHOR:  Orion Hodson 
 *
 * Copyright (c)      2003 University of Southern California
 * Copyright (c) 1996-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */

#include "config_unix.h"
#include "config_win32.h"
#include "audio_types.h"
#include "audio_util.h"
#include "audio_fmt.h"
#include "audio_hw/null.h"
#include "memory.h"
#include "debug.h"

static audio_format ifmt, ofmt;
static int audio_fd = -1;
static struct timeval last_read_time;

static int avail_bytes; /* Number of bytes available because of under read */

static int read_virgin = 1;

static int igain = 50, ogain = 50;

#ifdef WIN32
HANDLE   hAudioWakeUp;   /* Event that gets blocked for in null_audio_wait_for on Win32 */ 
#endif

int
null_audio_open(audio_desc_t ad, audio_format *infmt, audio_format *outfmt)
{
        if (audio_fd != -1) {
                debug_msg("Warning device not closed before opening.\n");
                null_audio_close(ad);
        }

        memcpy(&ifmt, infmt,  sizeof(ifmt));
        memcpy(&ofmt, outfmt, sizeof(ofmt));

        avail_bytes = 0;
        read_virgin  = TRUE;

#ifdef WIN32
        hAudioWakeUp   = CreateEvent(NULL, FALSE, FALSE, "RAT Null Audio Device Event");
#endif

	return TRUE;
}

/*
 * Shutdown.
 */
void
null_audio_close(audio_desc_t ad)
{
        UNUSED(ad);
	if (audio_fd > 0)
                audio_fd = -1;
#ifdef WIN32
        CloseHandle(hAudioWakeUp);
#endif

        return;
}

/*
 * Flush input buffer.
 */
void
null_audio_drain(audio_desc_t ad)
{
        UNUSED(ad);
        read_virgin = TRUE;
	return;
}

/*
 * Set record gain.
 */
void
null_audio_set_igain(audio_desc_t ad, int gain)
{
        UNUSED(ad);
        igain = gain;
	return;
}

/*
 * Get record gain.
 */
int
null_audio_get_igain(audio_desc_t ad)
{
        UNUSED(ad);
	return igain;
}

int
null_audio_duplex(audio_desc_t ad)
{
        UNUSED(ad);
        return TRUE;
}

/*
 * Set play gain.
 */
void
null_audio_set_ogain(audio_desc_t ad, int vol)
{
        UNUSED(ad);
        ogain = vol;
	return;
}

/*
 * Get play gain.
 */
int
null_audio_get_ogain(audio_desc_t ad)
{
        UNUSED(ad);
	return ogain;
}

static int
time_diff_to_bytes(struct timeval *start, struct timeval *end)
{
        int diff_ms, diff_bytes;
        diff_ms = (end->tv_sec  - start->tv_sec) * 1000 + (end->tv_usec - start->tv_usec) / 1000;
        diff_bytes = diff_ms * (ifmt.bits_per_sample / 8 ) * (ifmt.sample_rate / 1000) * ifmt.channels;

        return diff_bytes;
}


/*
 * Record audio data.
 */
int
null_audio_read(audio_desc_t ad, u_char *buf, int buf_bytes)
{
	int	                read_bytes;
	struct timeval          curr_time;

        UNUSED(ad);

	if (read_virgin) {
		gettimeofday(&last_read_time, NULL);
                avail_bytes = 0;
		read_virgin = FALSE;
	}

	gettimeofday(&curr_time, NULL);
        read_bytes = time_diff_to_bytes(&last_read_time, &curr_time);

        if (read_bytes + avail_bytes < ifmt.bytes_per_block) {
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
null_audio_write(audio_desc_t ad, u_char *buf, int write_bytes)
{
        UNUSED(buf);
        UNUSED(ad);

	return write_bytes;
}

/*
 * Set options on audio device to be non-blocking.
 */
void
null_audio_non_block(audio_desc_t ad)
{
        UNUSED(ad);
}

/*
 * Set options on audio device to be blocking.
 */
void
null_audio_block(audio_desc_t ad)
{
        UNUSED(ad);
}

#define NULL_SPEAKER    0x0101
#define NULL_MICROPHONE 0x0201

static audio_port_details_t out_ports[] = {
        {NULL_SPEAKER, AUDIO_PORT_SPEAKER}
};

static audio_port_details_t in_ports[] = {
        {NULL_MICROPHONE, AUDIO_PORT_MICROPHONE}
};

/*
 * Set output port.
 */
void
null_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
        UNUSED(ad); UNUSED(port);

	return;
}

/*
 * Get output port.
 */

audio_port_t
null_audio_oport_get(audio_desc_t ad)
{
        UNUSED(ad);
	return out_ports[0].port;
}

int
null_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return 1;
}

const audio_port_details_t*
null_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert(idx == 0);
        return &out_ports[0];
}

/*
 * Set input port.
 */
void
null_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
        UNUSED(ad);
        UNUSED(port);
	return;
}

/*
 * Get input port.
 */
audio_port_t
null_audio_iport_get(audio_desc_t ad)
{
        UNUSED(ad);
	return in_ports[0].port;
}

int
null_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return 1;
}

const audio_port_details_t*
null_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert(idx == 0);
        return &in_ports[0];
}

/*
 * Enable hardware loopback
 */
void 
null_audio_loopback(audio_desc_t ad, int gain)
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
null_audio_is_ready(audio_desc_t ad)
{
        struct timeval now;
        uint32_t diff;

        UNUSED(ad);

        gettimeofday(&now,NULL);
	diff = (now.tv_sec  - last_read_time.tv_sec) * 1000 + (now.tv_usec - last_read_time.tv_usec)/1000;
        diff *= (ifmt.bits_per_sample / 8) * ifmt.sample_rate / 1000 * ifmt.channels;

        if ((diff + avail_bytes) > (unsigned)ifmt.bytes_per_block) return TRUE;

        return FALSE;
}

static void 
null_audio_select(audio_desc_t ad, int delay_ms)
{
        int needed, dur;

        UNUSED(ad);

        needed = ifmt.bytes_per_block - avail_bytes;
        assert(needed >= 0);

        dur = needed * 1000 * 8 / (ifmt.sample_rate * ifmt.bits_per_sample * ifmt.channels);
        dur = min(dur, delay_ms);
#ifndef WIN32
        usleep(dur * 1000);
#else
        WaitForSingleObject(hAudioWakeUp, 100); /* Will time out since event never triggered */
#endif
}

void
null_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        if (null_audio_is_ready(ad) == FALSE) {
                null_audio_select(ad, delay_ms);
        }
}

int
null_audio_device_count()
{
        return 1;
}

char*
null_audio_device_name(audio_desc_t ad)
{
        UNUSED(ad);
        return "No Audio Device";
}

int
null_audio_supports(audio_desc_t ad, audio_format *fmt)
{
        UNUSED(ad);
        if ((!(fmt->sample_rate % 8000) || !(fmt->sample_rate % 11025)) && 
            (fmt->channels == 1 || fmt->channels == 2)) {
                return TRUE;
        }
        return FALSE;
}
