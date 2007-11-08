/*
 * FILE: audio_hw/freebsd_luigi.c - Sound interface for Luigi Rizzo's FreeBSD driver
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1996-2001 University College London
 * All rights reserved.
 *
 * AUTHOR: Orion Hodson
 * MODIFIED: Colin Perkins
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config_unix.h"
#include "config_win32.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/freebsd_luigi.h"
#include "memory.h"
#include "debug.h"

#include <machine/pcaudioio.h>
#include <sys/soundcard.h>

#define LUIGI_SPEAKER    0x101
#define LUIGI_MICROPHONE 0x201
#define LUIGI_LINE_IN    0x202
#define LUIGI_CD         0x203

static int iport = LUIGI_MICROPHONE;
static snd_chan_param pa;
static struct snd_size sz;
static int audio_fd = -1;

#define RAT_TO_DEVICE(x) ((x) * 100 / MAX_AMP)
#define DEVICE_TO_RAT(x) ((x) * MAX_AMP / 100)

#define LUIGI_AUDIO_IOCTL(fd, cmd, val) if (ioctl((fd), (cmd), (val)) < 0) { \
                                            debug_msg("Failed %s\n",#cmd); \
                                            luigi_error = __LINE__; \
                                               }

#define LUIGI_MAX_AUDIO_NAME_LEN 32
#define LUIGI_MAX_AUDIO_DEVICES  3

static int dev_ids[LUIGI_MAX_AUDIO_DEVICES];
static char names[LUIGI_MAX_AUDIO_DEVICES][LUIGI_MAX_AUDIO_NAME_LEN];
static int ndev = 0;
static int luigi_error;
static audio_format *input_format, *output_format, *tmp_format;
static snd_capabilities soundcaps[LUIGI_MAX_AUDIO_DEVICES];

int 
luigi_audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
        int             reclb = 0;
        
        char            thedev[64];
        
        assert(ad >= 0 && ad < ndev); 
	sprintf(thedev, "/dev/audio%d", dev_ids[ad]);

        debug_msg("Opening %s\n", thedev);

        audio_fd = open(thedev, O_RDWR);
        if (audio_fd >= 0) {
                /* Ignore any earlier errors */
                luigi_error = 0;

                LUIGI_AUDIO_IOCTL(audio_fd, AIOGCAP, &soundcaps[ad]);
		debug_msg("soundcaps[%d].rate_min = %d\n", ad, soundcaps[ad].rate_min);
		debug_msg("soundcaps[%d].rate_max = %d\n", ad, soundcaps[ad].rate_max);
		debug_msg("soundcaps[%d].formats  = 0x%08lx\n", ad, soundcaps[ad].formats);
                debug_msg("soundcaps[%d].bufsize  = %d\n", ad, soundcaps[ad].bufsize);
		debug_msg("soundcaps[%d].mixers   = 0x%08lx\n", ad, soundcaps[ad].mixers);
		debug_msg("soundcaps[%d].inputs   = 0x%08lx\n", ad, soundcaps[ad].inputs);
		debug_msg("soundcaps[%d].left     = 0x%04lx\n", ad, soundcaps[ad].left);
		debug_msg("soundcaps[%d].right    = 0x%04lx\n", ad, soundcaps[ad].right);

		/* XXX why do we reset here ??? [oh] */
                LUIGI_AUDIO_IOCTL(audio_fd,SNDCTL_DSP_RESET,0);

		/* Check card is full duplex - need for Luigi driver only */
		if ((soundcaps[ad].formats & AFMT_FULLDUPLEX) == 0) {
			     fprintf(stderr, "Sorry driver does support full duplex for this soundcard\n");
			     luigi_audio_close(ad);
			     return FALSE;
		}

		if (soundcaps[ad].formats & AFMT_WEIRD) {
                        /* this is a sb16/32/64... 
                         * you can change either ifmt or ofmt to U8 
                         * NOTE: No other format supported in driver at this time!
                         * to work around broken hardware here.  By default
                         * we use the 16bit channel for output and 8bit
                         * for input since most people probably want to
                         * listen to the radio. 
                         */
                        debug_msg("Weird Hardware\n");

                        audio_format_change_encoding(ifmt, DEV_U8);
		}

                /* Setup input and output format settings */
                assert(ofmt->channels == ifmt->channels);
                memset(&pa, 0, sizeof(pa));
                if (ifmt->channels == 2) {
                        if (!soundcaps[ad].formats & AFMT_STEREO) {
                                fprintf(stderr,"Driver does not support stereo for this soundcard\n");
                                luigi_audio_close(ad);
                                return FALSE;
                        }
                        pa.rec_format  = AFMT_STEREO;
                        pa.play_format = AFMT_STEREO;
                }

                switch(ifmt->encoding) {
                case DEV_PCMU: pa.rec_format |= AFMT_MU_LAW; break;
                case DEV_PCMA: pa.rec_format |= AFMT_A_LAW;  break;
                case DEV_S8:   pa.rec_format |= AFMT_S8;     break;
                case DEV_S16:  pa.rec_format |= AFMT_S16_LE; break;
                case DEV_U8:   pa.rec_format |= AFMT_U8;     break;
                }

                switch(ofmt->encoding) {
                case DEV_PCMU: pa.play_format |= AFMT_MU_LAW; break;
                case DEV_PCMA: pa.play_format |= AFMT_A_LAW;  break;
                case DEV_S8:   pa.play_format |= AFMT_S8;     break;
                case DEV_S16:  pa.play_format |= AFMT_S16_LE; break;
                case DEV_U8:   pa.play_format |= AFMT_U8;     break;
                }
                pa.play_rate = ofmt->sample_rate;
                pa.rec_rate = ifmt->sample_rate;
                LUIGI_AUDIO_IOCTL(audio_fd, AIOSFMT, &pa);

                sz.play_size = ofmt->bytes_per_block;
                sz.rec_size  = ifmt->bytes_per_block;
                LUIGI_AUDIO_IOCTL(audio_fd, AIOSSIZE, &sz);

                LUIGI_AUDIO_IOCTL(audio_fd, AIOGSIZE, &sz);
                debug_msg("rec size %d, play size %d bytes\n",
                          sz.rec_size, sz.play_size);
                
                /* Set global gain/volume to maximum values. This may
                 * fail on some cards, but shouldn't cause any harm
                 * when it does..... */

                /* Select microphone input. We can't select output source...  */
                luigi_audio_iport_set(audio_fd, iport);

                if (luigi_error != 0) {
                        /* Failed somewhere in initialization - reset error and exit*/
                        luigi_audio_close(ad);
                        luigi_error = 0;
                        return FALSE;
                }

                /* Store format in case we have to re-open device because
                 * of driver bug.  Careful with freeing format as input format
                 * could be static input_format if device reset during write.
                 */
                tmp_format = audio_format_dup(ifmt);
                if (input_format != NULL) {
                        audio_format_free(&input_format);
                }
                input_format = tmp_format;

                tmp_format = audio_format_dup(ofmt);
                if (output_format != NULL) {
                        audio_format_free(&output_format);
                }
                output_format = tmp_format;

                /* Turn off loopback from input to output... not fatal so
                 * after error check.
                 */
                LUIGI_AUDIO_IOCTL(audio_fd, MIXER_WRITE(SOUND_MIXER_IMIX), &reclb);

                read(audio_fd, thedev, 64);
                return TRUE;
        } else {
		fprintf(stderr, 
			"Could not open device: %s (half-duplex?)\n", 
			names[ad]);
		perror("luigi_audio_open");
                luigi_audio_close(ad);
                return FALSE;
        }
}

/* Close the audio device */
void
luigi_audio_close(audio_desc_t ad)
{
        UNUSED(ad);
	if (audio_fd < 0) {
                debug_msg("Device already closed!\n");
                return;
        }
        if (input_format != NULL) {
                audio_format_free(&input_format);
        }
        if (output_format != NULL) {
                audio_format_free(&output_format);
        }

	luigi_audio_drain(audio_fd);
	close(audio_fd);
        audio_fd = -1;
}

/* Flush input buffer */
void
luigi_audio_drain(audio_desc_t ad)
{
        u_char buf[4];
        int pre, post;
        
        assert(audio_fd > 0);

        LUIGI_AUDIO_IOCTL(audio_fd, FIONREAD, &pre);
        LUIGI_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_RESET, 0);
        LUIGI_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_SYNC, 0);
        LUIGI_AUDIO_IOCTL(audio_fd, FIONREAD, &post);
        debug_msg("audio drain: %d -> %d\n", pre, post);
        read(audio_fd, buf, sizeof(buf));

        UNUSED(ad);
}

int
luigi_audio_duplex(audio_desc_t ad)
{
        /* We only ever open device full duplex! */
        UNUSED(ad);
        return TRUE;
}

int
luigi_audio_read(audio_desc_t ad, u_char *buf, int read_bytes)
{
        int done, this_read;
        int len;
        /* Figure out how many bytes we can read before blocking... */

        UNUSED(ad); assert(audio_fd > 0);

        LUIGI_AUDIO_IOCTL(audio_fd, FIONREAD, &len);

        len = min(len, read_bytes);

        /* Read the data... */
        done = 0;
        while(done < len) {
                this_read = read(audio_fd, (void*)buf, len - done);
                done += this_read;
                buf  += this_read;
        }
        return done;
}

int
luigi_audio_write(audio_desc_t ad, u_char *buf, int write_bytes)
{
	int done;

        UNUSED(ad); assert(audio_fd > 0);

        done = write(audio_fd, (void*)buf, write_bytes);
        if (done != write_bytes && errno != EINTR) {
                /* Only ever seen this with soundblaster cards.
                 * Driver occasionally packs in reading.  Seems to be
                 * no way to reset cleanly whilst running, even
                 * closing device, waiting a few 100ms and re-opening
                 * seems to fail.  
                 */
                perror("Error writing device.");
                return (write_bytes - done);
        }

        return write_bytes;
}

/* Set ops on audio device to be non-blocking */
void
luigi_audio_non_block(audio_desc_t ad)
{
	int             frag = 1;

	UNUSED(ad); assert(audio_fd != -1);

        LUIGI_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_NONBLOCK, &frag);
}

/* Set ops on audio device to be blocking */
void
luigi_audio_block(audio_desc_t ad)
{
  	int             frag = 0;
        
        UNUSED(ad); assert(audio_fd > 0);
        
        LUIGI_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_NONBLOCK, &frag);
} 

/* Gain and volume values are in the range 0 - MAX_AMP */
void
luigi_audio_set_ogain(audio_desc_t ad, int vol)
{
	int volume;

        UNUSED(ad); assert(audio_fd > 0);

	volume = vol << 8 | vol;
	LUIGI_AUDIO_IOCTL(audio_fd, MIXER_WRITE(SOUND_MIXER_PCM), &volume);
}

int
luigi_audio_get_ogain(audio_desc_t ad)
{
	int volume;

        UNUSED(ad); assert(audio_fd > 0);

	LUIGI_AUDIO_IOCTL(audio_fd, MIXER_READ(SOUND_MIXER_PCM), &volume);

	return DEVICE_TO_RAT(volume & 0xff); /* Extract left channel volume */
}

void
luigi_audio_loopback(audio_desc_t ad, int gain)
{
        UNUSED(ad); assert(audio_fd > 0);

        gain = gain << 8 | gain;

        LUIGI_AUDIO_IOCTL(audio_fd, MIXER_WRITE(SOUND_MIXER_IMIX), &gain);
}

void
luigi_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
	UNUSED(ad); assert(audio_fd > 0);
        UNUSED(port);
	return;
}

audio_port_t
luigi_audio_oport_get(audio_desc_t ad)
{
	UNUSED(ad); assert(audio_fd > 0);
	return LUIGI_SPEAKER;
}

int
luigi_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return 1;
}

static const audio_port_details_t out_ports[] = {{ LUIGI_SPEAKER, AUDIO_PORT_SPEAKER }};

const audio_port_details_t*
luigi_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        UNUSED(idx);
        return out_ports;
}

void
luigi_audio_set_igain(audio_desc_t ad, int gain)
{
	int volume = RAT_TO_DEVICE(gain) << 8 | RAT_TO_DEVICE(gain);

        UNUSED(ad); assert(audio_fd > 0);

	switch (iport) {
	case LUIGI_MICROPHONE:
                LUIGI_AUDIO_IOCTL(audio_fd, MIXER_WRITE(SOUND_MIXER_MIC), &volume);
	case LUIGI_LINE_IN:
                LUIGI_AUDIO_IOCTL(audio_fd, MIXER_WRITE(SOUND_MIXER_LINE), &volume);
		break;
	case LUIGI_CD:
                LUIGI_AUDIO_IOCTL(audio_fd, MIXER_WRITE(SOUND_MIXER_CD), &volume);
		break;
	}
	return;
}

int
luigi_audio_get_igain(audio_desc_t ad)
{
	int volume;

        UNUSED(ad); assert(audio_fd > 0);

	switch (iport) {
	case LUIGI_MICROPHONE:
		LUIGI_AUDIO_IOCTL(audio_fd, MIXER_READ(SOUND_MIXER_MIC), &volume);
		break;
	case LUIGI_LINE_IN:
		LUIGI_AUDIO_IOCTL(audio_fd, MIXER_READ(SOUND_MIXER_LINE), &volume);
		break;
	case LUIGI_CD:
		LUIGI_AUDIO_IOCTL(audio_fd, MIXER_READ(SOUND_MIXER_CD), &volume);
		break;
	default:
		debug_msg("ERROR: Unknown iport in audio_set_igain!\n");
	}
	return (DEVICE_TO_RAT(volume & 0xff));
}

static audio_port_details_t in_ports[] = {
        { LUIGI_MICROPHONE, AUDIO_PORT_MICROPHONE},
        { LUIGI_LINE_IN,    AUDIO_PORT_LINE_IN},
        { LUIGI_CD,         AUDIO_PORT_CD}
};

#define NUM_IN_PORTS (sizeof(in_ports)/sizeof(in_ports[0]))

void
luigi_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	int recmask, gain, src;

        UNUSED(ad); assert(audio_fd > 0);

	if (ioctl(audio_fd, MIXER_READ(SOUND_MIXER_RECMASK), &recmask) == -1) {
		perror("Unable to read recording mask");
		return;
	}

	switch (port) {
	case LUIGI_MICROPHONE:
		src = SOUND_MASK_MIC;
		break;
	case LUIGI_LINE_IN:
		src = SOUND_MASK_LINE;
		break;
	case LUIGI_CD:
		src = SOUND_MASK_CD;
		break;
	}

	gain = luigi_audio_get_igain(ad);
	luigi_audio_set_igain(ad, 0);

	if ((ioctl(audio_fd, MIXER_WRITE(SOUND_MIXER_RECSRC), &src) < 0)) {
		return;
	}

	iport = port;
	luigi_audio_set_igain(ad, gain);
}

audio_port_t
luigi_audio_iport_get(audio_desc_t ad)
{
	UNUSED(ad); assert(audio_fd > 0);
	return iport;
}

int
luigi_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return NUM_IN_PORTS;
}

const audio_port_details_t *
luigi_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert(idx < (int)NUM_IN_PORTS && idx >= 0);
        return in_ports + idx;
}

void
luigi_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        if (!luigi_audio_is_ready(ad)) {
                usleep((unsigned int)delay_ms * 1000);
        }
}

int 
luigi_audio_is_ready(audio_desc_t ad)
{
        int avail;

        UNUSED(ad);

        LUIGI_AUDIO_IOCTL(audio_fd, FIONREAD, &avail);

        return (avail >= sz.rec_size);
}

int 
luigi_audio_supports(audio_desc_t ad, audio_format *fmt)
{
        snd_capabilities s;

        UNUSED(ad);

        if (luigi_error) debug_msg("Device error!");
        luigi_error = 0;
        LUIGI_AUDIO_IOCTL(audio_fd, AIOGCAP, &s);
        if (!luigi_error) {
                if ((unsigned)fmt->sample_rate < s.rate_min || (unsigned)fmt->sample_rate > s.rate_max) return FALSE;
                if (fmt->channels == 1) return TRUE;                    /* Always supports mono */
                assert(fmt->channels == 2);
                if (s.formats & AFMT_STEREO) return TRUE;
        }
        return FALSE;
}

int
luigi_audio_query_devices()
{
        FILE *f;
        char buf[128], *p;
        int n;

        f = fopen("/dev/sndstat", "r");
        if (f) {
                while (!feof(f) && ndev < LUIGI_MAX_AUDIO_DEVICES) {
                        p = fgets(buf, 128, f);
                        n = sscanf(buf, "pcm%d: <%[A-z0-9 ]>", dev_ids + ndev, names[ndev]);
                        if (p && n == 2) {
                                debug_msg("dev (%d) name (%s)\n", dev_ids[ndev], names[ndev]);
                                ndev++;
                        } else if (strstr(buf, "newpcm")) {
				/* This is a clunky check for the newpcm driver.  Don't use luigi in this case */
				debug_msg("Using newpcm driver\n");
				ndev = 0;
				break;
			}
                }
                fclose(f);
        }

        return (ndev);
}

int
luigi_get_device_count()
{
        return ndev;
}

char *
luigi_get_device_name(audio_desc_t idx)
{
        if (idx >=0 && idx < ndev) {
                return names[idx];
        }
        return NULL;
}
