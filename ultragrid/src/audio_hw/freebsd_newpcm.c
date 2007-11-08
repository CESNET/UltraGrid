/*
 * FILE:     audio_hw/freebsd__newpcm.c - Sound interface for newpcm FreeBSD driver.
 * AUTHOR:   Orion Hodson
 * MODIFIED: Colin Perkins
 *
 * Modified to support newpcm (July 2000).
 *
 * Copyright (c) 2002 University of Southern California
 * Copyright (c) 1996-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config_unix.h"
#include "config_win32.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/freebsd_newpcm.h"
#include "memory.h"
#include "debug.h"

#include <sys/soundcard.h>
#include <sys/types.h>

#include <dirent.h>
#include <errno.h>

/* #define DEBUG_JUST_NEWPCM if not using debug-enable and want err msgs */
#ifdef DEBUG_JUST_NEWPCM
#undef debug_msg
#define debug_msg(x...) fprintf(stderr, x)
#endif /* DEBUG_JUST_NEWPCM */

static char *port_names[] = SOUND_DEVICE_LABELS;
static int  iport, oport, loop;
static snd_chan_param pa;
static struct snd_size sz;
static int audio_fd = -1;
static int mixer_fd = -1;

#define RAT_TO_DEVICE(x) ((x) * 100 / MAX_AMP)
#define DEVICE_TO_RAT(x) ((x) * MAX_AMP / 100)

#define MIXER_CHECK0(fd)	if ((fd) < 0) {	\
		debug_msg("Failed mixer checked\n"); \
		return; \
	}

#define MIXER_CHECK1(fd, err)	if ((fd) < 0) {	\
		debug_msg("Failed mixer checked\n"); \
		return (err); \
	}

#define NEWPCM_AUDIO_IOCTL(fd, cmd, val) \
	newpcm_error = 0; \
	if (ioctl((fd), (cmd), (val)) < 0) { \
		debug_msg("Failed %s - line %d\n",#cmd, __LINE__); \
		newpcm_error = __LINE__; \
	}

#define IN_RANGE(x,l,u) ((x) >= (l) && (x) <= (u))

#define NEWPCM_MAX_AUDIO_NAME_LEN 32
#define NEWPCM_MAX_AUDIO_DEVICES  10

static char names[NEWPCM_MAX_AUDIO_DEVICES][NEWPCM_MAX_AUDIO_NAME_LEN];
static int ndev = 0;
static int newpcm_error;
static audio_format *input_format, *output_format, *tmp_format;
static snd_capabilities soundcaps[NEWPCM_MAX_AUDIO_DEVICES];

static int  newpcm_mixer_open(const char* audiodev);
static void newpcm_mixer_save(int fd);
static void newpcm_mixer_restore(int fd);
static void newpcm_mixer_init(int fd);
static void newpcm_audio_loopback_config(int gain);

int 
newpcm_audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
	audio_buf_info	abi;
	const char	*thedev;
	char		kick_off[64]; /* initial read buffer */
	int		frag;
 
	assert(ad >= 0 && ad < ndev); 
	thedev = names[ad];

	debug_msg("Opening \"%s\" device index %d\n", thedev, ad);

	audio_fd = open(thedev, O_RDWR);
	if (audio_fd >= 0) {
		/* Ignore any earlier errors */
		newpcm_error = 0;

		NEWPCM_AUDIO_IOCTL(audio_fd, AIOGCAP, &soundcaps[ad]);
		debug_msg("soundcaps[%d].rate_min = %ld\n", ad, soundcaps[ad].rate_min);
		debug_msg("soundcaps[%d].rate_max = %ld\n", ad, soundcaps[ad].rate_max);
		debug_msg("soundcaps[%d].formats  = 0x%08lx\n", ad, soundcaps[ad].formats);
		debug_msg("soundcaps[%d].bufsize  = %ld\n", ad, soundcaps[ad].bufsize);
		debug_msg("soundcaps[%d].mixers   = 0x%08lx\n", ad, soundcaps[ad].mixers);
		debug_msg("soundcaps[%d].inputs   = 0x%08lx\n", ad, soundcaps[ad].inputs);
		debug_msg("soundcaps[%d].left     = 0x%04x\n", ad, soundcaps[ad].left);
		debug_msg("soundcaps[%d].right    = 0x%04x\n", ad, soundcaps[ad].right);

		/* Setup input and output format settings */
		assert(ofmt->channels == ifmt->channels);
		memset(&pa, 0, sizeof(pa));
		if (ifmt->channels == 2) {
			if (!soundcaps[ad].formats & AFMT_STEREO) {
				fprintf(stderr,"Driver does not support stereo for this soundcard\n");
				newpcm_audio_close(ad);
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

		/* Check rate is supported (driver does not and
		 * appears to mis-report actual rate) */
		if ((IN_RANGE((uint32_t)ofmt->sample_rate, 
			      soundcaps[ad].rate_min, 
			      soundcaps[ad].rate_max) == 0) || 
		    (IN_RANGE((uint32_t)ifmt->sample_rate, 
			      soundcaps[ad].rate_min, 
			      soundcaps[ad].rate_max) == 0)) {
			debug_msg("(%d or %d) out of range %ld -- %ld Hz\n",
				  ofmt->sample_rate,
				  ifmt->sample_rate,
				  soundcaps[ad].rate_min,
				  soundcaps[ad].rate_max);
			if (ofmt->sample_rate == 8000 || ifmt->sample_rate == 8000) {
				fprintf(stderr, "8000Hz sampling not supported by soundcard.  This is the default rate\nfor RTP voice applications.\n");
			}
			newpcm_audio_close(ad);
			return FALSE;
		}

		/* Now set rate */
		pa.play_rate = ofmt->sample_rate;
		pa.rec_rate = ifmt->sample_rate;
		NEWPCM_AUDIO_IOCTL(audio_fd, AIOSFMT, &pa);

		/* Device buffer allocation - use
		 * SNDCTL_DSP_SETFRAGMENT because it creates as more
		 * frags for secondary buffer, making it possible to
		 * achieve fill levels set by cushion.  */

		/* fragsz [15:0] = log2(fragsz) */
		frag = 2;
		while((1 << frag) < ifmt->bytes_per_block)
			frag ++;
		frag --; /* Round down to nearest less than blocksize */
		
		/* fragsz [31:16] = number of frags (just lots, okay :-) */
		frag |= 0x00ff0000;

		NEWPCM_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_SETFRAGMENT, &frag);
		NEWPCM_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_GETOSPACE, &abi);
		debug_msg("fragments %d fragstotal %d fragsize %d bytes %d\n",
			  abi.fragments, abi.fragstotal, 
			  abi.fragsize, abi.bytes);

		/* sz is global and used throughout fn's */
		NEWPCM_AUDIO_IOCTL(audio_fd, AIOGSIZE, &sz);
		debug_msg("rec size %d, play size %d bytes\n",
			  sz.rec_size, sz.play_size);

		if (newpcm_error != 0) {
			/* Failed somewhere in initialization - reset error and exit*/
			newpcm_audio_close(ad);
			newpcm_error = 0;
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

		mixer_fd = newpcm_mixer_open(thedev);
		newpcm_mixer_save(mixer_fd);
		newpcm_mixer_init(mixer_fd);
		/* Turn off loopback from input to output... not fatal so
		 * after error check.
		 */
		newpcm_audio_loopback(ad, 0);

		read(audio_fd, kick_off, sizeof(kick_off)/sizeof(kick_off[0]));
		return TRUE;
	} else {
		fprintf(stderr, 
			"Could not open device: \"%s\" (half-duplex?)\n", 
			names[ad]);
		perror("newpcm_audio_open");
		newpcm_audio_close(ad);
		return FALSE;
	}
}

/* Close the audio device */
void
newpcm_audio_close(audio_desc_t ad)
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

	newpcm_audio_drain(audio_fd);
	close(audio_fd);
	audio_fd = -1;

	newpcm_mixer_restore(mixer_fd);
	close(mixer_fd);
	mixer_fd = -1;
}

/* Flush input buffer */
void
newpcm_audio_drain(audio_desc_t ad)
{
	u_char buf[4];
	int pre, post;
	 
	assert(audio_fd > 0);

	NEWPCM_AUDIO_IOCTL(audio_fd, FIONREAD, &pre);
	NEWPCM_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_RESET, 0);
	NEWPCM_AUDIO_IOCTL(audio_fd, FIONREAD, &post);
	debug_msg("audio drain: %d -> %d\n", pre, post);
	read(audio_fd, buf, sizeof(buf));

	UNUSED(ad);
}

int
newpcm_audio_duplex(audio_desc_t ad)
{
	/* We only ever open device full duplex! */
	UNUSED(ad);
	return TRUE;
}

int
newpcm_audio_read(audio_desc_t ad, u_char *buf, int read_bytes)
{
	int done, this_read;
	int len;

	UNUSED(ad); assert(audio_fd > 0);

	done = 0;
	len = min(read_bytes, sz.rec_size);
	do {
		this_read = read(audio_fd, buf + done, len);
		if (this_read == -1) break;
		done += this_read;
	} while (this_read == len && (done + this_read < read_bytes));

	return done;
}

int
newpcm_audio_write(audio_desc_t ad, u_char *buf, int write_bytes)
{
	int done, wrote;

	UNUSED(ad); assert(audio_fd > 0);

	done = 0;
	while (done < write_bytes) {
		wrote = write(audio_fd, 
			      buf + done, 
			      min(sz.play_size, write_bytes - done));
		if (wrote == -1) break;
		done += wrote;
	}
	return done;
}

/* Set ops on audio device to be non-blocking */
void
newpcm_audio_non_block(audio_desc_t ad)
{
	int	      frag = 1;

	UNUSED(ad); assert(audio_fd != -1);

	NEWPCM_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_NONBLOCK, &frag);
}

/* Set ops on audio device to be blocking */
void
newpcm_audio_block(audio_desc_t ad)
{
  	int	      frag = 0;
	 
	UNUSED(ad); assert(audio_fd > 0);
	 
	NEWPCM_AUDIO_IOCTL(audio_fd, SNDCTL_DSP_NONBLOCK, &frag);
} 


static int recmask, playmask;

static void
newpcm_mixer_init(int fd) 
{
	int devmask;

	NEWPCM_AUDIO_IOCTL(fd, SOUND_MIXER_READ_RECMASK, &recmask);

	/* Remove Vol from Rec mask - it is a play control! */
	recmask = recmask & ~SOUND_MASK_VOLUME;
	if (recmask & SOUND_MASK_MIC) {
		iport = SOUND_MASK_MIC;
	} else {
		iport = 1;
		while ((iport & recmask) == 0) {
			iport <<= 1;
		}
	}

	NEWPCM_AUDIO_IOCTL(fd, SOUND_MIXER_READ_DEVMASK, &devmask);
	playmask = devmask & ~recmask & ~SOUND_MASK_RECLEV ;
	debug_msg("devmask 0x%08x recmask 0x%08x playmask 0x%08x\n",
		  devmask,
		  recmask,
		  playmask);
}

static int
newpcm_count_ports(int mask) 
{
	int n = 0, m = mask;

	while (m > 0) {
		n += (m & 0x01);
		m >>= 1;
	}

	return n;
}

static int
newpcm_get_nth_port_mask(int mask, int n)
{
	static int lgmask;

	lgmask = -1;
	do {
		lgmask ++;
		if ((1 << lgmask) & mask) {
			n--;
		}
	} while (n >= 0);

	assert((1 << lgmask) & mask);
	return lgmask;
}

/* Gain and volume values are in the range 0 - MAX_AMP */
void
newpcm_audio_set_ogain(audio_desc_t ad, int vol)
{
	int volume, lgport, op;

	MIXER_CHECK0(mixer_fd);
		
	UNUSED(ad);
	vol = RAT_TO_DEVICE(vol);
	volume = vol << 8 | vol;
	lgport = -1;
	op = oport;
	while (op > 0) {
		op >>= 1;
		lgport ++;
	}

	NEWPCM_AUDIO_IOCTL(mixer_fd, MIXER_WRITE(lgport), &volume);
}

int
newpcm_audio_get_ogain(audio_desc_t ad)
{
	int volume, lgport, op;

	UNUSED(ad); 
	MIXER_CHECK1(mixer_fd, 0);

	lgport = -1;
	op     = oport;
	while (op > 0) {
		op >>= 1;
		lgport ++;
	}

	NEWPCM_AUDIO_IOCTL(mixer_fd, MIXER_READ(lgport), &volume);
	volume = DEVICE_TO_RAT(volume & 0xff);
	if (volume > 100 || volume < 0) {
		debug_msg("gain out of bounds (%08x %d--%d)" \
			  "mixer entry not implemented?", volume, 0, 100);  
		volume = 100;
	} 

	return volume;
}

void
newpcm_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
	UNUSED(ad);
	oport = port;
	return;
}

audio_port_t
newpcm_audio_oport_get(audio_desc_t ad)
{
	UNUSED(ad);
	return oport;
}

int
newpcm_audio_oport_count(audio_desc_t ad)
{
	UNUSED(ad);
	return newpcm_count_ports(playmask);
}

const audio_port_details_t*
newpcm_audio_oport_details(audio_desc_t ad, int idx)
{
	static audio_port_details_t ap;
	int lgmask;

	UNUSED(ad);

	lgmask = newpcm_get_nth_port_mask(playmask, idx);
	ap.port = 1 << lgmask;
	sprintf(ap.name, "%s", port_names[lgmask]);

	return &ap;
}

/* newpcm_audio_{get,set}_igain are used to control input gain.  There
 * are 2 key mixer types for newpcm: those based on ac97 and those
 * not.  AC97 mixers control input gain using RECLEV, whereas non-AC97
 * set gain on current input port and do not (usually) have a RECLEV
 * mixer line. */

void
newpcm_audio_set_igain(audio_desc_t ad, int gain)
{
	int volume = RAT_TO_DEVICE(gain);
	volume |= (volume << 8);

	UNUSED(ad); 
	MIXER_CHECK0(mixer_fd);

	newpcm_audio_loopback_config(gain);
	/* Try AC97 */
	NEWPCM_AUDIO_IOCTL(mixer_fd, SOUND_MIXER_WRITE_RECLEV, &volume);

	if (newpcm_error != 0 && iport != 0) { 
		/* Fallback to non-ac97 */
		int idx = 1;
		while ((1 << idx) != iport)
			idx++;
		NEWPCM_AUDIO_IOCTL(mixer_fd, MIXER_WRITE(idx), &volume);
	}
}

int
newpcm_audio_get_igain(audio_desc_t ad)
{
	int volume = 0;

	UNUSED(ad);
	MIXER_CHECK1(mixer_fd, 0);

	/* Try AC97 */
	NEWPCM_AUDIO_IOCTL(mixer_fd, SOUND_MIXER_READ_RECLEV, &volume);
	if (newpcm_error != 0 && iport != 0) {
		/* Fallback to non-ac97 */
		int idx = 1;
		while ((1 << idx) != iport)
			idx++;
		NEWPCM_AUDIO_IOCTL(mixer_fd, MIXER_READ(idx), &volume);
	}
	volume = DEVICE_TO_RAT(volume & 0xff); 
	if (volume > 100 || volume < 0) {
		debug_msg("gain out of bounds (%d %d--%d)" \
			  "mixer entry not implemented?", volume, 0, 100);  
		volume = 100;
	} 
	return volume;
}

void
newpcm_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	/* Check port is in record mask */
	int gain;

	MIXER_CHECK0(mixer_fd);

	debug_msg("port 0x%08x recmask 0x%08x\n", port, recmask);

	if ((port & recmask) == 0) {
		debug_msg("Port 0x%08d not in recmask 0x%08d\n",
			  port, recmask);
		return;
	}

	if (ioctl(mixer_fd, SOUND_MIXER_WRITE_RECSRC, &port) < 0) {
		perror("Unable to write record mask\n");
		return;
	}
	iport = port;
	gain = newpcm_audio_get_igain(ad);
	newpcm_audio_loopback_config(gain);
	UNUSED(ad);
}

audio_port_t
newpcm_audio_iport_get(audio_desc_t ad)
{
	UNUSED(ad); assert(audio_fd > 0);
	return iport;
}

int
newpcm_audio_iport_count(audio_desc_t ad)
{
	UNUSED(ad);
	return newpcm_count_ports(recmask);
}

const audio_port_details_t *
newpcm_audio_iport_details(audio_desc_t ad, int idx)
{
	static audio_port_details_t ap;
	int lgmask;

	UNUSED(ad);

	lgmask = newpcm_get_nth_port_mask(recmask, idx);
	ap.port = 1 << lgmask;
	sprintf(ap.name, "%s", port_names[lgmask]);

	return &ap;
}

void
newpcm_audio_loopback(audio_desc_t ad, int gain)
{
	UNUSED(ad); assert(audio_fd > 0);
	loop = gain;
}

static void
newpcm_audio_loopback_config(int gain) 
{
	int lgport, vol;

	MIXER_CHECK0(mixer_fd);

	/* Find current input port id */
	lgport = newpcm_get_nth_port_mask(iport, 0);

	if (loop) {
		vol = RAT_TO_DEVICE(gain) << 8 | RAT_TO_DEVICE(gain);
	} else {
		vol = 0;
	}

	NEWPCM_AUDIO_IOCTL(mixer_fd, MIXER_WRITE(lgport), &vol);
}

void
newpcm_audio_wait_for(audio_desc_t ad, int delay_ms)
{
	struct timeval	timeout;
	fd_set		fds;

	timeout.tv_sec	= 0;
	timeout.tv_usec	= 1000 * delay_ms;
	
	FD_ZERO(&fds);
	FD_SET(audio_fd, &fds);
	select(audio_fd + 1, &fds, 0, 0, &timeout);
	UNUSED(ad);
}

int 
newpcm_audio_is_ready(audio_desc_t ad)
{
	int avail;

	UNUSED(ad);

	NEWPCM_AUDIO_IOCTL(audio_fd, FIONREAD, &avail);

	return (avail >= sz.rec_size);
}

int 
newpcm_audio_supports(audio_desc_t ad, audio_format *fmt)
{
	snd_capabilities s;

	UNUSED(ad);

	NEWPCM_AUDIO_IOCTL(audio_fd, AIOGCAP, &s);
	if (!newpcm_error) {
		if ((unsigned)fmt->sample_rate < s.rate_min || (unsigned)fmt->sample_rate > s.rate_max) return FALSE;
		if (fmt->channels == 1) 
			return TRUE;	/* Always supports mono */
		assert(fmt->channels == 2);
		if (s.formats & AFMT_STEREO) 
			return TRUE;
	}
	return FALSE;
}

static int
newpcm_is_driver() {
	FILE	*f;
	char	buf[128], *p;
	int	newpcm = FALSE;

	f = fopen("/dev/sndstat", "r");
	if (f == NULL) return FALSE;

	while(!feof(f)) {
		p = fgets(buf, 128, f);		
		if (p && strstr(buf, "newpcm")) {
			newpcm = TRUE;
			break;
		}
	}
	fclose(f);
	return newpcm;
}

int
newpcm_audio_query_devices()
{

	DIR  		*d;
	struct dirent	*de;
	int		tfd;

	if (newpcm_is_driver() == 0)
		return 0;

	if (ndev)
		return ndev;

	d = opendir("/dev");
	if (d == NULL) {
		perror("opendir /dev");
		return 0;
	}

	while ((de = readdir(d)) != NULL && ndev < NEWPCM_MAX_AUDIO_DEVICES) {
		if (de->d_type != DT_CHR) continue;

		if (strncmp(de->d_name, "audio", 5) != 0) continue;

		sprintf(names[ndev], "/dev/%s", de->d_name); 
		tfd = open(names[ndev], O_RDWR);
		if (tfd < 0) {
		 	/* If device is busy it's an audio device, otherwise it's (probably) an invalid device description */	
			if (errno != EBUSY) 
				continue;
		} else {
			close(tfd);
		}
		ndev++;
	}
	closedir(d);
	return (ndev);
}

int
newpcm_get_device_count()
{
	return ndev;
}

char *
newpcm_get_device_name(audio_desc_t idx)
{
	if (idx >=0 && idx < ndev) {
		/* XXX this fn used to return (const char*) - grrr! */
		static char dummy[NEWPCM_MAX_AUDIO_NAME_LEN];
		strcpy(dummy, names[idx]);
		return dummy;
	}	
	return NULL;
}

/* Mixer open / close related */

static int
newpcm_mixer_device(const char* audiodev)
{
	const char* p = audiodev;
	int devno = 0;

	/* 
	 * Audio device looks like "/dev/fooN" or "dev/foo/N.n"
	 * and we want "N"
	 */
	while (p && !isnumber(*p))
		p++;
	while (p && isnumber(*p)) {
		devno = devno * 10 + (*p - '0');
		p++;
	}
	assert(devno < 20);
	return devno;
}

static int
newpcm_mixer_open(const char* audiodev)
{
	char mixer_name[32] = "/dev/mixerXXX";
	int m;

#define END_OF_DEV_MIXER 10
	sprintf(mixer_name + END_OF_DEV_MIXER, 
		"%d", newpcm_mixer_device(audiodev));

	m = open(mixer_name, O_RDWR);
	if (m < 0) {
		fprintf(stderr, "Could not open %s (%s): "
			"mixer operations will not work.\n",
			mixer_name, strerror(errno));
		return 0;
	}
	return m;
}

/* Functions to save and restore recording source and mixer levels */

static int saved_rec_mask, saved_gain_values[SOUND_MIXER_NRDEVICES];

static void
newpcm_mixer_save(int fd)
{
	int devmask, i;
	NEWPCM_AUDIO_IOCTL(fd, SOUND_MIXER_READ_RECSRC, &saved_rec_mask); 
	NEWPCM_AUDIO_IOCTL(fd, SOUND_MIXER_READ_DEVMASK, &devmask);
	for (i = 0; i < SOUND_MIXER_NRDEVICES; i++) {
		if ((1 << i) & devmask) {
			NEWPCM_AUDIO_IOCTL(fd, MIXER_READ(i), &saved_gain_values[i]);
		} else {
			saved_gain_values[i] = 0;
		}
	}
}

static void
newpcm_mixer_restore(int fd)
{
	int devmask, i;

	NEWPCM_AUDIO_IOCTL(fd, SOUND_MIXER_WRITE_RECSRC, &saved_rec_mask); 
	NEWPCM_AUDIO_IOCTL(fd, SOUND_MIXER_READ_DEVMASK, &devmask);
	for (i = 0; i < SOUND_MIXER_NRDEVICES; i++) {
		if ((1 << i) & devmask) {
			NEWPCM_AUDIO_IOCTL(fd, MIXER_WRITE(i), &saved_gain_values[i]);
		}
	}
}
