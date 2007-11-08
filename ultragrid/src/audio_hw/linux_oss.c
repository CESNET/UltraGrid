/*
 * FILE:     audio_hw/linux_oss.c - Open Sound System audio device driver
 * PROGRAM:  RAT
 * AUTHOR:   Colin Perkins
 * MODIFIED: Orion Hodson + Robert Olson
 *
 * Copyright (c) 2002-2003 University of Southern California
 * Copyright (c) 1996-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/linux_oss.h"

#ifdef HAVE_SOUNDCARD_H
#  include <soundcard.h>
#else
#ifdef HAVE_SYS_SOUNDCARD_H
#  include <sys/soundcard.h>
#endif
#endif

enum { AUDIO_SPEAKER, AUDIO_HEADPHONE, AUDIO_LINE_OUT, AUDIO_MICROPHONE, AUDIO_LINE_IN, AUDIO_CD};

/* Info to match port id's to port name's */
static audio_port_details_t in_ports[] = {
        {AUDIO_MICROPHONE, AUDIO_PORT_MICROPHONE},
        {AUDIO_LINE_IN,    AUDIO_PORT_LINE_IN},
        {AUDIO_CD,         AUDIO_PORT_CD}
};

#define NUM_IN_PORTS (sizeof(in_ports)/(sizeof(in_ports[0])))

static audio_port_details_t out_ports[] = {
        {AUDIO_SPEAKER,   AUDIO_PORT_SPEAKER}
};

#define NUM_OUT_PORTS (sizeof(out_ports)/(sizeof(out_ports[0])))

int	iport     = AUDIO_MICROPHONE;

char *oss_mixer_channels[] = SOUND_DEVICE_LABELS;

#define OSS_MAX_DEVICES   		 4	/* Magic constant based on the OSS source */
#define OSS_MAX_NAME_LEN 		64	/* Magic constant based on the OSS source */
#define OSS_MAX_SUPPORTED_FORMATS 	14 	/* == 8k,11k,16k,22k,32k,44.1k,48k * mono, stereo */

#define OSS_DUPLEX_HALF   		 0
#define OSS_DUPLEX_FULL   		 1
#define OSS_DUPLEX_PAIR			 2	/* ...a pair of half duplex devices emulating full duplex */

struct oss_device {
	char		*name;
	char		 audio_rdev[16];
	char		 audio_wdev[16];
	char		 mixer_rdev[16];
	char		 mixer_wdev[16];
	int		 audio_rfd;
	int		 audio_wfd;
	int		 mixer_rfd;
	int		 mixer_wfd;
	int		 duplex;	/* TRUE if the device supports full duplex */
	audio_format	 supported_formats[OSS_MAX_SUPPORTED_FORMATS];
	int		 num_supported_formats;
	int		 dev_mask;	/* Supported mixer channels     */
	int		 rec_mask;	/* Supported recording channels */
	int		 is_ac97;
	int		 is_latemodel_opensound;
};

static struct oss_device	devices[OSS_MAX_DEVICES];
static int			num_devices;


/*
 * Device names that we're going to consider as AC97 compatible.
 * That means we only touch igain, not the individual
 * input gains. We also don't touch the master output gains,
 * and only set the PCM output gain for ogain adjustment.
 */
static char *ac97_devices[] = {
	"OSS: AudioPCI 97 (STAC9708)",
	"OSS: AudioPCI 97 (CRY13/0x43525913)",
	"OSS: AudioPCI 97 (CS4297A)",
	0
};
 
static int
oss_probe_mixer_device(int i, struct oss_device *device)
{
	/* Probe /dev/mixerX, and fill in mixer related parts of device.   */
	/* If we are requested to probe /dev/mixer0, and that file doesn't */
	/* exist, we probe /dev/mixer instead (if that is not a symlink).  */
	/* This is for compatibility with some old Linux distributions,    */
	/* which have a broken /dev.                                       */
	mixer_info	info;
	struct stat	s;
	int		valid_mixer = FALSE;
	int		fd, d;

	sprintf(device->mixer_rdev, "/dev/mixer%d", i);
	fd = open(device->mixer_rdev, O_RDWR);
	if ((fd < 0) && (i == 0)) {
		if ((stat("/dev/mixer", &s) == 0) && !S_ISLNK(s.st_mode)) {
			sprintf(device->mixer_rdev, "/dev/mixer");
			fd = open(device->mixer_rdev, O_RDWR);
		}
	}
	if (fd < 0) {
		/* We can't find a working mixer device - this is not  */
		/* necessarily a problem, some soundcards don't have a */
		/* corresponding mixer device.                         */
		device->name = (char *) malloc(17);
		sprintf(device->name, "%s", device->mixer_rdev);
		debug_msg("cannot open %s - %s\n", device->mixer_rdev, strerror(errno));
		memset(device->mixer_rdev, 0, 16);
		memset(device->mixer_wdev, 0, 16);
		return FALSE;
	}
	strncpy(device->mixer_wdev, device->mixer_rdev, 16);

	/* Okay, the mixer device is open. Probe its capabilities... */
	if (ioctl(fd, SOUND_MIXER_INFO, &info) != 0) {
		debug_msg("cannot query mixer capabilities\n");
	} else {
		device->name = (char *) malloc(OSS_MAX_NAME_LEN + 6);
		memset(device->name, 0, OSS_MAX_NAME_LEN + 6);
		sprintf(device->name, "OSS: ");
		strncpy(device->name + 5, info.name, OSS_MAX_NAME_LEN);
		valid_mixer = TRUE;
	}

	if (ioctl(fd, SOUND_MIXER_READ_DEVMASK, &device->dev_mask) != 0) {
		debug_msg("cannot query mixer dev mask %s\n", strerror(errno));
		device->dev_mask = 0;
	} else {
		debug_msg("device mask for %s:\n", device->mixer_rdev);
		for (d = 0; d < SOUND_MIXER_NRDEVICES; d++) {
			if (device->dev_mask & (1 << d)) {
				debug_msg("  %2d %s\n", d, oss_mixer_channels[d]);
			}
		}
	}

	if (ioctl(fd, SOUND_MIXER_READ_RECMASK, &device->rec_mask) != 0) {
		debug_msg("cannot query mixer rec mask %s\n", strerror(errno));
		device->rec_mask = 0;
	} else {
		debug_msg("recording mask for %s:\n", device->mixer_rdev);
		for (d = 0; d < SOUND_MIXER_NRDEVICES; d++) {
			if (device->rec_mask & (1 << d)) {
				debug_msg("  %2d %s\n", d, oss_mixer_channels[d]);
			}
		}
	}

	device->mixer_rfd = -1;
	device->mixer_wfd = -1;

	close(fd);
	return valid_mixer;
}

static int
oss_test_mode(int fd, int speed, int stereo)
{
        int sp, st;

        ioctl(fd, SNDCTL_DSP_RESET, 0);

        st = stereo;
        if (ioctl(fd, SNDCTL_DSP_STEREO, &st) == -1 || st != stereo) {
		debug_msg("  disabled (%d channels not supported)\n", stereo + 1);
		return FALSE;
        }

        sp = speed;
        if (ioctl(fd, SNDCTL_DSP_SPEED, &sp) == -1) {
		debug_msg("  disabled (%dHz sampling not supported)\n", speed);
		return FALSE;
        }
	if (((100 * abs(sp - speed)) / speed) > 5) {
		debug_msg("  disabled (clock skew >5%: %dHz vs %dHz)\n", sp, speed);
		return FALSE;
	}
        
        return TRUE;
}

static int
oss_probe_audio_device(int i, struct oss_device *device)
{
	/* Probe /dev/dspX, and fill in mixer related parts of the device. */
	/* If we are requested to probe /dev/dsp0, and that file doesn't   */
	/* exist, we probe /dev/dsp instead (if that is not a symlink).    */
	/* This is for compatibility with some old Linux distributions,    */
	/* which have a broken /dev.                                       */
	struct stat	s;
	int		speed[] = {8000, 11025, 16000, 22050, 32000, 44100, 48000};
        int 		stereo, speed_index, fd;

	sprintf(device->audio_rdev, "/dev/dsp%d", i);
	fd = open(device->audio_rdev, O_RDWR);
	if ((fd < 0) && (i == 0)) {
		if ((stat("/dev/dsp", &s) == 0) && !S_ISLNK(s.st_mode)) {
			sprintf(device->audio_rdev, "/dev/dsp");
			fd = open(device->audio_rdev, O_RDWR);
		}
	}
	if (fd < 0) {
		debug_msg("cannot open %s - %s\n", device->audio_rdev, strerror(errno));
		return FALSE;
	}
	strncpy(device->audio_wdev, device->audio_rdev, 16);

	/* Check if the device is full duplex. This MUST be the first test   */
	/* after the audio device is opened.                                 */
	if (ioctl(fd, SNDCTL_DSP_SETDUPLEX, 0) == -1) {
		debug_msg("testing %s support for full duplex operation: no\n", device->audio_rdev);
		device->duplex = OSS_DUPLEX_HALF;
	} else {
		debug_msg("testing %s support for full duplex operation: yes\n", device->audio_rdev);
		device->duplex = OSS_DUPLEX_FULL;
	}

	/* Check which sampling modes are supported...                       */
	device->num_supported_formats = 0;
	for (speed_index = 0; speed_index < 7; speed_index++) {
                for (stereo = 0; stereo < 2; stereo++) {
			debug_msg("testing %s support for %5dHz %s\n", device->audio_rdev, speed[speed_index], stereo?"stereo":"mono");
                        if (oss_test_mode(fd, speed[speed_index], stereo)) {
				device->supported_formats[device->num_supported_formats].sample_rate = speed[speed_index];
				device->supported_formats[device->num_supported_formats].channels    = stereo + 1;
				device->num_supported_formats++;
			}
                }
        }
	/* Check for OSS version number */

	{
#ifdef OSS_GETVERSION
	    int version = -1;
	    if (ioctl(fd, OSS_GETVERSION, &version) < 0) {
		debug_msg("%s doesn't support OSS_GETVERSION\n",
			  device->audio_rdev);
	    } else {
		int level;
		debug_msg("OSS version is %x\n", version);

		/*
		 * Check to see if we can use the MIXER_RECLEV setting
		 * If we can, this might be an ES1371-based card that uses
		 * RECLEV to adjust input levels.
		 */

		if (ioctl(fd, MIXER_READ(SOUND_MIXER_RECLEV), &level) >= 0) {
		    debug_msg("Can use reclev. \n");
		
		    if (version >= 0x030903) {
			debug_msg("Enabling latemodel opensound\n");
			device->is_latemodel_opensound = 1;
		    }
		}
	    }
#endif
	}

	close(fd);
	device->audio_rfd = -1;
	device->audio_wfd = -1;
	return TRUE;
}

static int
oss_test_device_pair(int rdev, int wdev)
{

	char buf[1024];
	int n;

	/* Attempt to open rdev read-only and wdev write-only, both half- */
	/* duplex, simultaneously. Return TRUE if we succeed.             */
	devices[rdev].audio_rfd = open(devices[rdev].audio_rdev, O_RDONLY | O_NDELAY);
	if (devices[rdev].audio_rfd == 0) {
		debug_msg("unable to open %s %s\n", devices[rdev].audio_rdev, strerror(errno));
		return FALSE;
	}

	n = read(devices[rdev].audio_rfd, buf, sizeof(buf));
	debug_msg("read in test_device_pair returns %d\n", n);
	if (n < 0) {
		debug_msg("cannot read audio from %s %s\n", devices[rdev].audio_rdev, strerror(errno));
		return FALSE;
	}
	

	devices[wdev].audio_wfd = open(devices[wdev].audio_wdev, O_WRONLY | O_NDELAY);
	if (devices[wdev].audio_wfd == 0) {
		debug_msg("unable to open %s %s\n", devices[wdev].audio_wdev, strerror(errno));
		close(devices[rdev].audio_rfd); devices[rdev].audio_rfd = 0;
		return FALSE;
	}

	close(devices[rdev].audio_rfd); devices[rdev].audio_rfd = -1;
	close(devices[wdev].audio_wfd); devices[wdev].audio_wfd = -1;
	return TRUE;
}

static void
oss_pair_devices(void)
{
	/* This function scans through the devices[] array, merging pairs   */
	/* of half-duplex audio devices into a single full duplex device    */
	/* entry (with one device providing half-duplex recording, and the  */
	/* other providing half-duplex playback).                           */
	int	i, j;

	if (num_devices < 2) {
		return;
	}

	for (i = 0; i < num_devices - 1; i++) {
		if ((devices[i].duplex == OSS_DUPLEX_HALF) && (devices[i+1].duplex == OSS_DUPLEX_HALF)) {
			if (oss_test_device_pair(i, i+1)) {
				debug_msg("Combining %s and %s\n", devices[i].audio_rdev, devices[i+1].audio_wdev);
				memcpy(devices[i].audio_wdev, devices[i+1].audio_wdev, 16);
				if (strlen(devices[i+1].mixer_wdev) != 0) {
					debug_msg("Second mixer is valid\n");
					memcpy(devices[i].mixer_wdev, devices[i+1].mixer_wdev, 16);
				} else {
					debug_msg("Second mixer is invalid - assuming first is okay\n");
				}
				devices[i].audio_rfd = -1;
				devices[i].audio_wfd = -1;
				devices[i].mixer_rfd = -1;
				devices[i].mixer_wfd = -1;
				devices[i].duplex    = OSS_DUPLEX_PAIR;
				devices[i].rec_mask  = devices[i+1].rec_mask;
				free(devices[i+1].name);
				/* Move the rest of the device table up... */
				for (j = i+1; j < (num_devices - 1); j++) {
					devices[j] = devices[j+1];
				}
				num_devices--;
				i--;
			} else {
				debug_msg("Cannot pair %s and %s\n", devices[i].audio_rdev, devices[i+1].audio_wdev);
			}
		}
	}
}

static void
oss_remove_half_duplex_devices(void)
{
	/* Any half-duplex devices which remain are removed from the list   */
	/* of valid devices, since rat requires full duplex. This excludes  */
	/* pairs of half-duplex devices which are emulating full-duplex.    */
	int	i, j;

	for (i = 0; i < num_devices; i++) {
		if (devices[i].duplex == OSS_DUPLEX_HALF) {
			debug_msg("%s,%s is half duplex, removed\n", devices[i].audio_rdev, devices[i].mixer_rdev);
			for (j = i + 1; j < num_devices; j++) {
				devices[j-1] = devices[j];
			}
			num_devices--;
			i--;
		}
	}
}

int
oss_audio_init(void)
{
	/* One time initialization of the OSS audio driver. We probe the    */
	/* available devices, to setup the devices[] array and num_devices. */
	/* Note that it is entirely legal to have an audio device without a */
	/* corresponding mixer device.                                      */
	int			i, mix_i;
	struct oss_device	device;
	int aidx;

	num_devices = 0;
	mix_i = 0;
	for (i = 0; i < OSS_MAX_DEVICES; i++, mix_i++) {
		device.is_latemodel_opensound = 0;
		oss_probe_mixer_device(mix_i, &device);
		if (oss_probe_audio_device(i, &device)) {
			/* Check to see if it's one of the AC97 devices. For now  */
			/* we will use a hardcoded list of device names for this. */
			/*                                                        */
			/* Also check the OSS_IS_AC97 environment variable. If it */
			/* is set, make all devices AC97.                         */
			device.is_ac97 = 0;
			for (aidx = 0; ac97_devices[aidx] != 0; aidx++) {
				if (strcmp(device.name, ac97_devices[aidx]) == 0) {
					device.is_ac97 = 1;
					debug_msg("Device %d  %s tagged as ac97\n", i, device.name);
					break;
				}
			}

			if (device.is_latemodel_opensound)
				device.is_ac97 = 0;

			if (getenv("OSS_IS_AC97") != 0) {
				debug_msg("Device %d  %s tagged as ac97 by environment var\n",
					  i, device.name);
				device.is_ac97 = 1;
			}
			
			devices[num_devices++] = device;
			debug_msg("found \"%s\" as %s,%s\n", device.name, device.audio_rdev, device.mixer_rdev);
			/* 
			 * Hack. If it's an Ensoniq AudioPCI, skip the halfduplex
			 * device.
			 */
			if (strncmp(device.name, "OSS: AudioPCI 97", 16) == 0) {
				i++;
			}

		}
	}
	oss_pair_devices();
	oss_remove_half_duplex_devices();

	debug_msg("number of valid devices: %d\n", num_devices);
	return num_devices;
}

static int
oss_configure_device(int fd, audio_format *ifmt, audio_format *ofmt)
{
	/* Configure the audio device. This function is called for both read */
	/* and write file descriptors, so the setup must not break if done   */
	/* twice on the same device.                                         */
	int  	mode, stereo, speed;

        switch(ifmt->encoding) {
        	case DEV_PCMU: 
			mode = AFMT_MU_LAW;
			break;
		case DEV_PCMA: 
			mode = AFMT_A_LAW;
			break;
        	case DEV_S8:   
			mode = AFMT_S8;
			break;
        	case DEV_S16:  
			mode = AFMT_S16_LE;
			break;
		case DEV_U8:   
			mode = AFMT_U8;
			break;
		default:
			abort();
        }
	if ((ioctl(fd, SNDCTL_DSP_SETFMT, &mode) == -1)) {
		if (ifmt->encoding == DEV_S16) {
			audio_format_change_encoding(ifmt, DEV_PCMU);
			audio_format_change_encoding(ofmt, DEV_PCMU);
			if ((ioctl(fd, SNDCTL_DSP_SETFMT, &mode) == -1)) {
				return FALSE;
			}
			debug_msg("device doesn't support 16bit audio, using 8 bit PCMU\n");
		}
	}

	stereo = ifmt->channels - 1; 
	assert(stereo == 0 || stereo == 1);
	if ((ioctl(fd, SNDCTL_DSP_STEREO, &stereo) == -1) || (stereo != (ifmt->channels - 1))) {
		debug_msg("device doesn't support %d channels!\n", ifmt->channels);
		return FALSE;
	}

	speed = ifmt->sample_rate;
	if (ioctl(fd, SNDCTL_DSP_SPEED, &speed) == -1) {
		debug_msg("device doesn't support %dHz sampling rate in full duplex!\n", ifmt->sample_rate);
		return FALSE;
	}
	return TRUE;
}

int
oss_audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
	/* Try to open the audio device, returning TRUE if successful. Note  */
	/* that the order in which the device is set up is important: don't  */
	/* reorder this code unless you really know what you're doing.       */
#if 0
	int  	volume   = (100<<8)|100;
#endif
	int  	frag     = 0x7fff0000; 			/* unlimited number of fragments */
	char 	buffer[128];				/* sigh. */
	int	bytes_per_block;

        if (ad < 0 || ad > OSS_MAX_DEVICES) {
                debug_msg("invalid audio descriptor (%d)", ad);
                return FALSE;
        }

	devices[ad].audio_rfd = -1;
	devices[ad].audio_wfd = -1;
	devices[ad].mixer_rfd = -1;
	devices[ad].mixer_wfd = -1;

	switch (devices[ad].duplex) {
		case OSS_DUPLEX_HALF:
			debug_msg("cannot open half duplex devices\n");
			return FALSE;
		case OSS_DUPLEX_FULL:
			/* Open the device, and switch to full duplex mode. */
			assert(strcmp(devices[ad].audio_rdev, devices[ad].audio_wdev) == 0);

			devices[ad].audio_rfd = open(devices[ad].audio_rdev, O_RDWR | O_NDELAY);
			if (devices[ad].audio_rfd == 0) {
				debug_msg("unable to open %s\n", devices[ad].audio_rdev);
				return FALSE;
			}
			devices[ad].audio_wfd = devices[ad].audio_rfd;

			if (ioctl(devices[ad].audio_rfd, SNDCTL_DSP_SETDUPLEX, 0) == -1) {
				debug_msg("device doesn't support full duplex operation\n");
				oss_audio_close(ad);
				return FALSE;
			}

			/* Now open the corresponding mixer device... */
			devices[ad].mixer_rfd = open(devices[ad].mixer_rdev, O_RDWR | O_NDELAY);
			if (devices[ad].mixer_rfd == 0) {
				debug_msg("unable to open %s\n", devices[ad].mixer_rdev);
				oss_audio_close(ad);
				return FALSE;
			}
			devices[ad].mixer_wfd = devices[ad].mixer_rfd;

			break;
		case OSS_DUPLEX_PAIR:
			devices[ad].audio_rfd = open(devices[ad].audio_rdev, O_RDONLY | O_NDELAY);
			if (devices[ad].audio_rfd == 0) {
				debug_msg("unable to open %s\n", devices[ad].audio_rdev);
				return FALSE;
			}

			devices[ad].audio_wfd = open(devices[ad].audio_wdev, O_WRONLY | O_NDELAY);
			if (devices[ad].audio_wfd == 0) {
				debug_msg("unable to open %s\n", devices[ad].audio_wdev);
				return FALSE;
			}

			/* Now open the corresponding mixer device... */
			devices[ad].mixer_rfd = open(devices[ad].mixer_rdev, O_RDWR | O_NDELAY);
			if (devices[ad].mixer_rfd == 0) {
				debug_msg("unable to open %s\n", devices[ad].mixer_rdev);
				oss_audio_close(ad);
				return FALSE;
			}
			devices[ad].mixer_wfd = open(devices[ad].mixer_wdev, O_RDWR | O_NDELAY);
			if (devices[ad].mixer_wfd == 0) {
				debug_msg("unable to open %s\n", devices[ad].mixer_wdev);
				oss_audio_close(ad);
				return FALSE;
			}
			break;
		default:
			debug_msg("invalid duplex mode - this can never happen\n");
			abort();
	}
	
	/* Set 20 ms blocksize - this only modulates read sizes, and hence only has to be done on audio_rfd */
	bytes_per_block = 20 * (ifmt->sample_rate / 1000) * (ifmt->bits_per_sample / 8);
	/* Round to the nearest legal frag size (next power of two lower...) */
	frag |= (int) (log(bytes_per_block)/log(2));
	if ((ioctl(devices[ad].audio_rfd, SNDCTL_DSP_SETFRAGMENT, &frag) == -1)) {
		debug_msg("cannot set fragement size (frag=%x bytes_per_block=%d)\n", frag, bytes_per_block);
	}

	if (!oss_configure_device(devices[ad].audio_rfd, ifmt, ofmt)) {
		oss_audio_close(ad);
		return FALSE;
	}
	if (!oss_configure_device(devices[ad].audio_wfd, ifmt, ofmt)) {
		oss_audio_close(ad);
		return FALSE;
	}

	/* Set global gain/volume to maximum values. This may fail on */
	/* some cards, but shouldn't cause any harm when it does..... */ 

#if 0
	ioctl(devices[ad].mixer_rfd, MIXER_WRITE(SOUND_MIXER_RECLEV), &volume);
	ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_VOLUME), &volume);
#endif
	/* Select microphone input. We can't select output source...  */
	oss_audio_iport_set(ad, iport);
	/* Device driver bug: we must read some data before the ioctl */
	/* to tell us how much data is waiting works....              */
	read(devices[ad].audio_rfd, buffer, 128);	

	return TRUE;
}

/* Close the audio device */
void
oss_audio_close(audio_desc_t ad)
{
	oss_audio_drain(ad);
	if (devices[ad].audio_rfd != -1) close(devices[ad].audio_rfd);
	if (devices[ad].mixer_rfd != -1) close(devices[ad].mixer_rfd);

	if (devices[ad].duplex == OSS_DUPLEX_PAIR) {
		if (devices[ad].audio_wfd != -1) close(devices[ad].audio_wfd);
		if (devices[ad].mixer_wfd != -1) close(devices[ad].mixer_wfd);
	}

        devices[ad].audio_rfd = -1;
        devices[ad].audio_wfd = -1;
        devices[ad].mixer_rfd = -1;
        devices[ad].mixer_wfd = -1;
}

/* Flush input buffer */
void
oss_audio_drain(audio_desc_t ad)
{
        u_char buf[160];

        assert(ad < OSS_MAX_DEVICES);
        assert(devices[ad].audio_rfd > 0);

        while(oss_audio_read(ad, buf, 160) == 160);
}

int
oss_audio_duplex(audio_desc_t ad)
{
        /* We don't open device if not full duplex. */
	return devices[ad].duplex;
}

/* Gain and volume values are in the range 0 - MAX_AMP */
void
oss_audio_set_igain(audio_desc_t ad, int gain)
{
	int which_port;
	int volume = (gain << 8) | gain;

        assert(ad < OSS_MAX_DEVICES);
        assert(devices[ad].audio_rfd > 0);

	switch (iport) {
		case AUDIO_MICROPHONE : 
			if (devices[ad].is_ac97) {
				which_port = SOUND_MIXER_IGAIN;
			} else if (devices[ad].is_latemodel_opensound) {
				which_port = SOUND_MIXER_RECLEV;
			} else {
				which_port = SOUND_MIXER_MIC;
			}
			if (ioctl(devices[ad].mixer_rfd, MIXER_WRITE(which_port), &volume) == -1) {
				perror("Setting gain");
			}
			break;
		case AUDIO_LINE_IN : 
			/* From Stuart Levy <slevy@ncsa.uiuc.edu>:                           */
			/* Finally, one completely untested (but plausible) change           */
			/* that may help some people quite a bit.                            */
			/*                                                                   */
			/* An access-grid user reported a problem that adjusting their rat's */
			/* "Line" level seemed to cause local audio loopback,                */
			/* but that it *didn't* control the level of LineIn->rat signal.     */
			/*                                                                   */
			/* Bob Olson <olson@mcs.anl.gov> suggested that their sound card     */
			/* might be "AC97 compliant", in which case the LINE mixer input     */
			/* just controls the LineIn -> LineOut gain -- i.e. loopback! --     */
			/* and LineIn capture level is controlled by Input Gain (IGAIN).     */
			/*                                                                   */
			/* So, in oss_audio_{set,get}_igain(), it first tries to             */
			/* {write,read} the SOUND_MIXER_IGAIN value.  Only if that fails --  */
			/* which I *hope* happens iff the card has no such control --        */
			/* does it {write,read} SOUND_MIXER_LINE.                            */
			if (devices[ad].is_ac97) {
				which_port = SOUND_MIXER_IGAIN;
			} else if (devices[ad].is_latemodel_opensound) {
				which_port = SOUND_MIXER_RECLEV;
			} else {
				which_port = SOUND_MIXER_LINE;
			}
			if (ioctl(devices[ad].mixer_rfd, MIXER_WRITE(which_port), &volume) == -1) {
				perror("Setting gain");
			}
			break;
		case AUDIO_CD:
			if (devices[ad].is_ac97) {
				which_port = SOUND_MIXER_IGAIN;
			} else if (devices[ad].is_latemodel_opensound) {
				which_port = SOUND_MIXER_RECLEV;
			} else {
				which_port = SOUND_MIXER_CD;
			}
			if (ioctl(devices[ad].mixer_rfd, MIXER_WRITE(which_port), &volume) < 0) {
				perror("Setting gain");
			}
			break;
		default:
			printf("ERROR: Unknown iport in audio_set_igain!\n");
			abort();
	}
}

int
oss_audio_get_igain(audio_desc_t ad)
{
	int volume;
	int which_port;

	if (devices[ad].is_ac97) {
		which_port = SOUND_MIXER_IGAIN;
	} else if (devices[ad].is_latemodel_opensound) {
	    which_port = SOUND_MIXER_RECLEV;
	} else {
		switch (iport) {
		case AUDIO_MICROPHONE:
			which_port = SOUND_MIXER_MIC;
			break;
		case AUDIO_LINE_IN:
			which_port = SOUND_MIXER_LINE;
			break;
		case AUDIO_CD:
			which_port = SOUND_MIXER_CD;
			break;
		default:
			printf("ERROR: Unknown iport in audio_set_igain!\n");
			abort();
	    	}
	}

        UNUSED(ad); assert(devices[ad].mixer_rfd > 0); assert(ad < OSS_MAX_DEVICES);

	if (ioctl(devices[ad].mixer_rfd, MIXER_READ(which_port), &volume) == -1) {
		perror("Getting gain");
	}
	debug_msg("getting igain; port=%d is_ac97=%d is_latemodel_opensound=%d vol=%d %d\n", which_port, devices[ad].is_ac97, devices[ad].is_latemodel_opensound, volume, volume & 0xff);
	return volume & 0xff;
}

void
oss_audio_set_ogain(audio_desc_t ad, int vol)
{
	/* From Stuart Levy <slevy@ncsa.uiuc.edu>:                           */
	/* Also for the SBLive, the useful outputs weren't the ones          */
	/* controlled by the rat-builtin mixer controls.                     */
	/* Not sure what best to do here, so I made                          */
	/* oss_audio_set_ogain() set *all* the level controls to the         */
	/* specified value (PCM, SPEAKER, OGAIN, LINE1, LINE2).              */
	/* Perhaps on some sound cards these values get multiplied together, */
	/* yielding quadratic volume controls?                               */
	/* And, oss_audio_get_ogain() tries PCM, SPEAKER, and OGAIN          */
	/* in turn, and returns the value from the *first* that succeeds.    */
	/* (Should OGAIN be first??)                                         */
	/*                                                                   */
	/* The above changes seem usable on our SBLive card, but I haven't   */
	/* been able to try it elsewhere.                                    */
	int volume;

        UNUSED(ad); assert(devices[ad].mixer_wfd > 0);

	volume = vol << 8 | vol;


	/*
	 * On the AC97-based cards, we want to just set the PCM gain.
	 */

	if (devices[ad].is_ac97) {
		if (ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_PCM), &volume) < 0) {
			perror("setting volume");
		}
	} else {
		/* Use & not && -- we want to execute all of these */
		if ((ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_PCM),     &volume) < 0)
		 &  (ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_SPEAKER), &volume) < 0)
		 &  (ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_OGAIN),   &volume) < 0)
		 &  (ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_LINE1),   &volume) < 0)
		 &  (ioctl(devices[ad].mixer_wfd, MIXER_WRITE(SOUND_MIXER_LINE2),   &volume) < 0)) {
			perror("Setting volume");
		}
	}
}

int
oss_audio_get_ogain(audio_desc_t ad)
{
	int volume;

        UNUSED(ad); assert(devices[ad].audio_wfd > 0);

	if (ioctl(devices[ad].mixer_wfd, MIXER_READ(SOUND_MIXER_PCM),     &volume) == -1
	&&  ioctl(devices[ad].mixer_wfd, MIXER_READ(SOUND_MIXER_SPEAKER), &volume) == -1
	&&  ioctl(devices[ad].mixer_wfd, MIXER_READ(SOUND_MIXER_OGAIN),   &volume) == -1) {
		perror("Getting volume");
	}
	return volume & 0x000000ff; /* Extract left channel volume */
}

void
oss_audio_loopback(audio_desc_t ad, int gain)
{
        UNUSED(ad); assert(devices[ad].audio_rfd > 0);

        gain = gain << 8 | gain;
        if (ioctl(devices[ad].mixer_rfd, MIXER_WRITE(SOUND_MIXER_IMIX), &gain) == -1) {
                perror("audio loopback");
        }
}

int
oss_audio_read(audio_desc_t ad, u_char *buf, int read_bytes)
{
        int 		read_len, available;
	audio_buf_info	info;

        assert(devices[ad].audio_rfd > 0);        

        /* Figure out how many bytes we can read before blocking... */
        ioctl(devices[ad].audio_rfd, SNDCTL_DSP_GETISPACE, &info);
        available = min(info.bytes, read_bytes);

        read_len  = read(devices[ad].audio_rfd, (char *)buf, available);
	if (read_len < 0) {
                perror("audio_read");
		return 0;
        }

        return read_len;
}

int
oss_audio_write(audio_desc_t ad, u_char *buf, int write_bytes)
{
        int    		 done, len;
        char  		*p;

        assert(devices[ad].audio_wfd > 0);
        
        p   = (char *) buf;
        len = write_bytes;
	errno = 0;
        while (1) {
                if ((done = write(devices[ad].audio_wfd, p, len)) == len) {
                        break;
                }
		if (errno != EINTR && errno != EAGAIN) {
			perror("audio_write");
			if(done < 0) done=0;
			return write_bytes - (len - done);
                }
                len -= done;
                p   += done;
        }
        return write_bytes;
}

/* Set ops on audio device to be non-blocking */
void
oss_audio_non_block(audio_desc_t ad)
{
	int  on = 1;

        assert(devices[ad].audio_rfd > 0);
        assert(devices[ad].audio_wfd > 0);

	if (ioctl(devices[ad].audio_rfd, FIONBIO, (char *)&on) < 0) {
		debug_msg("Failed to set non-blocking mode on audio device!\n");
	}
	if (ioctl(devices[ad].audio_wfd, FIONBIO, (char *)&on) < 0) {
		debug_msg("Failed to set non-blocking mode on audio device!\n");
	}
}

/* Set ops on audio device to block */
void
oss_audio_block(audio_desc_t ad)
{
	int  on = 0;

        assert(devices[ad].audio_rfd > 0);
        assert(devices[ad].audio_wfd > 0);

	if (ioctl(devices[ad].audio_rfd, FIONBIO, (char *)&on) < 0) {
		debug_msg("Failed to set blocking mode on audio device!\n");
	}
	if (ioctl(devices[ad].audio_wfd, FIONBIO, (char *)&on) < 0) {
		debug_msg("Failed to set blocking mode on audio device!\n");
	}
}

void
oss_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
	/* There appears to be no-way to select this with OSS... */
        assert(devices[ad].audio_wfd > 0);
	UNUSED(port);
	return;
}

audio_port_t
oss_audio_oport_get(audio_desc_t ad)
{
	/* There appears to be no-way to select this with OSS... */
        assert(devices[ad].audio_wfd > 0);
	return out_ports[0].port;
}

int 
oss_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_OUT_PORTS;
}

const audio_port_details_t*
oss_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        if (idx >= 0 && idx < (int)NUM_OUT_PORTS) {
                return &out_ports[idx];
        }
        return NULL;
}

void
oss_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	int portmask;
	int recsrc;
	int gain;

        UNUSED(ad); assert(devices[ad].mixer_rfd > 0);

        switch (port) {
		case AUDIO_MICROPHONE: 
			debug_msg("Trying to select microphone input...\n");
			recsrc = SOUND_MASK_MIC;  
			break;
		case AUDIO_LINE_IN:    
			debug_msg("Trying to select line input...\n");
			recsrc = SOUND_MASK_LINE; 
			break;
		case AUDIO_CD:         
			debug_msg("Trying to select CD input...\n");
			recsrc = SOUND_MASK_CD;   
			break;
		default:
			debug_msg("Port not recognized\n");
			return;
        }

        /* Can we select chosen port ? */
        if (devices[ad].rec_mask & recsrc) {
                portmask = recsrc;
                if ((ioctl(devices[ad].mixer_rfd, MIXER_WRITE(SOUND_MIXER_RECSRC), &recsrc) == -1) && !(recsrc & portmask)) {
                        debug_msg("WARNING: Unable to select recording source!\n");
                        return;
                }
                gain = oss_audio_get_igain(ad);
                iport = port;
                oss_audio_set_igain(ad, gain);
		debug_msg("...okay\n");
        } else {
                debug_msg("Audio device doesn't support recording from port %d (%s)\n", port, oss_mixer_channels[port]);
        }
}

audio_port_t
oss_audio_iport_get(audio_desc_t ad)
{
        assert(devices[ad].audio_rfd > 0);
	return iport;
}

int
oss_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_IN_PORTS;
}

const audio_port_details_t*
oss_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        if (idx >= 0 && idx < (int)NUM_IN_PORTS) {
                return &in_ports[idx];
        }
        return NULL;
}

static int
oss_audio_select(audio_desc_t ad, int delay_us)
{
        fd_set rfds;
        struct timeval tv;

        assert(devices[ad].audio_rfd > 0);
        
        tv.tv_sec = 0;
        tv.tv_usec = delay_us;

        FD_ZERO(&rfds);
        FD_SET(devices[ad].audio_rfd, &rfds);

        select(devices[ad].audio_rfd+1, &rfds, NULL, NULL, &tv);

        return FD_ISSET(devices[ad].audio_rfd, &rfds);
}

void
oss_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        oss_audio_select(ad, delay_ms * 1000);
}

int 
oss_audio_is_ready(audio_desc_t ad)
{
        return oss_audio_select(ad, 0);
}

int
oss_audio_supports(audio_desc_t ad, audio_format *fmt)
{
        int i;

        for(i = 0; i < devices[ad].num_supported_formats; i++) {
                if (devices[ad].supported_formats[i].channels    == fmt->channels 
		&&  devices[ad].supported_formats[i].sample_rate == fmt->sample_rate) {
			return TRUE;
		}
        }
        return FALSE;
}

int
oss_get_device_count()
{
	return num_devices;
}

char *
oss_get_device_name(audio_desc_t ad)
{
        assert((ad >= 0) && (ad < num_devices));
	return devices[ad].name;
}

