/*
 * FILE:     audio_hw/linux_ixj.c - QuickNet audio device driver
 * PROGRAM:  RAT
 * AUTHOR:   James MacLean 
 * MODIFIED: Orion Hodson + Robert Olson + Colin Perkins
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1996-2001 University College London
 * All rights reserved.
 *
 * NOTE : Environement variable IXJ_AEC sets Echo Cancellation 
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */

#define __user
#include <time.h>
#include <linux/telephony.h>
#include <linux/ixjuser.h>
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "memory.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/linux_ixj.h"

enum { AUDIO_PHONE, AUDIO_SPEAKER, AUDIO_MICROPHONE};

/* Info to match port id's to port name's */
static audio_port_details_t in_ports[] = {
        {AUDIO_PHONE,    AUDIO_PORT_PHONE},
        {AUDIO_MICROPHONE, AUDIO_PORT_MICROPHONE}
};

#define NUM_IN_PORTS (sizeof(in_ports)/(sizeof(in_ports[0])))

static audio_port_details_t out_ports[] = {
        {AUDIO_PHONE,   AUDIO_PORT_PHONE},
        {AUDIO_SPEAKER,   AUDIO_PORT_SPEAKER}
};

#define NUM_OUT_PORTS (sizeof(out_ports)/(sizeof(out_ports[0])))

int	ixjiport     = PORT_POTS;

char *ixj_mixer_channels[1];

#define IXJ_MAX_DEVICES   		 4	/* Magic constant based on the OSS source */
#define IXJ_MAX_NAME_LEN 		64	/* Magic constant based on the OSS source */
#define IXJ_MAX_SUPPORTED_FORMATS 	1 	/* == 8k */

#define IXJ_DUPLEX_HALF   		 0
#define IXJ_DUPLEX_FULL   		 1
#define IXJ_DUPLEX_PAIR			 2	/* ...a pair of half duplex devices emulating full duplex */

struct ixj_device {
	char		*name;
	char		 audio_dev[16];
	char		 mixer_dev[16];
	int		 audio_fd;
	int		 mixer_fd;
	int		 duplex;	/* TRUE if the device supports full duplex */
	audio_format	 supported_formats[IXJ_MAX_SUPPORTED_FORMATS];
	int		 num_supported_formats;
	int		 frame_len;
};

static struct ixj_device	devices[IXJ_MAX_DEVICES];
static int			num_devices;

static int
deve2oss(deve_e encoding)
{
        switch(encoding) {
        	case DEV_PCMU: return ULAW;
		case DEV_PCMA: return ALAW;
        	case DEV_S8:   return LINEAR8;
        	case DEV_S16:  return LINEAR16;
                case DEV_U8:   return LINEAR8;
        }
        abort();
	return 0;
}

static int
ixj_probe_mixer_device(int i, struct ixj_device *device)
{
	int dsp, ver, fd;
	/* We can't find a working mixer device - this is not  */
	/* necessarily a problem, some soundcards don't have a */
	/* corresponding mixer device.                         */
	UNUSED(i);
	debug_msg("QuickNet has no mixer\n");
	sprintf(device->mixer_dev, "/dev/phone0");
	fd = open(device->mixer_dev, O_RDWR);
	if (fd < 0) {
		device->mixer_fd = -1;
		debug_msg("Can not open %s\n", device->mixer_dev);
		return FALSE;
	}
	device->name = (char *) malloc(IXJ_MAX_NAME_LEN + 6);
	memset(device->name, 0, IXJ_MAX_NAME_LEN + 6);
	ioctl(fd, IXJCTL_DSP_TYPE, &dsp);
	ioctl(fd, IXJCTL_DSP_VERSION, &ver);
	close(fd);
	sprintf(device->name, "IXJ: DSP type 0x%x, DSP Version 0x%x", dsp, ver);
	return TRUE;
}

static int
ixj_test_mode(int fd, int speed, int stereo)
{
	UNUSED(fd);

        if (stereo) {
		debug_msg("  disabled (%d channels not supported)\n", stereo + 1);
		return FALSE;
        }

// QuickNet only handles 8k 
        if (speed != 8000) {
		debug_msg("  disabled (%dHz sampling not supported)\n", speed);
		return FALSE;
        }
        
        return TRUE;
}

static int
ixj_probe_audio_device(int i, struct ixj_device *device)
{
	/* Probe /dev/phoneX, and fill in mixer related parts of the device. */
	/* If we are requested to probe /dev/phone0, and that file doesn't   */
	/* exist, we probe /dev/phone instead (if that is not a symlink).    */
	/* This is for compatibility with some old Linux distributions,    */
	/* which have a broken /dev.                                       */
	struct stat	s;
	int		speed[] = {8000};
        int 		stereo, speed_index, fd;

	sprintf(device->audio_dev, "/dev/phone%d", i);
	fd = open(device->audio_dev, O_RDWR);
	if ((fd < 0) && (i == 0)) {
		if ((stat("/dev/phone", &s) == 0) && !S_ISLNK(s.st_mode)) {
			sprintf(device->audio_dev, "/dev/phone");
			fd = open(device->audio_dev, O_RDWR);
		}
	}
	if (fd < 0) {
		debug_msg("cannot open %s - %s\n", device->audio_dev, strerror(errno));
		return FALSE;
	}

	/* The device is full duplex. */
	device->duplex = IXJ_DUPLEX_FULL;

	/* Check which sampling modes are supported...                       */
	device->num_supported_formats = 0;
	for (speed_index = 0; speed_index < 1; speed_index++) {
                for (stereo = 0; stereo < 2; stereo++) {
			debug_msg("testing %s support for %5dHz %s\n", device->audio_dev, speed[speed_index], stereo?"stereo":"mono");
                        if (ixj_test_mode(fd, speed[speed_index], stereo)) {
				device->supported_formats[device->num_supported_formats].sample_rate = speed[speed_index];
				device->supported_formats[device->num_supported_formats].channels    = stereo + 1;
				device->num_supported_formats++;
			}
                }
        }

	close(fd);
	device->audio_fd = -1;
	return TRUE;
}

int
ixj_audio_init(void)
{
	/* One time initialization of the OSS audio driver. We probe the    */
	/* available devices, to setup the devices[] array and num_devices. */
	/* Note that it is entirely legal to have an audio device without a */
	/* corresponding mixer device.                                      */
	int			i;
	struct ixj_device	device;

	num_devices = 0;
	for (i = 0; i < IXJ_MAX_DEVICES; i++) {
		ixj_probe_mixer_device(i, &device);
		if (ixj_probe_audio_device(i, &device)) {
			devices[num_devices++] = device;
			debug_msg("found \"%s\" as %s,%s\n", device.name, device.audio_dev, device.mixer_dev);

		}
	}
	debug_msg("number of valid devices: %d\n", num_devices);
	return num_devices;
}

static int
ixj_configure_device(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
	/* Configure the audio device. This function is called for both read */
	/* and write file descriptors, so the setup must not break if done   */
	/* twice on the same device.                                         */
	int  	mode, stereo, speed;

	mode = deve2oss(ifmt->encoding);
	switch(mode)
	  {
	    case ULAW:
	      devices[ad].frame_len = 160;
	      break;
	    case ALAW:
	      devices[ad].frame_len = 160;
	      break;
	    case LINEAR16:
	      devices[ad].frame_len = 320;
	      break;
	    case LINEAR8:
	      devices[ad].frame_len = 160;
	      break;
	    default:
	      debug_msg("Illegal CODEC %d\n", mode);
	  }
	/* This low-level IOCTL calling sets the sample to 20ms */
	ioctl(devices[ad].audio_fd, PHONE_FRAME, 20);
	ioctl(devices[ad].audio_fd, PHONE_PLAY_DEPTH, 2);
	ioctl(devices[ad].audio_fd, PHONE_REC_DEPTH, 2);
	ioctl(devices[ad].audio_fd, IXJCTL_DSP_IDLE);
	if((ioctl(devices[ad].audio_fd, PHONE_REC_CODEC, mode) == -1)) {

		if (ifmt->encoding == DEV_S16) {
			audio_format_change_encoding(ifmt, DEV_PCMU);
			audio_format_change_encoding(ofmt, DEV_PCMU);
			devices[ad].frame_len = 160;
			if ((ioctl(devices[ad].audio_fd, PHONE_REC_CODEC, mode) == -1)) {
				debug_msg("No Luck setting format\n");
				return FALSE;
			}
			debug_msg("device doesn't support 16bit audio, using 8 bit PCMU\n");
		}
	}
	ioctl(devices[ad].audio_fd, PHONE_REC_START);
//	if(ixjiport == 1 && ioctl(devices[ad].audio_fd, PHONE_RING)) {
//		debug_msg("Rang the phone\n");
//	}
	ioctl(devices[ad].audio_fd, PHONE_PLAY_CODEC, mode);
	ioctl(devices[ad].audio_fd, PHONE_PLAY_START);
// This sets the card up OK for EC, but it appears these cards like to jitter :(
	if(getenv("IXJ_AEC") != 0) {
		debug_msg("AEC environment found\n");
		mode = atoi(getenv("IXJ_AEC"));
		switch(mode) {
			case 0:
      				mode = AEC_OFF;
      				break;
			case 1:
      				mode = AEC_LOW;
      				break;
			case 2:
      				mode = AEC_MED;
      				break;
			case 3:
      				mode = AEC_HIGH;
      				break;
			case 4:
      				mode = AEC_AUTO;
      				break;
			default:
      				mode = AEC_AGC;
      				break;
		}
		ioctl(devices[ad].audio_fd, IXJCTL_AEC_START, 1);
		ioctl(devices[ad].audio_fd, IXJCTL_AEC_START, mode);
		debug_msg("AEC set to %d\n", mode);
	} else {
		ioctl(devices[ad].audio_fd, IXJCTL_AEC_START, 1);
		ioctl(devices[ad].audio_fd, IXJCTL_AEC_STOP);
		debug_msg("AEC turned off\n");
	}

	stereo = ifmt->channels - 1; 
	assert(stereo == 0 || stereo == 1);
	if (stereo) {
		debug_msg("device doesn't support %d channels!\n", ifmt->channels);
		return FALSE;
	}

	speed = ifmt->sample_rate;
	if (speed != 8000) {
		debug_msg("device doesn't support %dHz sampling rate in full duplex!\n", ifmt->sample_rate);
		return FALSE;
	}
	return TRUE;
}

int
ixj_audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
	/* Try to open the audio device, returning TRUE if successful. Note  */
	/* that the order in which the device is set up is important: don't  */
	/* reorder this code unless you really know what you're doing.       */
//	int  	frag     = 0x7fff0000; 			/* unlimited number of fragments */
//	int	bytes_per_block;

        if (ad < 0 || ad > IXJ_MAX_DEVICES) {
                debug_msg("invalid audio descriptor (%d)", ad);
                return FALSE;
        }

	devices[ad].audio_fd = -1;
	devices[ad].mixer_fd = -1;

	switch (devices[ad].duplex) {
		case IXJ_DUPLEX_HALF:
			debug_msg("cannot open half duplex devices\n");
			return FALSE;
		case IXJ_DUPLEX_FULL:
			/* Open the device, and switch to full duplex mode. */
			assert(strcmp(devices[ad].audio_dev, devices[ad].audio_dev) == 0);

			devices[ad].audio_fd = open(devices[ad].audio_dev, O_RDWR);
			if (devices[ad].audio_fd <= 0) {
				debug_msg("unable to open %s\n", devices[ad].audio_dev);
				return FALSE;
			}

			/* Now open the corresponding mixer device... */
			devices[ad].mixer_fd = devices[ad].audio_fd;
			break;
		default:
			debug_msg("invalid duplex mode - this can never happen\n");
			abort();
	}
	
// Maybe PhoneJack can be altered to adhere to this? I have not persued it :(
	/* Set 30 ms blocksize - this only modulates read sizes, and hence only has to be done on audio_fd */
	// bytes_per_block = 30 * (ifmt->sample_rate / 1000) * (ifmt->bits_per_sample / 8);
	/* Round to the nearest legal frag size (next power of two lower...) */
	// frag |= (int) (log(bytes_per_block)/log(2));

	if (!ixj_configure_device(ad, ifmt, ofmt)) {
		ixj_audio_close(ad);
		return FALSE;
	}

	/* Select microphone input. We can't select output source...  */
//	ixj_audio_iport_set(ad, ixjiport);

	return TRUE;
}

/* Close the audio device */
void
ixj_audio_close(audio_desc_t ad)
{
	ixj_audio_drain(ad);
	ioctl(devices[ad].audio_fd, IXJCTL_PORT, PORT_POTS);
	ioctl(devices[ad].audio_fd, PHONE_PLAY_STOP);
	ioctl(devices[ad].audio_fd, PHONE_REC_STOP);
	if (devices[ad].audio_fd != -1) close(devices[ad].audio_fd);

        devices[ad].audio_fd = -1;
        devices[ad].mixer_fd = -1;
	debug_msg("All closed down!\n");
}

/* Flush input buffer */
void
ixj_audio_drain(audio_desc_t ad)
{
        u_char buf[1024];

        assert(ad < IXJ_MAX_DEVICES);
        assert(devices[ad].audio_fd > 0);

        while(ixj_audio_read(ad, buf, devices[ad].frame_len) == devices[ad].frame_len);
}

int
ixj_audio_duplex(audio_desc_t ad)
{
        /* We don't open device if not full duplex. */
	return devices[ad].duplex;
}

/* Gain and volume values are in the range 0 - MAX_AMP */
void
ixj_audio_set_igain(audio_desc_t ad, int gain)
{
	int volume = gain;

        assert(ad < IXJ_MAX_DEVICES);
        assert(devices[ad].audio_fd > 0);

	if (ioctl(devices[ad].audio_fd, PHONE_REC_VOLUME_LINEAR, volume) == -1) {
		perror("Setting gain");
	}
	debug_msg("Record volume set to %d, gain was %d\n", volume, gain);
}

int
ixj_audio_get_igain(audio_desc_t ad)
{
	int volume=100;

        UNUSED(ad); assert(devices[ad].audio_fd > 0); assert(ad < IXJ_MAX_DEVICES);

	if ((volume = ioctl(devices[ad].audio_fd, PHONE_REC_VOLUME_LINEAR, -1)) == -1) {
		perror("Getting gain");
	}
	debug_msg("getting igain vol=%d\n",  volume);
	return volume;
}

void
ixj_audio_set_ogain(audio_desc_t ad, int vol)
{
	int volume;
	volume = vol;

        UNUSED(ad); assert(devices[ad].audio_fd > 0);

	if ((ioctl(devices[ad].audio_fd, PHONE_PLAY_VOLUME_LINEAR,  volume) < 0)) {
			perror("Setting volume");
	}
	debug_msg("Speaker volume set to %d, request was %d\n", volume, vol);
}

int
ixj_audio_get_ogain(audio_desc_t ad)
{
	int volume=100;

        UNUSED(ad); assert(devices[ad].audio_fd > 0);

	if ((volume = ioctl(devices[ad].audio_fd, PHONE_PLAY_VOLUME_LINEAR, -1)) == -1) {
		perror("Getting volume");
	}
	return volume;
}

void
ixj_audio_loopback(audio_desc_t ad, int gain)
{
        UNUSED(ad); assert(devices[ad].audio_fd > 0);
        gain = gain << 8 | gain;
}

// READ some bytes
// . Must be read in pre-determined frame size chunks
// . If read buffer is not ready, EAGAIN returns
//
#define AUDIO_READ_BUFFER 2048
int
ixj_audio_read(audio_desc_t ad, u_char *buf, int read_bytes)
{
        int 		read_len=0, available=0;
	static char mybuf[AUDIO_READ_BUFFER];
	static int offset=0;

        assert(devices[ad].audio_fd > 0);        

        /* Figure out how many bytes we can read before blocking... */
	if(offset > read_bytes) {
		memcpy(buf, mybuf, read_bytes);
		memcpy(mybuf, &mybuf[read_bytes], offset - read_bytes);
		offset = offset - read_bytes;
//		debug_msg("Return lie %d\n", read_bytes);
		return read_bytes;
	}

//	debug_msg("Reading %d bytes requested\n", read_bytes);
	read_len=0;
	if(ioctl(devices[ad].audio_fd, PHONE_HOOKSTATE) || ixjiport != 1) {
	  read_len  = read(devices[ad].audio_fd, &mybuf[offset], devices[ad].frame_len);
	  if (read_len < 0 && errno != EAGAIN && errno != EINTR) {
		perror("audio_read");
		return 0;
	  }
	}
	if(read_len > 0) {
		offset += read_len;
		available = min(offset, read_bytes);
		memcpy(buf, mybuf, available);
		memcpy(mybuf, &mybuf[available], offset - available);
		offset = offset - available;
	}

//	if(read_len > 0) debug_msg("Got %d of %d available  frame_len=%d , read_bytes=%d\n", read_len,  available, devices[ad].frame_len, read_bytes);

        return available;
}

// WRITE some bytes
// . Must be written in pre-determined frame size chunks
// . If write device is not ready, EAGAIN returns
//
#define AUDIO_WRITE_BUFFER 4096
int
ixj_audio_write(audio_desc_t ad, u_char *buf, int write_bytes)
{
        int    		 done;
	static char mybuf[AUDIO_WRITE_BUFFER];
	static int offset=0;

        assert(devices[ad].audio_fd > 0);
        assert(AUDIO_WRITE_BUFFER > (offset + write_bytes));

	memcpy(&mybuf[offset], buf, write_bytes);
	offset += write_bytes;
	if(offset < devices[ad].frame_len) {
//		debug_msg("Writing lie of %d bytes\n", write_bytes);
		return write_bytes;
	}
        
//	debug_msg("Writing %d bytes to %d, offset=%d\n", write_bytes, devices[ad].audio_fd, offset);
        while (1) {
		errno=0;
		if(offset < devices[ad].frame_len) break;

		if(ioctl(devices[ad].audio_fd, PHONE_HOOKSTATE) || ixjiport != 1) {
		  done = write(devices[ad].audio_fd, mybuf, devices[ad].frame_len);
                  if (done != devices[ad].frame_len && errno != EINTR && errno != EAGAIN) {
			if(done > 0) {
				memcpy(mybuf, &mybuf[done], offset - done);
				offset -= done;
			} else done=0;
			return min(write_bytes, done);
                  }
		} else {
			offset = 0;
			done = 0;
		}
		if(done > 0) {
			memcpy(mybuf, &mybuf[done], offset - done);
			offset -= done;
		}
//		if(errno == 0) debug_msg("Offset now %d, done=%d, errno=%d, frame=%d\n", offset, done, errno, devices[ad].frame_len);
        }
//	debug_msg("Writing %d bytes, offset left at %d errno=%d\n", write_bytes, offset, errno);
        return write_bytes;
}

/* Set ops on audio device to be non-blocking */
void
ixj_audio_non_block(audio_desc_t ad)
{
	int  on = 1;

        assert(devices[ad].audio_fd > 0);

	on = 1;
	if (ioctl(devices[ad].audio_fd, FIONBIO, (char *)&on) < 0) {
		debug_msg("Failed to set non-blocking mode on audio device!\n");
	}
	debug_msg("Set non-blocking\n");
}

/* Set ops on audio device to block */
void
ixj_audio_block(audio_desc_t ad)
{
	int  on = 0;

        assert(devices[ad].audio_fd > 0);

	on = 0;
	if (ioctl(devices[ad].audio_fd, FIONBIO, (char *)&on) < 0) {
		debug_msg("Failed to set blocking mode on audio device!\n");
	}
	debug_msg("Set blocking\n");
}

void
ixj_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
	/* There appears to be no-way to select this with OSS... */
        assert(devices[ad].audio_fd > 0);
	UNUSED(port);
	return;
}

audio_port_t
ixj_audio_oport_get(audio_desc_t ad)
{
	/* There appears to be no-way to select this with OSS... */
        assert(devices[ad].audio_fd > 0);
	return out_ports[0].port;
}

int 
ixj_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_OUT_PORTS;
}

const audio_port_details_t*
ixj_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        if (idx >= 0 && idx < (int)NUM_OUT_PORTS) {
                return &out_ports[idx];
        }
        return NULL;
}

void
ixj_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	int recsrc;
	int gain;

        UNUSED(ad); assert(devices[ad].audio_fd > 0);

        switch (port) {
		case AUDIO_MICROPHONE: 
			debug_msg("Trying to select microphone input...\n");
			recsrc = PORT_SPEAKER;  
			break;
		case AUDIO_PHONE: 
			debug_msg("Trying to select handset input...\n");
			recsrc = PORT_POTS;  
			break;
		default:
			debug_msg("Port not recognized\n");
			return;
        }

	if ((ioctl(devices[ad].audio_fd, IXJCTL_PORT, recsrc) == -1)) {
		debug_msg("WARNING: Unable to select recording source!\n");
		return;
	}
	ixjiport = port;
	gain = ixj_audio_get_igain(ad);
	ixj_audio_set_igain(ad, gain);
	debug_msg("...okay\n");
}

audio_port_t
ixj_audio_iport_get(audio_desc_t ad)
{
        assert(devices[ad].audio_fd > 0);
	return ixjiport;
}

int
ixj_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_IN_PORTS;
}

const audio_port_details_t*
ixj_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        if (idx >= 0 && idx < (int)NUM_IN_PORTS) {
                return &in_ports[idx];
        }
        return NULL;
}

static int
ixj_audio_select(audio_desc_t ad, int delay_us)
{
        fd_set rfds;
        struct timeval tv;

        assert(devices[ad].audio_fd > 0);
        
        tv.tv_sec = 0;
        tv.tv_usec = delay_us;

        FD_ZERO(&rfds);
        FD_SET(devices[ad].audio_fd, &rfds);

        select(devices[ad].audio_fd+1, &rfds, NULL, NULL, &tv);

        return FD_ISSET(devices[ad].audio_fd, &rfds);
}

void
ixj_audio_wait_for(audio_desc_t ad, int delay_ms)
{
        ixj_audio_select(ad, delay_ms * 1000);
}

int 
ixj_audio_is_ready(audio_desc_t ad)
{
        return ixj_audio_select(ad, 0);
}

int
ixj_audio_supports(audio_desc_t ad, audio_format *fmt)
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
ixj_get_device_count()
{
	return num_devices;
}

char *
ixj_get_device_name(audio_desc_t ad)
{
        assert((ad >= 0) && (ad < num_devices));
	return devices[ad].name;
}

