/* sdiaudiocfg.c
 *
 * SDI audio configuration program.
 *
 * Copyright (C) 2009-2010 Linear Systems Ltd. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of Linear Systems Ltd. nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LINEAR SYSTEMS LTD. "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LINEAR SYSTEMS LTD. OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Linear Systems can be contacted at <http://www.linsys.ca/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

#include "sdiaudio.h"
#include "master.h"
#include "../util.h"

#define MAXLEN 256
#define BUFFERS_FLAG		0x00000001
#define BUFSIZE_FLAG		0x00000002
#define SAMPLESIZE_FLAG		0x00000010
#define CHANNELS_FLAG		0x00000020
#define SAMPLERATE_FLAG		0x00000040
#define NONAUDIO_FLAG		0x00000080

static const char progname[] = "sdiaudiocfg";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/sdiaudio/sdiaudio%cx%i/%s";
	int opt;
	unsigned int write_flags;
	struct stat buf;
	int num;
	char type, name[MAXLEN], data[MAXLEN];
	unsigned long int buffers, bufsize, samplesize, channels, samplerate, nonaudio;
	int retcode;
	char *endptr;

	/* Parse the command line */
	write_flags = 0;
	buffers = 0;
	bufsize = 0;
	samplesize = 0;
	channels = 0;
	samplerate = 0;
	nonaudio = 0;
	while ((opt = getopt (argc, argv, "b:c:hn:r:s:Vz:")) != -1) {
		switch (opt) {
		case 'b':
			write_flags |= BUFFERS_FLAG;
			buffers = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of buffers: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'c':
			write_flags |= CHANNELS_FLAG;
			channels = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid audio channel enable: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Configure an SDI audio interface.\n\n");
			printf ("  -b BUFFERS\tset the number of buffers\n");
			printf ("  -c CHANNELS\tset the audio channel enable\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -n NONAUDIO\tset PCM or non-audio "
				"(transmitters only)\n");
			printf ("  -r SAMPLERATE\tset the audio sample rate "
				"(transmitters only)\n");
			printf ("  -s BUFSIZE\tset the buffer size\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("  -z SAMPLESIZE set the audio sample size\n");
			printf ("\nIf no options are specified, "
				"the current configuration is displayed.\n");
			printf ("\nBUFFERS must be two or more.\n");
			printf ("\nCHANNELS may be:\n"
				"\t0 (audio disabled)\n"
				"\t2 (2 channels)\n"
				"\t4 (4 channels)\n"
				"\t6 (6 channels)\n"
				"\t8 (8 channels)\n");
			printf ("\nNONAUDIO may be:\n"
				"\t0x0000 (PCM)\n"
				"\t0x00ff (non-audio)\n"
				"\tbetween 0x0000 and 0x00ff (mixed audio and data)\n");
			printf ("\nSAMPLERATE may be:\n"
				"\t32000 (32 kHz)\n"
				"\t44100 (44.1 kHz)\n"
				"\t48000 (48 kHz)\n");
			printf ("\nBUFSIZE must be "
				"a positive multiple of four,\n"
				"and at least 1024 bytes for transmitters.\n");
			printf ("\nSAMPLESIZE may be:\n"
				"\t16 (16-bit)\n"
				"\t24 (24-bit)\n"
				"\t32 (32-bit)\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'n':
			write_flags |= NONAUDIO_FLAG;
			nonaudio = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid non-audio: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'r':
			write_flags |= SAMPLERATE_FLAG;
			samplerate = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid audio sample rate: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 's':
			write_flags |= BUFSIZE_FLAG;
			bufsize = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid buffer size: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2009-2010 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case 'z':
			write_flags |= SAMPLESIZE_FLAG;
			samplesize = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid audio sample size: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) < 1) {
		fprintf (stderr, "%s: missing arguments\n", argv[0]);
		goto USAGE;
	} else if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Get the sysfs info */
	memset (&buf, 0, sizeof (buf));
	if (stat (argv[optind], &buf) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get the file status");
		return -1;
	}
	if (!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", argv[0]);
		return -1;
	}
	type = (buf.st_rdev & 0x0080) ? 'r' : 't';
	num = buf.st_rdev & 0x007f;
	snprintf (name, sizeof (name), fmt, type, num, "dev");
	memset (data, 0, sizeof (data));
	if (util_read (name, data, sizeof (data)) < 0) {
		fprintf (stderr, "%s: error reading %s: ", argv[0], name);
		perror (NULL);
		return -1;
	}
	if (strtoul (data, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not an SDI audio device\n", argv[0]);
		return -1;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", argv[0], name);
		return -1;
	}

	retcode = 0;
	printf ("%s:\n", argv[optind]);
	if (write_flags) {
		if (write_flags & BUFFERS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "buffers");
			snprintf (data, sizeof (data), "%lu\n", buffers);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the number of buffers");
				return -1;
			}
			printf ("\tSet number of buffers = %lu.\n", buffers);
		}
		if (write_flags & BUFSIZE_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "bufsize");
			snprintf (data, sizeof (data), "%lu\n", bufsize);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the buffer size");
				return -1;
			}
			printf ("\tSet buffer size = %lu bytes.\n", bufsize);
		}
		if (write_flags & SAMPLESIZE_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "sample_size");
			snprintf (data, sizeof (data), "%lu\n", samplesize);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface audio sample size");
				return -1;
			}
			switch (samplesize) {
			case SDIAUDIO_CTL_AUDSAMP_SZ_16:
				printf ("\tAssuming 16-bit audio.\n");
				break;
			case SDIAUDIO_CTL_AUDSAMP_SZ_24:
				printf ("\tAssuming 24-bit audio.\n");
				break;
			case SDIAUDIO_CTL_AUDSAMP_SZ_32:
				printf ("\tAssuming 32-bit audio.\n");
				break;
			default:
				printf ("\tSet audio sample size = %lu.\n",
					samplesize);
				break;
			}
		}
		if (write_flags & SAMPLERATE_FLAG) {
			if (type == 'r') {
				fprintf (stderr, "%s: "
					"unable to set the "
					"interface audio sample rate: "
					"Not a transmitter\n", argv[0]);
				return -1;
			}
			snprintf (name, sizeof (name),
				fmt, type, num, "sample_rate");
			snprintf (data, sizeof (data), "%lu\n", samplerate);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface audio sample rate");
				return -1;
			}
			switch (samplerate) {
			case 32000:
				printf ("\tAssuming 32 kHz audio.\n");
				break;
			case 44100:
				printf ("\tAssuming 44.1 kHz audio.\n");
				break;
			case 48000:
				printf ("\tAssuming 48 kHz audio.\n");
				break;
			default:
				printf ("\tSet audio sample rate = %lu.\n",
					samplerate);
				break;
			}
		}
		if (write_flags & CHANNELS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "channels");
			snprintf (data, sizeof (data), "%lu\n", channels);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface audio channel enable");
				return -1;
			}
			switch (channels) {
			case SDIAUDIO_CTL_AUDCH_EN_0:
				printf ("\tDisabling audio.\n");
				break;
			case SDIAUDIO_CTL_AUDCH_EN_2:
				printf ("\tAssuming 2 channels of audio.\n");
				break;
			case SDIAUDIO_CTL_AUDCH_EN_4:
				printf ("\tAssuming 4 channels of audio.\n");
				break;
			case SDIAUDIO_CTL_AUDCH_EN_6:
				printf ("\tAssuming 6 channels of audio.\n");
				break;
			case SDIAUDIO_CTL_AUDCH_EN_8:
				printf ("\tAssuming 8 channels of audio.\n");
				break;
			default:
				printf ("\tSet audio channel enable = %lu.\n",
					channels);
				break;
			}
		}
		if (write_flags & NONAUDIO_FLAG) {
			if (type == 'r') {
				fprintf (stderr, "%s: "
					"unable to set the "
					"interface non-audio: "
					"Not a transmitter\n", argv[0]);
				return -1;
			}
			snprintf (name, sizeof (name),
				fmt, type, num, "non_audio");
			snprintf (data, sizeof (data), "0x%04lX\n", nonaudio);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface non-audio");
				return -1;
			}
			switch (nonaudio) {
			case 0x0000:
				printf ("\tPassing PCM audio.\n");
				break;
			case 0x00ff:
				printf ("\tPassing non-audio.\n");
				break;
			default:
				printf ("\tSet non-audio = 0x%04lX.\n", nonaudio);
				break;
			}
		}
	} else {
		snprintf (name, sizeof (name),
			fmt, type, num, "buffers");
		if (util_strtoul (name, &buffers) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the number of buffers");
			retcode = -1;
		}
		snprintf (name, sizeof (name),
			fmt, type, num, "bufsize");
		if (util_strtoul (name, &bufsize) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the buffer size");
			retcode = -1;
		}
		printf ("\t%lu x %lu-byte buffers\n",
			buffers, bufsize);

		snprintf (name, sizeof (name),
			fmt, type, num, "sample_size");
		if (util_strtoul (name, &samplesize) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the audio sample size");
			retcode = -1;
		}
		printf ("\tAudio sample size: %lu ", samplesize);
		switch (samplesize) {
		case SDIAUDIO_CTL_AUDSAMP_SZ_16:
			printf ("(assume 16-bit audio)\n");
			break;
		case SDIAUDIO_CTL_AUDSAMP_SZ_24:
			printf ("(assume 24-bit audio)\n");
			break;
		case SDIAUDIO_CTL_AUDSAMP_SZ_32:
			printf ("(assume 32-bit audio)\n");
			break;
		default:
			printf ("(unknown)\n");
			break;
		}

		if (type == 't') {
			snprintf (name, sizeof (name),
				fmt, type, num, "sample_rate");
			if (util_strtoul (name, &samplerate) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the audio sample rate");
				retcode = -1;
			}
			printf ("\tAudio sample rate: %lu ", samplerate);
			switch (samplerate) {
			case 32000:
				printf ("(assume 32 kHz audio)\n");
				break;
			case 44100:
				printf ("(assume 44.1 kHz audio)\n");
				break;
			case 48000:
				printf ("(assume 48 kHz audio)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "channels");
		if (util_strtoul (name, &channels) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get "
				"the audio channel information");
			retcode = -1;
		}
		printf ("\tAudio channels: %lu ", channels);
		switch (channels) {
		case SDIAUDIO_CTL_AUDCH_EN_0:
			printf ("(audio disabled)\n");
			break;
		case SDIAUDIO_CTL_AUDCH_EN_2:
			printf ("(assume 2 channel audio)\n");
			break;
		case SDIAUDIO_CTL_AUDCH_EN_4:
			printf ("(assume 4 channel audio)\n");
			break;
		case SDIAUDIO_CTL_AUDCH_EN_6:
			printf ("(assume 6 channel audio)\n");
			break;
		case SDIAUDIO_CTL_AUDCH_EN_8:
			printf ("(assume 8 channel audio)\n");
			break;
		default:
			printf ("(unknown)\n");
			break;
		}

		if (type == 't') {
			snprintf (name, sizeof (name),
				fmt, type, num, "non_audio");
			if (util_strtoul (name, &nonaudio) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the non-audio information");
				retcode = -1;
			}
			printf ("\tPCM/non-audio: 0x%04lX ", nonaudio);
			switch (nonaudio) {
			case 0x0000:
				printf ("(PCM audio)\n");
				break;
			case 0x00ff:
				printf ("(non-audio)\n");
				break;
			default:
				printf ("(mixed audio and data)\n");
				break;
			}
		}
	}
	return retcode;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

