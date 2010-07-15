/* sdiaudiorxcfg.c
 *
 * Demonstrate SMPTE 292M and SMPTE 259M-C receiver ioctls.
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
#include <sys/ioctl.h>

#include "sdiaudio.h"
#include "master.h"
#include "../util.h"

#define TMP_BUFLEN 80

static const char *argv0;
static const char progname[] = "sdiaudiorxcfg";

static void
show_cap (unsigned int cap)
{
	char str[TMP_BUFLEN] = "[ ] ";

	str[1] = (cap & SDIAUDIO_CAP_RX_CD) ? 'X' : ' ';
	printf ("%sCarrier detect\n", str);

	str[1] = (cap & SDIAUDIO_CAP_RX_DATA) ? 'X' : ' ';
	printf ("%sReceive data status\n", str);

	str[1] = (cap & SDIAUDIO_CAP_RX_STATS) ? 'X' : ' ';
	printf ("%sAudio delay and error statistics\n", str);

	str[1] = (cap & SDIAUDIO_CAP_RX_NONAUDIO) ? 'X' : ' ';
	printf ("%sAES3 non-audio flags\n", str);

	str[1] = (cap & SDIAUDIO_CAP_RX_24BIT) ? 'X' : ' ';
	printf ("%s24-bit sample packing\n", str);

	printf ("\nPress Enter to continue: ");
	fgets (str, TMP_BUFLEN, stdin);
	return;
}

static void
get_events (int fd)
{
	unsigned int val;
	char str[TMP_BUFLEN];

	printf ("Getting the receiver event flags.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETEVENTS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver event flags");
	} else if (val) {
		if (val & SDIAUDIO_EVENT_RX_BUFFER) {
			printf ("Driver receive buffer queue "
				"overrun detected.\n");
		}
		if (val & SDIAUDIO_EVENT_RX_FIFO) {
			printf ("Onboard receive FIFO overrun detected.\n");
		}
		if (val & SDIAUDIO_EVENT_RX_CARRIER) {
			printf ("Carrier status change detected.\n");
		}
		if (val & SDIAUDIO_EVENT_RX_DATA) {
			printf ("Receive data change detected.\n");
		}
	} else {
		printf ("No receiver events detected.\n");
	}
	printf ("\nPress Enter to continue: ");
	fgets (str, TMP_BUFLEN, stdin);
	return;
}

static void
get_buflevel (int fd)
{
	unsigned int val;

	printf ("Getting the driver receive buffer queue length.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETBUFLEVEL, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the driver receive buffer queue length");
	} else {
		printf ("Driver receive buffer queue length = %u.\n", val);
	}
	return;
}

static void
get_nonaudio (int fd)
{
	unsigned int val;

	printf ("Getting the PCM/non-audio status.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETNONAUDIO, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the PCM/non-audio status");
	} else {
		printf ("PCM/non-audio = 0x%04X\n", val);
	}
	return;
}

static void
get_audiostatus (int fd)
{
	unsigned int val;

	printf ("Getting the audio status.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETAUDSTAT, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the audio status");
	} else {
		switch (val) {
		case SDIAUDIO_CTL_ACT_CHAN_0:
			printf ("No audio control packets.\n");
			break;
		case SDIAUDIO_CTL_ACT_CHAN_2:
			printf ("Two channels.\n");
			break;
		case SDIAUDIO_CTL_ACT_CHAN_4:
			printf ("Four channels.\n");
			break;
		case SDIAUDIO_CTL_ACT_CHAN_6:
			printf ("Six channels.\n");
			break;
		case SDIAUDIO_CTL_ACT_CHAN_8:
			printf ("Eight channels.\n");
			break;
		default:
			printf ("Active audio channel flags = 0x%08X.\n", val);
			break;
		}
	}
	return;
}

static void
get_audiorate (int fd)
{
	unsigned int val;

	printf ("Getting the audio sampling rate.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETAUDRATE, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the audio rate");
	} else {
		switch (val) {
		case SDIAUDIO_CTL_ASYNC_48_KHZ:
			printf ("Asynchronous, 48 kHz.\n");
			break;
		case SDIAUDIO_CTL_ASYNC_44_1_KHZ:
			printf ("Asynchronous, 44.1 kHz.\n");
			break;
		case SDIAUDIO_CTL_ASYNC_32_KHZ:
			printf ("Asynchronous, 32 kHz.\n");
			break;
		case SDIAUDIO_CTL_ASYNC_96_KHZ:
			printf ("Asynchronous, 96 kHz.\n");
			break;
		case SDIAUDIO_CTL_ASYNC_FREE_RUNNING:
			printf ("Asynchronous, free running.\n");
			break;
		case SDIAUDIO_CTL_SYNC_48_KHZ:
			printf ("Synchronous, 48 kHz.\n");
			break;
		case SDIAUDIO_CTL_SYNC_44_1_KHZ:
			printf ("Synchronous, 44.1 kHz.\n");
			break;
		case SDIAUDIO_CTL_SYNC_32_KHZ:
			printf ("Synchronous, 32 kHz.\n");
			break;
		case SDIAUDIO_CTL_SYNC_96_KHZ:
			printf ("Synchronous, 96 kHz.\n");
			break;
		case SDIAUDIO_CTL_SYNC_FREE_RUNNING:
			printf ("Synchronous, free running.\n");
			break;
		default:
			printf ("Sampling rate unknown.\n");
			break;
		}
	}
	return;
}

static void
get_carrier (int fd)
{
	int val;

	printf ("Getting the carrier status.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETCARRIER, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the carrier status");
	} else if (val) {
		printf ("Carrier detected.\n");
	} else {
		printf ("No carrier.\n");
	}
	return;
}

static void
get_status (int fd)
{
	int val;

	printf ("Getting the receiver data status.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETSTATUS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver data status");
	} else {
		printf ("Receiver is ");
		if (val) {
			printf ("passing data.\n");
		} else {
			printf ("blocking data.\n");
		}
	}
	return;
}

static void
get_audiogroup0_error (int fd)
{
	int val;

	printf ("Getting the audio group 0 error count.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETAUDIOGR0ERROR, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the audio group 0 error count");
	} else {
		printf ("Parity error count (Y channel) = %u \n", (val >> 24) & 0xff);
		printf ("Checksum error count (Y channel) = %u \n", (val >> 16) & 0xff);
		printf ("Parity error count (C channel) = %u \n", (val >> 8) & 0xff);
		printf ("Checksum error count (C channel) = %u \n", val & 0xff);
	}
	return;
}

static void
get_audiogroup0_delay_a (int fd)
{
	int val;

	printf ("Getting the audio group 0 delay A.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETAUDIOGR0DELAYA, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the audio group 0 delay A");
	} else {
		printf ("Delay A = %u \n", val);
	}
	return;
}

static void
get_audiogroup0_delay_b (int fd)
{
	int val;

	printf ("Getting the audio group 0 delay B.\n");
	if (ioctl (fd, SDIAUDIO_IOC_RXGETAUDIOGR0DELAYB, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the audio group 0 delay B");
	} else {
		printf ("Delay B = %u \n", val);
	}
	return;
}


int
main (int argc, char **argv)
{
	int opt, fd, choice = 0;
	unsigned int id, version, cap;
	struct util_info *info;
	char str[TMP_BUFLEN];

	argv0 = argv[0];
	while ((opt = getopt (argc, argv, "hV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv0);
			printf ("Interactively issue SMPTE 292M and SMPTE 259M-C \n"
				"receiver ioctls to DEVICE_FILE.\n\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -V\toutput version information "
				"and exit\n\n");
			printf ("Report bugs to <support@linsys.ca>.\n");
			return 0;
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
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) < 1) {
		fprintf (stderr, "%s: missing argument\n", argv0);
		goto USAGE;
	} else if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra argument\n", argv0);
		goto USAGE;
	}

	/* Open the file */
	if ((fd = open (argv[optind], O_RDONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to open file for reading");
		return -1;
	}

	/* Get the device ID */
	if (ioctl (fd, SDIAUDIO_IOC_GETID, &id) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device ID");
		close (fd);
		return -1;
	}

	/* Get the firmware version */
	if (ioctl (fd, SDIAUDIO_IOC_GETVERSION, &version) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device firmware version");
		close (fd);
		return -1;
	}

	if (((info = getinfo (id)) == NULL) ||
		!(info->flags & UTIL_SDIRX)) {
		fprintf (stderr, "%s: invalid device ID\n", argv0);
		close (fd);
		return -1;
	}

	/* Get the receiver capabilities */
	if (ioctl (fd, SDIAUDIO_IOC_RXGETCAP, &cap) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver capabilities");
		close (fd);
		return -1;
	}

	while (choice != 13) {
		printf ("\n\t%s, firmware version %u.%u (0x%04X)\n",
			info->name, version >> 8, version & 0x00ff, version);
		printf ("\ton %s\n\n", argv[optind]);
		printf ("\t 1. Show the receiver capabilities\n");
		printf ("\t 2. Get the receiver event flags\n");
		printf ("\t 3. Get the driver receive buffer "
			"queue length\n");
		if (cap & SDIAUDIO_CAP_RX_CD) {
			printf ("\t 4. Get the carrier status\n");
		} else {
			printf ("\t 4.\n");
		}
		if (cap & SDIAUDIO_CAP_RX_DATA) {
			printf ("\t 5. Get the receiver data status\n");
		} else {
			printf ("\t 5.\n");
		}
		if (cap & SDIAUDIO_CAP_RX_STATS) {
			printf ("\t 6. Get the audio group 0 error count\n");
			printf ("\t 7. Get the audio group 0 delay A\n");
			printf ("\t 8. Get the audio group 0 delay B\n");
		} else {
			printf ("\t 6.\n");
			printf ("\t 7.\n");
			printf ("\t 8.\n");
		}
		if (cap & SDIAUDIO_CAP_RX_NONAUDIO) {
			printf ("\t 9. Get the PCM/non-audio status\n");
		} else {
			printf ("\t 9.\n");
		}
		printf ("\t 10. Get the audio status\n");
		printf ("\t 11. Get the audio sampling rate\n");
		printf ("\t 12. Fsync\n");
		printf ("\t 13. Quit\n");
		printf ("\nEnter choice: ");
		fgets (str, TMP_BUFLEN, stdin);
		choice = strtol (str, NULL, 0);
		printf ("\n");
		switch (choice) {
		case 1:
			show_cap (cap);
			break;
		case 2:
			get_events (fd);
			break;
		case 3:
			get_buflevel (fd);
			break;
		case 4:
			if (cap & SDIAUDIO_CAP_RX_CD) {
				get_carrier (fd);
			}
			break;
		case 5:
			if (cap & SDIAUDIO_CAP_RX_DATA) {
				get_status (fd);
			}
			break;
		case 6:
			if (cap & SDIAUDIO_CAP_RX_STATS) {
				get_audiogroup0_error (fd);
			}
			break;
		case 7:
			if (cap & SDIAUDIO_CAP_RX_STATS) {
				get_audiogroup0_delay_a (fd);
			}
			break;
		case 8:
			if (cap & SDIAUDIO_CAP_RX_STATS) {
				get_audiogroup0_delay_b (fd);
			}
			break;
		case 9:
			if (cap & SDIAUDIO_CAP_RX_NONAUDIO) {
				get_nonaudio (fd);
			}
			break;
		case 10:
			get_audiostatus (fd);
			break;
		case 11:
			get_audiorate (fd);
			break;
		case 12:
			printf ("Fsyncing.\n");
			if (fsync (fd) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to fsync");
			} else {
				printf ("Fsync successful.\n");
			}
			break;
		default:
			break;
		}
	}
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv0);
	return -1;
}

