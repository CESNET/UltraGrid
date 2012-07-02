/* sdirxcfg.c
 *
 * Demonstrate SMPTE 292M receiver ioctls.
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

#include "sdivideo.h"
#include "master.h"
#include "../util.h"

#define TMP_BUFLEN 80

static const char *argv0;
static const char progname[] = "sdivideorxcfg";

static void
show_cap (unsigned int cap)
{
	char str[TMP_BUFLEN] = "[ ] ";

	str[1] = (cap & SDIVIDEO_CAP_RX_CD) ? 'X' : ' ';
	printf ("%sCarrier detect\n", str);

	str[1] = (cap & SDIVIDEO_CAP_RX_DATA) ? 'X' : ' ';
	printf ("%sReceive data status\n", str);

	str[1] = (cap & SDIVIDEO_CAP_RX_ERR_COUNT) ? 'X' : ' ';
	printf ("%sError count\n", str);

	str[1] = (cap & SDIVIDEO_CAP_RX_VBI) ? 'X' : ' ';
	printf ("%sDirect access to vertical blanking region\n", str);

	str[1] = (cap & SDIVIDEO_CAP_RX_RAWMODE) ? 'X' : ' ';
	printf ("%sRaw mode\n", str);

	str[1] = (cap & SDIVIDEO_CAP_RX_DEINTERLACING) ? 'X' : ' ';
	printf ("%sWeave deinterlacing\n", str);

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
	if (ioctl (fd, SDIVIDEO_IOC_RXGETEVENTS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver event flags");
	} else if (val) {
		if (val & SDIVIDEO_EVENT_RX_BUFFER) {
			printf ("Driver receive buffer queue "
				"overrun detected.\n");
		}
		if (val & SDIVIDEO_EVENT_RX_FIFO) {
			printf ("Onboard receive FIFO overrun detected.\n");
		}
		if (val & SDIVIDEO_EVENT_RX_CARRIER) {
			printf ("Carrier status change detected.\n");
		}
		if (val & SDIVIDEO_EVENT_RX_DATA) {
			printf ("Receive data change detected.\n");
		}
		if (val & SDIVIDEO_EVENT_RX_STD) {
			printf ("Receive format change detected.\n");
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
	if (ioctl (fd, SDIVIDEO_IOC_RXGETBUFLEVEL, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the driver receive buffer queue length");
	} else {
		printf ("Driver receive buffer queue length = %u.\n", val);
	}
	return;
}

static void
get_carrier (int fd)
{
	int val;

	printf ("Getting the carrier status.\n");
	if (ioctl (fd, SDIVIDEO_IOC_RXGETCARRIER, &val) < 0) {
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
	if (ioctl (fd, SDIVIDEO_IOC_RXGETSTATUS, &val) < 0) {
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
get_ycrc_errorcount (int fd)
{
	int val;

	printf ("Getting the Y channel CRC Error count.\n");
	if (ioctl (fd, SDIVIDEO_IOC_RXGETYCRCERROR, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the Y channel CRC Error count");
	} else {
		printf ("Y channel CRC Error count = %u.\n", val);
	}
	return;
}

static void
get_ccrc_errorcount (int fd)
{
	int val;

	printf ("Getting the C channel CRC Error count.\n");
	if (ioctl (fd, SDIVIDEO_IOC_RXGETCCRCERROR, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the C channel CRC Error count");
	} else {
		printf ("C channel CRC Error count = %u.\n", val);
	}
	return;
}

static void
get_video_standard (int fd)
{
	unsigned int val;

	printf ("Getting the receive video standard detected.\n");
	if (ioctl (fd, SDIVIDEO_IOC_RXGETVIDSTATUS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receive video standard detected");
	} else {
		switch (val) {
		case SDIVIDEO_CTL_UNLOCKED:
			printf ("No video standard locked.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ:
			printf ("SMPTE 125M 486i 59.94 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_BT_601_576I_50HZ:
			printf ("ITU-R BT.601 720x576i 50 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ:
			printf ("SMPTE 260M 1035i 60 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ:
			printf ("SMPTE 260M 1035i 59.94 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ:
			printf ("SMPTE 295M 1080i 50 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ:
		case SDIVIDEO_CTL_SMPTE_274M_1080PSF_30HZ:
			printf ("SMPTE 274M 1080i 60 Hz or 1080psf 30 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ:
		case SDIVIDEO_CTL_SMPTE_274M_1080PSF_29_97HZ:
			printf ("SMPTE 274M 1080i 59.94 Hz or 1080psf 29.97 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ:
		case SDIVIDEO_CTL_SMPTE_274M_1080PSF_25HZ:
			printf ("SMPTE 274M 1080i 50 Hz or 1080psf 25 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ:
			printf ("SMPTE 274M 1080psf 24 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ:
			printf ("SMPTE 274M 1080psf 23.98 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ:
			printf ("SMPTE 274M 1080p 30 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ:
			printf ("SMPTE 274M 1080p 29.97 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ:
			printf ("SMPTE 274M 1080p 25 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ:
			printf ("SMPTE 274M 1080p 24 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ:
			printf ("SMPTE 274M 1080p 23.98 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_60HZ:
			printf ("SMPTE 296M 720p 60 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ:
			printf ("SMPTE 296M 720p 59.94 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_50HZ:
			printf ("SMPTE 296M 720p 50 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_30HZ:
			printf ("SMPTE 296M 720p 30 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ:
			printf ("SMPTE 296M 720p 29.97 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_25HZ:
			printf ("SMPTE 296M 720p 25 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_24HZ:
			printf ("SMPTE 296M 720p 24 Hz detected.\n");
			break;
		case SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ:
			printf ("SMPTE 296M 720p 23.98 Hz detected.\n");
			break;
		default:
			printf ("Unknown video standard detected.\n");
			break;
		}
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
			printf ("Interactively issue SMPTE 292M "
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
	if (ioctl (fd, SDIVIDEO_IOC_GETID, &id) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device ID");
		close (fd);
		return -1;
	}

	/* Get the firmware version */
	if (ioctl (fd, SDIVIDEO_IOC_GETVERSION, &version) < 0) {
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
	if (ioctl (fd, SDIVIDEO_IOC_RXGETCAP, &cap) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver capabilities");
		close (fd);
		return -1;
	}

	while (choice != 10) {
		printf ("\n\t%s, firmware version %u.%u (0x%04X)\n",
			info->name, version >> 8, version & 0x00ff, version);
		printf ("\ton %s\n\n", argv[optind]);
		printf ("\t 1. Show the receiver capabilities\n");
		printf ("\t 2. Get the receiver event flags\n");
		printf ("\t 3. Get the driver receive buffer "
			"queue length\n");
		if (cap & SDIVIDEO_CAP_RX_CD) {
			printf ("\t 4. Get the carrier status\n");
		} else {
			printf ("\t 4.\n");
		}
		if (cap & SDIVIDEO_CAP_RX_DATA) {
			printf ("\t 5. Get the receiver data status\n");
		} else {
			printf ("\t 5.\n");
		}
		printf ("\t 6. Get the receive video standard detected\n");
		if (cap & SDIVIDEO_CAP_RX_ERR_COUNT) {
			printf ("\t 7. Get the Y channel CRC error count\n");
			printf ("\t 8. Get the C channel CRC error count\n");
		} else {
			printf ("\t 7.\n");
			printf ("\t 8.\n");
		}
		printf ("\t 9. Fsync\n");
		printf ("\t 10. Quit\n");
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
			if (cap & SDIVIDEO_CAP_RX_CD) {
				get_carrier (fd);
			}
			break;
		case 5:
			if (cap & SDIVIDEO_CAP_RX_DATA) {
				get_status (fd);
			}
			break;
		case 6:
			get_video_standard (fd);
			break;
		case 7:
			if (cap & SDIVIDEO_CAP_RX_ERR_COUNT) {
				get_ycrc_errorcount (fd);
			}
			break;
		case 8:
			if (cap & SDIVIDEO_CAP_RX_ERR_COUNT) {
				get_ccrc_errorcount (fd);
			}
			break;
		case 9:
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

