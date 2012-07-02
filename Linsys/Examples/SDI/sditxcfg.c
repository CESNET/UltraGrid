/* sditxcfg.c
 * 
 * Demonstrate SMPTE 259M-C transmitter ioctls.
 *
 * Copyright (C) 2004-2006 Linear Systems Ltd. All rights reserved.
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

#include "sdi.h"
#include "master.h"
#include "../util.h"

#define TMP_BUFLEN 80

static const char *argv0;
static const char progname[] = "sditxcfg";

static void
show_cap (unsigned int cap)
{
	char str[TMP_BUFLEN] = "[ ] ";

	str[1] = (cap & SDI_CAP_TX_RXCLKSRC) ? 'X' : ' ';
	printf ("%sRecovered receive clock\n", str);
	return;
}

static void
get_events (int fd)
{
	unsigned int val;

	printf ("Getting the transmitter event flags.\n");
	if (ioctl (fd, SDI_IOC_TXGETEVENTS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transmitter event flags");
	} else if (val) {
		if (val & SDI_EVENT_TX_BUFFER) {
			printf ("Driver transmit buffer queue underrun "
				"detected.\n");
		}
		if (val & SDI_EVENT_TX_FIFO) {
			printf ("Onboard transmit FIFO underrun detected.\n");
		}
		if (val & SDI_EVENT_TX_DATA) {
			printf ("Transmit data change detected.\n");
		}
	} else {
		printf ("No transmitter events detected.\n");
	}
	return;
}

static void
get_buflevel (int fd)
{
	unsigned int val;

	printf ("Getting the driver transmit buffer queue length.\n");
	if (ioctl (fd, SDI_IOC_TXGETBUFLEVEL, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the driver transmit buffer queue length");
	} else {
		printf ("Driver transmit buffer queue length = %u.\n", val);
	}
	return;
}

static void
get_data (int fd)
{
	int val;

	printf ("Getting the transmit data status.\n");
	if (ioctl (fd, SDI_IOC_TXGETTXD, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transmit data status");
	} else if (val) {
		printf ("Data is being transmitted.\n");
	} else {
		printf ("Data is not being transmitted.\n");
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
			printf ("Interactively issue SMPTE 259M-C "
				"transmitter ioctls to DEVICE_FILE.\n\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -V\toutput version information "
				"and exit\n\n");
			printf ("Report bugs to <support@linsys.ca>.\n");
			return 0;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2004-2006 "
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
	if ((fd = open (argv[optind], O_WRONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to open file for writing");
		return -1;
	}

	/* Get the device ID */
	if (ioctl (fd, SDI_IOC_GETID, &id) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device ID");
		close (fd);
		return -1;
	}

	/* Get the firmware version */
	if (ioctl (fd, SDI_IOC_GETVERSION, &version) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device firmware version");
		close (fd);
		return -1;
	}

	if (((info = getinfo (id)) == NULL) ||
		!(info->flags & UTIL_SDITX)) {
		fprintf (stderr, "%s: invalid device ID\n", argv0);
		close (fd);
		return -1;
	}

	/* Get the transmitter capabilities */
	if (ioctl (fd, SDI_IOC_TXGETCAP, &cap) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transmitter capabilities");
		close (fd);
		return -1;
	}

	while (choice != 6) {
		printf ("\n\t%s, firmware version %u.%u (0x%04X)\n",
			info->name, version >> 8, version & 0x00ff, version);
		printf ("\ton %s\n\n", argv[optind]);
		printf ("\t 1. Show the transmitter capabilities\n");
		printf ("\t 2. Get the event flags\n");
		printf ("\t 3. Get the driver transmit buffer "
			"queue length\n");
		printf ("\t 4. Get the transmit data status\n");
		printf ("\t 5. Fsync\n");
		printf ("\t 6. Quit\n");
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
			get_data (fd);
			break;
		case 5:
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

