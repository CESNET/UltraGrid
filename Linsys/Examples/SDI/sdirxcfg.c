/* sdirxcfg.c
 * 
 * Demonstrate SMPTE 259M-C receiver ioctls.
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
static const char progname[] = "sdirxcfg";

static void
show_cap (unsigned int cap)
{
	if (cap) {
		printf ("Capabilities = 0x%08X\n", cap);
	} else {
		printf ("No capabilities flags set.\n");
	}
	return;
}

static void
get_events (int fd)
{
	unsigned int val;
	char str[TMP_BUFLEN];

	printf ("Getting the receiver event flags.\n");
	if (ioctl (fd, SDI_IOC_RXGETEVENTS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver event flags");
	} else if (val) {
		if (val & SDI_EVENT_RX_BUFFER) {
			printf ("Driver receive buffer queue "
				"overrun detected.\n");
		}
		if (val & SDI_EVENT_RX_FIFO) {
			printf ("Onboard receive FIFO overrun detected.\n");
		}
		if (val & SDI_EVENT_RX_CARRIER) {
			printf ("Carrier status change detected.\n");
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
	if (ioctl (fd, SDI_IOC_RXGETBUFLEVEL, &val) < 0) {
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
	if (ioctl (fd, SDI_IOC_RXGETCARRIER, &val) < 0) {
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

	printf ("Getting the receiver status.\n");
	if (ioctl (fd, SDI_IOC_RXGETSTATUS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver status");
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
	if ((fd = open (argv[optind], O_RDONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to open file for reading");
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
		!(info->flags & UTIL_SDIRX)) {
		fprintf (stderr, "%s: invalid device ID\n", argv0);
		close (fd);
		return -1;
	}

	/* Get the receiver capabilities */
	if (ioctl (fd, SDI_IOC_RXGETCAP, &cap) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver capabilities");
		close (fd);
		return -1;
	}

	while (choice != 7) {
		printf ("\n\t%s, firmware version %u.%u (0x%04X)\n",
			info->name, version >> 8, version & 0x00ff, version);
		printf ("\ton %s\n\n", argv[optind]);
		printf ("\t 1. Show the receiver capabilities\n");
		printf ("\t 2. Get the receiver event flags\n");
		printf ("\t 3. Get the driver receive buffer "
			"queue length\n");
		printf ("\t 4. Get the carrier status\n");
		printf ("\t 5. Get the receiver status\n");
		printf ("\t 6. Fsync\n");
		printf ("\t 7. Quit\n");
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
			get_carrier (fd);
			break;
		case 5:
			get_status (fd);
			break;
		case 6:
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

