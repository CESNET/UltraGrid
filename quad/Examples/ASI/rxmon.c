/* rxmon.c
 *
 * Monitor a DVB ASI receiver for events.
 *
 * Copyright (C) 2001-2004 Linear Systems Ltd. All rights reserved.
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
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/poll.h>

#include "asi.h"
#include "master.h"
#include "../util.h"

static const char progname[] = "rxmon";

int
main (int argc, char **argv)
{
	int opt, fd, val;
	struct pollfd pfd;

	while ((opt = getopt (argc, argv, "hV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Monitor DEVICE_FILE for "
				"DVB ASI receiver events.\n\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -V\toutput version information "
				"and exit\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2001-2004 "
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
		fprintf (stderr, "%s: missing arguments\n", argv[0]);
		goto USAGE;
	} else if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Open the file */
	if ((fd = open (argv[optind], O_RDONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to open file for reading");
		return -1;
	}

	/* Monitor the interface */
	pfd.fd = fd;
	pfd.events = POLLPRI;
	for (;;) {
		if (poll (&pfd, 1, -1) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to poll device file");
			close (fd);
			return -1;
		}
		if (pfd.revents & POLLPRI) {
			if (ioctl (fd, ASI_IOC_RXGETEVENTS, &val) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the receiver event flags");
				close (fd);
				return -1;
			}
			if (val & ASI_EVENT_RX_BUFFER) {
				fprinttime (stdout, progname);
				printf ("driver receive buffer queue "
					"overrun detected\n");
			}
			if (val & ASI_EVENT_RX_FIFO) {
				fprinttime (stdout, progname);
				printf ("onboard receive FIFO overrun "
					"detected\n");
			}
			if (val & ASI_EVENT_RX_CARRIER) {
				fprinttime (stdout, progname);
				printf ("carrier status "
					"change detected\n");
			}
			if (val & ASI_EVENT_RX_LOS) {
				fprinttime (stdout, progname);
				printf ("loss of packet "
					"synchronization detected\n");
			}
			if (val & ASI_EVENT_RX_AOS) {
				fprinttime (stdout, progname);
				printf ("acquisition of packet "
					"synchronization detected\n");
			}
			if (val & ASI_EVENT_RX_DATA) {
				fprinttime (stdout, progname);
				printf ("receive data status "
					"change detected\n");
			}
		}
	}
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

