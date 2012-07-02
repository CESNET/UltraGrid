/* mvideocapture.c
 *
 * Mmap capture example for Linear Systems Ltd. SMPTE 292M boards.
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
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#include "sdivideo.h"
#include "master.h"
#include "../util.h"

#define BUFLEN 256

static const char progname[] = "mvideocapture";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/sdivideo/sdivideorx%i/%s";
	const int pagesize = getpagesize ();
	int opt, fd, num, seconds;
	struct stat buf;
	char name[BUFLEN], str[BUFLEN], *endptr;
	unsigned long int buffers, bufsize;
	unsigned int bufmemsize, i, j, bufnum, bytes, val;
	ssize_t ret;
	char **address;
	struct pollfd pfd;
	struct timeval tv;
	double lasttime, time_sec;

	/* Parse the command line */
	seconds = -1;
	while ((opt = getopt (argc, argv, "hn:V")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Capture from DEVICE_FILE and monitor for "
				"SMPTE 292M receiver events.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -n TIME\tstop receiving after "
				"TIME seconds\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'n':
			seconds = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid timeout: %s\n",
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
		goto NO_STAT;
	}
	if (!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", argv[0]);
		goto NO_STAT;
	}
	if (!(buf.st_rdev & 0x0080)) {
		fprintf (stderr, "%s: not a receiver\n", argv[0]);
		goto NO_STAT;
	}
	num = buf.st_rdev & 0x007f;
	snprintf (name, sizeof (name), fmt, num, "dev");
	memset (str, 0, sizeof (str));
	if (util_read (name, str, sizeof (str)) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get the device number");
		goto NO_STAT;
	}
	if (strtoul (str, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not a SMPTE 292M device\n", argv[0]);
		goto NO_STAT;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n",
			argv[0], name);
		goto NO_STAT;
	}

	/* Open the file */
	if ((fd = open (argv[optind], O_RDONLY)) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to open file for reading");
		goto NO_STAT;
	}

	/* Get the number of buffers */
	snprintf (name, sizeof (name), fmt, num, "buffers");
	if (util_strtoul (name, &buffers) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get the number of buffers");
		goto NO_BUFS;
	}

	/* Get the buffer size */
	snprintf (name, sizeof (name), fmt, num, "bufsize");
	if (util_strtoul (name, &bufsize) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get the receiver buffer size");
		goto NO_BUFS;
	}

	/* Mmap */
	bufmemsize = bufsize +
		((bufsize % pagesize) ? (pagesize - bufsize % pagesize) : 0);
	if ((address = (char **)malloc (buffers * sizeof (char *))) == NULL) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to allocate memory");
		goto NO_BUFS;
	}
	for (i = 0; i < buffers; i++) {
		if ((address[i] = (char *)mmap (NULL,
			bufsize,
			PROT_READ,
			MAP_SHARED,
			fd,
			i * bufmemsize)) == (char *)MAP_FAILED) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to map memory");
			for (j = 0; j < i; j++) {
				munmap (address[j], bufsize);
			}
			free (address);
			goto NO_BUFS;
		}
	}

	/* Wait for data */
	pfd.fd = fd;
	pfd.events = POLLIN | POLLPRI;
	if (gettimeofday (&tv, NULL) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get time");
		goto NO_RUN;
	}
	lasttime = tv.tv_sec + (double)tv.tv_usec / 1000000;
	bufnum = 0;
	while (seconds) {
		if (poll (&pfd, 1, 1000) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to poll device file");
			goto NO_RUN;
		}
		if (pfd.revents & POLLIN) {
			if (ioctl (fd, SDIVIDEO_IOC_DQBUF, bufnum) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to dequeue buffer");
				goto NO_RUN;
			}
			bytes = 0;
			while (bytes < bufsize) {
				if ((ret = write (STDOUT_FILENO,
					address[bufnum] + bytes,
					bufsize - bytes)) < 0) {
					fprintf (stderr, "%s: ", argv[0]);
					perror ("unable to write to output");
					goto NO_RUN;
				}
				bytes += ret;
			}
			if (ioctl (fd, SDIVIDEO_IOC_QBUF, bufnum) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to queue buffer");
				goto NO_RUN;
			}
			bufnum = (bufnum + 1) % buffers;
		}
		if (pfd.revents & POLLPRI) {
			if (ioctl (fd, SDIVIDEO_IOC_RXGETEVENTS, &val) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get receiver event flags");
				close (fd);
				return -1;
			}
			if (val & SDIVIDEO_EVENT_RX_BUFFER) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"driver receive buffer queue "
					"overrun detected\n");
			}
			if (val & SDIVIDEO_EVENT_RX_FIFO) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"onboard receive FIFO "
					"overrun detected\n");
			}
			if (val & SDIVIDEO_EVENT_RX_CARRIER) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"carrier status "
					"change detected\n");
			}
			if (val & SDIVIDEO_EVENT_RX_STD) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"format "
					"change detected\n");
			}
		}
		if (gettimeofday (&tv, NULL) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get time");
			goto NO_RUN;
		}
		time_sec = tv.tv_sec + (double)tv.tv_usec / 1000000;
		if ((time_sec - lasttime) >= 1) {
			lasttime = time_sec;
			if (seconds > 0) {
				seconds--;
			}
		}
	}

	for (i = 0; i < buffers; i++) {
		munmap (address[i], buffers);
	}
	free (address);
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;

NO_RUN:
	for (i = 0; i < buffers; i++) {
		munmap (address[i], buffers);
	}
	free (address);
NO_BUFS:
	close (fd);
NO_STAT:
	return -1;
}

