/* audioplayout.c
 *
 * Audio playout example for Linear Systems Ltd. SMPTE 292M and SMPTE 259M-C boards.
 *
 * Copyright (C) 2009 Linear Systems Ltd. All rights reserved.
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
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <sys/time.h>

#include "sdiaudio.h"
#include "master.h"
#include "../util.h"

#define BUFLEN 256

static const char progname[] = "audioplayout";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/sdiaudio/sdiaudiotx%i/%s";
	int opt, fd, seconds;
	int num;
	struct stat buf;
	unsigned char *data;
	struct timeval tv;
	unsigned long int bufsize;
	unsigned int val;
	ssize_t read_ret, write_ret, bytes_read, bytes_written;
	struct pollfd pfd;
	double lasttime, time_sec;
	char name[BUFLEN], str[BUFLEN], *endptr;

	/* Parse the command line */
	seconds = -1;
	while ((opt = getopt (argc, argv, "hn:V")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Copy standard input to DEVICE_FILE\n"
				"and monitor for "
				"SMPTE 292M and SMPTE 259M-C transmitter events.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -n TIME\tstop transmitting "
				"after TIME seconds\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nIf TIME < 0, transmission never stops "
				"(default).\n");
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
			printf ("\nCopyright (C) 2009 "
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
		fprintf (stderr, "%s: not a character device\n",
			argv[0]);
		goto NO_STAT;
	}
	if (buf.st_rdev & 0x0080) {
		fprintf (stderr, "%s: not a transmitter\n", argv[0]);
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
		fprintf (stderr, "%s: not a SMPTE 292M/SMPTE 259M-C device\n", argv[0]);
		goto NO_STAT;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", argv[0], name);
		goto NO_STAT;
	}

	/* Open the file */
	if ((fd = open (argv[optind], O_WRONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to open file for writing");
		goto NO_STAT;
	}

	/* Get the buffer size */
	snprintf (name, sizeof (name), fmt, num, "bufsize");
	if (util_strtoul (name, &bufsize) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get "
			"the transmitter buffer size");
		goto NO_BUFS;
	}

	/* Allocate some memory */
	if (bufsize < BUFSIZ) {
		bufsize = BUFSIZ;
	}
	if ((data = (unsigned char *)malloc (bufsize)) == NULL) {
		fprintf (stderr, "%s: unable to allocate memory\n",
			argv[0]);
		goto NO_BUFS;
	}

	/* Copy stdin to fd */
	pfd.fd = fd;
	pfd.events = POLLOUT | POLLPRI;
	bytes_read = 0;
	while ((read_ret = read (STDIN_FILENO,
		data + bytes_read, bufsize - bytes_read))) {
		if (read_ret < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to read from input");
			goto NO_RUN;
		}
		bytes_read += read_ret;
	}
	if (gettimeofday (&tv, NULL) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get time");
		goto NO_RUN;
	}
	lasttime = tv.tv_sec + (double)tv.tv_usec / 1000000;
	while (seconds && bytes_read) {
		if (poll (&pfd, 1, -1) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to poll device file");
			goto NO_RUN;
		}
		if (pfd.revents & POLLOUT) {
			bytes_written = 0;
			while (bytes_written < bytes_read) {
				if ((write_ret = write (fd,
					data + bytes_written,
					bytes_read - bytes_written)) < 0) {
					fprintf (stderr, "%s: ", argv[0]);
					perror ("unable to write to "
						"device file");
					goto NO_RUN;
				}
				bytes_written += write_ret;
			}
			bytes_read = 0;
			while ((read_ret = read (STDIN_FILENO,
				data + bytes_read,
				bufsize - bytes_read)) != 0) {
				if (read_ret < 0) {
					fprintf (stderr, "%s: ", argv[0]);
					perror ("unable to read from input");
					goto NO_RUN;
				}
				bytes_read += read_ret;
			}
		}
		if (pfd.revents & POLLPRI) {
			if (ioctl (fd, SDIAUDIO_IOC_TXGETEVENTS, &val) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the transmitter event flags");
				goto NO_RUN;
			}
			if (val & SDIAUDIO_EVENT_TX_BUFFER) {
				fprinttime (stdout, progname);
				printf ("driver transmit buffer queue "
					"underrun detected\n");
			}
			if (val & SDIAUDIO_EVENT_TX_FIFO) {
				fprinttime (stdout, progname);
				printf ("onboard transmit FIFO "
					"underrun detected\n");
			}
			if (val & SDIAUDIO_EVENT_TX_DATA) {
				fprinttime (stdout, progname);
				printf ("transmit data status "
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
	fsync (fd);
	free (data);
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;

NO_RUN:
	free (data);
NO_BUFS:
	close (fd);
NO_STAT:
	return -1;
}

