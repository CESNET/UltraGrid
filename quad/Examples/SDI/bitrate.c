/* rxtest.c
 * 
 * Example program for DVB ASI receivers.
 *
 * Copyright (C) 2000-2005 Linear Systems Ltd. All rights reserved.
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
#include <sys/stat.h>
#include <sys/poll.h>
#include <sys/time.h>

#include "sdi.h"
#include "master.h"
#include "../util.h"

#ifndef _LFS_LARGEFILE
#error Large file support not found
#endif

#define BUFLEN 256

static const char *argv0;
static const char progname[] = "bitrate";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/sdi/sdirx%i/%s";
	int opt;
	int fd, val;
	int period, seconds, quiet, verbose, num;
	unsigned long int bufsize;
	int ofd, read_ret, write_ret;
	struct stat buf;
	unsigned char *data;
	unsigned int bytes_written;
	struct timeval tv;
	struct pollfd pfd;
	double bytes, time_sec, lasttime, dt, status_time;
	char name[BUFLEN], str[BUFLEN], *endptr;

	argv0 = argv[0];

	/* Parse the command line */
	period = 0;
	seconds = -1;
	quiet = 0;
	verbose = 0;
	while ((opt = getopt (argc, argv, "dDhiIn:p:qs:vV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE [FILE]\n",
				argv0);
			printf ("Copy DEVICE_FILE to FILE and monitor for "
				"DVB ASI receiver events.\n\n");
			printf ("  -d\t\tgain packet synchronization "
				"after two sychronization bytes\n");
			printf ("  -D\t\tgain packet synchronization "
				"after one synchronization byte\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -i\t\tsynchronize on packets starting with "
				"0x47 or 0xB8\n");
			printf ("  -I\t\tsynchronize on packets starting with "
				"0x47 only\n");
			printf ("  -n TIME\tstop receiving after "
				"TIME seconds\n");
			printf ("  -p PID\tcount packets "
				"with the given PID\n");
			printf ("  -q\t\tquiet operation\n");
			printf ("  -s PERIOD\tdisplay status "
				"every PERIOD seconds\n");
			printf ("  -v\t\tverbose output\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'n':
			seconds = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid timeout: %s\n",
					argv0, optarg);
				return -1;
			}
			break;
		case 'q':
			quiet = 1;
			break;
		case 's':
			period = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid period: %s\n",
					argv0, optarg);
				return -1;
			}
			break;
		case 'v':
			verbose = 1;
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2000-2005 "
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
		if (!quiet) {
			fprintf (stderr, "%s: missing arguments\n", argv0);
			goto USAGE;
		}
		return -1;
	} else if ((argc - optind) > 2) {
		if (!quiet) {
			fprintf (stderr, "%s: extra operand\n", argv0);
			goto USAGE;
		}
		return -1;
	}

	/* Get the sysfs info */
	memset (&buf, 0, sizeof (buf));
	if (stat (argv[optind], &buf) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get the file status");
		}
		return -1;
	}
	if (!S_ISCHR (buf.st_mode)) {
		if (!quiet) {
			fprintf (stderr, "%s: not a character device\n",
				argv0);
		}
		return -1;
	}
	if (!(buf.st_rdev & 0x0080)) {
		if (!quiet) {
			fprintf (stderr, "%s: not a receiver\n", argv0);
		}
		return -1;
	}
	num = buf.st_rdev & 0x007f;
	snprintf (name, sizeof (name), fmt, num, "dev");
	memset (str, 0, sizeof (str));
	if (util_read (name, str, sizeof (str)) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get the device number");
		}
		return -1;
	}
	if (strtoul (str, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not a SMPTE 259M-C device\n", argv[0]);
		goto NO_STAT;
	}
	if (*endptr != ':') {
		if (!quiet) {
			fprintf (stderr, "%s: error reading %s\n",
				argv0, name);
		}
		return -1;
	}

	/* Open the file. */
	if (verbose && !quiet) {
		printf ("Opening %s.\n", argv[optind]);
	}
	if ((fd = open (argv[optind], O_RDONLY, 0)) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to open file for reading");
		}
		return -1;
	}

	/* Get the buffer size */
	snprintf (name, sizeof (name), fmt, num, "bufsize");
	if (util_strtoul (name, &bufsize) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get "
				"the receiver buffer size");
		}
		close (fd);
		return -1;
	}

	/* Allocate some memory */
	if (verbose && !quiet) {
		printf ("Allocating %lu bytes of memory.\n", bufsize);
	}
	if ((data = (unsigned char *)malloc (bufsize)) == NULL) {
		if (!quiet) {
			fprintf (stderr, "%s: unable to allocate memory\n",
				argv0);
		}
		close (fd);
		return -1;
	}

	/* Open output file */
	ofd = -1;
	if ((argc - optind) == 2) {
		if (verbose && !quiet) {
			printf ("Opening %s.\n", argv[optind + 1]);
		}
		if ((ofd = open (argv[optind + 1],
			O_CREAT | O_WRONLY | O_TRUNC,
			0664)) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to open output file");
			}
			free (data);
			close (fd);
			return -1;
		}
	}

	/* Receive the data and estimate the throughput */
	bytes = 0.0;
	status_time = 0.0;
	pfd.fd = fd;
	pfd.events = POLLIN | POLLPRI;
	if (verbose && !quiet) {
		printf ("Listening for data...\n");
	}
	if (gettimeofday (&tv, NULL) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get time");
		}
		free (data);
		if (ofd >= 0) {
			close (ofd);
		}
		close (fd);
		return -1;
	}
	lasttime = tv.tv_sec + (double)tv.tv_usec / 1000000;
	while (seconds) {
		if (poll (&pfd, 1, 1000) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to poll device file");
			}
			free (data);
			if (ofd >= 0) {
				close (ofd);
			}
			close (fd);
			return -1;
		}
		if (pfd.revents & POLLIN) {
			if ((read_ret = read (fd, data, bufsize)) < 0) {
				if (!quiet) {
					fprintf (stderr, "%s: ", argv0);
					perror ("unable to read "
						"from device file");
				}
				free (data);
				if (ofd >= 0) {
					close (ofd);
				}
				close (fd);
				return -1;
			}
			bytes += read_ret;
			if (ofd >= 0) {
				bytes_written = 0;
				while ((read_ret - bytes_written) > 0) {
					if ((write_ret = write (ofd,
						data + bytes_written,
						read_ret - bytes_written)) < 0) {
						if (!quiet) {
							fprintf (stderr,
								"%s: ",
								argv0);
							perror ("unable to "
								"write to "
								"output "
								"file");
						}
						free (data);
						close (ofd);
						close (fd);
						return -1;
					}
					bytes_written += write_ret;
				}
			}
		}




		if (pfd.revents & POLLPRI) {
			if (ioctl (fd, SDI_IOC_RXGETEVENTS, &val) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get receiver event flags");
				goto NO_RUN;
			}
			if (val & SDI_EVENT_RX_BUFFER) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"driver receive buffer queue "
					"overrun detected\n");
			}
			if (val & SDI_EVENT_RX_FIFO) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"onboard receive FIFO "
					"overrun detected\n");
			}
			if (val & SDI_EVENT_RX_CARRIER) {
				fprinttime (stderr, progname);
				fprintf (stderr,
					"carrier status "
					"change detected\n");
			}
		}

		gettimeofday (&tv, NULL);
		time_sec = tv.tv_sec + (double)tv.tv_usec / 1000000;
		dt = time_sec - lasttime;
		if (dt >= 1) {
			status_time += dt;
			lasttime = time_sec;
			if (seconds > 0) {
				seconds--;
			}
		}
		if ((period > 0) && (!quiet) && (status_time >= period)) {
			printf ("%8.0f bytes in %f seconds = "
				"%9.0f bps, status = %i.\n",
				bytes, status_time,
				8 * bytes / status_time, val);
			bytes = 0.0;
			status_time = 0.0;
		}
	}
	free (data);
	if (ofd >= 0) {
		close (ofd);
	}
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv0);
	return -1;

NO_RUN:
	free (data);
NO_STAT:
	return -1;
}

