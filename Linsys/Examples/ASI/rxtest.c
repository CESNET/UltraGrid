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

#include "asi.h"
#include "master.h"
#include "../util.h"

#ifndef _LFS_LARGEFILE
#error Large file support not found
#endif

#define BUFLEN 256

static const char *argv0;
static const char progname[] = "rxtest";

static int
set_invsync (int fd,
	int inv_sync,
	unsigned int cap,
	int verbose,
	int quiet)
{
	switch (inv_sync) {
	case 0:
		if (verbose && !quiet) {
			printf ("Synchronizing on 0x47 packets.\n");
		}
		break;
	case 1:
		if (!(cap & ASI_CAP_RX_INVSYNC)) {
			if (!quiet) {
				fprintf (stderr,
					"%s: synchronization on 0xB8 packets "
					"not supported\n", argv0);
			}
			return -1;
		}
		if (verbose && !quiet) {
			printf ("Synchronizing on 0x47 or 0xB8 packets.\n");
		}
		break;
	default:
		if (!quiet) {
			fprintf (stderr, "%s: invalid 0xB8 "
				"packet synchronization mode\n", argv0);
		}
		return -1;
	}
	if (ioctl (fd, ASI_IOC_RXSETINVSYNC, &inv_sync) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to set the 0xB8 "
				"packet synchronization mode");
		}
		return -1;
	}
	return 0;
}

static int
set_dsync (int fd,
	int double_sync,
	unsigned int cap,
	int verbose,
	int quiet)
{
	switch (double_sync) {
	case 0:
		if (verbose && !quiet) {
			printf ("Disabling double packet synchronization.\n");
		}
		break;
	case 1:
		if (!(cap & ASI_CAP_RX_DSYNC)) {
			if (!quiet) {
				fprintf (stderr,
					"%s: double packet synchronization "
					"not supported\n", argv0);
			}
			return -1;
		}
		if (verbose && !quiet) {
			printf ("Enabling double packet synchronization.\n");
		}
		break;
	default:
		if (!quiet) {
			fprintf (stderr, "%s: invalid double "
				"packet synchronization mode\n", argv0);
		}
		return -1;
	}
	if (ioctl (fd, ASI_IOC_RXSETDSYNC, &double_sync) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to set the "
				"double packet synchronization mode");
		}
		return -1;
	}
	return 0;
}

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/asi/asirx%i/%s";
	int opt;
	int fd, val;
	int double_sync, inv_sync;
	int period, seconds, pid, quiet, verbose, num;
	unsigned long int bufsize, mode, timestamps;
	int ofd, read_ret, write_ret;
	struct stat buf;
	unsigned char *data;
	unsigned int cap, bytes_written, uval;
	struct timeval tv;
	struct pollfd pfd;
	double bytes, time_sec, lasttime, dt, status_time;
	char name[BUFLEN], str[BUFLEN], *endptr;

	argv0 = argv[0];

	/* Parse the command line */
	double_sync = -1;
	inv_sync = -1;
	period = 0;
	seconds = -1;
	pid = -1;
	quiet = 0;
	verbose = 0;
	while ((opt = getopt (argc, argv, "dDhiIn:p:qs:vV")) != -1) {
		switch (opt) {
		case 'd':
			double_sync = 1;
			break;
		case 'D':
			double_sync = 0;
			break;
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
		case 'i':
			inv_sync = 1;
			break;
		case 'I':
			inv_sync = 0;
			break;
		case 'n':
			seconds = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid timeout: %s\n",
					argv0, optarg);
				return -1;
			}
			break;
		case 'p':
			pid = strtol (optarg, &endptr, 0);
			if ((*endptr != '\0') || (pid < 0) || (pid > 0x1fff)) {
				fprintf (stderr, "%s: invalid PID: %s\n",
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
		if (!quiet) {
			fprintf (stderr, "%s: not an ASI device\n", argv0);
		}
		return -1;
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

	/* Get the receiver capabilities */
	if (ioctl (fd, ASI_IOC_RXGETCAP, &cap) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get the receiver capabilities");
		}
		close (fd);
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

	/* Get the receiver operating mode */
	if (cap & ASI_CAP_RX_SYNC) {
		snprintf (name, sizeof (name), fmt, num, "mode");
		if (util_strtoul (name, &mode) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to get "
					"the receiver operating mode");
			}
			close (fd);
			return -1;
		}
	} else {
		mode = ASI_CTL_RX_MODE_RAW;
	}
	if (verbose && !quiet) {
		switch (mode) {
		case ASI_CTL_RX_MODE_RAW:
			printf ("Receiving in raw mode.\n");
			break;
		case ASI_CTL_RX_MODE_188:
			printf ("Synchronizing on 188-byte packets.\n");
			break;
		case ASI_CTL_RX_MODE_204:
			printf ("Synchronizing on 204-byte packets.\n");
			break;
		case ASI_CTL_RX_MODE_AUTO:
			printf ("Synchronizing on detected packet size.\n");
			break;
		case ASI_CTL_RX_MODE_AUTOMAKE188:
			printf ("Synchronizing on "
				"detected packet size\n"
				"    and stripping the last "
				"sixteen bytes from "
				"each 204-byte packet.\n");
			break;
		case ASI_CTL_RX_MODE_204MAKE188:
			printf ("Synchronizing on "
				"204-byte packets\n"
				"    and stripping the last "
				"sixteen bytes from "
				"each packet.\n");
			break;
		default:
			printf ("Receiving in unknown mode.\n");
			break;
		}
	}

	/* Get the packet timestamping mode */
	if (cap & ASI_CAP_RX_TIMESTAMPS) {
		snprintf (name, sizeof (name), fmt, num, "timestamps");
		if (util_strtoul (name, &timestamps) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to get "
					"the packet timestamping mode");
			}
			close (fd);
			return -1;
		}
	} else {
		timestamps = ASI_CTL_TSTAMP_NONE;
	}
	if (verbose && !quiet) {
		switch (timestamps) {
		case ASI_CTL_TSTAMP_NONE:
			break;
		case ASI_CTL_TSTAMP_APPEND:
			printf ("Appending an eight-byte timestamp "
				"to each packet.\n");
			break;
		case ASI_CTL_TSTAMP_PREPEND:
			printf ("Prepending an eight-byte timestamp "
				"to each packet.\n");
			break;
		default:
			printf ("Unknown timestamping mode.\n");
			break;
		}
	}

	/* Enable/disable synchronization on 0xB8 */
	if (inv_sync >= 0) {
		if (set_invsync (fd, inv_sync, cap, verbose, quiet)) {
			close (fd);
			return -1;
		}
	} else if (verbose && !quiet) {
		printf ("Packet synchronization byte(s) not specified, "
			"using default.\n");
	}

	/* Enable/disable double packet synchronization */
	if (double_sync >= 0) {
		if (set_dsync (fd, double_sync, cap, verbose, quiet)) {
			close (fd);
			return -1;
		}
	} else if (verbose && !quiet) {
		printf ("Double packet synchronization mode not specified, "
			"using default.\n");
	}

	/* Set the PID counter */
	if (!quiet) {
		if (pid >= 0) {
			if (mode == ASI_CTL_RX_MODE_RAW) {
				fprintf (stderr, "%s: "
					"PID counter not supported "
					"in raw mode\n", argv0);
				close (fd);
				return -1;
			}
			if (cap & ASI_CAP_RX_PIDCOUNTER) {
				if (ioctl (fd, ASI_IOC_RXSETPID0, &pid) < 0) {
					fprintf (stderr, "%s: ", argv0);
					perror ("unable to set "
						"the PID counter");
					close (fd);
					return -1;
				} else if (verbose) {
					printf ("Counting PID %4Xh.\n", pid);
				}
			} else {
				fprintf (stderr,
					"%s: PID counter not supported\n",
					argv0);
				close (fd);
				return -1;
			}
		} else if (verbose) {
			printf ("Ignoring the PID counter.\n");
		}
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
		if ((pfd.revents & POLLPRI) && !quiet) {
			if (ioctl (fd, ASI_IOC_RXGETEVENTS, &val) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to get "
					"the receiver event flags");
				free (data);
				if (ofd >= 0) {
					close (ofd);
				}
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
			if (ioctl (fd, ASI_IOC_RXGETSTATUS, &val) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to get "
					"the receiver status");
				free (data);
				if (ofd >= 0) {
					close (ofd);
				}
				close (fd);
				return -1;
			}
			printf ("%8.0f bytes in %f seconds = "
				"%9.0f bps, status = %i.\n",
				bytes, status_time,
				8 * bytes / status_time, val);
			if (pid >= 0) {
				if (ioctl (fd, ASI_IOC_RXGETPID0COUNT, &uval)
					< 0) {
					fprintf (stderr, "%s: ", argv0);
					perror ("unable to get "
						"the PID count");
					free (data);
					if (ofd >= 0) {
						close (ofd);
					}
					close (fd);
					return -1;
				} else if (uval) {
					printf ("PID %4Xh count = %u.\n",
						pid, uval);
				}
			}
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
}

