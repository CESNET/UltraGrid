/* txtest.c
 *
 * Example program for DVB ASI transmitters.
 *
 * Copyright (C) 2000-2010 Linear Systems Ltd. All rights reserved.
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

#define BUFLEN 256

static const char progname[] = "txtest";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/asi/asitx%i/%s";
	int opt;
	int fd, period, quiet, seconds, verbose, packetsize, num;
	unsigned long int bufsize, mode, clksrc, timestamps, transport;
	int read_ret, write_ret, val;
	unsigned int bytes_per_sec;
	struct stat buf;
	unsigned char *data;
	struct asi_txstuffing stuffing;
	unsigned long int bypass_status;
	struct timeval tv;
	unsigned int cap, bytes_written, bytes_read, bytes;
	struct pollfd pfd;
	double status_bytes, bitrate, time_sec, lasttime, dt;
	char name[BUFLEN], str[BUFLEN], *endptr;

	/* Parse the command line */
	period = 0;
	quiet = 0;
	seconds = -1;
	verbose = 0;
	while ((opt = getopt (argc, argv, "hn:qs:vV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE [IB IP\n"
				"\t[NORMAL_IP BIG_IP [IL_NORMAL IL_BIG]]]\n",
				argv[0]);
			printf ("Copy standard input to DEVICE_FILE "
				"with interbyte stuffing IB,\n"
				"interpacket stuffing "
				"IP + (BIG_IP / (BIG_IP + NORMAL_IP)),\n"
				"and interleaved finetuning parameters "
				"IL_NORMAL and IL_BIG\n"
				"while monitoring for "
				"DVB ASI transmitter events.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -n TIME\tstop transmitting "
				"after TIME seconds\n");
			printf ("  -q\t\tquiet operation\n");
			printf ("  -s PERIOD\tdisplay status "
				"every PERIOD seconds\n");
			printf ("  -v\t\tverbose output\n");
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
		case 'q':
			quiet = 1;
			break;
		case 's':
			period = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid period: %s\n",
					argv[0], optarg);
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
			printf ("\nCopyright (C) 2000-2010 "
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
	if (((argc - optind) == 0) ||
		((argc - optind) == 2) ||
		((argc - optind) == 4) ||
		((argc - optind) == 6)) {
		if (!quiet) {
			fprintf (stderr, "%s: missing arguments\n", argv[0]);
			goto USAGE;
		}
		return -1;
	} else if ((argc - optind) > 7) {
		if (!quiet) {
			fprintf (stderr, "%s: extra operand\n", argv[0]);
			goto USAGE;
		}
		return -1;
	}

	/* Read the stuffing parameters */
	memset (&stuffing, 0, sizeof (stuffing));
	if ((argc - optind) > 2) {
		stuffing.ib = strtol (argv[optind + 1], &endptr, 0);
		if (*endptr != '\0') {
			fprintf (stderr,
				"%s: invalid interbyte stuffing: %s\n",
				argv[0], argv[optind + 1]);
			return -1;
		}
		stuffing.ip = strtol (argv[optind + 2], &endptr, 0);
		if (*endptr != '\0') {
			fprintf (stderr,
				"%s: invalid interpacket stuffing: %s\n",
				argv[0], argv[optind + 2]);
			return -1;
		}
	}
	if ((argc - optind) > 4) {
		stuffing.normal_ip = strtol (argv[optind + 3], &endptr, 0);
		if (*endptr != '\0') {
			fprintf (stderr,
				"%s: invalid finetuning parameter: %s\n",
				argv[0], argv[optind + 3]);
			return -1;
		}
		stuffing.big_ip = strtol (argv [optind + 4], &endptr, 0);
		if (*endptr != '\0') {
			fprintf (stderr,
				"%s: invalid finetuning parameter: %s\n",
				argv[0], argv[optind + 4]);
			return -1;
		}
		if (stuffing.normal_ip == 0) {
			stuffing.big_ip = 0;
		}
		if ((argc - optind) > 6) {
			stuffing.il_normal = strtol (argv[optind + 5],
				&endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid interleaving parameter: "
					"%s\n",
					argv[0], argv[optind + 5]);
				return -1;
			}
			stuffing.il_big = strtol (argv [optind + 6],
				&endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid interleaving parameter: "
					"%s\n",
					argv[0], argv[optind + 6]);
				return -1;
			}
		}
	}

	/* Get the sysfs info */
	memset (&buf, 0, sizeof (buf));
	if (stat (argv[optind], &buf) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the file status");
		}
		return -1;
	}
	if (!S_ISCHR (buf.st_mode)) {
		if (!quiet) {
			fprintf (stderr, "%s: not a character device\n",
				argv[0]);
		}
		return -1;
	}
	if (buf.st_rdev & 0x0080) {
		if (!quiet) {
			fprintf (stderr, "%s: not a transmitter\n", argv[0]);
		}
		return -1;
	}
	num = buf.st_rdev & 0x007f;
	snprintf (name, sizeof (name), fmt, num, "dev");
	memset (str, 0, sizeof (str));
	if (util_read (name, str, sizeof (str)) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the device number");
		}
		return -1;
	}
	if (strtoul (str, &endptr, 0) != (buf.st_rdev >> 8)) {
		if (!quiet) {
			fprintf (stderr, "%s: not an ASI device\n", argv[0]);
		}
		return -1;
	}
	if (*endptr != ':') {
		if (!quiet) {
			fprintf (stderr, "%s: error reading %s\n",
				argv[0], name);
		}
		return -1;
	}

	/* Open the file */
	if (verbose && !quiet) {
		printf ("Opening %s.\n", argv[optind]);
	}
	if ((fd = open (argv[optind], O_WRONLY, 0)) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to open file for writing");
		}
		return -1;
	}

	/* Get the transport type */
	snprintf (name, sizeof (name), fmt, num, "transport");
	if (util_strtoul (name, &transport) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get "
				"the transmitter transport type");
		}
		close (fd);
		return -1;
	}

	/* Get the transmitter capabilities */
	if (ioctl (fd, ASI_IOC_TXGETCAP, &cap) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the transmitter capabilities");
		}
		close (fd);
		return -1;
	}

	/* Get the buffer size */
	snprintf (name, sizeof (name), fmt, num, "bufsize");
	if (util_strtoul (name, &bufsize) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get "
				"the transmitter buffer size");
		}
		close (fd);
		return -1;
	}

	/* Get the output packet size */
	snprintf (name, sizeof (name), fmt, num, "mode");
	if (util_strtoul (name, &mode) < 0) {
		if (!quiet) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get "
				"the transmitter operating mode");
		}
		close (fd);
		return -1;
	}
	switch (mode) {
	case ASI_CTL_TX_MODE_188:
		if (verbose && !quiet) {
			printf ("Assuming 188-byte packets.\n");
		}
		packetsize = 188;
		break;
	case ASI_CTL_TX_MODE_204:
		if (verbose && !quiet) {
			printf ("Assuming 204-byte packets.\n");
		}
		packetsize = 204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		if (verbose && !quiet) {
			printf ("Appending sixteen 0x00 bytes to each "
				"188-byte packet.\n");
		}
		packetsize = 204;
		break;
	default:
		if (!quiet) {
			fprintf (stderr, "%s: "
				"unknown transmitter operating mode\n",
				argv[0]);
		}
		close (fd);
		return -1;
	}

	/* Get the clock source */
	if (cap & ASI_CAP_TX_SETCLKSRC) {
		snprintf (name, sizeof (name), fmt, num, "clock_source");
		if (util_strtoul (name, &clksrc) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the clock source");
			}
			close (fd);
			return -1;
		}
	} else {
		clksrc = 0;
	}
	if (verbose && !quiet) {
		switch (clksrc) {
		case ASI_CTL_TX_CLKSRC_ONBOARD:
			printf ("Using onboard oscillator.\n");
			break;
		case ASI_CTL_TX_CLKSRC_EXT:
			printf ("Using external reference clock.\n");
			break;
		case ASI_CTL_TX_CLKSRC_RX:
			printf ("Using recovered receive clock.\n");
			break;
		default:
			printf ("Unknown clock source.\n");
			break;
		}
	}

	/* Get the packet timestamping mode */
	if (cap & ASI_CAP_TX_TIMESTAMPS) {
		snprintf (name, sizeof (name), fmt, num, "timestamps");
		if (util_strtoul (name, &timestamps) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv[0]);
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
			printf ("Stripping eight bytes "
				"from the end of each packet.\n");
			break;
		case ASI_CTL_TSTAMP_PREPEND:
			printf ("Releasing packets according to "
				"prepended timestamps.\n");
			break;
		default:
			printf ("Unknown timestamping mode.\n");
			break;
		}
	}

	switch (transport) {
	default:
	case ASI_CTL_TRANSPORT_DVB_ASI:
		/* Set the stuffing parameters */
		if (verbose && !quiet) {
			printf ("Setting %i K28.5 character(s) between bytes,\n",
				stuffing.ib);
			printf ("    and %i + 2 K28.5 characters between packets.\n",
				stuffing.ip);
			if (cap & ASI_CAP_TX_FINETUNING) {
				printf ("Adding a K28.5 character to "
					"%i / %i packets.\n",
					stuffing.big_ip,
					stuffing.normal_ip + stuffing.big_ip);
				if (cap & ASI_CAP_TX_INTERLEAVING) {
					if (stuffing.il_normal && stuffing.il_big) {
						int normal_mult, big_mult, mult;

						normal_mult = stuffing.normal_ip /
							stuffing.il_normal;
						big_mult = stuffing.big_ip /
							stuffing.il_big;
						mult = (normal_mult > big_mult) ?
							big_mult : normal_mult;
						printf ("Interleaving %i x %i = %i "
							"K28.5 characters "
							"over %i x %i = %i packets.\n",
							mult, stuffing.il_big,
							mult * stuffing.il_big,
							mult, stuffing.il_normal +
							stuffing.il_big,
							mult * (stuffing.il_normal +
							stuffing.il_big));
					}
				} else {
					printf ("Interleaved bitrate finetuning "
						"not supported.\n");
				}
			} else {
				printf ("Bitrate finetuning not supported.\n");
			}
		}
		if (ioctl (fd, ASI_IOC_TXSETSTUFFING, &stuffing) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set the stuffing parameters");
			}
			close (fd);
			return -1;
		}
		/* Calculate the target interface bitrate */
		if ((cap & ASI_CAP_TX_FINETUNING) &&
			(stuffing.normal_ip != 0) && (stuffing.big_ip != 0)) {
			bitrate = 270000000 * 0.8 * packetsize /
				(packetsize + (packetsize - 1) * stuffing.ib +
				 stuffing.ip + (double)stuffing.big_ip /
				 (stuffing.normal_ip + stuffing.big_ip) + 2);
		} else {
			bitrate = 270000000 * 0.8 * (double)packetsize /
				(packetsize + (packetsize - 1) * stuffing.ib +
				 stuffing.ip + 2);
		}
		break;
	case ASI_CTL_TRANSPORT_SMPTE_310M:
		bitrate = 19392658.46;
		break;
	}
	if (verbose && !quiet) {
		printf ("Target interface bitrate = %.0f bps.\n",
			bitrate);
	}
	bytes_per_sec = bitrate / 8;

	/* Check the bypass status */
	snprintf (name, sizeof (name),
		fmt, num, "device/bypass_status");
	if (util_strtoul (name, &bypass_status) > 0) {
		/* Don't complain on an error,
		 * since this parameter may not exist. */
		if (!bypass_status && !quiet) {
			printf ("WARNING: Bypass enabled, "
				"data will not be transmitted.\n");
		}
	}

	/* Allocate some memory */
	if (bufsize < BUFSIZ) {
		bufsize = BUFSIZ;
	}
	if (verbose && !quiet) {
		printf ("Allocating %lu bytes of memory.\n", bufsize);
	}
	if ((data = (unsigned char *)malloc (bufsize)) == NULL) {
		if (!quiet) {
			fprintf (stderr, "%s: unable to allocate memory\n",
				argv[0]);
		}
		close (fd);
		return -1;
	}

	/* Repeatedly send the data and estimate the throughput */
	lasttime = 0;
	if (!quiet) {
		if (verbose) {
			printf ("Transmitting from standard input...\n");
		}
		if (gettimeofday (&tv, NULL) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get time");
			free (data);
			close (fd);
			return -1;
		}
		lasttime = tv.tv_sec + (double)tv.tv_usec / 1000000;
	}
	bytes = 0;
	status_bytes = 0.0;
	pfd.fd = fd;
	pfd.events = POLLOUT | POLLPRI;
	bytes_read = 0;
	while ((bytes_read < bufsize) &&
		(read_ret = read (STDIN_FILENO,
		data + bytes_read, bufsize - bytes_read))) {
		if (read_ret < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to read from input");
			}
			free (data);
			close (fd);
			return -1;
		}
		bytes_read += read_ret;
	}
	while (seconds && bytes_read) {
		if (poll (&pfd, 1, -1) < 0) {
			if (!quiet) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to poll device file");
			}
			free (data);
			close (fd);
			return -1;
		}
		if (pfd.revents & POLLOUT) {
			bytes_written = 0;
			while (bytes_written < bytes_read) {
				if ((write_ret =
					write (fd, data + bytes_written,
					bytes_read - bytes_written)) < 0) {
					if (!quiet) {
						fprintf (stderr, "%s: ",
							argv[0]);
						perror ("unable to write to "
							"device file");
					}
					free (data);
					close (fd);
					return -1;
				}
				bytes_written += write_ret;
			}
			bytes += bytes_written;
			bytes_read = 0;
			while ((bytes_read < bufsize) &&
				(read_ret = read (STDIN_FILENO,
				data + bytes_read, bufsize - bytes_read))) {
				if (read_ret < 0) {
					if (!quiet) {
						fprintf (stderr, "%s: ",
							argv[0]);
						perror ("unable to read "
							"from input");
					}
					free (data);
					close (fd);
					return -1;
				}
				bytes_read += read_ret;
			}
		}
		if ((pfd.revents & POLLPRI) && !quiet) {
			if (ioctl (fd, ASI_IOC_TXGETEVENTS, &val) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the transmitter event flags");
				free (data);
				close (fd);
				return -1;
			}
			if (val & ASI_EVENT_TX_BUFFER) {
				fprinttime (stdout, progname);
				printf ("driver transmit buffer queue "
					"underrun detected\n");
			}
			if (val & ASI_EVENT_TX_FIFO) {
				fprinttime (stdout, progname);
				printf ("onboard transmit FIFO "
					"underrun detected\n");
			}
			if (val & ASI_EVENT_TX_DATA) {
				fprinttime (stdout, progname);
				printf ("transmit data status "
					"change detected\n");
			}
		}
		if (bytes >= bytes_per_sec) {
			status_bytes += bytes;
			bytes = 0;
			if (seconds > 0) {
				seconds--;
			}
		}
		if ((period > 0) && (!quiet) &&
			(status_bytes >= (bytes_per_sec * period))) {
			gettimeofday (&tv, NULL);
			time_sec = tv.tv_sec + (double)tv.tv_usec / 1000000;
			dt = time_sec - lasttime;
			printf ("%8.0f bytes in %f seconds = "
				"%9.0f bps.\n",
				status_bytes, dt, 8 * status_bytes / dt);
			lasttime = time_sec;
			status_bytes = 0.0;
		}
	}
	fsync (fd);
	free (data);
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

