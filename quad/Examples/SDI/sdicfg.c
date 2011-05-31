/* sdicfg.c
 *
 * Raw SDI configuration program.
 *
 * Copyright (C) 2004-2010 Linear Systems Ltd. All rights reserved.
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
#include <sys/stat.h>

#include "sdi.h"
#include "master.h"
#include "../util.h"

#define MAXLEN 256
#define BUFFERS_FLAG		0x00000001
#define BUFSIZE_FLAG		0x00000002
#define CLKSRC_FLAG		0x00000004
#define MODE_FLAG		0x00000008

static const char progname[] = "sdicfg";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/sdi/sdi%cx%i/%s";
	int opt;
	unsigned int write_flags;
	struct stat buf;
	int num;
	char type, name[MAXLEN], data[MAXLEN];
	unsigned long int buffers, bufsize, clksrc, mode;
	int retcode;
	char *endptr;

	/* Parse the command line */
	write_flags = 0;
	buffers = 0;
	bufsize = 0;
	clksrc = 0;
	mode = 0;
	while ((opt = getopt (argc, argv, "b:hm:s:Vx:")) != -1) {
		switch (opt) {
		case 'b':
			write_flags |= BUFFERS_FLAG;
			buffers = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of buffers: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Configure a raw SDI interface.\n\n");
			printf ("  -b BUFFERS\tset the number of buffers\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -m MODE\tset the operating mode\n");
			printf ("  -s BUFSIZE\tset the buffer size\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("  -x CLKSRC\tset the clock source "
				"(transmitters only)\n");
			printf ("\nIf no options are specified, "
				"the current configuration is displayed.\n");
			printf ("\nBUFFERS must be two or more.\n");
			printf ("\nCLKSRC may be:\n"
				"\t0 (onboard oscillator)\n"
				"\t1 (external reference)\n"
				"\t2 (recovered receive clock)\n");
			printf ("\nMODE may be:\n"
				"\t0 (8-bit precision)\n"
				"\t1 (pack four 10-bit words "
				"into five bytes)\n");
			printf ("\nBUFSIZE must be "
				"a positive multiple of four,\n"
				"and at least 1024 bytes for transmitters.\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'm':
			write_flags |= MODE_FLAG;
			mode = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid mode: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 's':
			write_flags |= BUFSIZE_FLAG;
			bufsize = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid buffer size: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2004-2010 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case 'x':
			write_flags |= CLKSRC_FLAG;
			clksrc = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid clock source: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
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
		return -1;
	}
	if (!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", argv[0]);
		return -1;
	}
	type = (buf.st_rdev & 0x0080) ? 'r' : 't';
	num = buf.st_rdev & 0x007f;
	snprintf (name, sizeof (name), fmt, type, num, "dev");
	memset (data, 0, sizeof (data));
	if (util_read (name, data, sizeof (data)) < 0) {
		fprintf (stderr, "%s: error reading %s: ", argv[0], name);
		perror (NULL);
		return -1;
	}
	if (strtoul (data, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not a raw SDI device\n", argv[0]);
		return -1;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", argv[0], name);
		return -1;
	}

	retcode = 0;
	printf ("%s:\n", argv[optind]);
	if (write_flags) {
		if (write_flags & BUFFERS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "buffers");
			snprintf (data, sizeof (data), "%lu\n", buffers);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the number of buffers");
				return -1;
			}
			printf ("\tSet number of buffers = %lu.\n", buffers);
		}
		if (write_flags & BUFSIZE_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "bufsize");
			snprintf (data, sizeof (data), "%lu\n", bufsize);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the buffer size");
				return -1;
			}
			printf ("\tSet buffer size = %lu bytes.\n", bufsize);
		}
		if (write_flags & CLKSRC_FLAG) {
			if (type == 'r') {
				fprintf (stderr, "%s: "
					"unable to set the clock source: "
					"Not a transmitter\n", argv[0]);
				return -1;
			}
			snprintf (name, sizeof (name),
				fmt, type, num, "clock_source");
			snprintf (data, sizeof (data), "%lu\n", clksrc);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the clock source");
				return -1;
			}
			switch (clksrc) {
			case SDI_CTL_TX_CLKSRC_ONBOARD:
				printf ("\tUsing onboard oscillator.\n");
				break;
			case SDI_CTL_TX_CLKSRC_EXT:
				printf ("\tUsing external reference.\n");
				break;
			case SDI_CTL_TX_CLKSRC_RX:
				printf ("\tUsing recovered receive clock.\n");
				break;
			default:
				printf ("\tSet clock source = %lu.\n", clksrc);
				break;
			}
		}
		if (write_flags & MODE_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "mode");
			snprintf (data, sizeof (data), "%lu\n", mode);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface operating mode");
				return -1;
			}
			if (type == 'r') {
				switch (mode) {
				case SDI_CTL_MODE_8BIT:
					printf ("\tReceiving with "
						"8-bit precision.\n");
					break;
				case SDI_CTL_MODE_10BIT:
					printf ("\tPacking four 10-bit words "
						"into every five bytes.\n");
					break;
				default:
					printf ("\tSet mode = %lu.\n", mode);
					break;
				}
			} else {
				switch (mode) {
				case SDI_CTL_MODE_8BIT:
					printf ("\tAssuming "
						"8-bit data.\n");
					break;
				case SDI_CTL_MODE_10BIT:
					printf ("\tAssuming "
						"10-bit data.\n");
					break;
				default:
					printf ("\tSet mode = %lu.\n", mode);
					break;
				}
			}
		}
	} else {
		snprintf (name, sizeof (name),
			fmt, type, num, "buffers");
		if (util_strtoul (name, &buffers) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the number of buffers");
			retcode = -1;
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "bufsize");
		if (util_strtoul (name, &bufsize) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the buffer size");
			retcode = -1;
		}

		printf ("\t%lu x %lu-byte buffers\n",
			buffers, bufsize);

		if (type == 'r') {
			snprintf (name, sizeof (name),
				fmt, type, num, "mode");
			if (util_strtoul (name, &mode) > 0) {
				/* Don't complain on an error,
				 * since this parameter may not exist. */
				printf ("\tMode: %lu ", mode);
				switch (mode) {
				case SDI_CTL_MODE_8BIT:
					printf ("(receiving with "
						"8-bit precision)\n");
					break;
				case SDI_CTL_MODE_10BIT:
					printf ("(packing four words "
						"into every five bytes)\n");
					break;
				default:
					printf ("(unknown)\n");
					break;
				}
			}
		} else {
			snprintf (name, sizeof (name),
				fmt, type, num, "clock_source");
			if (util_strtoul (name, &clksrc) > 0) {
				/* Don't complain on an error,
				 * since this parameter may not exist. */
				printf ("\tClock source: %lu ", clksrc);
				switch (clksrc) {
				case SDI_CTL_TX_CLKSRC_ONBOARD:
					printf ("(onboard oscillator)\n");
					break;
				case SDI_CTL_TX_CLKSRC_EXT:
					printf ("(external reference)\n");
					break;
				case SDI_CTL_TX_CLKSRC_RX:
					printf ("(recovered receive clock)\n");
					break;
				default:
					printf ("(unknown)\n");
					break;
				}
			}

			snprintf (name, sizeof (name),
				fmt, type, num, "mode");
			if (util_strtoul (name, &mode) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get "
					"the interface operating mode");
				retcode = -1;
			}
			printf ("\tMode: %lu ", mode);
			switch (mode) {
			case SDI_CTL_MODE_8BIT:
				printf ("(assume 8-bit data)\n");
				break;
			case SDI_CTL_MODE_10BIT:
				printf ("(assume 10-bit data)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}
	}
	return retcode;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

