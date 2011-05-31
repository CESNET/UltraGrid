/* asicfg.c
 *
 * DVB ASI configuration program.
 *
 * Copyright (C) 2001-2010 Linear Systems Ltd. All rights reserved.
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

#include "asi.h"
#include "master.h"
#include "../util.h"

#define MAXLEN 256
#define BUFFERS_FLAG		0x00000001
#define BUFSIZE_FLAG		0x00000002
#define CLKSRC_FLAG		0x00000004
#define COUNT27_FLAG		0x00000008
#define MODE_FLAG		0x00000010
#define NULLPACKETS_FLAG	0x00000020
#define TIMESTAMPS_FLAG		0x00000040

static const char progname[] = "asicfg";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/asi/asi%cx%i/%s";
	int opt;
	unsigned int write_flags;
	struct stat buf;
	int num;
	char type, name[MAXLEN], data[MAXLEN];
	unsigned long int buffers, bufsize, granularity, transport;
	unsigned long int clksrc, mode, count27, timestamps, null_packets;
	char *endptr;

	/* Parse the command line */
	write_flags = 0;
	buffers = 0;
	bufsize = 0;
	clksrc = 0;
	count27 = 0;
	granularity = 0;
	mode = 0;
	null_packets = 0;
	timestamps = 0;
	transport = 0;
	while ((opt = getopt (argc, argv, "b:cChm:nNs:t:Vx:")) != -1) {
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
		case 'c':
			write_flags |= COUNT27_FLAG;
			count27 = 1;
			break;
		case 'C':
			write_flags |= COUNT27_FLAG;
			count27 = 0;
			break;
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Configure a DVB ASI interface.\n\n");
			printf ("  -b BUFFERS\tset the number of buffers\n");
			printf ("  -c\t\tenable 27 MHz counter\n");
			printf ("  -C\t\tdisable 27 MHz counter\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -m MODE\tset the operating mode\n");
			printf ("  -n\t\tenable null packet insertion\n");
			printf ("  -N\t\tdisable null packet insertion\n");
			printf ("  -s BUFSIZE\tset the buffer size\n");
			printf ("  -t MODE"
				"\tset the packet timestamping mode\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("  -x CLKSRC\tset the clock source "
				"(transmitters only)\n");
			printf ("\nIf no options are specified, "
				"the current configuration is displayed.\n");
			printf ("\nBUFFERS must be two or more.\n");
			printf ("\nCLKSRC may be:\n"
				"\t0 (onboard oscillator)\n"
				"\t1 (external reference clock)\n"
				"\t2 (recovered receive clock)\n"
				"\t3 (external reference clock 2)\n");
			printf ("\nFor transmitters, MODE may be:\n"
				"\t0 (assume 188-byte packets)\n"
				"\t1 (assume 204-byte packets)\n"
				"\t2 (append sixteen 0x00 bytes to "
				"each 188-byte packet)\n");
			printf ("\nFor receivers, MODE may be:\n"
				"\t0 (raw mode)\n"
				"\t1 (synchronize on 188-byte packets)\n"
				"\t2 (synchronize on 204-byte packets)\n"
				"\t3 (synchronize on detected packet size)\n"
				"\t4 (synchronize on detected packet size\n"
				"\t\tand strip the last sixteen bytes "
				"from each 204-byte packet)\n"
				"\t5 (synchronize on 204-byte packets\n"
				"\t\tand strip the last sixteen bytes "
				"from each packet)\n");
			printf ("\nBUFSIZE must be "
				"a positive multiple of the granularity,\n"
				"and at least 1024 bytes for transmitters.\n");
			printf ("\nTIMESTAMPS may be:\n"
				"\t0 (disabled)\n"
				"\t1 (enable appended PCR-format "
				"packet timestamps)\n"
				"\t2 (enable prepended 63-bit "
				"packet timestamps)\n");
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
		case 'n':
			write_flags |= NULLPACKETS_FLAG;
			null_packets = 1;
			break;
		case 'N':
			write_flags |= NULLPACKETS_FLAG;
			null_packets = 0;
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
		case 't':
			write_flags |= TIMESTAMPS_FLAG;
			timestamps = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid timestamp mode: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2001-2010 "
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
		fprintf (stderr, "%s: not an ASI device\n", argv[0]);
		return -1;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", argv[0], name);
		return -1;
	}

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
			case ASI_CTL_TX_CLKSRC_ONBOARD:
				printf ("\tUsing onboard oscillator.\n");
				break;
			case ASI_CTL_TX_CLKSRC_EXT:
				printf ("\tUsing external reference clock.\n");
				break;
			case ASI_CTL_TX_CLKSRC_RX:
				printf ("\tUsing recovered receive clock.\n");
				break;
			case ASI_CTL_TX_CLKSRC_EXT2:
				printf ("\tUsing external reference clock 2.\n");
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
				case ASI_CTL_RX_MODE_RAW:
					printf ("\tReceiving in raw mode.\n");
					break;
				case ASI_CTL_RX_MODE_188:
					printf ("\tSynchronizing on "
						"188-byte packets.\n");
					break;
				case ASI_CTL_RX_MODE_204:
					printf ("\tSynchronizing on "
						"204-byte packets.\n");
					break;
				case ASI_CTL_RX_MODE_AUTO:
					printf ("\tSynchronizing on "
						"detected packet size.\n");
					break;
				case ASI_CTL_RX_MODE_AUTOMAKE188:
					printf ("\tSynchronizing on "
						"detected packet size\n"
						"\tand stripping the last "
						"sixteen bytes from "
						"each 204-byte packet.\n");
					break;
				case ASI_CTL_RX_MODE_204MAKE188:
					printf ("\tSynchronizing on "
						"204-byte packets\n"
						"\tand stripping the last "
						"sixteen bytes from "
						"each packet.\n");
					break;
				default:
					printf ("\tSet mode = %lu.\n", mode);
					break;
				}
			} else {
				switch (mode) {
				case ASI_CTL_TX_MODE_188:
					printf ("\tAssuming "
						"188-byte packets.\n");
					break;
				case ASI_CTL_TX_MODE_204:
					printf ("\tAssuming "
						"204-byte packets.\n");
					break;
				case ASI_CTL_TX_MODE_MAKE204:
					printf ("\tAppending "
						"sixteen 0x00 bytes to "
						"each 188-byte packet.\n");
					break;
				default:
					printf ("\tSet mode = %lu.\n", mode);
					break;
				}
			}
		}
		if (write_flags & COUNT27_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "count27");
			snprintf (data, sizeof (data), "%lu\n", count27);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				if (count27) {
					perror ("unable to enable "
						"27 MHz counter");
				} else {
					perror ("unable to disable "
						"27 MHz counter");
				}
				return -1;
			}
			printf ("\t%sabled 27 MHz counter.\n",
				count27 ? "En" : "Dis");
		}
		if (write_flags & TIMESTAMPS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "timestamps");
			snprintf (data, sizeof (data), "%lu\n", timestamps);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the packet timestamping mode");
				return -1;
			}
			switch (timestamps) {
			case ASI_CTL_TSTAMP_NONE:
				printf ("\tDisabled packet timestamps.\n");
				break;
			case ASI_CTL_TSTAMP_APPEND:
				printf ("\tEnabled "
					"appended packet timestamps.\n");
				break;
			case ASI_CTL_TSTAMP_PREPEND:
				printf ("\tEnabled "
					"prepended packet timestamps.\n");
				break;
			default:
				printf ("\tSet packet timestamping mode = "
					"%lu.\n", timestamps);
				break;
			}
		}
		if (write_flags & NULLPACKETS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "null_packets");
			snprintf (data, sizeof (data), "%lu\n", null_packets);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				if (null_packets) {
					perror ("unable to enable "
						"null packet insertion");
				} else {
					perror ("unable to disable "
						"null packet insertion");
				}
				return -1;
			}
			printf ("\t%sabled null packet insertion.\n",
				null_packets ? "En" : "Dis");
		}
	} else {
		snprintf (name, sizeof (name),
			fmt, type, num, "transport");
		if (util_strtoul (name, &transport) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the transport type");
			return -1;
		}
		printf ("\tTransport type: %lu ", transport);
		switch (transport) {
		case ASI_CTL_TRANSPORT_DVB_ASI:
			printf ("(DVB ASI)\n");
			break;
		case ASI_CTL_TRANSPORT_SMPTE_310M:
			printf ("(SMPTE 310M)\n");
			break;
		default:
			printf ("(unknown)\n");
			break;
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "buffers");
		if (util_strtoul (name, &buffers) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the number of buffers");
			return -1;
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "bufsize");
		if (util_strtoul (name, &bufsize) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the buffer size");
			return -1;
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "granularity");
		if (util_strtoul (name, &granularity) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the buffer granularity");
			return -1;
		}
		printf ("\t%lu x %lu-byte buffers, granularity %lu\n",
			buffers, bufsize, granularity);

		if (type == 'r') {
			snprintf (name, sizeof (name),
				fmt, type, num, "mode");
			if (util_strtoul (name, &mode) > 0) {
				/* Don't complain on an error,
				 * since this parameter may not exist. */
				printf ("\tMode: %lu ", mode);
				switch (mode) {
				case ASI_CTL_RX_MODE_RAW:
					printf ("(raw mode)\n");
					break;
				case ASI_CTL_RX_MODE_188:
					printf ("(synchronize on "
						"188-byte packets)\n");
					break;
				case ASI_CTL_RX_MODE_204:
					printf ("(synchronize on "
						"204-byte packets)\n");
					break;
				case ASI_CTL_RX_MODE_AUTO:
					printf ("(synchronize on "
						"detected packet size)\n");
					break;
				case ASI_CTL_RX_MODE_AUTOMAKE188:
					printf ("(synchronize on "
						"detected packet size\n"
						"\t\tand strip the last "
						"sixteen bytes from "
						"each 204-byte packet)\n");
					break;
				case ASI_CTL_RX_MODE_204MAKE188:
					printf ("(synchronize on "
						"204-byte packets\n"
						"\t\tand strip the last "
						"sixteen bytes from "
						"each packet)\n");
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
				case ASI_CTL_TX_CLKSRC_ONBOARD:
					printf ("(onboard oscillator)\n");
					break;
				case ASI_CTL_TX_CLKSRC_EXT:
					printf ("(external reference)\n");
					break;
				case ASI_CTL_TX_CLKSRC_RX:
					printf ("(recovered receive clock)\n");
					break;
				case ASI_CTL_TX_CLKSRC_EXT2:
					printf ("(external reference 2)\n");
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
				return -1;
			}
			printf ("\tMode: %lu ", mode);
			switch (mode) {
			case ASI_CTL_TX_MODE_188:
				printf ("(assume 188-byte packets)\n");
				break;
			case ASI_CTL_TX_MODE_204:
				printf ("(assume 204-byte packets)\n");
				break;
			case ASI_CTL_TX_MODE_MAKE204:
				printf ("(append sixteen 0x00 bytes to "
					"each 188-byte packet)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "count27");
		if (util_strtoul (name, &count27) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\t27 MHz counter %sabled\n",
				count27 ? "en" : "dis");
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "timestamps");
		if (util_strtoul (name, &timestamps) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tPacket timestamping mode: %lu ", timestamps);
			switch (timestamps) {
			case 0:
				printf ("(disabled)\n");
				break;
			case 1:
				printf ("(appended timestamps)\n");
				break;
			case 2:
				printf ("(prepended timestamps)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "null_packets");
		if (util_strtoul (name, &null_packets) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tNull packet insertion %sabled\n",
				null_packets ? "en" : "dis");
		}
	}
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

