/* mknull.c
 * 
 * MPEG-2 transport stream generation utility.
 *
 * Copyright (C) 2001-2006 Linear Systems Ltd. All rights reserved.
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
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include "master.h"

/* This must be a multiple of 16 */
#define BUFFERED_PACKETS 16

static const char progname[] = "mknull";

/**
 * Update the timestamps prepended to a number of packets
 * based on an initial value, a running sum,
 * the desired bitrate, and the packet size.
 **/
static void
update_timestamps (unsigned char *p,
	unsigned long long int *timestamp,
	unsigned long long int *sum,
	unsigned long long int bitrate,
	int packetsize)
{
	const unsigned long long int n = bitrate;
	const unsigned long long int d = 216000000ULL;
	const unsigned long long int ip = d * packetsize / n;
	int i;
	uint64_t *ts = (uint64_t *)p;

	for (i = 0; i < BUFFERED_PACKETS; i++) {
		*ts = *timestamp;
		p += sizeof (uint64_t) + packetsize;
		ts = (uint64_t *)p;
		*sum += ip * n;
		*timestamp += ip;
		while (*sum < (d * packetsize)) {
			*sum += n;
			(*timestamp)++;
		}
		*sum -= d * packetsize;
	}
	return;
}

int
main (int argc, char **argv)
{
	int opt, i, j, bufsize, retcode;
	int bitrate, invert, packets, packetsize, tssize, packetnum, bytes;
	unsigned char *data, *p;
	char *endptr;
	unsigned long long int timestamp, sum;

	bitrate = 0;
	invert = 0;
	packets = -1; /* Generate an infinite number of packets */
	packetsize = 188;
	while ((opt = getopt (argc, argv, "b:hin:V2")) != -1) {
		switch (opt) {
		case 'b':
			bitrate = strtol (optarg, &endptr, 0);
			if ((*endptr != '\0') ||
				(bitrate <= 0) || (bitrate > 216000000)) {
				fprintf (stderr,
					"%s: invalid bitrate: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]...\n", argv[0]);
			printf ("Repeatedly output null "
				"MPEG-2 transport stream packets\n"
				"with an ascending continuity_counter "
				"and a payload of 0xff.\n\n");
			printf ("  -b BITRATE\tprepend timestamps "
				"at BITRATE bits per second\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -i\t\tinvert every eighth "
				"packet synchronization byte\n");
			printf ("  -n NUM\tstop after NUM packets\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("  -2\t\tappend sixteen 0x00 bytes "
				"to each 188-byte packet\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'i':
			invert = 1;
			break;
		case 'n':
			packets = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of packets: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2001-2006 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case '2':
			packetsize = 204;
			break;
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) > 0) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Allocate a buffer */
	tssize = (bitrate > 0) ? 8 : 0;
	bufsize = BUFFERED_PACKETS * (packetsize + tssize);
	if ((data = (unsigned char *)malloc (bufsize)) == NULL) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		return -1;
	}

	/* Initialize the buffer */
	p = data;
	packetnum = 0;
	for (i = 0; i < BUFFERED_PACKETS; i++) {
		if (bitrate > 0) {
			p += 8;
		}
		if (invert) {
			*p++ = (packetnum % 8) ? 0x47 : 0xB8;
		} else {
			*p++ = 0x47;
		}
		*p++ = 0x1f;
		*p++ = 0xff;
		*p++ = 0x10 | packetnum;
		packetnum = (packetnum + 1) & 0x0f;
		for (j = 4; j < 188; j++) {
			*p++ = 0xff;
		}
		for (j = 188; j < packetsize; j++) {
			*p++ = 0x00;
		}
	}
	timestamp = 0ULL;
	sum = 0ULL;
	if (bitrate > 0) {
		uint64_t *ts = (uint64_t *)data;

		update_timestamps (data,
			&timestamp,
			&sum,
			bitrate,
			packetsize);
		*ts |= UINT64_C(0x8000000000000000);
	}

	/* Generate the data */
	while (packets) {
		bytes = 0;
		if ((packets < BUFFERED_PACKETS) && (packets > 0)) {
			/* The number of packets to write is less than
			 * the buffer size and is not infinite (-1). */
			while (bytes < (packets * (packetsize + tssize))) {
				if ((retcode = write (STDOUT_FILENO,
					data + bytes,
					(packets * (packetsize + tssize)) -
					bytes)) < 0) {
					fprintf (stderr, "%s: ", argv[0]);
					perror ("unable to write to output");
					free (data);
					return -1;
				}
				bytes += retcode;
			}
			packets = 0;
		} else {
			/* The number of packets to write is greater than
			 * or equal to the buffer size or is infinite (-1). */
			while (bytes < bufsize) {
				if ((retcode = write (STDOUT_FILENO,
					data + bytes, bufsize - bytes)) < 0) {
					fprintf (stderr, "%s: ", argv[0]);
					perror ("unable to write to output");
					free (data);
					return -1;
				}
				bytes += retcode;
			}
			if (packets > 0) {
				packets -= BUFFERED_PACKETS;
			}
			if (bitrate > 0) {
				update_timestamps (data,
					&timestamp,
					&sum,
					bitrate,
					packetsize);
			}
		}
	}
	free (data);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

