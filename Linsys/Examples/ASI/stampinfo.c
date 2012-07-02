/**
 * stampinfo.c
 *
 * Read the timestamps attached to each MPEG-2 transport stream packet.
 *
 * Copyright (C) 2005-2010 Linear Systems Ltd.
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
 **/

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <limits.h>
#include <signal.h>

#include "master.h"
#include "../util.h"

#ifndef _LFS_LARGEFILE
#error Large file support not found
#endif

#ifndef LLONG_MAX
#define LLONG_MAX ((long long int)(~0ULL>>1))
#endif

#define PACKET_SIZE 188
#define ECC_SIZE 16
#define TIMESTAMP_SIZE 8

#if __BYTE_ORDER == __BIG_ENDIAN
#error Big endian architecture not supported
#endif

static const char progname[] = "stampinfo";

/* Static function prototypes */
static ssize_t readbuf (int fd, uint8_t *buf, size_t count);
static unsigned long long int getpcr (uint8_t *buf);
static void handler (int sig __attribute__((unused)));

/**
 * readbuf - read until @buf is full or an error occurs
 * @fd: file descriptor
 * @buf: buffer
 * @count: sizeof (@buf)
 **/
static ssize_t
readbuf (int fd, uint8_t *buf, size_t count)
{
	uint8_t *p;
	ssize_t retcode;

	p = buf;
	while (p < (buf + count)) {
		retcode = read (fd, p, buf + count - p);
		if (retcode <= 0) {
			return retcode;
		}
		p += retcode;
	}
	return count;
}

/**
 * getpcr - read a PCR
 * @buf: pointer to the PCR
 **/
static unsigned long long int
getpcr (uint8_t *buf)
{
	uint8_t *p = buf;
	unsigned long long int pcr;

	pcr = *p++;
	pcr <<= 8;
	pcr += *p++;
	pcr <<= 8;
	pcr += *p++;
	pcr <<= 8;
	pcr += *p++;
	pcr <<= 1;
	pcr += *p >> 7;
	pcr *= 300;
	pcr += (*p++ & 0x01) << 8;
	pcr += *p;
	return pcr;
}

int flag;

/**
 * handler - signal handler
 * @sig: signal number
 **/
static void
handler (int sig __attribute__((unused)))
{
	flag = 1;
	return;
}

int
main (int argc, char **argv)
{
	int bufsize, opt, fd, append, verbose;
	uint8_t buf[PACKET_SIZE + ECC_SIZE + TIMESTAMP_SIZE], *packet, *stamp;
	ssize_t retcode;
	unsigned int i, mindiffpos, maxdiffpos;
	uint64_t ts, oldts;
	long long int diff, mindiff, maxdiff;
	struct sigaction act;

	/* Parse the command line */
	bufsize = PACKET_SIZE + TIMESTAMP_SIZE;
	append = -1;
	verbose = 0;
	while ((opt = getopt (argc, argv, "ahpvV2")) != -1) {
		switch (opt) {
		case 'a':
			append = 1;
			break;
		case 'h':
			printf ("Usage: %s -a | -p [OPTION]... [FILE]\n",
				argv[0]);
			printf ("Read the timestamps attached to "
				"each MPEG-2 transport stream packet\n"
				"in FILE or standard input.\n\n");
			printf ("  -a\tread appended timestamps\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -p\tread prepended timestamps\n");
			printf ("  -v\tverbose output\n");
			printf ("  -V\toutput version information and exit\n");
			printf ("  -2\tassume 204-byte packets\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'p':
			append = 0;
			break;
		case 'v':
			verbose = 1;
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2005-2010 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case '2':
			bufsize = PACKET_SIZE + ECC_SIZE + TIMESTAMP_SIZE;
			break;
		case '?':
			goto USAGE;
		}
	}

	/* Check for mandatory argument */
	if (append < 0) {
		fprintf (stderr, "%s: "
			"you must specify one of the '-ap' options\n",
			argv[0]);
		goto USAGE;
	}
	if (append) {
		packet = buf;
		stamp = buf + bufsize - TIMESTAMP_SIZE;
	} else {
		packet = buf + TIMESTAMP_SIZE;
		stamp = buf;
	}

	/* Check the number of arguments */
	if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Open the file */
	fd = STDIN_FILENO;
	if (argc > optind) {
		if ((fd = open (argv[optind], O_RDONLY)) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to open file for reading");
			return -1;
		}
	}

	/* Print the first packet header and timestamp */
	i = 0;
	ts = UINT64_C(0);
	retcode = readbuf (fd, buf, bufsize);
	if (retcode == bufsize) {
		i++;
		ts = append ?
			getpcr (stamp) :
			*(uint64_t *)stamp; /* Little endian only */
		if (verbose) {
			printf ("%u: hdr=%08X ts=%"PRIu64"\n",
				i,
				ntohl (*(uint32_t *)packet),
				ts);
		}
	} else if (retcode < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to read");
		if (argc > optind) {
			close (fd);
		}
		return -1;
	} else {
		printf ("No packets found.\n");
		if (argc > optind) {
			close (fd);
		}
		return 0;
	}

	/* Print the packet headers, timestamps, and
	 * differences between consecutive timestamps */
	oldts = ts;
	mindiff = LLONG_MAX;
	mindiffpos = 2;
	maxdiff = 0LL;
	maxdiffpos = 2;
	flag = 0;
	act.sa_handler = handler;
	sigemptyset (&act.sa_mask);
	act.sa_flags = SA_RESETHAND;
	sigaction (SIGINT, &act, NULL);
	while (!flag && ((retcode = readbuf (fd, buf, bufsize)) == bufsize)) {
		i++;
		ts = append ?
			getpcr (stamp) :
			*(uint64_t *)stamp; /* Little endian only */
		diff = ts - oldts;
		if (verbose) {
			printf ("%u: hdr=%08X ts=%"PRIu64" diff=%lli\n",
				i,
				ntohl (*(uint32_t *)packet),
				ts,
				diff);
		}
		if (diff < mindiff) {
			mindiff = diff;
			mindiffpos = i;
			fprinttime (stdout, progname);
			printf ("min interval = %lli (%.3f us) at packet %u\n",
				mindiff,
				(double)mindiff / 27,
				mindiffpos);
		}
		if (diff > maxdiff) {
			maxdiff = diff;
			maxdiffpos = i;
			fprinttime (stdout, progname);
			printf ("max interval = %lli (%.3f us) at packet %u\n",
				maxdiff,
				(double)maxdiff / 27,
				maxdiffpos);
		}
		oldts = ts;
	}
	if (!flag && (retcode < 0)) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to read");
		if (argc > optind) {
			close (fd);
		}
		return -1;
	}
	if (argc > optind) {
		close (fd);
	}

	/* Print a summary */
	printf ("\n--- Timestamp summary ---\n");
	printf ("%u timestamps read\n", i);
	if (i > 1) {
		printf ("min interval = %lli (%.3f us) at packet %u\n",
			mindiff, (double)mindiff / 27, mindiffpos);
		printf ("max interval = %lli (%.3f us) at packet %u\n",
			maxdiff, (double)maxdiff / 27, maxdiffpos);
	}

	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

