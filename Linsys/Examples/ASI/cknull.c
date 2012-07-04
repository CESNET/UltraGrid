/* cknull.c
 * 
 * MPEG-2 transport stream testing utility.
 *
 * Copyright (C) 2001-2009 Linear Systems Ltd. All rights reserved.
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

#include "master.h"
#include "../util.h"

#ifndef _LFS_LARGEFILE
#error Large file support not found
#endif

static const char progname[] = "cknull";

int
main (int argc, char **argv)
{
	int opt, fd, i, retcode;
	int started, packetbyte, packetnum, packetsize;
	unsigned long long int bytes;
	unsigned char *data;

	while ((opt = getopt (argc, argv, "hV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... [FILE]\n", argv[0]);
			printf ("Test FILE or standard input "
				"for null MPEG-2 transport stream packets\n"
				"with an ascending continuity_counter "
				"and a payload of 0xff.\n\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -V\toutput version information and exit\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2001-2009 "
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
	if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Open the input file */
	fd = STDIN_FILENO;
	if (argc - optind) {
		if ((fd = open (argv[optind], O_RDONLY, 0)) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to open file for reading");
			return -1;
		}
	}

	/* Allocate some memory */
	if ((data = (unsigned char *)malloc (BUFSIZ)) == NULL) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		goto NO_DATA;
	}

	started = 0;
	packetbyte = 0;
	packetnum = 0;
	packetsize = 0;
	bytes = 0;
	while ((retcode = read (fd, data, BUFSIZ))) {
		if (retcode < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to read from file");
			goto NO_READ;
		}
		i = 0;
		/* Search for the first packet and get the
		 * packet size and continuity_counter value. */
		while ((!started) && (i < retcode)) {
			if (packetbyte == 0) {
				if ((data[i] == 0x47) || (data[i] == 0xB8)) {
					packetbyte++;
				} else {
					fprinttime (stdout, progname);
					printf ("packet not found at "
						"byte %llu\n", bytes);
				}
			} else if (packetbyte == 1) {
				if (data[i] == 0x1f) {
					packetbyte++;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("header error at "
						"byte %llu\n", bytes);
				}
			} else if (packetbyte == 2) {
				if (data[i] == 0xff) {
					packetbyte++;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("header error at "
						"byte %llu\n", bytes);
				}
			} else if (packetbyte == 3) {
				if ((data[i] & 0xf0) == 0x10) {
					packetbyte++;
					packetnum = ((data[i] & 0x0f) + 1) %
						16;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("header error at "
						"byte %llu\n", bytes);
				}
			} else if ((packetbyte >= 4) && (packetbyte < 188)) {
				if (data[i] == 0xff) {
					packetbyte++;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("payload error at "
						"byte %llu\n", bytes);
				}
			} else if (packetbyte == 188) {
				if ((data[i] == 0x47) || (data[i] == 0xB8)) {
					packetbyte = 1;
					packetsize = 188;
					started = 1;
				} else if (data[i] == 0x00) {
					packetbyte++;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("payload error at "
						"byte %llu\n", bytes);
				}
			} else if ((packetbyte > 188) && (packetbyte < 204)) {
				if (data[i] == 0x00) {
					packetbyte++;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("payload error at "
						"byte %llu\n", bytes);
				}
			} else {
				if ((data[i] == 0x47) || (data[i] == 0xB8)) {
					packetbyte = 1;
					packetsize = 204;
					started = 1;
				} else {
					packetbyte = 0;
					fprinttime (stdout, progname);
					printf ("invalid packet length at "
						"byte %llu\n", bytes);
				}
			}
			i++;
			bytes++;
		}
		/* Test all packets after the first,
		 * assuming the packet size is the same as the first. */
		while (started && (i < retcode)) {
			if ((packetbyte >= 4) && (packetbyte < 188)) {
				if (data[i] != 0xff) {
					fprinttime (stdout, progname);
					printf ("payload error at "
						"byte %llu\n", bytes);
				}
				packetbyte++;
			} else if (packetbyte == packetsize) {
				if ((data[i] != 0x47) && (data[i] != 0xB8)) {
					fprinttime (stdout, progname);
					printf ("packet not found at "
						"byte %llu\n", bytes);
				} else {
					packetbyte = 1;
				}
			} else if (packetbyte == 1) {
				if (data[i] != 0x1f) {
					fprinttime (stdout, progname);
					printf ("header error at "
						"byte %llu\n", bytes);
				}
				packetbyte++;
			} else if (packetbyte == 2) {
				if (data[i] != 0xff) {
					fprinttime (stdout, progname);
					printf ("header error at "
						"byte %llu\n", bytes);
				}
				packetbyte++;
			} else if (packetbyte == 3) {
				if ((data[i] & 0xf0) != 0x10) {
					fprinttime (stdout, progname);
					printf ("header error at "
						"byte %llu\n", bytes);
				} else if ((data[i] & 0x0f) != packetnum) {
					fprinttime (stdout, progname);
					printf ("sequence error at "
						"byte %llu\n", bytes);
				}
				packetbyte++;
				packetnum = ((data[i] & 0x0f) + 1) % 16;
			} else if ((packetbyte >= 188) && (packetbyte < 204)) {
				if (data[i] != 0x00) {
					fprinttime (stdout, progname);
					printf ("payload error at "
						"byte %llu\n", bytes);
				}
				packetbyte++;
			}
			i++;
			bytes++;
		}
	}
	free (data);
	if (argc - optind) {
		close (fd);
	}
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;

NO_READ:
	free (data);
NO_DATA:
	if (argc - optind) {
		close (fd);
	}
	return -1;
}

