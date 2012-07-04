/* aes3data.c
 * 
 * Extract data bursts from AES3 serial digital audio.
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
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/in.h>

#include "master.h"

#ifndef _LFS_LARGEFILE
#error Large file support not found
#endif

static const char progname[] = "aes3data";

static const char *
data_type_str (int data_type)
{
	switch (data_type) {
	case 0:
		return "Null data";
	case 1:
		return "ATSC A/52B (AC-3) data";
	case 2:
		return "Time stamp data";
	case 3:
		return "Pause data";
	case 4:
		return "MPEG-1 layer 1 data";
	case 5:
		return "MPEG-1 layer 2 or 3 data, MPEG-2 data without extension";
	case 6:
		return "MPEG-2 data with extension";
	case 8:
		return "MPEG-2 layer 1 data low-sampling frequency";
	case 9:
		return "MPEG-2 layer 2 or 3 data low-sampling frequency";
	case 10:
		return "MPEG-4 AAC data";
	case 11:
		return "MPEG-4 HE-AAC data";
	case 16:
		return "ATSC A/52B (Enhanced AC-3) data";
	case 26:
		return "Utility data type";
	case 27:
		return "SMPTE KLV data";
	case 28:
		return "Dolby E data";
	case 29:
		return "Captioning data";
	case 30:
		return "User defined data";
	default:
		return "Reserved";
	}
}

static ssize_t
write_all (int fd, const unsigned char *buf, size_t count)
{
	int bytes_written = 0;
	ssize_t ret;

	while ((count - bytes_written) > 0) {
		ret = write (fd, buf + bytes_written, count - bytes_written);
		if (ret < 0) {
			return ret;
		}
		bytes_written += ret;
	}
	return bytes_written;
}

int
main (int argc, char **argv)
{
	int opt, fd;
	ssize_t retcode;
	char *endptr;
	int word, bits;
	unsigned char *indata, *outdata;
	uint16_t *inp, *outp;
	int info, data_stream_number;
	int samp, oldsamp, bytes;

	info = 0;
	data_stream_number = 0;
	while ((opt = getopt (argc, argv, "his:V")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... [FILE]\n", argv[0]);
			printf ("Extract data bursts "
				"from AES3 serial digital audio "
				"in FILE or standard input.\n"
				"Only one data stream in 16-bit frame mode "
				"in exactly two channels\n"
				"of 16-bit little-endian words "
				"is supported.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -i\t\tdisplay data burst information only\n");
			printf ("  -s STREAM\textract data stream number "
				"STREAM (default 0)\n");
			printf ("  -V\t\toutput version information and exit\n");
			printf ("\nSTREAM may be 0-7.\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'i':
			info = 1;
			break;
		case 's':
			data_stream_number = strtol (optarg, &endptr, 0);
			if (*endptr != '\0' ||
				data_stream_number < 0 ||
				data_stream_number > 7) {
				fprintf (stderr,
					"%s: invalid data stream number: %s\n",
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
	indata = (unsigned char *)malloc (BUFSIZ);
	if (indata == NULL) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		goto NO_INDATA;
	}
	outdata = (unsigned char *)malloc (BUFSIZ);
	if (outdata == NULL) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		goto NO_OUTDATA;
	}

	word = 0;
	bits = 0;
	samp = 0;
	oldsamp = 3;
	outp = (uint16_t *)outdata;
	bytes = 0;
	while ((retcode = read (fd, indata, BUFSIZ))) {
		if (retcode < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to read from file");
			goto NO_IO;
		}
		/* Assume BUFSIZ and retcode are even */
		/* and process two bytes per iteration */
		inp = (uint16_t *)indata;
		while ((unsigned char *)inp < (indata + retcode)) {
			switch (word) {
			case 0:
				if (*inp == 0xf872) {
					word++;
				} else {
					word = 0;
				}
				break;
			case 1:
				if (*inp == 0x4e1f) {
					word++;
				} else {
					word = 0;
				}
				break;
			case 2:
				if (((*inp) >> 13) == data_stream_number) {
					if ((*inp & 0x60) != 0) {
						fprintf (stderr,
							"Unsupported data mode.\n");
						word = 0;
						break;
					}
					word++;
				} else {
					word = 0;
				}
				break;
			case 3:
				bits = *inp;
				word++;
				if (info) {
					printf ("Word %i (%+i): "
						"%s, %i bits\n",
						samp - 3, samp - oldsamp,
						data_type_str (*(inp - 1) & 0x1f),
						bits);
				}
				oldsamp = samp;
				break;
			default:
				*outp++ = ntohs (*inp);
				if (bits > 8) {
					bytes += 2;
				} else {
					bytes++;
				}
				bits -= 16;
				if (bits <= 0) {
					if (!info &&
						write_all (STDOUT_FILENO,
						outdata, bytes) < 0) {
						fprintf (stderr, "%s: ", argv[0]);
						perror ("unable to write to file");
						goto NO_IO;
					}
					word = 0;
					outp = (uint16_t *)outdata;
					bytes = 0;
				}
				break;
			}
			inp++;
			samp++;
		}
	}
	free (outdata);
	free (indata);
	if (argc - optind) {
		close (fd);
	}
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;

NO_IO:
	free (outdata);
NO_OUTDATA:
	free (indata);
NO_INDATA:
	if (argc - optind) {
		close (fd);
	}
	return -1;
}

