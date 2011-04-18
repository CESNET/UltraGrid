/* ac3aes3.c
 * 
 * Embed AC-3 frames in AES3 serial digital audio.
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
#include <string.h>
#include <netinet/in.h>

#include "master.h"

#ifndef _LFS_LARGEFILE
#error Large file support not found
#endif

#define CHANNELS 2
#define SAMPLES 1536
#define SAMPLE_SIZE (sizeof(uint16_t))
#define PERIOD_SIZE (CHANNELS * SAMPLES * SAMPLE_SIZE)
#define DATA_TYPE_AC3 1

static const char progname[] = "ac3aes3";

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
	const uint16_t frmsize[][3] = {
		{  64,   69,   96},
		{  64,   70,   96},
		{  80,   87,  120},
		{  80,   88,  120},
		{  96,  104,  144},
		{  96,  105,  144},
		{ 112,  121,  168},
		{ 112,  122,  168},
		{ 128,  139,  192},
		{ 128,  140,  192},
		{ 160,  174,  240},
		{ 160,  175,  240},
		{ 192,  208,  288},
		{ 192,  209,  288},
		{ 224,  243,  336},
		{ 224,  244,  336},
		{ 256,  278,  384},
		{ 256,  279,  384},
		{ 320,  348,  480},
		{ 320,  349,  480},
		{ 384,  417,  576},
		{ 384,  418,  576},
		{ 448,  487,  672},
		{ 448,  488,  672},
		{ 512,  557,  768},
		{ 512,  558,  768},
		{ 640,  696,  960},
		{ 640,  697,  960},
		{ 768,  835, 1152},
		{ 768,  836, 1152},
		{ 896,  975, 1344},
		{ 896,  976, 1344},
		{1024, 1114, 1536},
		{1024, 1115, 1536},
		{1152, 1253, 1728},
		{1152, 1254, 1728},
		{1280, 1393, 1920},
		{1280, 1394, 1920}
	};
	const char *fscod_str[] = {"48", "44.1", "32"};
	const unsigned int bitrate[] = {32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640};
	int opt, fd;
	ssize_t retcode;
	char *endptr;
	int word;
	unsigned char *indata, *outdata;
	uint16_t *inp, *outp, *length_code_ptr;
	int info, data_stream_number;
	unsigned int fscod;
	unsigned int frmsizecod;
	unsigned int framesize;

	info = 0;
	data_stream_number = 0;
	while ((opt = getopt (argc, argv, "his:V")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... [FILE]\n", argv[0]);
			printf ("Embed AC-3 frames "
				"from FILE or standard input "
				"in AES3 serial digital audio.\n"
				"Only 16-bit frame mode "
				"in exactly two channels\n"
				"of 16-bit little-endian words "
				"is supported.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -i\t\tdisplay AC-3 frame information only\n");
			printf ("  -s STREAM\tembed data stream number "
				"STREAM (default 0)\n");
			printf ("  -V\t\toutput version information and exit\n");
			printf ("\nSTREAM may be 0-6.\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'i':
			info = 1;
			break;
		case 's':
			data_stream_number = strtol (optarg, &endptr, 0);
			if (*endptr != '\0' ||
				data_stream_number < 0 ||
				data_stream_number > 6) {
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
	outdata = (unsigned char *)malloc (PERIOD_SIZE);
	if (outdata == NULL) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		goto NO_OUTDATA;
	}

	word = 0;
	memset (outdata, 0, SAMPLES * SAMPLE_SIZE);
	outp = (uint16_t *)outdata;
	*outp++ = 0xf872;
	*outp++ = 0x4e1f;
	*outp++ = (data_stream_number << 13) | DATA_TYPE_AC3;
	length_code_ptr = outp;
	*outp++ = 0;
	fscod = 0;
	frmsizecod = 0;
	framesize = 0;
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
				if (htons (*inp) == 0x0b77) {
					word++;
					*outp++ = htons (*inp);
				} else {
					word = 0;
				}
				break;
			case 1:
				word++;
				*outp++ = htons (*inp);
				break;
			case 2:
				fscod = (htons (*inp) >> 14) & 0x03;
				frmsizecod = (htons (*inp) >> 8) & 0x3f;
				if (fscod >= 0x03) {
					fprintf (stderr, "fscod reserved.\n");
					word = 0;
					break;
				}
				if (frmsizecod >= 38) {
					fprintf (stderr, "Unsupported frmsizcod.\n");
					word = 0;
					break;
				}
				framesize = frmsize[frmsizecod][fscod];
				if (info) {
					printf ("%s kHz, %u kbps, %u words\n",
						fscod_str[fscod],
						bitrate[frmsizecod >> 1],
						framesize);
				}
				word++;
				*length_code_ptr = framesize << 4;
				*outp++ = htons (*inp);
				break;
			default:
				*outp++ = htons (*inp);
				if (outp >= ((uint16_t *)outdata) + 4 + framesize) {
					memset (outp, 0, CHANNELS * (SAMPLES - framesize) * SAMPLE_SIZE);
					if (!info &&
						write_all (STDOUT_FILENO,
						outdata,
						PERIOD_SIZE) < 0) {
						fprintf (stderr, "%s: ", argv[0]);
						perror ("unable to write to file");
						goto NO_IO;
					}
					word = 0;
					outp = (uint16_t *)outdata;
					outp += 4;
				}
				break;
			}
			inp++;
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

