/* eg1.c
 * 
 * SMPTE EG 1 color bar generator for Linear Systems Ltd. SMPTE 259M-C boards.
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
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include "master.h"

#define TOTAL_SAMPLES 1716
#define TOTAL_LINES 525
#define FIELD_1 1
#define FIELD_2 2
#define VERT_BLANKING 0
#define ACTIVE_MAIN 1
#define ACTIVE_CHROMA_SET 2
#define ACTIVE_BLACK_SET 3

/* Static function prototypes */
static int mkline (unsigned short int *buf,
	int field,
	int active);
static uint8_t *pack8 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);
static uint8_t *pack10 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);

static const char progname[] = "eg1";

/**
 * mkline - generate one line
 * @buf: pointer to a buffer
 * @count: number of elements in the buffer
 * @field: field number
 * @active: line type
 *
 * Returns a negative error code on failure and zero on success.
 **/
static int
mkline (unsigned short int *buf,
	int field,
	int active)
{
	const unsigned int b = 205;
	unsigned short int *p = buf, sav, eav;

	switch (field) {
	case FIELD_1:
		switch (active) {
		case VERT_BLANKING:
			eav = 0xb6 << 2;
			sav = 0xab << 2;
			break;
		case ACTIVE_MAIN:
		case ACTIVE_CHROMA_SET:
		case ACTIVE_BLACK_SET:
			eav = 0x9d << 2;
			sav = 0x80 << 2;
			break;
		default:
			return -EINVAL;
		}
		break;
	case FIELD_2:
		switch (active) {
		case VERT_BLANKING:
			eav = 0xf1 << 2;
			sav = 0xec << 2;
			break;
		case ACTIVE_MAIN:
		case ACTIVE_CHROMA_SET:
		case ACTIVE_BLACK_SET:
			eav = 0xda << 2;
			sav = 0xc7 << 2;
			break;
		default:
			return -EINVAL;
		}
		break;
	default:
		return -EINVAL;
	}
	*p++ = 0x3ff;
	*p++ = 0x000;
	*p++ = 0x000;
	*p++ = eav;
	while (p < (buf + 272)) {
		*p++ = 0x200;
		*p++ = 0x040;
		*p++ = 0x200;
		*p++ = 0x040;
	}
	*p++ = 0x3ff;
	*p++ = 0x000;
	*p++ = 0x000;
	*p++ = sav;
	switch (active) {
	default:
	case VERT_BLANKING:
		while (p < (buf + TOTAL_SAMPLES)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		break;
	case ACTIVE_MAIN:
		/* 75% gray */
		while (p < (buf + 276 + b + 1)) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		/* 75% yellow */
		while (p < (buf + 276 + 2 * b + 2)) {
			*p++ = 176;
			*p++ = 646;
			*p++ = 567;
			*p++ = 646;
		}
		/* 75% cyan */
		while (p < (buf + 276 + 3 * b + 3)) {
			*p++ = 625;
			*p++ = 525;
			*p++ = 176;
			*p++ = 525;
		}
		/* 75% green */
		while (p < (buf + 276 + 4 * b + 2)) {
			*p++ = 289;
			*p++ = 450;
			*p++ = 231;
			*p++ = 450;
		}
		/* 75% magenta */
		while (p < (buf + 276 + 5 * b + 3)) {
			*p++ = 735;
			*p++ = 335;
			*p++ = 793;
			*p++ = 335;
		}
		/* 75% red */
		while (p < (buf + 276 + 6 * b + 4)) {
			*p++ = 399;
			*p++ = 260;
			*p++ = 848;
			*p++ = 260;
		}
		/* 75% blue */
		while (p < (buf + TOTAL_SAMPLES)) {
			*p++ = 848;
			*p++ = 139;
			*p++ = 457;
			*p++ = 139;
		}
		break;
	case ACTIVE_CHROMA_SET:
		/* 75% blue */
		while (p < (buf + 276 + b + 1)) {
			*p++ = 848;
			*p++ = 139;
			*p++ = 457;
			*p++ = 139;
		}
		/* black */
		while (p < (buf + 276 + 2 * b + 2)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* 75% magenta */
		while (p < (buf + 276 + 3 * b + 3)) {
			*p++ = 735;
			*p++ = 335;
			*p++ = 793;
			*p++ = 335;
		}
		/* black */
		while (p < (buf + 276 + 4 * b + 2)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* 75% cyan */
		while (p < (buf + 276 + 5 * b + 3)) {
			*p++ = 625;
			*p++ = 525;
			*p++ = 176;
			*p++ = 525;
		}
		/* black */
		while (p < (buf + 276 + 6 * b + 4)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* 75% gray */
		while (p < (buf + TOTAL_SAMPLES)) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		break;
	case ACTIVE_BLACK_SET:
		/* -I */
		while (p < (buf + 276 + 257)) {
			*p++ = 624;
			*p++ = 231;
			*p++ = 390;
			*p++ = 231;
		}
		/* white */
		while (p < (buf + 276 + 2 * 257)) {
			*p++ = 0x200;
			*p++ = 940;
			*p++ = 0x200;
			*p++ = 940;
		}
		/* +Q */
		while (p < (buf + 276 + 3 * 257)) {
			*p++ = 684;
			*p++ = 177;
			*p++ = 591;
			*p++ = 177;
		}
		/* black */
		while (p < (buf + 276 + 4 * 257)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* blacker than black */
		while (p < (buf + 276 + 4 * 257 + 68)) {
			*p++ = 0x200;
			*p++ = 29;
			*p++ = 0x200;
			*p++ = 29;
		}
		/* black */
		while (p < (buf + 276 + 4 * 257 + 2 * 68 + 2)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* whiter than black */
		while (p < (buf + 276 + 4 * 257 + 3 * 68 + 2)) {
			*p++ = 0x200;
			*p++ = 99;
			*p++ = 0x200;
			*p++ = 99;
		}
		/* black */
		while (p < (buf + TOTAL_SAMPLES)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		break;
	}
	return 0;
}

/**
 * pack8 - pack a line of 8-bit data
 * @outbuf: pointer to the output buffer
 * @inbuf: pointer to the input buffer
 * @count: number of elements in the buffer
 *
 * Returns a pointer to the next output location.
 **/
static uint8_t *
pack8 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count)
{
	unsigned short int *inp = inbuf;
	uint8_t *outp = outbuf;

	while (inp < (inbuf + count)) {
		*outp++ = *inp++ >> 2;
	}
	return outp;
}

/**
 * pack10 - pack a line of 10-bit data
 * @outbuf: pointer to the output buffer
 * @inbuf: pointer to the input buffer
 * @count: number of elements in the buffer
 *
 * Returns a pointer to the next output location.
 **/
static uint8_t *
pack10 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count)
{
	unsigned short int *inp = inbuf;
	uint8_t *outp = outbuf;

	while (inp < (inbuf + count)) {
		*outp++ = *inp & 0xff;
		*outp = *inp++ >> 8;
		*outp++ += (*inp << 2) & 0xfc;
		*outp = *inp++ >> 6;
		*outp++ += (*inp << 4) & 0xf0;
		*outp = *inp++ >> 4;
		*outp++ += (*inp << 6) & 0xc0;
		*outp++ = *inp++ >> 2;
	}

	return outp;
}

int
main (int argc, char **argv)
{
	int opt, frames;
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	char *endptr;
	unsigned short int buf[TOTAL_SAMPLES];
	uint8_t *data, *p;
	size_t framesize, bytes;
	int i, ret;

	frames = -1; /* Generate an infinite number of frames */
	pack = pack10;
	framesize = TOTAL_SAMPLES * 10 / 8 * TOTAL_LINES;
	while ((opt = getopt (argc, argv, "hn:V8")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]...\n", argv[0]);
			printf ("Output a SMPTE EG 1 color bar pattern "
				"for transmission by\n"
				"a Linear Systems Ltd. "
				"SMPTE 259M-C board.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -n NUM\tstop after NUM frames\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("  -8\t\toutput an 8-bit signal\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'n':
			frames = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of frames: %s\n",
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
		case '8':
			pack = pack8;
			framesize = TOTAL_SAMPLES * TOTAL_LINES;
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

	/* Allocate memory */
	data = malloc (framesize);
	if (!data) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		return -1;
	}

	/* Generate a frame */
	memset (buf, 0, sizeof (buf));
	p = data;
	for (i = 10; i <= 11; i++) {
		mkline (buf, FIELD_1, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 12; i <= 19; i++) {
		mkline (buf, FIELD_1, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 20; i <= 182; i++) {
		mkline (buf, FIELD_1, ACTIVE_MAIN);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 183; i <= 202; i++) {
		mkline (buf, FIELD_1, ACTIVE_CHROMA_SET);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 203; i <= 263; i++) {
		mkline (buf, FIELD_1, ACTIVE_BLACK_SET);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 264; i <= 265; i++) {
		mkline (buf, FIELD_1, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 266; i <= 271; i++) {
		mkline (buf, FIELD_2, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	mkline (buf, FIELD_2, VERT_BLANKING);
	p = pack (p, buf, TOTAL_SAMPLES);
	for (i = 273; i <= 274; i++) {
		mkline (buf, FIELD_2, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 275; i <= 282; i++) {
		mkline (buf, FIELD_2, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 283; i <= 445; i++) {
		mkline (buf, FIELD_2, ACTIVE_MAIN);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 446; i <= 464; i++) {
		mkline (buf, FIELD_2, ACTIVE_CHROMA_SET);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 465; i <= 525; i++) {
		mkline (buf, FIELD_2, ACTIVE_BLACK_SET);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 1; i <= 3; i++) {
		mkline (buf, FIELD_2, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 4; i <= 8; i++) {
		mkline (buf, FIELD_1, VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	mkline (buf, FIELD_1, VERT_BLANKING);
	p = pack (p, buf, TOTAL_SAMPLES);

	while (frames) {
		/* Output the frame */
		bytes = 0;
		while (bytes < framesize) {
			if ((ret = write (STDOUT_FILENO,
				data + bytes, framesize - bytes)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to write");
				free (data);
				return -1;
			}
			bytes += ret;
		}
		if (frames > 0) {
			frames--;
		}
	}
	free (data);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

