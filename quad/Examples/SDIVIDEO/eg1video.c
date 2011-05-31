/* eg1video.c
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

#if __BYTE_ORDER == __BIG_ENDIAN
#error Big endian architecture not supported
#endif

#define TOTAL_SAMPLES 1716
#define ACTIVE_SAMPLES 1440
#define TOTAL_LINES 525
#define ACTIVE_LINES 487
#define PRODUCTION_LINES 486

#define VERT_BLANKING 0
#define MAIN_SET 1
#define CHROMA_SET 2
#define BLACK_SET 3
#define CEA_608 4

struct trs {
	unsigned short int sav;
	unsigned short int eav;
};

static const struct trs FIELD_1_ACTIVE = {
	.sav = 0x200,
	.eav = 0x274
};

static const struct trs FIELD_1_VERT_BLANKING = {
	.sav = 0x2ac,
	.eav = 0x2d8
};

static const struct trs FIELD_2_ACTIVE = {
	.sav = 0x31c,
	.eav = 0x368
};

static const struct trs FIELD_2_VERT_BLANKING = {
	.sav = 0x3b0,
	.eav = 0x3c4
};

struct line_info {
	const struct trs *xyz;
	unsigned int blanking;
};

/* Static function prototypes */
static int mkline (unsigned short int *buf,
	const struct line_info *info,
	unsigned int pattern);
static uint8_t *pack_v210 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);
static uint8_t *pack_uyvy (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);
static uint8_t *pack10 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);
static uint8_t *pack_v216 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);

static const char progname[] = "eg1video";

/**
 * mkline - generate one line
 * @buf: pointer to a buffer
 * @info: pointer to a line information structure
 * @pattern: pattern
 *
 * Returns a negative error code on failure and zero on success.
 **/
static int
mkline (unsigned short int *buf,
	const struct line_info *info __attribute__((unused)),
	unsigned int pattern)
{
	const unsigned int b = 205;
	unsigned short int *p = buf, *endp, sum;
	unsigned int samples = ACTIVE_SAMPLES;

	endp = p;
	switch (pattern) {
	default:
	case VERT_BLANKING:
		while (p < (buf + samples)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		break;
	case MAIN_SET:
		/* 75% gray */
		endp += b + 1;
		while (p < endp) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		/* 75% yellow */
		endp += b + 1;
		while (p < endp) {
			*p++ = 176;
			*p++ = 646;
			*p++ = 567;
			*p++ = 646;
		}
		/* 75% cyan */
		endp += b + 1;
		while (p < endp) {
			*p++ = 625;
			*p++ = 525;
			*p++ = 176;
			*p++ = 525;
		}
		/* 75% green */
		endp += b - 1;
		while (p < endp) {
			*p++ = 289;
			*p++ = 450;
			*p++ = 231;
			*p++ = 450;
		}
		/* 75% magenta */
		endp += b + 1;
		while (p < endp) {
			*p++ = 735;
			*p++ = 335;
			*p++ = 793;
			*p++ = 335;
		}
		/* 75% red */
		endp += b + 1;
		while (p < endp) {
			*p++ = 399;
			*p++ = 260;
			*p++ = 848;
			*p++ = 260;
		}
		/* 75% blue */
		while (p < (buf + samples)) {
			*p++ = 848;
			*p++ = 139;
			*p++ = 457;
			*p++ = 139;
		}
		break;
	case CHROMA_SET:
		/* 75% blue */
		endp += b + 1;
		while (p < endp) {
			*p++ = 848;
			*p++ = 139;
			*p++ = 457;
			*p++ = 139;
		}
		/* black */
		endp += b + 1;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* 75% magenta */
		endp += b + 1;
		while (p < endp) {
			*p++ = 735;
			*p++ = 335;
			*p++ = 793;
			*p++ = 335;
		}
		/* black */
		endp += b - 1;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* 75% cyan */
		endp += b + 1;
		while (p < endp) {
			*p++ = 625;
			*p++ = 525;
			*p++ = 176;
			*p++ = 525;
		}
		/* black */
		endp += b + 1;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* 75% gray */
		while (p < (buf + samples)) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		break;
	case BLACK_SET:
		/* -I */
		endp += 257;
		while (p < endp) {
			*p++ = 624;
			*p++ = 231;
			*p++ = 390;
			*p++ = 231;
		}
		/* white */
		endp += 257;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 940;
			*p++ = 0x200;
			*p++ = 940;
		}
		/* +Q */
		endp += 257;
		while (p < endp) {
			*p++ = 684;
			*p++ = 177;
			*p++ = 591;
			*p++ = 177;
		}
		/* black */
		endp += 257;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* blacker than black */
		endp += 68;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 29;
			*p++ = 0x200;
			*p++ = 29;
		}
		/* black */
		endp += 68 + 2;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		/* whiter than black */
		endp += 68;
		while (p < endp) {
			*p++ = 0x200;
			*p++ = 99;
			*p++ = 0x200;
			*p++ = 99;
		}
		/* black */
		while (p < (buf + samples)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
		break;
	case CEA_608:
		/* CEA-608 packet */
		*p++ = 0;
		*p++ = 0x3ff;
		*p++ = 0x3ff;
		*p++ = 0x161; sum = 0x161 & 0x1ff; /* DID */
		*p++ = 0x102; sum += 0x102 & 0x1ff; /* SDID */
		*p++ = 0x203; sum += 0x203 & 0x1ff; /* DC */
		*p++ = 0x18a; sum += 0x18a & 0x1ff;
		*p++ = 0x180; sum += 0x180 & 0x1ff;
		*p++ = 0x180; sum += 0x180 & 0x1ff;
		*p++ = (sum & 0x1ff) | ((sum & 0x100) ? 0 : 0x200);
		*p++ = 0x200;
		*p++ = 0x040;
		while (p < (buf + samples)) {
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
 * mkframe - generate one frame
 * @p: pointer to a buffer
 * @info: line information structure
 * @pat1: pattern 1
 * @pat2: pattern 2
 * @pat3: pattern 3
 * @pack: pointer to packing function
 *
 * Returns a negative error code on failure and zero on success.
 **/
static int
mkframe (uint8_t *p,
	struct line_info info,
	unsigned int pat1,
	unsigned int pat2,
	unsigned int pat3,
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count))
{
	uint8_t *(*vanc_pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	unsigned short int buf[ACTIVE_SAMPLES];
	size_t elements = ACTIVE_SAMPLES;
	unsigned int i;

	if (pack == pack_v210) {
		vanc_pack = pack_v210;
	} else {
		vanc_pack = pack_v216;
	}
	memset (buf, 0, sizeof (buf));
	if (info.blanking) {
		info.xyz = &FIELD_2_VERT_BLANKING;
		for (i = 1; i <= 3; i++) {
			mkline (buf, &info, VERT_BLANKING);
			p = vanc_pack (p, buf, elements);
		}
		info.xyz = &FIELD_1_VERT_BLANKING;
		for (i = 4; i <= 18; i++) {
			mkline (buf, &info, VERT_BLANKING);
			p = vanc_pack (p, buf, elements);
		}
		mkline (buf, &info, CEA_608);
		p = vanc_pack (p, buf, elements);
		info.xyz = &FIELD_1_ACTIVE;
		mkline (buf, &info, pat1);
		p = vanc_pack (p, buf, elements);
	}
	info.xyz = &FIELD_1_ACTIVE;
	for (i = 21; i <= 182; i++) {
		mkline (buf, &info, pat1);
		p = pack (p, buf, elements);
	}
	for (i = 183; i <= 202; i++) {
		mkline (buf, &info, pat2);
		p = pack (p, buf, elements);
	}
	for (i = 203; i <= 263; i++) {
		mkline (buf, &info, pat3);
		p = pack (p, buf, elements);
	}
	if (info.blanking) {
		info.xyz = &FIELD_1_VERT_BLANKING;
		for (i = 264; i <= 265; i++) {
			mkline (buf, &info, VERT_BLANKING);
			p = vanc_pack (p, buf, elements);
		}
		info.xyz = &FIELD_2_VERT_BLANKING;
		for (i = 266; i <= 272; i++) {
			mkline (buf, &info, VERT_BLANKING);
			p = vanc_pack (p, buf, elements);
		}
		for (i = 273; i <= 282; i++) {
			mkline (buf, &info, VERT_BLANKING);
			p = vanc_pack (p, buf, elements);
		}
	}
	info.xyz = &FIELD_2_ACTIVE;
	for (i = 283; i <= 445; i++) {
		mkline (buf, &info, pat1);
		p = pack (p, buf, elements);
	}
	for (i = 446; i <= 464; i++) {
		mkline (buf, &info, pat2);
		p = pack (p, buf, elements);
	}
	for (i = 465; i <= 525; i++) {
		mkline (buf, &info, pat3);
		p = pack (p, buf, elements);
	}
	return 0;
}

/**
 * pack_v210 - pack a line of v210 data
 * @outbuf: pointer to the output buffer
 * @inbuf: pointer to the input buffer
 * @count: number of elements in the buffer
 *
 * Returns a pointer to the next output location.
 **/
static uint8_t *
pack_v210 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count)
{
	unsigned short int *inp = inbuf;
	uint8_t *outp = outbuf;

	count = (count / 48) * 48 + ((count % 48) ? 48 : 0);
	while (inp < (inbuf + count)) {
		*outp++ = *inp & 0xff;
		*outp = *inp++ >> 8;
		*outp++ += (*inp << 2) & 0xfc;
		*outp = *inp++ >> 6;
		*outp++ += (*inp << 4) & 0xf0;
		*outp++ = *inp++ >> 4;
	}
	return outp;
}

/**
 * pack_uyvy - pack a line of uyvy data
 * @outbuf: pointer to the output buffer
 * @inbuf: pointer to the input buffer
 * @count: number of elements in the buffer
 *
 * Returns a pointer to the next output location.
 **/
static uint8_t *
pack_uyvy (uint8_t *outbuf,
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

/**
 * pack_v216 - pack a line of v216 data
 * @outbuf: pointer to the output buffer
 * @inbuf: pointer to the input buffer
 * @count: number of elements in the buffer
 *
 * Returns a pointer to the next output location.
 **/
static uint8_t *
pack_v216 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count)
{
	unsigned short int *inp = inbuf;
	uint16_t *outp = (uint16_t *)outbuf;

	while (inp < (inbuf + count)) {
		*outp++ = *inp++ << 6; /* Little endian only */
	}
	return (uint8_t *)outp;
}

int
main (int argc, char **argv)
{
	int opt, frames, avsync_period;
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	struct line_info info;
	char *endptr;
	uint8_t *data, *black_frame, *p;
	size_t samples, framesize, bytes;
	unsigned int black, avsync_sum, n, d;
	int ret;

	avsync_period = 0;
	frames = -1; /* Generate an infinite number of frames */
	pack = pack_uyvy;
	info.blanking = 0;
	while ((opt = getopt (argc, argv, "ahm:n:p:V")) != -1) {
		switch (opt) {
		case 'a':
			info.blanking = 1;
			break;
		case 'h':
			printf ("Usage: %s [OPTION]...\n", argv[0]);
			printf ("Output a SMPTE EG 1 color bar pattern "
				"for transmission by\n"
				"a Linear Systems Ltd. "
				"SMPTE 259M-C board.\n\n");
			printf ("  -a\t\tinclude vertical ancillary space\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -m PERIOD\tswitch between bars and black "
				"every PERIOD seconds\n");
			printf ("  -n NUM\tstop after NUM frames\n");
			printf ("  -p PACKING\tpixel packing\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nPACKING may be:\n"
				"\tuyvy (default)\n"
				"\tv210\n"
				"\traw\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'm':
			avsync_period = strtol (optarg, &endptr, 0);
			if ((*endptr != '\0') ||
				(avsync_period > 35000)) {
				fprintf (stderr,
					"%s: invalid period: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'n':
			frames = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of frames: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'p':
			if (!strcmp (optarg, "v210")) {
				pack = pack_v210;
			} else if (!strcmp (optarg, "uyvy")) {
				pack = pack_uyvy;
			} else if (!strcmp (optarg, "raw")) {
				pack = pack10;
			} else {
				fprintf (stderr,
					"%s: invalid packing: %s\n",
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
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) > 0) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Calculate the frame size */
	samples = ACTIVE_SAMPLES;
	if (info.blanking) {
		if (pack == pack_v210) {
			framesize = samples * 4 / 3 * TOTAL_LINES;
		} else if (pack == pack_uyvy) {
			framesize = samples * PRODUCTION_LINES +
				samples * (TOTAL_LINES - PRODUCTION_LINES) * 2;
		} else {
			framesize = samples * 10 / 8 * TOTAL_LINES;
		}
	} else {
		if (pack == pack_v210) {
			framesize = samples * 4 / 3 * PRODUCTION_LINES;
		} else if (pack == pack_uyvy) {
			framesize = samples * PRODUCTION_LINES;
		} else {
			framesize = samples * 10 / 8 * PRODUCTION_LINES;
		}
	}

	/* Allocate memory */
	data = malloc (framesize);
	if (!data) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		return -1;
	}

	/* Generate a frame */
	mkframe (data,
		info,
		MAIN_SET,
		CHROMA_SET,
		BLACK_SET,
		pack);

	if (avsync_period > 0) {
		/* Allocate memory */
		black_frame = malloc (framesize);
		if (!black_frame) {
			fprintf (stderr, "%s: unable to allocate memory\n",
				argv[0]);
			free (data);
			return -1;
		}

		/* Generate a black frame */
		mkframe (black_frame,
			info,
			VERT_BLANKING,
			VERT_BLANKING,
			VERT_BLANKING,
			pack);
	} else {
		black_frame = NULL;
	}

	black = 0;
	avsync_sum = 0;
	n = 30 * 1000 * avsync_period;
	d = 1001;
	p = data;
	while (frames) {
		/* Output the frame */
		bytes = 0;
		while (bytes < framesize) {
			if ((ret = write (STDOUT_FILENO,
				p + bytes, framesize - bytes)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to write");
				free (data);
				free (black_frame);
				return -1;
			}
			bytes += ret;
		}
		if (avsync_period > 0) {
			if (avsync_sum >= n) {
				avsync_sum = avsync_sum - n + d;
				black = !black;
				p = black ? black_frame : data;
			} else {
				avsync_sum += d;
			}
		}
		if (frames > 0) {
			frames--;
		}
	}
	free (data);
	free (black_frame);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

