/* bt801video.c
 *
 * ITU-R BT.801-1 625-line, 50 field/s, 100/0/75/0 colour bar generator for
 * Linear Systems Ltd. SMPTE 259M-C boards.
 *
 * Copyright (C) 2008-2010 Linear Systems Ltd. All rights reserved.
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
#include <string.h>

#include "master.h"

#if __BYTE_ORDER == __BIG_ENDIAN
#error Big endian architecture not supported
#endif

#define TOTAL_SAMPLES 1728
#define ACTIVE_SAMPLES 1440
#define TOTAL_LINES 625
#define ACTIVE_LINES 576

static const char progname[] = "bt801video";

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

unsigned short int y[720];
unsigned short int cr[360];
unsigned short int cb[360];

/* Static function prototypes */
static int mkline (unsigned short int *buf,
	const struct line_info *info,
	int black);
static int mkframe (uint8_t *p,
	struct line_info info,
	int black,
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count));
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

/**
 * mkline - generate one line
 * @buf: pointer to a buffer
 * @info: pointer to a line information structure
 * @black: black frame flag
 *
 * Returns a negative error code on failure and zero on success.
 **/
static int
mkline (unsigned short int *buf,
	const struct line_info *info,
	int black)
{
	unsigned short int *p = buf;
	unsigned short int *py = y, *pcr = cr, *pcb = cb, sum;
	unsigned int samples = ACTIVE_SAMPLES;

	if (black) {
		while (p < (buf + samples)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
	} else if ((info->xyz == &FIELD_1_VERT_BLANKING) ||
		(info->xyz == &FIELD_2_VERT_BLANKING)) {
		/* Ancillary data packet marked for deletion */
		*p++ = 0;
		*p++ = 0x3ff;
		*p++ = 0x3ff;
		*p++ = 0x180; sum = 0x180 & 0x1ff; /* DID */
		*p++ = 0x200; sum += 0x200 & 0x1ff;
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
	} else {
		while (p < (buf + samples)) {
			*p++ = *pcb++ << 2;
			*p++ = *py++ << 2;
			*p++ = *pcr++ << 2;
			*p++ = *py++ << 2;
		}
	}
	return 0;
}

/**
 * mkframe - generate one frame
 * @p: pointer to a buffer
 * @info: line information structure
 * @black: black frame flag
 * @pack: pointer to packing function
 *
 * Returns a negative error code on failure and zero on success.
 **/
static int
mkframe (uint8_t *p,
	struct line_info info,
	int black,
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count))
{
	uint8_t *(*vanc_pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	unsigned short int buf[TOTAL_SAMPLES];
	size_t elements = ACTIVE_SAMPLES;
	unsigned int i;

	if (pack == pack_v210) {
		vanc_pack = pack_v210;
	} else {
		vanc_pack = pack_v216;
	}
	memset (buf, 0, sizeof (buf));
	if (info.blanking) {
		info.xyz = &FIELD_1_VERT_BLANKING;
		for (i = 1; i <= 21; i++) {
			mkline (buf, &info, 1);
			p = vanc_pack (p, buf, elements);
		}
		mkline (buf, &info, 0);
		p = vanc_pack (p, buf, elements);
	}
	info.xyz = &FIELD_1_ACTIVE;
	for (i = 23; i <= 310; i++) {
		mkline (buf, &info, black);
		p = pack (p, buf, elements);
	}
	if (info.blanking) {
		info.xyz = &FIELD_1_VERT_BLANKING;
		for (i = 311; i <= 312; i++) {
			mkline (buf, &info, 1);
			p = vanc_pack (p, buf, elements);
		}
		info.xyz = &FIELD_2_VERT_BLANKING;
		for (i = 313; i <= 335; i++) {
			mkline (buf, &info, 1);
			p = vanc_pack (p, buf, elements);
		}
	}
	info.xyz = &FIELD_2_ACTIVE;
	for (i = 336; i <= 623; i++) {
		mkline (buf, &info, black);
		p = pack (p, buf, elements);
	}
	if (info.blanking) {
		info.xyz = &FIELD_2_VERT_BLANKING;
		for (i = 624; i <= 625; i++) {
			mkline (buf, &info, 1);
			p = vanc_pack (p, buf, elements);
		}
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
	size_t framesize, bytes;
	unsigned int black, avsync_sum, n, d;
	int i, ret;

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
			printf ("Output an ITU-R BT.801-1 625-line, "
				"50 field/s 100/0/75/0 colour bar pattern\n"
				"for transmission by "
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
			printf ("\nCopyright (C) 2008-2010 "
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

	/* Initialize the active line arrays */
	for (i = 0; i <= 14; i++) y[i] = 16;
	y[15] = 39;
	y[16] = 126;
	y[17] = 212;
	for (i = 18; i <= 100; i++) y[i] = 235;
	y[101] = 227;
	y[102] = 198;
	y[103] = 169;
	for (i = 104; i <= 185; i++) y[i] = 162;
	y[186] = 161;
	y[187] = 158;
	y[188] = 146;
	y[189] = 134;
	for (i = 190; i <= 272; i++) y[i] = 131;
	y[273] = 129;
	y[274] = 122;
	y[275] = 114;
	for (i = 276; i <= 358; i++) y[i] = 112;
	y[359] = 109;
	y[360] = 98;
	y[361] = 87;
	for (i = 362; i <= 444; i++) y[i] = 84;
	y[445] = 82;
	y[446] = 74;
	y[447] = 67;
	for (i = 448; i <= 530; i++) y[i] = 65;
	y[531] = 62;
	y[532] = 50;
	y[533] = 38;
	for (i = 534; i <= 616; i++) y[i] = 35;
	y[617] = 33;
	y[618] = 25;
	y[619] = 18;
	for (i = 620; i <= 719; i++) y[i] = 16;
	for (i = 0; i <= 49; i++) cr[i] = 128;
	cr[50] = 129;
	cr[51] = 135;
	cr[52] = 140;
	for (i = 53; i <= 91; i++) cr[i] = 142;
	cr[92] = 141;
	cr[93] = 132;
	cr[94] = 93;
	cr[95] = 54;
	for (i = 96; i <= 135; i++) cr[i] = 44;
	cr[136] = 45;
	cr[137] = 51;
	cr[138] = 56;
	for (i = 139; i <= 178; i++) cr[i] = 58;
	cr[179] = 72;
	cr[180] = 128;
	cr[181] = 184;
	for (i = 182; i <= 221; i++) cr[i] = 198;
	cr[222] = 200;
	cr[223] = 205;
	cr[224] = 211;
	for (i = 225; i <= 264; i++) cr[i] = 212;
	cr[265] = 202;
	cr[266] = 163;
	cr[267] = 124;
	cr[268] = 115;
	for (i = 269; i <= 307; i++) cr[i] = 114;
	cr[308] = 116;
	cr[309] = 121;
	cr[310] = 127;
	for (i = 311; i <= 359; i++) cr[i] = 128;
	for (i = 0; i <= 49; i++) cb[i] = 128;
	cb[50] = 119;
	cb[51] = 86;
	cb[52] = 53;
	for (i = 53; i <= 92; i++) cb[i] = 44;
	cb[93] = 56;
	cb[94] = 100;
	cb[95] = 145;
	for (i = 96; i <= 135; i++) cb[i] = 156;
	cb[136] = 148;
	cb[137] = 114;
	cb[138] = 81;
	cb[139] = 73;
	for (i = 140; i <= 177; i++) cb[i] = 72;
	cb[178] = 73;
	cb[179] = 84;
	cb[180] = 128;
	cb[181] = 172;
	cb[182] = 183;
	for (i = 183; i <= 220; i++) cb[i] = 184;
	cb[221] = 183;
	cb[222] = 175;
	cb[223] = 142;
	cb[224] = 108;
	for (i = 225; i <= 264; i++) cb[i] = 100;
	cb[265] = 111;
	cb[266] = 156;
	cb[267] = 200;
	for (i = 268; i <= 307; i++) cb[i] = 212;
	cb[308] = 203;
	cb[309] = 170;
	cb[310] = 137;
	for (i = 311; i <= 359; i++) cb[i] = 128;

	/* Calculate the frame size */
	if (info.blanking) {
		if (pack == pack_v210) {
			framesize = ACTIVE_SAMPLES * 4 / 3 * TOTAL_LINES;
		} else if (pack == pack_uyvy) {
			framesize = ACTIVE_SAMPLES * ACTIVE_LINES +
				ACTIVE_SAMPLES * (TOTAL_LINES - ACTIVE_LINES) * 2;
		} else {
			framesize = ACTIVE_SAMPLES * 10 / 8 * TOTAL_LINES;
		}
	} else {
		if (pack == pack_v210) {
			framesize = ACTIVE_SAMPLES * 4 / 3 * ACTIVE_LINES;
		} else if (pack == pack_uyvy) {
			framesize = ACTIVE_SAMPLES * ACTIVE_LINES;
		} else {
			framesize = ACTIVE_SAMPLES * 10 / 8 * ACTIVE_LINES;
		}
	}

	/* Allocate memory */
	data = malloc (framesize);
	if (!data) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		return -1;
	}

	/* Generate a frame */
	mkframe (data, info, 0, pack);

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
		mkframe (black_frame, info, 1, pack);
	} else {
		black_frame = NULL;
	}

	black = 0;
	avsync_sum = 0;
	n = 25 * avsync_period;
	d = 1;
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

