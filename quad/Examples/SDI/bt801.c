/* bt801.c
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

#define TOTAL_SAMPLES 1728
#define TOTAL_LINES 625

static const char progname[] = "bt801";

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

unsigned short int y[720];
unsigned short int cr[360];
unsigned short int cb[360];

/* Static function prototypes */
static int mkline (unsigned short int *buf,
	const struct trs *info);
static uint8_t *pack8 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);
static uint8_t *pack10 (uint8_t *outbuf,
	unsigned short int *inbuf,
	size_t count);

/**
 * mkline - generate one line
 * @buf: pointer to a buffer
 * @info: pointer to a TRS structure
 * 
 * Returns a negative error code on failure and zero on success.
 **/
static int
mkline (unsigned short int *buf,
	const struct trs *info)
{
	unsigned short int *p = buf;
	unsigned short int *py = y, *pcr = cr, *pcb = cb;

	/* EAV */
	*p++ = 0x3ff;
	*p++ = 0x000;
	*p++ = 0x000;
	*p++ = info->eav;
	/* Horizontal blanking */
	while (p < (buf + 284)) {
		*p++ = 0x200;
		*p++ = 0x040;
		*p++ = 0x200;
		*p++ = 0x040;
	}
	/* SAV */
	*p++ = 0x3ff;
	*p++ = 0x000;
	*p++ = 0x000;
	*p++ = info->sav;
	/* Active region */
	if ((info == &FIELD_1_VERT_BLANKING) ||
		(info == &FIELD_2_VERT_BLANKING)) {
		while (p < (buf + TOTAL_SAMPLES)) {
			*p++ = 0x200;
			*p++ = 0x040;
			*p++ = 0x200;
			*p++ = 0x040;
		}
	} else {
		while (p < (buf + TOTAL_SAMPLES)) {
			*p++ = *pcb++ << 2;
			*p++ = *py++ << 2;
			*p++ = *pcr++ << 2;
			*p++ = *py++ << 2;
		}
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
			printf ("Output an ITU-R BT.801-1 625-line, "
				"50 field/s 100/0/75/0 colour bar pattern\n"
				"for transmission by "
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
			printf ("\nCopyright (C) 2008-2010 "
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

	/* Allocate memory */
	data = malloc (framesize);
	if (!data) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		return -1;
	}

	/* Generate a frame */
	memset (buf, 0, sizeof (buf));
	p = data;
	for (i = 1; i <= 22; i++) {
		mkline (buf, &FIELD_1_VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 23; i <= 310; i++) {
		mkline (buf, &FIELD_1_ACTIVE);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 311; i <= 312; i++) {
		mkline (buf, &FIELD_1_VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 313; i <= 335; i++) {
		mkline (buf, &FIELD_2_VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 336; i <= 623; i++) {
		mkline (buf, &FIELD_2_ACTIVE);
		p = pack (p, buf, TOTAL_SAMPLES);
	}
	for (i = 624; i <= 625; i++) {
		mkline (buf, &FIELD_2_VERT_BLANKING);
		p = pack (p, buf, TOTAL_SAMPLES);
	}

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

