/* rp219.c
 *
 * SMPTE RP 219 color bar generator for Linear Systems Ltd. SMPTE 292M boards.
 *
 * Copyright (C) 2008-2009 Linear Systems Ltd. All rights reserved.
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

#define MAX_SAMPLES_PER_LINE (2*2750)
#define MAX_LINES_PER_FRAME 1125

#define VERT_BLANKING 0
#define PATTERN_1 1
#define PATTERN_2 2
#define PATTERN_3 3
#define PATTERN_4 4

struct source_format {
	unsigned int lines_per_frame;
	unsigned int active_lines_per_frame;
	unsigned int samples_per_line;
	unsigned int active_samples_per_line;
	unsigned int interlaced;
	unsigned int rp219_d;
	unsigned int rp219_f;
	unsigned int rp219_c;
	unsigned int rp219_e;
	unsigned int rp219_k;
	unsigned int rp219_g;
	unsigned int rp219_h;
	unsigned int rp219_i_side;
	unsigned int rp219_i_middle;
	unsigned int rp219_j_left;
	unsigned int rp219_j_right;
	unsigned int rp219_m;
};

const struct source_format FMT_1080i60 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2200,
	.active_samples_per_line = 2*1920,
	.interlaced = 1,
	.rp219_d = 2*240,
	.rp219_f = 2*206,
	.rp219_c = 2*206,
	.rp219_e = 2*204,
	.rp219_k = 2*308,
	.rp219_g = 2*412,
	.rp219_h = 2*170,
	.rp219_i_side = 2*68,
	.rp219_i_middle = 2*70,
	.rp219_j_left = 2*70,
	.rp219_j_right = 2*68,
	.rp219_m = 2*206
};

const struct source_format FMT_1080i50 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2640,
	.active_samples_per_line = 2*1920,
	.interlaced = 1,
	.rp219_d = 2*240,
	.rp219_f = 2*206,
	.rp219_c = 2*206,
	.rp219_e = 2*204,
	.rp219_k = 2*308,
	.rp219_g = 2*412,
	.rp219_h = 2*170,
	.rp219_i_side = 2*68,
	.rp219_i_middle = 2*70,
	.rp219_j_left = 2*70,
	.rp219_j_right = 2*68,
	.rp219_m = 2*206
};

const struct source_format FMT_1080p30 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2200,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.rp219_d = 2*240,
	.rp219_f = 2*206,
	.rp219_c = 2*206,
	.rp219_e = 2*204,
	.rp219_k = 2*308,
	.rp219_g = 2*412,
	.rp219_h = 2*170,
	.rp219_i_side = 2*68,
	.rp219_i_middle = 2*70,
	.rp219_j_left = 2*70,
	.rp219_j_right = 2*68,
	.rp219_m = 2*206
};

const struct source_format FMT_1080p25 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2640,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.rp219_d = 2*240,
	.rp219_f = 2*206,
	.rp219_c = 2*206,
	.rp219_e = 2*204,
	.rp219_k = 2*308,
	.rp219_g = 2*412,
	.rp219_h = 2*170,
	.rp219_i_side = 2*68,
	.rp219_i_middle = 2*70,
	.rp219_j_left = 2*70,
	.rp219_j_right = 2*68,
	.rp219_m = 2*206
};

const struct source_format FMT_1080p24 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2750,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.rp219_d = 2*240,
	.rp219_f = 2*206,
	.rp219_c = 2*206,
	.rp219_e = 2*204,
	.rp219_k = 2*308,
	.rp219_g = 2*412,
	.rp219_h = 2*170,
	.rp219_i_side = 2*68,
	.rp219_i_middle = 2*70,
	.rp219_j_left = 2*70,
	.rp219_j_right = 2*68,
	.rp219_m = 2*206
};

const struct source_format FMT_720p60 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*1650,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.rp219_d = 2*160,
	.rp219_f = 2*136,
	.rp219_c = 2*138,
	.rp219_e = 2*136,
	.rp219_k = 2*206,
	.rp219_g = 2*274,
	.rp219_h = 2*114,
	.rp219_i_side = 2*46,
	.rp219_i_middle = 2*46,
	.rp219_j_left = 2*46,
	.rp219_j_right = 2*46,
	.rp219_m = 2*136
};

struct trs {
	unsigned short int sav;
	unsigned short int eav;
};

const struct trs FIELD_1_ACTIVE = {
	.sav = 0x200,
	.eav = 0x274
};

const struct trs FIELD_1_VERT_BLANKING = {
	.sav = 0x2ac,
	.eav = 0x2d8
};

const struct trs FIELD_2_ACTIVE = {
	.sav = 0x31c,
	.eav = 0x368
};

const struct trs FIELD_2_VERT_BLANKING = {
	.sav = 0x3b0,
	.eav = 0x3c4
};

struct rp219_parameters {
	unsigned short int star1_y;
	unsigned short int star1_cb;
	unsigned short int star1_cr;
	unsigned short int star2_y;
	unsigned short int star2_cb;
	unsigned short int star2_cr;
	unsigned short int star3_y;
	unsigned short int star3_cb;
	unsigned short int star3_cr;
};

const struct rp219_parameters WHITE_75 = {
	.star1_y = 414,
	.star1_cb = 512,
	.star1_cr = 512,
	.star2_y = 721,
	.star2_cb = 512,
	.star2_cr = 512,
	.star3_y = 64,
	.star3_cb = 512,
	.star3_cr = 512
};

const struct rp219_parameters WHITE_100 = {
	.star1_y = 414,
	.star1_cb = 512,
	.star1_cr = 512,
	.star2_y = 940,
	.star2_cb = 512,
	.star2_cr = 512,
	.star3_y = 64,
	.star3_cb = 512,
	.star3_cr = 512
};

const struct rp219_parameters PLUS_I = {
	.star1_y = 414,
	.star1_cb = 512,
	.star1_cr = 512,
	.star2_y = 245,
	.star2_cb = 412,
	.star2_cr = 629,
	.star3_y = 64,
	.star3_cb = 512,
	.star3_cr = 512
};

const struct rp219_parameters MINUS_I = {
	.star1_y = 414,
	.star1_cb = 512,
	.star1_cr = 512,
	.star2_y = 244,
	.star2_cb = 612,
	.star2_cr = 395,
	.star3_y = 141,
	.star3_cb = 697,
	.star3_cr = 606
};

struct line_info {
	const struct source_format *fmt;
	unsigned int ln;
	const struct trs *xyz;
	const struct rp219_parameters *rp219;
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

static const char progname[] = "rp219";

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
	const struct line_info *info,
	unsigned int pattern)
{
	unsigned short int *p = buf, *endp, ln, y;
	unsigned int samples = info->blanking ?
		info->fmt->samples_per_line :
		info->fmt->active_samples_per_line;
	double slope;

	if (info->blanking) {
		/* EAV */
		*p++ = 1023;
		*p++ = 1023;
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
		*p++ = info->xyz->eav;
		*p++ = info->xyz->eav;
		/* LN */
		ln = ((info->ln & 0x07f) << 2) | (~info->ln & 0x040) << 3;
		*p++ = ln;
		*p++ = ln;
		ln = ((info->ln & 0x780) >> 5) | 0x200;
		*p++ = ln;
		*p++ = ln;
		/* CRC, added by serializer */
		*p++ = 512;
		*p++ = 64;
		*p++ = 512;
		*p++ = 64;
		/* Horizontal blanking */
		while (p < (buf + info->fmt->samples_per_line -
			info->fmt->active_samples_per_line - 8)) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* SAV */
		*p++ = 1023;
		*p++ = 1023;
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
		*p++ = info->xyz->sav;
		*p++ = info->xyz->sav;
	}
	/* Active region */
	endp = p;
	switch (pattern) {
	default:
	case VERT_BLANKING:
		while (p < (buf + samples)) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		break;
	case PATTERN_1:
		/* 40% gray (star 1) */
		endp += info->fmt->rp219_d;
		while (p < endp) {
			*p++ = info->rp219->star1_cb;
			*p++ = info->rp219->star1_y;
			*p++ = info->rp219->star1_cr;
			*p++ = info->rp219->star1_y;
		}
		/* 75% white */
		endp += info->fmt->rp219_f;
		while (p < endp) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		/* 75% yellow */
		endp += info->fmt->rp219_c;
		while (p < endp) {
			*p++ = 176;
			*p++ = 674;
			*p++ = 543;
			*p++ = 674;
		}
		/* 75% cyan */
		endp += info->fmt->rp219_c;
		while (p < endp) {
			*p++ = 589;
			*p++ = 581;
			*p++ = 176;
			*p++ = 581;
		}
		/* 75% green */
		endp += info->fmt->rp219_e;
		while (p < endp) {
			*p++ = 253;
			*p++ = 534;
			*p++ = 207;
			*p++ = 534;
		}
		/* 75% magenta */
		endp += info->fmt->rp219_c;
		while (p < endp) {
			*p++ = 771;
			*p++ = 251;
			*p++ = 817;
			*p++ = 251;
		}
		/* 75% red */
		endp += info->fmt->rp219_c;
		while (p < endp) {
			*p++ = 435;
			*p++ = 204;
			*p++ = 848;
			*p++ = 204;
		}
		/* 75% blue */
		endp += info->fmt->rp219_f;
		while (p < endp) {
			*p++ = 848;
			*p++ = 111;
			*p++ = 481;
			*p++ = 111;
		}
		/* 40% gray */
		while (p < (buf + samples)) {
			*p++ = info->rp219->star1_cb;
			*p++ = info->rp219->star1_y;
			*p++ = info->rp219->star1_cr;
			*p++ = info->rp219->star1_y;
		}
		break;
	case PATTERN_2:
		/* 100% cyan */
		endp += info->fmt->rp219_d;
		while (p < endp) {
			*p++ = 615;
			*p++ = 754;
			*p++ = 64;
			*p++ = 754;
		}
		/* 75% white (star 2) */
		endp += info->fmt->rp219_f;
		while (p < endp) {
			*p++ = info->rp219->star2_cb;
			*p++ = info->rp219->star2_y;
			*p++ = info->rp219->star2_cr;
			*p++ = info->rp219->star2_y;
		}
		/* 75% white */
		endp = buf + samples - info->fmt->rp219_d;
		while (p < endp) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		/* 100% blue */
		while (p < (buf + samples)) {
			*p++ = 960;
			*p++ = 127;
			*p++ = 471;
			*p++ = 127;
		}
		break;
	case PATTERN_3:
		/* 100% yellow */
		endp += info->fmt->rp219_d;
		while (p < endp) {
			*p++ = 64;
			*p++ = 877;
			*p++ = 553;
			*p++ = 877;
		}
		/* black (star 3) */
		endp += info->fmt->rp219_f;
		while (p < endp) {
			*p++ = info->rp219->star3_cb;
			*p++ = info->rp219->star3_y;
			*p++ = info->rp219->star3_cr;
			*p++ = info->rp219->star3_y;
		}
		/* Y-ramp */
		endp = buf + samples - info->fmt->rp219_d;
		slope = (double)(940 - 64) / (endp - p);
		while (p < endp) {
			*p++ = 512;
			y = 940 - (int)(slope * (endp - p) + 0.5);
			*p++ = y;
		}
		/* 100% red */
		while (p < (buf + samples)) {
			*p++ = 409;
			*p++ = 250;
			*p++ = 960;
			*p++ = 250;
		}
		break;
	case PATTERN_4:
		/* 15% gray */
		endp += info->fmt->rp219_d;
		while (p < endp) {
			*p++ = 512;
			*p++ = 195;
			*p++ = 512;
			*p++ = 195;
		}
		/* black */
		endp += info->fmt->rp219_k;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* 100% white */
		endp += info->fmt->rp219_g;
		while (p < endp) {
			*p++ = 512;
			*p++ = 940;
			*p++ = 512;
			*p++ = 940;
		}
		/* black */
		endp += info->fmt->rp219_h;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* -2% black */
		endp += info->fmt->rp219_i_side;
		while (p < endp) {
			*p++ = 512;
			*p++ = 46;
			*p++ = 512;
			*p++ = 46;
		}
		/* black */
		endp += info->fmt->rp219_i_middle;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* +2% black */
		endp += info->fmt->rp219_i_side;
		while (p < endp) {
			*p++ = 512;
			*p++ = 82;
			*p++ = 512;
			*p++ = 82;
		}
		/* black */
		endp += info->fmt->rp219_j_left;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* +4% black */
		endp += info->fmt->rp219_j_right;
		while (p < endp) {
			*p++ = 512;
			*p++ = 99;
			*p++ = 512;
			*p++ = 99;
		}
		/* black */
		endp += info->fmt->rp219_m;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* 15% gray */
		while (p < (buf + samples)) {
			*p++ = 512;
			*p++ = 195;
			*p++ = 512;
			*p++ = 195;
		}
		break;
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

	count = (count / 96) * 96 + ((count % 96) ? 96 : 0);
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

int
main (int argc, char **argv)
{
	int opt;
	struct line_info info;
	int frames;
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	char *endptr;
	unsigned short int buf[MAX_SAMPLES_PER_LINE];
	uint8_t data[MAX_SAMPLES_PER_LINE*10/8*MAX_LINES_PER_FRAME], *p = data;
	size_t elements, framesize, bytes;
	unsigned int samples;
	int ret;

	info.fmt = &FMT_720p60;
	info.rp219 = &WHITE_75;
	info.blanking = 0;
	frames = -1; /* Generate an infinite number of frames */
	pack = pack_uyvy;
	while ((opt = getopt (argc, argv, "f:hin:p:qwV")) != -1) {
		switch (opt) {
		case 'f':
			if (!strcmp (optarg, "1080i")) {
				info.fmt = &FMT_1080i60;
			} else if (!strcmp (optarg, "1080p")) {
				info.fmt = &FMT_1080p30;
			} else if (!strcmp (optarg, "720p")) {
				info.fmt = &FMT_720p60;
			} else {
				fprintf (stderr,
					"%s: invalid source format: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]...\n", argv[0]);
			printf ("Output a SMPTE RP 219 color bar pattern "
				"for transmission by\n"
				"a Linear Systems Ltd. "
				"SMPTE 292M board.\n\n");
			printf ("  -f FORMAT\tselect source format\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -i\t\tselect +I signal in pattern 2\n");
			printf ("  -n NUM\tstop after NUM frames\n");
			printf ("  -p PACKING\tpixel packing\n");
			printf ("  -q\t\tselect -I signal in pattern 2 "
				"and +Q signal in pattern 3\n");
			printf ("  -w\t\tselect 100%% white signal "
				"in pattern 2\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nFORMAT may be:\n"
				"\t720p (default)\n"
				"\t1080i\n"
				"\t1080p\n");
			printf ("\nPACKING may be:\n"
				"\tuyvy (default)\n"
				"\tv210\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'i':
			info.rp219 = &PLUS_I;
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
			} else {
				fprintf (stderr,
					"%s: invalid packing: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'q':
			info.rp219 = &MINUS_I;
			break;
		case 'w':
			info.rp219 = &WHITE_100;
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2008-2009 "
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

	/* Generate a frame */
	if (info.fmt->interlaced) {
		if (info.blanking) {
			elements = info.fmt->samples_per_line;
			info.xyz = &FIELD_1_VERT_BLANKING;
			for (info.ln = 1; info.ln <= 20; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = pack (p, buf, elements);
			}
		} else {
			elements = info.fmt->active_samples_per_line;
		}
		info.xyz = &FIELD_1_ACTIVE;
		for (info.ln = 21; info.ln <= 335; info.ln++) {
			mkline (buf, &info, PATTERN_1);
			p = pack (p, buf, elements);
		}
		for (info.ln = 336; info.ln <= 380; info.ln++) {
			mkline (buf, &info, PATTERN_2);
			p = pack (p, buf, elements);
		}
		for (info.ln = 381; info.ln <= 425; info.ln++) {
			mkline (buf, &info, PATTERN_3);
			p = pack (p, buf, elements);
		}
		for (info.ln = 426; info.ln <= 560; info.ln++) {
			mkline (buf, &info, PATTERN_4);
			p = pack (p, buf, elements);
		}
		if (info.blanking) {
			info.xyz = &FIELD_1_VERT_BLANKING;
			for (info.ln = 561; info.ln <= 563; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = pack (p, buf, elements);
			}
			info.xyz = &FIELD_2_VERT_BLANKING;
			for (info.ln = 564; info.ln <= 583; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = pack (p, buf, elements);
			}
		}
		info.xyz = &FIELD_2_ACTIVE;
		for (info.ln = 584; info.ln <= 898; info.ln++) {
			mkline (buf, &info, PATTERN_1);
			p = pack (p, buf, elements);
		}
		for (info.ln = 899; info.ln <= 943; info.ln++) {
			mkline (buf, &info, PATTERN_2);
			p = pack (p, buf, elements);
		}
		for (info.ln = 944; info.ln <= 988; info.ln++) {
			mkline (buf, &info, PATTERN_3);
			p = pack (p, buf, elements);
		}
		for (info.ln = 989; info.ln <= 1123; info.ln++) {
			mkline (buf, &info, PATTERN_4);
			p = pack (p, buf, elements);
		}
		if (info.blanking) {
			info.xyz = &FIELD_2_VERT_BLANKING;
			for (info.ln = 1124; info.ln <= 1125; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = pack (p, buf, elements);
			}
		}
	} else {
		if (info.fmt->lines_per_frame == 1125) {
			if (info.blanking) {
				elements = info.fmt->samples_per_line;
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 1; info.ln <= 41; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = pack (p, buf, elements);
				}
			} else {
				elements = info.fmt->active_samples_per_line;
			}
			info.xyz = &FIELD_1_ACTIVE;
			for (info.ln = 42; info.ln <= 671; info.ln++) {
				mkline (buf, &info, PATTERN_1);
				p = pack (p, buf, elements);
			}
			for (info.ln = 672; info.ln <= 761; info.ln++) {
				mkline (buf, &info, PATTERN_2);
				p = pack (p, buf, elements);
			}
			for (info.ln = 762; info.ln <= 851; info.ln++) {
				mkline (buf, &info, PATTERN_3);
				p = pack (p, buf, elements);
			}
			for (info.ln = 852; info.ln <= 1121; info.ln++) {
				mkline (buf, &info, PATTERN_4);
				p = pack (p, buf, elements);
			}
			if (info.blanking) {
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 1122; info.ln <= 1125; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = pack (p, buf, elements);
				}
			}
		} else {
			if (info.blanking) {
				elements = info.fmt->samples_per_line;
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 1; info.ln <= 25; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = pack (p, buf, elements);
				}
			} else {
				elements = info.fmt->active_samples_per_line;
			}
			info.xyz = &FIELD_1_ACTIVE;
			for (info.ln = 26; info.ln <= 445; info.ln++) {
				mkline (buf, &info, PATTERN_1);
				p = pack (p, buf, elements);
			}
			for (info.ln = 446; info.ln <= 505; info.ln++) {
				mkline (buf, &info, PATTERN_2);
				p = pack (p, buf, elements);
			}
			for (info.ln = 506; info.ln <= 565; info.ln++) {
				mkline (buf, &info, PATTERN_3);
				p = pack (p, buf, elements);
			}
			for (info.ln = 566; info.ln <= 745; info.ln++) {
				mkline (buf, &info, PATTERN_4);
				p = pack (p, buf, elements);
			}
			if (info.blanking) {
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 746; info.ln <= 750; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = pack (p, buf, elements);
				}
			}
		}
	}

	if (info.blanking) {
		if (pack == pack_v210) {
			samples = (info.fmt->samples_per_line / 96 * 48) +
				((info.fmt->samples_per_line % 96) ? 48 : 0);
			framesize = samples *
				info.fmt->lines_per_frame * 8 / 3;
		} else {
			framesize = info.fmt->samples_per_line *
				info.fmt->lines_per_frame;
		}
	} else {
		if (pack == pack_v210) {
			samples = (info.fmt->active_samples_per_line / 96 * 48) +
				((info.fmt->active_samples_per_line % 96) ? 48 : 0);
			framesize = samples *
				info.fmt->active_lines_per_frame * 8 / 3;
		} else {
			framesize = info.fmt->active_samples_per_line *
				info.fmt->active_lines_per_frame;
		}
	}
	while (frames) {
		/* Output the frame */
		bytes = 0;
		while (bytes < framesize) {
			if ((ret = write (STDOUT_FILENO,
				data + bytes, framesize - bytes)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to write");
				return -1;
			}
			bytes += ret;
		}
		if (frames > 0) {
			frames--;
		}
	}
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

