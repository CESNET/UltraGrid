/* rp219.c
 *
 * SMPTE RP 219 color bar generator for Linear Systems Ltd. SMPTE 292M boards.
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

#define MAX_SAMPLES_PER_LINE (2*2750)

#define VERT_BLANKING 0
#define PATTERN_1 1
#define PATTERN_2 2
#define PATTERN_3 3
#define PATTERN_4 4
#define DELETED_PACKET 5

struct rp219_bars {
	unsigned int d;
	unsigned int f;
	unsigned int c;
	unsigned int e;
	unsigned int k;
	unsigned int g;
	unsigned int h;
	unsigned int i_side;
	unsigned int i_middle;
	unsigned int j_left;
	unsigned int j_right;
	unsigned int m;
};

static const struct rp219_bars RP219_1080 = {
	.d = 2*240,
	.f = 2*206,
	.c = 2*206,
	.e = 2*204,
	.k = 2*308,
	.g = 2*412,
	.h = 2*170,
	.i_side = 2*68,
	.i_middle = 2*70,
	.j_left = 2*70,
	.j_right = 2*68,
	.m = 2*206
};

static const struct rp219_bars RP219_720 = {
	.d = 2*160,
	.f = 2*136,
	.c = 2*138,
	.e = 2*136,
	.k = 2*206,
	.g = 2*274,
	.h = 2*114,
	.i_side = 2*46,
	.i_middle = 2*46,
	.j_left = 2*46,
	.j_right = 2*46,
	.m = 2*136
};

struct source_format {
	unsigned int lines_per_frame;
	unsigned int active_lines_per_frame;
	unsigned int samples_per_line;
	unsigned int active_samples_per_line;
	unsigned int interlaced;
	unsigned int frame_rate;
	unsigned int m;
	const struct rp219_bars *rp219;
};

static const struct source_format FMT_1080i60 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2200,
	.active_samples_per_line = 2*1920,
	.interlaced = 1,
	.frame_rate = 30,
	.m = 0,
	.rp219 = &RP219_1080,
};

static const struct source_format FMT_1080i59 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2200,
	.active_samples_per_line = 2*1920,
	.interlaced = 1,
	.frame_rate = 30,
	.m = 1,
	.rp219 = &RP219_1080,
};

static const struct source_format FMT_1080i50 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2640,
	.active_samples_per_line = 2*1920,
	.interlaced = 1,
	.frame_rate = 25,
	.m = 0,
	.rp219 = &RP219_1080
};

static const struct source_format FMT_1080p30 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2200,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.frame_rate = 30,
	.m = 0,
	.rp219 = &RP219_1080
};

static const struct source_format FMT_1080p29 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2200,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.frame_rate = 30,
	.m = 1,
	.rp219 = &RP219_1080
};

static const struct source_format FMT_1080p25 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2640,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.frame_rate = 25,
	.m = 0,
	.rp219 = &RP219_1080
};

static const struct source_format FMT_1080p24 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2750,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.frame_rate = 24,
	.m = 0,
	.rp219 = &RP219_1080
};

static const struct source_format FMT_1080p23 = {
	.lines_per_frame = 1125,
	.active_lines_per_frame = 1080,
	.samples_per_line = 2*2750,
	.active_samples_per_line = 2*1920,
	.interlaced = 0,
	.frame_rate = 24,
	.m = 1,
	.rp219 = &RP219_1080
};

static const struct source_format FMT_720p60 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*1650,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 60,
	.m = 0,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p59 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*1650,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 60,
	.m = 1,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p50 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*1980,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 50,
	.m = 0,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p30 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*3300,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 30,
	.m = 0,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p29 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*3300,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 30,
	.m = 1,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p25 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*3960,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 25,
	.m = 0,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p24 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*4125,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 24,
	.m = 0,
	.rp219 = &RP219_720
};

static const struct source_format FMT_720p23 = {
	.lines_per_frame = 750,
	.active_lines_per_frame = 720,
	.samples_per_line = 2*4125,
	.active_samples_per_line = 2*1280,
	.interlaced = 0,
	.frame_rate = 24,
	.m = 1,
	.rp219 = &RP219_720
};

struct format_map {
	const char *name;
	const struct source_format *fmt;
};

static const struct format_map fmt_map[] = {
	{.name = "1080i", .fmt = &FMT_1080i59},
	{.name = "1080p", .fmt = &FMT_1080p29},
	{.name = "720p", .fmt = &FMT_720p59},
	{.name = "1080i60", .fmt = &FMT_1080i60},
	{.name = "1080i59", .fmt = &FMT_1080i59},
	{.name = "1080i50", .fmt = &FMT_1080i50},
	{.name = "1080p30", .fmt = &FMT_1080p30},
	{.name = "1080p29", .fmt = &FMT_1080p29},
	{.name = "1080p25", .fmt = &FMT_1080p25},
	{.name = "1080p24", .fmt = &FMT_1080p24},
	{.name = "1080p23", .fmt = &FMT_1080p23},
	{.name = "720p60", .fmt = &FMT_720p60},
	{.name = "720p59", .fmt = &FMT_720p59},
	{.name = "720p50", .fmt = &FMT_720p50},
	{.name = "720p30", .fmt = &FMT_720p30},
	{.name = "720p29", .fmt = &FMT_720p29},
	{.name = "720p25", .fmt = &FMT_720p25},
	{.name = "720p24", .fmt = &FMT_720p24},
	{.name = "720p23", .fmt = &FMT_720p23}
};

static const unsigned int fmt_count =
	sizeof (fmt_map) / sizeof (struct format_map);

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

struct rp219_opts {
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

static const struct rp219_opts WHITE_75 = {
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

static const struct rp219_opts WHITE_100 = {
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

static const struct rp219_opts PLUS_I = {
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

static const struct rp219_opts MINUS_I = {
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
	const struct rp219_opts *opt;
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
static uint8_t *pack_v216 (uint8_t *outbuf,
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
	unsigned short int *p = buf, *endp, y, sum;
	unsigned int samples = info->fmt->active_samples_per_line;
	double slope;

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
		endp += info->fmt->rp219->d;
		while (p < endp) {
			*p++ = info->opt->star1_cb;
			*p++ = info->opt->star1_y;
			*p++ = info->opt->star1_cr;
			*p++ = info->opt->star1_y;
		}
		/* 75% white */
		endp += info->fmt->rp219->f;
		while (p < endp) {
			*p++ = 512;
			*p++ = 721;
			*p++ = 512;
			*p++ = 721;
		}
		/* 75% yellow */
		endp += info->fmt->rp219->c;
		while (p < endp) {
			*p++ = 176;
			*p++ = 674;
			*p++ = 543;
			*p++ = 674;
		}
		/* 75% cyan */
		endp += info->fmt->rp219->c;
		while (p < endp) {
			*p++ = 589;
			*p++ = 581;
			*p++ = 176;
			*p++ = 581;
		}
		/* 75% green */
		endp += info->fmt->rp219->e;
		while (p < endp) {
			*p++ = 253;
			*p++ = 534;
			*p++ = 207;
			*p++ = 534;
		}
		/* 75% magenta */
		endp += info->fmt->rp219->c;
		while (p < endp) {
			*p++ = 771;
			*p++ = 251;
			*p++ = 817;
			*p++ = 251;
		}
		/* 75% red */
		endp += info->fmt->rp219->c;
		while (p < endp) {
			*p++ = 435;
			*p++ = 204;
			*p++ = 848;
			*p++ = 204;
		}
		/* 75% blue */
		endp += info->fmt->rp219->f;
		while (p < endp) {
			*p++ = 848;
			*p++ = 111;
			*p++ = 481;
			*p++ = 111;
		}
		/* 40% gray */
		while (p < (buf + samples)) {
			*p++ = info->opt->star1_cb;
			*p++ = info->opt->star1_y;
			*p++ = info->opt->star1_cr;
			*p++ = info->opt->star1_y;
		}
		break;
	case PATTERN_2:
		/* 100% cyan */
		endp += info->fmt->rp219->d;
		while (p < endp) {
			*p++ = 615;
			*p++ = 754;
			*p++ = 64;
			*p++ = 754;
		}
		/* 75% white (star 2) */
		endp += info->fmt->rp219->f;
		while (p < endp) {
			*p++ = info->opt->star2_cb;
			*p++ = info->opt->star2_y;
			*p++ = info->opt->star2_cr;
			*p++ = info->opt->star2_y;
		}
		/* 75% white */
		endp = buf + samples - info->fmt->rp219->d;
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
		endp += info->fmt->rp219->d;
		while (p < endp) {
			*p++ = 64;
			*p++ = 877;
			*p++ = 553;
			*p++ = 877;
		}
		/* black (star 3) */
		endp += info->fmt->rp219->f;
		while (p < endp) {
			*p++ = info->opt->star3_cb;
			*p++ = info->opt->star3_y;
			*p++ = info->opt->star3_cr;
			*p++ = info->opt->star3_y;
		}
		/* Y-ramp */
		endp = buf + samples - info->fmt->rp219->d;
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
		endp += info->fmt->rp219->d;
		while (p < endp) {
			*p++ = 512;
			*p++ = 195;
			*p++ = 512;
			*p++ = 195;
		}
		/* black */
		endp += info->fmt->rp219->k;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* 100% white */
		endp += info->fmt->rp219->g;
		while (p < endp) {
			*p++ = 512;
			*p++ = 940;
			*p++ = 512;
			*p++ = 940;
		}
		/* black */
		endp += info->fmt->rp219->h;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* -2% black */
		endp += info->fmt->rp219->i_side;
		while (p < endp) {
			*p++ = 512;
			*p++ = 46;
			*p++ = 512;
			*p++ = 46;
		}
		/* black */
		endp += info->fmt->rp219->i_middle;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* +2% black */
		endp += info->fmt->rp219->i_side;
		while (p < endp) {
			*p++ = 512;
			*p++ = 82;
			*p++ = 512;
			*p++ = 82;
		}
		/* black */
		endp += info->fmt->rp219->j_left;
		while (p < endp) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
		}
		/* +4% black */
		endp += info->fmt->rp219->j_right;
		while (p < endp) {
			*p++ = 512;
			*p++ = 99;
			*p++ = 512;
			*p++ = 99;
		}
		/* black */
		endp += info->fmt->rp219->m;
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
	case DELETED_PACKET:
		/* Ancillary data packet marked for deletion */
		*p++ = 512;
		*p++ = 0;
		*p++ = 512;
		*p++ = 0x3ff;
		*p++ = 512;
		*p++ = 0x3ff;
		*p++ = 512;
		*p++ = 0x180; sum = 0x180 & 0x1ff; /* DID */
		*p++ = 512;
		*p++ = 0x200; sum += 0x200 & 0x1ff;
		*p++ = 512;
		*p++ = 0x203; sum += 0x203 & 0x1ff; /* DC */
		*p++ = 512;
		*p++ = 0x18a; sum += 0x18a & 0x1ff;
		*p++ = 512;
		*p++ = 0x180; sum += 0x180 & 0x1ff;
		*p++ = 512;
		*p++ = 0x180; sum += 0x180 & 0x1ff;
		*p++ = 512;
		*p++ = (sum & 0x1ff) | ((sum & 0x100) ? 0 : 0x200);
		while (p < (buf + samples)) {
			*p++ = 512;
			*p++ = 64;
			*p++ = 512;
			*p++ = 64;
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
 * @pat4: pattern 4
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
	unsigned int pat4,
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count))
{
	uint8_t *(*vanc_pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	unsigned short int buf[MAX_SAMPLES_PER_LINE];
	size_t elements = info.fmt->active_samples_per_line;

	if (pack == pack_v210) {
		vanc_pack = pack_v210;
	} else {
		vanc_pack = pack_v216;
	}
	memset (buf, 0, sizeof (buf));
	if (info.fmt->interlaced) {
		if (info.blanking) {
			info.xyz = &FIELD_1_VERT_BLANKING;
			for (info.ln = 1; info.ln <= 19; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = vanc_pack (p, buf, elements);
			}
			mkline (buf, &info, DELETED_PACKET);
			p = vanc_pack (p, buf, elements);
		}
		info.xyz = &FIELD_1_ACTIVE;
		for (info.ln = 21; info.ln <= 335; info.ln++) {
			mkline (buf, &info, pat1);
			p = pack (p, buf, elements);
		}
		for (info.ln = 336; info.ln <= 380; info.ln++) {
			mkline (buf, &info, pat2);
			p = pack (p, buf, elements);
		}
		for (info.ln = 381; info.ln <= 425; info.ln++) {
			mkline (buf, &info, pat3);
			p = pack (p, buf, elements);
		}
		for (info.ln = 426; info.ln <= 560; info.ln++) {
			mkline (buf, &info, pat4);
			p = pack (p, buf, elements);
		}
		if (info.blanking) {
			info.xyz = &FIELD_1_VERT_BLANKING;
			for (info.ln = 561; info.ln <= 563; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = vanc_pack (p, buf, elements);
			}
			info.xyz = &FIELD_2_VERT_BLANKING;
			for (info.ln = 564; info.ln <= 583; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = vanc_pack (p, buf, elements);
			}
		}
		info.xyz = &FIELD_2_ACTIVE;
		for (info.ln = 584; info.ln <= 898; info.ln++) {
			mkline (buf, &info, pat1);
			p = pack (p, buf, elements);
		}
		for (info.ln = 899; info.ln <= 943; info.ln++) {
			mkline (buf, &info, pat2);
			p = pack (p, buf, elements);
		}
		for (info.ln = 944; info.ln <= 988; info.ln++) {
			mkline (buf, &info, pat3);
			p = pack (p, buf, elements);
		}
		for (info.ln = 989; info.ln <= 1123; info.ln++) {
			mkline (buf, &info, pat4);
			p = pack (p, buf, elements);
		}
		if (info.blanking) {
			info.xyz = &FIELD_2_VERT_BLANKING;
			for (info.ln = 1124; info.ln <= 1125; info.ln++) {
				mkline (buf, &info, VERT_BLANKING);
				p = vanc_pack (p, buf, elements);
			}
		}
	} else {
		if (info.fmt->lines_per_frame == 1125) {
			if (info.blanking) {
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 1; info.ln <= 40; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = vanc_pack (p, buf, elements);
				}
				mkline (buf, &info, DELETED_PACKET);
				p = vanc_pack (p, buf, elements);
			}
			info.xyz = &FIELD_1_ACTIVE;
			for (info.ln = 42; info.ln <= 671; info.ln++) {
				mkline (buf, &info, pat1);
				p = pack (p, buf, elements);
			}
			for (info.ln = 672; info.ln <= 761; info.ln++) {
				mkline (buf, &info, pat2);
				p = pack (p, buf, elements);
			}
			for (info.ln = 762; info.ln <= 851; info.ln++) {
				mkline (buf, &info, pat3);
				p = pack (p, buf, elements);
			}
			for (info.ln = 852; info.ln <= 1121; info.ln++) {
				mkline (buf, &info, pat4);
				p = pack (p, buf, elements);
			}
			if (info.blanking) {
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 1122; info.ln <= 1125; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = vanc_pack (p, buf, elements);
				}
			}
		} else {
			if (info.blanking) {
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 1; info.ln <= 24; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = vanc_pack (p, buf, elements);
				}
				mkline (buf, &info, DELETED_PACKET);
				p = vanc_pack (p, buf, elements);
			}
			info.xyz = &FIELD_1_ACTIVE;
			for (info.ln = 26; info.ln <= 445; info.ln++) {
				mkline (buf, &info, pat1);
				p = pack (p, buf, elements);
			}
			for (info.ln = 446; info.ln <= 505; info.ln++) {
				mkline (buf, &info, pat2);
				p = pack (p, buf, elements);
			}
			for (info.ln = 506; info.ln <= 565; info.ln++) {
				mkline (buf, &info, pat3);
				p = pack (p, buf, elements);
			}
			for (info.ln = 566; info.ln <= 745; info.ln++) {
				mkline (buf, &info, pat4);
				p = pack (p, buf, elements);
			}
			if (info.blanking) {
				info.xyz = &FIELD_1_VERT_BLANKING;
				for (info.ln = 746; info.ln <= 750; info.ln++) {
					mkline (buf, &info, VERT_BLANKING);
					p = vanc_pack (p, buf, elements);
				}
			}
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
	int opt;
	struct line_info info;
	int frames, avsync_period;
	uint8_t *(*pack)(uint8_t *outbuf,
		unsigned short int *inbuf,
		size_t count);
	char *endptr;
	uint8_t *data, *black_frame, *p;
	size_t samples, framesize, bytes;
	unsigned int i, black, avsync_sum, n, d;
	int ret;

	info.blanking = 0;
	info.fmt = &FMT_720p59;
	info.opt = &WHITE_75;
	avsync_period = 0;
	frames = -1; /* Generate an infinite number of frames */
	pack = pack_uyvy;
	while ((opt = getopt (argc, argv, "af:him:n:p:qwV")) != -1) {
		switch (opt) {
		case 'a':
			info.blanking = 1;
			break;
		case 'f':
			for (i = 0; i < fmt_count; i++) {
				if (!strcmp (optarg, fmt_map[i].name)) {
					info.fmt = fmt_map[i].fmt;
					break;
				}
			}
			if (i == fmt_count) {
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
			printf ("  -a\t\tinclude vertical ancillary space\n");
			printf ("  -f FORMAT\tsource format\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -i\t\tselect +I signal in pattern 2\n");
			printf ("  -m PERIOD\tswitch between bars and black "
				"every PERIOD seconds\n");
			printf ("  -n NUM\tstop after NUM frames\n");
			printf ("  -p PACKING\tpixel packing\n");
			printf ("  -q\t\tselect -I signal in pattern 2 "
				"and +Q signal in pattern 3\n");
			printf ("  -w\t\tselect 100%% white signal "
				"in pattern 2\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nFORMAT may be:\n"
				"\t1080i60,\n"
				"\t1080i59 or 1080i,\n"
				"\t1080i50,\n"
				"\t1080p30,\n"
				"\t1080p29 or 1080p,\n"
				"\t1080p25,\n"
				"\t1080p24,\n"
				"\t1080p23,\n"
				"\t720p60,\n"
				"\t720p59 or 720p (default),\n"
				"\t720p50,\n"
				"\t720p30,\n"
				"\t720p29,\n"
				"\t720p25,\n"
				"\t720p24,\n"
				"\t720p23\n"
				"\tThe frame rate is only meaningful\n"
				"\twhen switching between bars and black.\n");
			printf ("\nPACKING may be:\n"
				"\tuyvy (default)\n"
				"\tv210\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'i':
			info.opt = &PLUS_I;
			break;
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
			} else {
				fprintf (stderr,
					"%s: invalid packing: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'q':
			info.opt = &MINUS_I;
			break;
		case 'w':
			info.opt = &WHITE_100;
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

	/* Calculate the frame size */
	if (info.blanking) {
		if (pack == pack_v210) {
			samples = (info.fmt->active_samples_per_line / 96 * 48) +
				((info.fmt->active_samples_per_line % 96) ? 48 : 0);
			framesize = samples *
				info.fmt->lines_per_frame * 8 / 3;
		} else {
			framesize = info.fmt->active_samples_per_line *
				info.fmt->active_lines_per_frame +
				info.fmt->active_samples_per_line *
				(info.fmt->lines_per_frame -
				info.fmt->active_lines_per_frame) * 2;
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

	/* Allocate memory */
	data = malloc (framesize);
	if (!data) {
		fprintf (stderr, "%s: unable to allocate memory\n", argv[0]);
		return -1;
	}

	/* Generate a frame */
	mkframe (data,
		info,
		PATTERN_1,
		PATTERN_2,
		PATTERN_3,
		PATTERN_4,
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
			VERT_BLANKING,
			pack);
	} else {
		black_frame = NULL;
	}

	black = 0;
	avsync_sum = 0;
	n = info.fmt->frame_rate * 1000 * avsync_period;
	d = info.fmt->m ? 1001 : 1000;
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

