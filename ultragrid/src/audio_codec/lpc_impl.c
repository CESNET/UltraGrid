/*
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * This code was originally pulled from VAT source.  Berkeley license 
 * removed since code did not originate at Berkeley/LBL.
 *
 * Main modifications are:
 * - Elimination static variable declarations that cause state overlap
 *   problems when coding/decoding multiple streams simultaneously.
 * - Array bound overrun fixes.
 * - Assorted complier warning fixes.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#include "config.h"
#include "config_win32.h"
#include "config_unix.h"
#include "debug.h"
#include "audio_codec/lpc_impl.h"

#define MAXWINDOW	1000	/* Max analysis window length */
#define FS		8000.0f	/* Sampling rate */

#define DOWN		5	/* Decimation for pitch analyzer */
#define PITCHORDER	4	/* Model order for pitch analyzer */
#define FC		600.0f	/* Pitch analyzer filter cutoff */
#define MINPIT		50.0	/* Minimum pitch */
#define MAXPIT		300.0	/* Maximum pitch */

#define MINPER		(int)(FS/(DOWN*MAXPIT)+.5)	/* Minimum period */
#define MAXPER		(int)(FS/(DOWN*MINPIT)+.5)	/* Maximum period */

#define WSCALE		1.5863	/* Energy loss due to windowing */

#define BUFLEN		((FRAMESIZE * 3) / 2)

static float h[MAXWINDOW];
static float fa[6];

#define BIAS 0x84		/* define the add-in bias for 16 bit samples */
#define CLIP 32635

static void
auto_correl(float *w, int n, int p, float *r)
{
	int i, k, nk;

	for (k = 0; k <= p; k++) {
		nk = n - k;
		r[k] = 0.0f;
		for (i = 0; i < nk; i++) 
			r[k] += w[i] * w[i + k];
	}
}

static void
durbin(float *r, int p, float *k, float *g)
{
	int i, j;
	float a[LPC_FILTORDER + 1], at[LPC_FILTORDER + 1], e;

	for (i = 0; i <= p; i++)
		a[i] = at[i] = 0.0f;

	e = r[0];
	for (i = 1; i <= p; i++) {
		k[i] = -r[i];
		for (j = 1; j < i; j++) {
			at[j] = a[j];
			k[i] -= a[j] * r[i - j];
		}
		k[i] /= e;
		a[i] = k[i];
		for (j = 1; j < i; j++)
			a[j] = at[j] + k[i] * at[i - j];
		e *= 1.0f - k[i] * k[i];
	}

	*g = (float)sqrt(e);
}

static void
inverse_filter(float *w, float *k)
{
	int i, j;
	float b[PITCHORDER + 1], bp[PITCHORDER + 1], f[PITCHORDER + 1];

	for (i = 0; i <= PITCHORDER; i++)
		b[i] = f[i] = bp[i] = 0.0;

	for (i = 0; i < BUFLEN / DOWN; i++) {
		f[0] = b[0] = w[i];
		for (j = 1; j <= PITCHORDER; j++) {
			f[j] = f[j - 1] + k[j] * bp[j - 1];
			b[j] = k[j] * f[j - 1] + bp[j - 1];
			bp[j - 1] = b[j - 1];
		}
		w[i] = f[PITCHORDER];
	}
}

static void
calc_pitch(float *w, float *per)
{
	int i, j, rpos;
	float d[MAXWINDOW / DOWN], k[PITCHORDER + 1], r[MAXPER + 1], g, rmax;
	float rval, rm, rp;
	float a, b, c, x, y;
	static int vuv = 0;

	for (i = 0, j = 0; i < BUFLEN; i += DOWN)
		d[j++] = w[i];
        assert(PITCHORDER <= MAXPER + 1);
	auto_correl(d, BUFLEN / DOWN, PITCHORDER, r);
	durbin(r, PITCHORDER, k, &g);
	inverse_filter(d, k);
        /* Note diff from lpc.c in VAT:                                   */
        /* Paramter 3 to auto_correl is MAXPER, rather than VAT's         */
        /* MAXPER + 1 which causes buffer overrun                         */
	auto_correl(d, BUFLEN / DOWN, MAXPER, r);

        /* Note diff from lpc.c in VAT:                                   */
        /* rpos is defined to be 1 at start to prevent potential illegal  */
        /* memory dereference when defining rm.  Same with rp so rmax     */
        /* search uses a terminating value of MAXPER rather than MAXPER+1 */
	rpos = 1;
	rmax = 0.0;
	for (i = MINPER; i < MAXPER; i++) {
		if (r[i] > rmax) {
			rmax = r[i];
			rpos = i;
		}
	}

	rm = r[rpos - 1];
	rp = r[rpos + 1];
	rval = rmax / r[0];

        a = 0.5f * rm - rmax + 0.5f * rp;
        b = -0.5f * rm * (2.0f * rpos + 1.0f) +
            2.0f * rpos * rmax + 0.5f * rp * (1.0f - 2.0f * rpos);
        c = 0.5f * rm * (rpos * rpos + rpos) +
	    rmax * (1.0f - rpos * rpos) + 0.5f * rp * (rpos * rpos - rpos);

        x = -b / (2.0f * a);
	y = a * x * x + b * x + c;
	x *= DOWN;

	rmax = y;
	rval = rmax / r[0];
	if (rval >= 0.4 || (vuv == 3 && rval >= 0.3)) {
		*per = x;
		vuv = (vuv & 1) * 2 + 1;
	} else {
		*per = 0.0;
		vuv = (vuv & 1) * 2;
	}
}

void
lpc_init()
{
        float   r, v, w, wcT;
        int     i;

        for (i = 0; i < BUFLEN; i++) {
                h[i] = (float)WSCALE * (0.54f - 0.46f * 
                                        (float)cos(2.0f * (float)M_PI * (float)i / (BUFLEN - 1.0f)));
        }
 
        wcT = 2 * (float)M_PI * FC / FS;
        r = 0.36891079f * wcT;
        v = 0.18445539f * wcT;
        w = 0.92307712f * wcT;
        fa[1] = -(float)exp(-r);
        fa[2] = 1.0f + fa[1];
        fa[3] = -2.0f * (float)exp(-v) * (float)cos(w);
        fa[4] = (float)exp(-2.0f * v);
        fa[5] = 1.0f + fa[3] + fa[4];
}

void
lpc_enc_init(lpc_encstate_t *enc)
{
	int        i;
        enc->u = enc->u1 = enc->yp1 = enc->yp2 = 0.0f;
        for(i = 0; i < BUFLEN; i++) {
                enc->raw[i]      = 0.0f;
                enc->filtered[i] = 0.0f;
        }
}

void
lpc_dec_init(lpc_intstate_t* state)
{
        int i;

	state->Oldper = 0.0f;
	state->OldG = 0.0f;
	for (i = 0; i <= LPC_FILTORDER; i++) {
		state->Oldk[i] = 0.0f;
		state->bp[i] = 0.0f;
	}
	state->pitchctr = 0;
}

void
lpc_analyze(const short *buf, lpc_encstate_t *enc, lpc_txstate_t *params)
{
	int     i, j;
	float   w[MAXWINDOW], r[LPC_FILTORDER + 1];
	float   per, G, k[LPC_FILTORDER + 1];
        float *s, *y, u, u1, yp1, yp2;

        u   = enc->u;
        u1  = enc->u1;
        yp1 = enc->yp1;
        yp2 = enc->yp2;

        s = enc->raw;
        y = enc->filtered;

        /* Removed i from loop - redundant */
	for (j = BUFLEN - FRAMESIZE; j < BUFLEN; j++) {
		s[j] = ((float)(*buf++)) / 32768.0f;
		u = fa[2] * s[j] - fa[1] * u1;
		y[j] = fa[5] * u1 - fa[3] * yp1 - fa[4] * yp2;
		u1  = u;
		yp2 = yp1;
		yp1 = y[j];
	}

        enc->u   = u;
        enc->u1  = u1;
        enc->yp1 = yp1;
        enc->yp2 = yp2;

	calc_pitch(y, &per);

	for (i = 0; i < BUFLEN; i++) {
		w[i] = s[i] * h[i];
        }
	auto_correl(w, BUFLEN, LPC_FILTORDER, r);
	durbin(r, LPC_FILTORDER, k, &G);

	params->period = (unsigned short)(per * 256.0f);
	params->gain   = (unsigned char) (G   * 256.0f);
	for (i = 0; i < LPC_FILTORDER; i++) {
		params->k[i] = (char)(k[i + 1] * 128.0f);
	}

	memcpy(s, s + FRAMESIZE, (BUFLEN - FRAMESIZE) * sizeof(s[0]));
	memcpy(y, y + FRAMESIZE, (BUFLEN - FRAMESIZE) * sizeof(y[0]));
}

double drand48(void);

void
lpc_synthesize(short *buf, lpc_txstate_t *params, lpc_intstate_t* state)
{
	int i, j;
	register double u, f, per, G, NewG, Ginc, Newper, perinc;
	double k[LPC_FILTORDER + 1], Newk[LPC_FILTORDER + 1],
	       kinc[LPC_FILTORDER + 1];

	per = (double) params->period / 256.;
	G = (double) params->gain / 256.;
	k[0] = 0.0;
	for (i = 0; i < LPC_FILTORDER; i++)
		k[i + 1] = (double) (params->k[i]) / 128.;

	G /= sqrt(BUFLEN / (per == 0.0? 3.0 : per));
	Newper = state->Oldper;
	NewG = state->OldG;
	for (i = 1; i <= LPC_FILTORDER; i++)
		Newk[i] = state->Oldk[i];

	if (state->Oldper != 0 && per != 0) {
		perinc = (per - state->Oldper) / (double)FRAMESIZE;
		Ginc = (G - state->OldG) / (double)FRAMESIZE;
		for (i = 1; i <= LPC_FILTORDER; i++)
			kinc[i] = (k[i] - state->Oldk[i]) / (double)FRAMESIZE;
	} else {
		perinc = 0.0;
		Ginc = 0.0;
		for (i = 1; i <= LPC_FILTORDER; i++)
			kinc[i] = 0.0;
	}

	if (Newper == 0)
		state->pitchctr = 0;

	for (i = 0; i < FRAMESIZE; i++) {
		if (Newper == 0) {
			u = drand48() * NewG;
		} else {
			if (state->pitchctr == 0) {
				u = NewG;
				state->pitchctr = (int) Newper;
			} else {
				u = 0.0;
				state->pitchctr--;
			}
		}

		f = u;
		for (j = LPC_FILTORDER; j >= 1; j--) {
			register double b = state->bp[j - 1];
			register double kj = Newk[j];
			Newk[j] = kj + kinc[j];
			f -= b * kj;
			b += f * kj;
			state->bp[j] = b;
		}
		state->bp[0] = f;
		
		*buf++ = (short)((int)(f * 32768) & 0xffff);

		Newper += perinc;
		NewG += Ginc;
	}

	state->Oldper = per;
	state->OldG = G;
	for (i = 1; i <= LPC_FILTORDER; i++)
		state->Oldk[i] = k[i];
}

/* this routine does non-interpolating lpc synthesis */
/* for the state passed [oth]                        */
void
lpc_extend_synthesize(short *buf, int len, lpc_intstate_t* s)
{
	int i, j;
        register double u, f;
        
	if (s->Oldper == 0) {
		s->pitchctr = 0;
        }
        
	for (i = 0; i < len; i++) {
		if (s->Oldper == 0) {
			u = drand48() * s->OldG;
		} else {
			if (s->pitchctr == 0) {
				u           = s->OldG;
				s->pitchctr = (int) s->Oldper;
			} else {
				u = 0.0;
				s->pitchctr--;
			}
		}

		f = u;
		for (j = LPC_FILTORDER; j >= 1; j--) {
			register double b  = s->bp[j - 1];
			register double kj = s->Oldk[j];
			f -= b * kj;
			b += f * kj;
			s->bp[j] = b;
		}
		s->bp[0] = f;
                *buf++ = (short)((int)(f * 32768) & 0xffff);
	}
}
