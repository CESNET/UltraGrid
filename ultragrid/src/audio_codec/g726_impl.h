/*
 * This source code is a product of Sun Microsystems, Inc. and is provided
 * for unrestricted use.  Users may copy or modify this source code without
 * charge.
 *
 * SUN SOURCE CODE IS PROVIDED AS IS WITH NO WARRANTIES OF ANY KIND INCLUDING
 * THE WARRANTIES OF DESIGN, MERCHANTIBILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE, OR ARISING FROM A COURSE OF DEALING, USAGE OR TRADE PRACTICE.
 *
 * Sun source code is provided with no support and without any obligation on
 * the part of Sun Microsystems, Inc. to assist in its use, correction,
 * modification or enhancement.
 *
 * SUN MICROSYSTEMS, INC. SHALL HAVE NO LIABILITY WITH RESPECT TO THE
 * INFRINGEMENT OF COPYRIGHTS, TRADE SECRETS OR ANY PATENTS BY THIS SOFTWARE
 * OR ANY PART THEREOF.
 *
 * In no event will Sun Microsystems, Inc. be liable for any lost revenue
 * or profits or other special, indirect and consequential damages, even if
 * Sun has been advised of the possibility of such damages.
 *
 * Sun Microsystems, Inc.
 * 2550 Garcia Avenue
 * Mountain View, California  94043
 *
 * $Id: g726_impl.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

/*
 * cx_g726.h
 *
 * Header file for CCITT conversion routines.
 */
#ifndef _G726_H
#define	_G726_H

#define	AUDIO_ENCODING_ULAW	(1)	/* ISDN u-law */
#define	AUDIO_ENCODING_ALAW	(2)	/* ISDN A-law */
#define	AUDIO_ENCODING_LINEAR	(3)	/* PCM 2's-complement (0-center) */

/*
 * The following is the definition of the state structure
 * used by the G.726_32/G.726 encoder and decoder to preserve their internal
 * state between successive calls.  The meanings of the majority
 * of the state structure fields are explained in detail in the
 * CCITT Recommendation G.726_32.  The field names are essentially indentical
 * to variable names in the bit level description of the coding algorithm
 * included in this Recommendation.
 */
struct g726_state {
	int yl;	/* Locked or steady state step size multiplier. */
	short yu;	/* Unlocked or non-steady state step size multiplier. */
	short dms;	/* Short term energy estimate. */
	short dml;	/* Long term energy estimate. */
	short ap;	/* Linear weighting coefficient of 'yl' and 'yu'. */

	short a[2];	/* Coefficients of pole portion of prediction filter. */
	short b[6];	/* Coefficients of zero portion of prediction filter. */
	short pk[2];	/*
			 * Signs of previous two samples of a partially
			 * reconstructed signal.
			 */
	short dq[6];	/*
			 * Previous 6 samples of the quantized difference
			 * signal represented in an internal floating point
			 * format.
			 */
	short sr[2];	/*
			 * Previous 2 samples of the quantized difference
			 * signal represented in an internal floating point
			 * format.
			 */
	char td;	/* delayed tone detect, new in 1988 version */
};

/* External function definitions. */

extern void g726_init_state(struct g726_state *);
extern int g726_16_encoder(
		int sample,
		int in_coding,
		struct g726_state *state_ptr);
extern int g726_16_decoder(
		int code,
		int out_coding,
		struct g726_state *state_ptr);
extern int g726_24_encoder(
		int sample,
		int in_coding,
		struct g726_state *state_ptr);
extern int g726_24_decoder(
		int code,
		int out_coding,
		struct g726_state *state_ptr);
extern int g726_32_encoder(
		int sample,
		int in_coding,
		struct g726_state *state_ptr);
extern int g726_32_decoder(
		int code,
		int out_coding,
		struct g726_state *state_ptr);
extern int g726_40_encoder(
		int sample,
		int in_coding,
		struct g726_state *state_ptr);
extern int g726_40_decoder(
		int code,
		int out_coding,
		struct g726_state *state_ptr);

/* Internal function definitions */
#ifdef _PRIVATE_G726_
int predictor_zero(struct g726_state *state_ptr);
int predictor_pole(struct g726_state *state_ptr);
int step_size(struct g726_state *state_ptr);
int quantize(int d, int	y, short *table, int size);
int reconstruct(int sign, int dqln, int y);
void
update(int code_size, int y, int wi, int fi, int dq, int sr, int dqsez, struct g726_state *state_ptr);

#endif /* _PRIVATE_G726_ */

#endif /* !_G726_H */
