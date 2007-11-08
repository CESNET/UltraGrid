/*
 *
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Id: lpc_impl.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _LPC_H_
#define _LPC_H_

#define FRAMESIZE 160
#define BUFLEN   ((FRAMESIZE * 3) / 2)

#define LPC_FILTORDER		10

/* lpc transmitted state */

typedef struct {
	unsigned short period;
	unsigned char gain;
	char k[LPC_FILTORDER];
	char pad;
} lpc_txstate_t;

#define LPCTXSIZE 14

/*
 * we can't use 'sizeof(lpcparams_t)' because some compilers
 * add random padding so define size of record that goes over net.
 */

/* lpc decoder internal state */
typedef struct {
	double Oldper, OldG, Oldk[LPC_FILTORDER + 1], bp[LPC_FILTORDER + 1];
	int pitchctr;
} lpc_intstate_t;

/* Added encoder state by removing static buffers
 * that are used for storing float represetantions
 * of audio from previous frames.  Multiple coders
 * will no longer interfere. It's all filter state.
 */
typedef struct {
        float u, u1, yp1, yp2;
        float raw[BUFLEN];
        float filtered[BUFLEN];
} lpc_encstate_t;

void lpc_init(void);
void lpc_enc_init(lpc_encstate_t* state);
void lpc_dec_init(lpc_intstate_t* state);
void lpc_analyze(const short *buf, lpc_encstate_t *enc, lpc_txstate_t *params);
void lpc_synthesize(short *buf, lpc_txstate_t *params, lpc_intstate_t* state);
void lpc_extend_synthesize(short *buf, int len, lpc_intstate_t* state);

#endif /* _LPC_H_ */



