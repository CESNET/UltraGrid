/*
 * FILE:     tfrc.c
 * AUTHOR:   Ladan Gharai <ladan@isi.edu>
 * MODIFIED: Colin Perkins <csp@isi.edu>
 *
 * Copyright (C) 2002 USC Information Sciences Institute
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/rtp.h"
#include "tv.h"
#include "tfrc.h"

#define TFRC_MAGIC	0xbaef03b7      /* For debugging */

#define MAX_HISTORY		1000
#define N			8       /* number of intervals */
#define MAX_DROPOUT		100
#define RTP_SEQ_MOD        	0x10000
#define MAX_MISORDER   		100

/*
 * The state of this TFRC connection, stored in a struct that is passed to 
 * all TFRC routines so we can have multiple connections active at once. 
 * See tfrc_init() for initialisation.
 *
 */
struct tfrc {
        int total_pckts;
        int ooo;
        int cycles;             /* number of times seq number cycles 65535 */
        uint32_t RTT;           /* received from sender in app packet */
        struct timeval feedback_timer;  /* indicates points in time when p should be computed */
        double p, p_prev;
        int s;
        struct timeval start_time;
        int loss_count;
        int interval_count;
        int gap[20];
        int jj;                 /* index for arrival -1<=i<=MAX_HISTORY */
        int ii;                 /* index for loss    -1<=ii<=10 */
        double W_tot;
        double weight[N + 5];   /* Weights for loss event calculation. In the RFC, the numbering is the other way around */
        uint32_t magic;         /* For debugging */
};

static void validate_tfrc_state(struct tfrc *state)
{
        /* Debugging routine. Called each time we enter TFRC code, */
        /* to ensure that the state information we've been given   */
        /* is valid.                                               */
        assert(state->magic == TFRC_MAGIC);
#ifdef DEBUG
        /* ...do some fancy debugging... */
#endif
}

static struct {
        uint32_t seq;
        uint32_t ts;
} arrival[MAX_HISTORY];

static struct {
        uint32_t seq;
        uint32_t ts;
} loss[MAX_HISTORY];

#ifdef NDEF
static double transfer_rate(double p)
{
        double t, t1, t2, t3, t4, rtt, tRTO;
        if (p == 0) {
                return 0;
        }

        /* convert RTT from usec to sec */
        rtt = ((double)RTT) / 1000000.0;
        tRTO = 4 * rtt;

        t1 = rtt * sqrt(2 * p / 3);
        t2 = (1 + 32 * p * p);
        t3 = 3 * sqrt(3 * p / 8);
        t4 = t1 + tRTO * t3 * p * t2;
        t = ((double)s) / t4;

        return (t);
}

static void compute_transfer_rate(void)
{
        double t1;
        struct timeval now;

        t1 = transfer_rate(p);
        gettimeofday(&now, NULL);
}
#endif

static int set_zero(int first, int last, uint16_t u)
{
        int i, count = 0;;

        assert((first >= 0) && (first < MAX_HISTORY));
        assert((last >= 0) && (last < MAX_HISTORY));

        if (first == last)
                return (0);
        if ((first + 1) % MAX_HISTORY == last)
                return (1);

        if (first < last) {
                for (i = first + 1; i < last; i++) {
                        arrival[i].seq = 0;
                        arrival[i].ts = 0;
                        count++;
                }
        } else {
                for (i = first + 1; i < MAX_HISTORY; i++) {
                        arrival[i].seq = 0;
                        arrival[i].ts = 0;
                        count++;
                }
                for (i = 0; i < last; i++) {
                        arrival[i].seq = 0;
                        arrival[i].ts = 0;
                        count++;
                }
        }
        assert(count == (u - 1));
        return (count);
}

static int arrived(int first, int last)
{
        int i, count = 0;

        assert((first >= 0) && (first < MAX_HISTORY));
        assert((last >= 0) && (last < MAX_HISTORY));

        if (first == last)
                return (0);
        if ((first + 1) % MAX_HISTORY == last)
                return (0);

        if (first < last) {
                for (i = first; i <= last; i++) {
                        if (arrival[i].seq != 0)
                                count++;
                }
        } else {
                for (i = first; i < MAX_HISTORY; i++) {
                        if (arrival[i].seq != 0)
                                count++;
                }
                for (i = 0; i <= last; i++) {
                        if (arrival[i].seq != 0)
                                count++;
                }
        }
        return (count);
}

static void
record_loss(struct tfrc *state, uint32_t s1, uint32_t s2, uint32_t ts1,
            uint32_t ts2)
{
        /* Mark all packets between s1 (which arrived at time ts1) and */
        /* s2 (which arrived at time ts2) as lost.                     */
        int i;
        uint32_t est;
        uint32_t seq = s1 + 1;

        est = ts1 + (ts2 - ts1) * ((seq - s1) / (s2 - seq));

        state->loss_count++;
        if (state->ii <= -1) {
                /* first loss! */
                state->ii++;
                loss[state->ii].seq = seq;
                loss[state->ii].ts = est;
                return;
        }

        if (est - loss[state->ii].ts <= state->RTT) {   /* not a new event */
                return;
        }

        state->interval_count++;
        if (state->ii > (N + 1)) {
                printf("how did this happen?\n");
        }

        if (state->ii >= (N + 1)) {     /* shift */
                for (i = 0; i < (N + 1); i++) {
                        loss[i].seq = loss[i + 1].seq;
                        loss[i].ts = loss[i + 1].ts;
                }
                state->ii = N;
        }

        state->ii++;
        loss[state->ii].seq = seq;
        loss[state->ii].ts = est;
}

static void
save_arrival(struct tfrc *state, struct timeval curr_time, uint16_t seq)
{
        int kk, inc, last_jj;
        uint16_t udelta;
        uint32_t now;
        uint32_t ext_seq;
        static uint16_t last_seq;
        static uint32_t ext_last_ack;
        static int last_ack_jj = 0;

        gettimeofday(&curr_time, NULL);
        now = tv_diff_usec(curr_time, state->start_time);

        if (state->jj == -1) {
                /* first packet arrival */
                state->jj = 0;
                last_seq = seq;
                ext_last_ack = seq;
                last_ack_jj = 0;
                arrival[state->jj].seq = seq;
                arrival[state->jj].ts = now;
                return;
        }

        udelta = seq - last_seq;

        state->total_pckts++;
        if (udelta < MAX_DROPOUT) {
                /* in order, with permissible gap */
                if (seq < last_seq) {
                        state->cycles++;
                }
                /* record arrival */
                last_jj = state->jj;
                state->jj = (state->jj + udelta) % MAX_HISTORY;
                set_zero(last_jj, state->jj, udelta);
                ext_seq = seq + state->cycles * RTP_SEQ_MOD;
                last_seq = seq;
                arrival[state->jj].seq = ext_seq;
                arrival[state->jj].ts = now;
                if (udelta < 10)
                        state->gap[udelta - 1]++;

                if ((ext_seq - ext_last_ack) == 1) {
                        /* We got two consecutive packets, no loss */
                        ext_last_ack = ext_seq;
                        last_ack_jj = state->jj;
                } else {
                        /* Sequence number jumped, we've missed a packet for some reason */
                        if (arrived(last_ack_jj, state->jj) >= 4) {
                                record_loss(state, ext_last_ack, ext_seq,
                                            arrival[last_ack_jj].ts, now);
                                ext_last_ack = ext_seq;
                                last_ack_jj = state->jj;
                        }
                }
        } else if (udelta <= RTP_SEQ_MOD - MAX_MISORDER) {
                printf(" -- seq:%u last seq:%u  ", seq, arrival[state->jj].seq);
                abort();        /* FIXME */
        } else {
                /* duplicate or reordered packet */
                ext_seq = seq + state->cycles * RTP_SEQ_MOD;
                state->ooo++;
                if (ext_seq > ext_last_ack) {
                        inc = ext_seq - arrival[state->jj].seq;

                        kk = (state->jj + inc) % MAX_HISTORY;
                        if (arrival[kk].seq == 0) {
                                arrival[kk].seq = ext_seq;
                                arrival[kk].ts = (arrival[last_ack_jj].ts + now) / 2;   /* NOT the best interpolation */
                        }
                        while (arrival[last_ack_jj + 1].seq != 0
                               && last_ack_jj < state->jj) {
                                last_ack_jj = (last_ack_jj + 1) % MAX_HISTORY;
                        }
                        ext_last_ack = arrival[last_ack_jj].seq;
                }
        }
}

static double compute_loss_event(struct tfrc *state)
{
        int i;
        uint32_t t, temp, I_tot0 = 0, I_tot1 = 0, I_tot = 0;
        struct timeval now;
        double I_mean, p;

        if (state->ii < N) {
                return 0;
        }

        for (i = state->ii - N; i < state->ii; i++) {
                temp = loss[i + 1].seq - loss[i].seq;
                I_tot0 = I_tot0 + temp * state->weight[i];
                if (i >= (state->ii - N + 1)) {
                        I_tot1 = I_tot1 + temp * state->weight[i - 1];
                }
        }
        I_tot1 =
            I_tot1 + (arrival[state->jj].seq -
                      loss[state->ii].seq) * state->weight[N - 1];

        I_tot = (I_tot1 > I_tot0) ? I_tot1 : I_tot0;
        I_mean = I_tot / state->W_tot;
        p = 1 / I_mean;

        gettimeofday(&now, NULL);
        t = tv_diff_usec(now, state->start_time);

        return p;

}

/*
 * External API follows...
 *
 */

struct tfrc *tfrc_init(struct timeval curr_time)
{
        struct tfrc *state;
        int i;

        state = (struct tfrc *)malloc(sizeof(struct tfrc));
        if (state != NULL) {
                state->magic = TFRC_MAGIC;
                state->total_pckts = 0;
                state->ooo = 0;
                state->cycles = 0;
                state->RTT = 0;
                state->feedback_timer = curr_time;
                state->p = 0.0;
                state->p_prev = 0.0;
                state->s = 0;
                state->start_time = curr_time;
                state->loss_count = 0;
                state->interval_count = 0;
                state->jj = -1;
                state->ii = -1;
                state->W_tot = 0;
                state->weight[0] = 0.2;
                state->weight[1] = 0.4;
                state->weight[2] = 0.6;
                state->weight[3] = 0.8;
                state->weight[4] = 1.0;
                state->weight[5] = 1.0;
                state->weight[6] = 1.0;
                state->weight[7] = 1.0;
                state->weight[8] = 1.0;

                for (i = 0; i < 20; i++) {
                        state->gap[i] = 0;
                }

                for (i = 0; i < N; i++) {
                        state->W_tot = state->W_tot + state->weight[i];
                }
        }

        for (i = 0; i < MAX_HISTORY; i++) {
                arrival[i].seq = 0;
                arrival[i].ts = 0;
                loss[i].seq = 0;
                loss[i].ts = 0;
        }
        return state;
}

void tfrc_done(struct tfrc *state)
{
        int i;

        validate_tfrc_state(state);

        for (i = 0; i < 10; i++) {
                printf("\n%2d %8d", i, state->gap[i]);
        }
        printf("\n");
        printf("\nLost:       %8d", state->loss_count);
        printf("\nIntervals:  %8d", state->interval_count);
        printf("\nTotal:      %8d", state->total_pckts);
        printf("\nooo:        %8d -- %7.5f\n\n", state->ooo,
               (state->ooo * 100) / (double)state->total_pckts);
}

void
tfrc_recv_data(struct tfrc *state, struct timeval curr_time, uint16_t seqnum,
               unsigned length)
{
        /* This is called each time an RTP packet is received. Accordingly, */
        /* it needs to be _very_ fast, otherwise we'll drop packets.        */

        validate_tfrc_state(state);

        if (state->RTT > 0) {
                save_arrival(state, curr_time, seqnum);
                state->p_prev = state->p;
#ifdef NDEF
                state->p = compute_loss_event(state);
                if (state->p - state->p_prev > 0.00000000001) {
                        gettimeofday(&(state->feedback_timer), NULL);
                        tv_add(&(state->feedback_timer),
                               (unsigned int)state->RTT);
                        compute_transfer_rate();
                }
#endif
        }
        state->s = length;      /* packet size is needed transfer_rate */
}

void tfrc_recv_rtt(struct tfrc *state, struct timeval curr_time, uint32_t rtt)
{
        /* Called whenever the receiver gets an RTCP APP packet telling */
        /* it the RTT to the sender. Not performance critical.          */
        /* Note: RTT is in microseconds.                                */

        validate_tfrc_state(state);

        if (state->RTT == 0) {
                state->feedback_timer = curr_time;
                tv_add(&(state->feedback_timer), rtt);
        }
        state->RTT = rtt;
}

int tfrc_feedback_is_due(struct tfrc *state, struct timeval curr_time)
{
        /* Determine if it is time to send feedback to the sender */
        validate_tfrc_state(state);

        if ((state->RTT == 0) || tv_gt(state->feedback_timer, curr_time)) {
                /* Not yet time to send feedback to the sender... */
                return FALSE;
        }
        return TRUE;
}

double tfrc_feedback_txrate(struct tfrc *state, struct timeval curr_time)
{
        /* Calculate the appropriate transmission rate, to be included */
        /* in a feedback message to the sender.                        */

        validate_tfrc_state(state);

        assert(tfrc_feedback_is_due(state, curr_time));

        state->feedback_timer.tv_sec = curr_time.tv_sec;
        state->feedback_timer.tv_usec = curr_time.tv_usec;
        tv_add(&(state->feedback_timer), state->RTT);
        state->p = compute_loss_event(state);
        //compute_transfer_rate ();
        if (state->ii >= N) {
                abort();        /* FIXME */
        }
        return 0.0;             /* FIXME */
}
