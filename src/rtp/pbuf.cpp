/*
 * AUTHOR:   N.Cihan Tas
 * MODIFIED: Ladan Gharai
 *           Colin Perkins
 *           Martin Benes     <martinbenesh@gmail.com>
 *           Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *           Petr Holub       <hopet@ics.muni.cz>
 *           Milos Liska      <xliska@fi.muni.cz>
 *           Jiri Matela      <matela@ics.muni.cz>
 *           Dalibor Matura   <255899@mail.muni.cz>
 *           Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 * 
 * This file implements a linked list for the playout buffer.
 *
 * Copyright (c) 2003-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 * Copyright (c) 2005-2014 CESNET z.s.p.o.
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
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/ptime.h"
#include "rtp/pbuf.h"

#define PBUF_MAGIC	0xcafebabe

#define STATS_INTERVAL 10

struct pbuf_node {
        struct pbuf_node *nxt;
        struct pbuf_node *prv;
        uint32_t rtp_timestamp; /* RTP timestamp for the frame           */
        std::chrono::high_resolution_clock::time_point arrival_time;    /* Arrival time of first packet in frame */
        std::chrono::high_resolution_clock::time_point playout_time;    /* Playout time for the frame            */
        struct coded_data *cdata;       /*                                       */
        int decoded;            /* Non-zero if we've decoded this frame  */
        int mbit;               /* determines if mbit of frame had been seen */
        uint32_t magic;         /* For debugging                         */
        bool completed;
};

struct pbuf {
        struct pbuf_node *frst;
        struct pbuf_node *last;
        long long int playout_delay_us;
        volatile int *offset_ms;

        // for statistics
        /// @todo figure out packet duplication
        int pkt_count[2]; // first buffer is for last_rtp_seq + STATS_INTERVAL
                          // second for last_rtp_seq + 2 * STATS_INTERVAL
        int last_rtp_seq;
        int received_pkts, expected_pkts; // currently computed values
        int received_pkts_last, expected_pkts_last; // values for last interval
        long long int received_pkts_cum, expected_pkts_cum; // cumulative values
        uint32_t last_display_ts;
};

static void free_cdata(struct coded_data *head);
static int frame_complete(struct pbuf_node *frame);

/*********************************************************************************/

static void pbuf_validate(struct pbuf *playout_buf)
{
        /* Run through the entire playout buffer, checking pointers, etc.  */
        /* Only used in debugging mode, since it's a lot of overhead [csp] */
#ifdef NDEF
        struct pbuf_node *cpb, *ppb;
        struct coded_data *ccd, *pcd;

        cpb = playout_buf->frst;
        ppb = NULL;
        while (cpb != NULL) {
                assert(cpb->magic == PBUF_MAGIC);
                assert(cpb->prv == ppb);
                if (cpb->prv != NULL) {
                        assert(cpb->prv->nxt == cpb);
                        /* stored in RTP timestamp order */
                        assert(cpb->rtp_timestamp > ppb->rtp_timestamp);
                        /* stored in playout time order  */
                        /* TODO: eventually check why is this assert always failng */
                        // assert(tv_gt(cpb->ptime, ppb->ptime));  
                }
                if (cpb->nxt != NULL) {
                        assert(cpb->nxt->prv == cpb);
                } else {
                        assert(cpb = playout_buf->last);
                }
                if (cpb->cdata != NULL) {
                        /* We have coded data... check all the pointers on that list too */
                        ccd = cpb->cdata;
                        pcd = NULL;
                        while (ccd != NULL) {
                                assert(ccd->prv == pcd);
                                if (ccd->prv != NULL) {
                                        assert(ccd->prv->nxt == ccd);
                                        /* list is descending - cant really check this now */
                                        //assert(ccd->seqno < pcd->seqno); 
                                        assert(ccd->data != NULL);
                                }
                                if (ccd->nxt != NULL) {
                                        assert(ccd->nxt->prv == ccd);
                                }
                                pcd = ccd;
                                ccd = ccd->nxt;
                        }
                }
                ppb = cpb;
                cpb = cpb->nxt;
        }
#else
        UNUSED(playout_buf);
#endif
}

struct pbuf *pbuf_init(volatile int *delay_ms)
{
        struct pbuf *playout_buf = NULL;

        playout_buf = (struct pbuf *) calloc(1, sizeof(struct pbuf));
        if (playout_buf != NULL) {
                playout_buf->frst = NULL;
                playout_buf->last = NULL;
                /* Playout delay... should really be adaptive, based on the */
                /* jitter, but we use a (conservative) fixed 32ms delay for */
                /* now (2 video frames at 60fps).                           */
                playout_buf->offset_ms = delay_ms;
                playout_buf->playout_delay_us = 0.032 * 1000 * 1000;
                playout_buf->last_rtp_seq = -1;
        } else {
                debug_msg("Failed to allocate memory for playout buffer\n");
        }
        return playout_buf;
}

void pbuf_destroy(struct pbuf *playout_buf) {
        if (playout_buf) {
                struct pbuf_node *curr, *temp;

                pbuf_validate(playout_buf);

                curr = playout_buf->frst;
                while (curr != NULL) {
                        temp = curr->nxt;
                        if (curr == playout_buf->frst) {
                                playout_buf->frst = curr->nxt;
                        }
                        if (curr == playout_buf->last) {
                                playout_buf->last = curr->prv;
                        }
                        if (curr->nxt != NULL) {
                                curr->nxt->prv = curr->prv;
                        }
                        if (curr->prv != NULL) {
                                curr->prv->nxt = curr->nxt;
                        }
                        free_cdata(curr->cdata);
                        delete curr;
                        curr = temp;
                }
                free(playout_buf);
        }
}

static void add_coded_unit(struct pbuf_node *node, rtp_packet * pkt)
{
        /* Add "pkt" to the frame represented by "node". The "node" has    */
        /* previously been created, and has some coded data already...     */

        /* New arrivals are added at the head of the list, which is stored */
        /* in descending order of packets as they arrive (NOT necessarily  */
        /* descending sequence number order, as the network might reorder) */

        struct coded_data *tmp, *curr, *prv;

        assert(node->rtp_timestamp == pkt->ts);
        assert(node->cdata != NULL);

        tmp = (struct coded_data *) malloc(sizeof(struct coded_data));
        if (tmp == NULL) {
                /* this is bad, out of memory, drop the packet... */
                free(pkt);
                return;
        }

        tmp->seqno = pkt->seq;
        tmp->data = pkt;
        node->mbit |= pkt->m;
        if((int16_t)(tmp->seqno - node->cdata->seqno) > 0){
                tmp->prv = NULL;
                tmp->nxt = node->cdata;
                node->cdata->prv = tmp;
                node->cdata = tmp;
        } else {
                curr = node->cdata;
                if (curr == NULL){
                        /* this is bad, out of memory, drop the packet... */
                        free(pkt);
                        free(tmp);
                } else {
                        while (curr != NULL &&  ((int16_t)(tmp->seqno - curr->seqno) < 0)){
                                prv = curr;
                                curr = curr->nxt;
                        }
                        if (curr == NULL) {
                                tmp->nxt = NULL;
                                tmp->prv = prv;
                                prv->nxt = tmp;
                        }else if ((int16_t)(tmp->seqno - curr->seqno) > 0){
                                tmp->nxt = curr;
                                tmp->prv = curr->prv;
                                tmp->prv->nxt = tmp;
                                curr->prv = tmp;
                        } else {
                                /* this is bad, something went terribly wrong... */
                                free(pkt);
                                free(tmp);
                        }
                }
        }
}

static struct pbuf_node *create_new_pnode(rtp_packet * pkt, long long playout_delay_us)
{
        struct pbuf_node *tmp;

        perf_record(UVP_CREATEPBUF, pkt->ts);

        tmp = new struct pbuf_node();
        if (tmp != NULL) {
                tmp->magic = PBUF_MAGIC;
                tmp->rtp_timestamp = pkt->ts;
                tmp->mbit = pkt->m;
                tmp->playout_time =
                        tmp->arrival_time = std::chrono::high_resolution_clock::now();
                tmp->playout_time += std::chrono::microseconds(playout_delay_us);

                tmp->cdata = (struct coded_data *) malloc(sizeof(struct coded_data));
                if (tmp->cdata != NULL) {
                        tmp->cdata->nxt = NULL;
                        tmp->cdata->prv = NULL;
                        tmp->cdata->seqno = pkt->seq;
                        tmp->cdata->data = pkt;
                } else {
                        free(pkt);
                        delete tmp;
                        return NULL;
                }
        } else {
                free(pkt);
        }
        return tmp;
}

void pbuf_insert(struct pbuf *playout_buf, rtp_packet * pkt)
{
        struct pbuf_node *tmp;

        pbuf_validate(playout_buf);

        // collect statistics
        if (playout_buf->last_rtp_seq == -1) {
                playout_buf->last_rtp_seq = pkt->seq;
                playout_buf->pkt_count[0] += 1;
        } else {
                if ((((int) pkt->seq - playout_buf->last_rtp_seq + (1<<16)) %
                                        (1<<16)) < STATS_INTERVAL * 2) {

                        if ((((int) pkt->seq - playout_buf->last_rtp_seq + (1<<16))
                                                % (1<<16)) < STATS_INTERVAL) {
                                playout_buf->pkt_count[0] += 1;
                        } else {
                                playout_buf->pkt_count[1] += 1;
                        }
                } else {
                        playout_buf->received_pkts += playout_buf->pkt_count[0];
                        playout_buf->expected_pkts += STATS_INTERVAL;
                        playout_buf->last_rtp_seq = (playout_buf->last_rtp_seq +
                                        STATS_INTERVAL) % (1<<16);
                        playout_buf->pkt_count[0] = playout_buf->pkt_count[1];
                        playout_buf->pkt_count[1] = 1;
                }
        }

        // print statistics after 5 seconds
        if ((pkt->ts - playout_buf->last_display_ts) > 90000 * 5 &&
                        playout_buf->expected_pkts > 0) {
                // print stats
                log_msg(LOG_LEVEL_INFO, "SSRC %08x: %d/%d packets received "
                                "(%.5f%%).\n",
                                pkt->ssrc,
                                playout_buf->received_pkts,
                                playout_buf->expected_pkts,
                                (double) playout_buf->received_pkts /
                                playout_buf->expected_pkts * 100.0);
                playout_buf->received_pkts_cum += playout_buf->received_pkts;
                playout_buf->expected_pkts_cum += playout_buf->expected_pkts;
                playout_buf->received_pkts_last = playout_buf->received_pkts;
                playout_buf->expected_pkts_last = playout_buf->expected_pkts;
                playout_buf->expected_pkts = playout_buf->received_pkts = 0;
                playout_buf->last_display_ts = pkt->ts;
        }

        if (playout_buf->frst == NULL && playout_buf->last == NULL) {
                /* playout buffer is empty - add new frame */
                playout_buf->frst = create_new_pnode(pkt, playout_buf->playout_delay_us + 1000 * (playout_buf->offset_ms ? *playout_buf->offset_ms : 0));
                playout_buf->last = playout_buf->frst;
                return;
        }

        if (playout_buf->last->rtp_timestamp == pkt->ts) {
                /* Packet belongs to last frame in playout_buf this is the */
                /* most likely scenario - although...                      */
                add_coded_unit(playout_buf->last, pkt);
        } else {
                if (playout_buf->last->rtp_timestamp < pkt->ts) {
                        /* Packet belongs to a new frame... */
                        tmp = create_new_pnode(pkt, playout_buf->playout_delay_us + 1000 * (playout_buf->offset_ms ? *playout_buf->offset_ms : 0));
                        playout_buf->last->nxt = tmp;
                        playout_buf->last->completed = true;
                        tmp->prv = playout_buf->last;
                        playout_buf->last = tmp;
                } else {
                        bool discard_pkt = false;
                        /* Packet belongs to a previous frame... */
                        if (playout_buf->frst->rtp_timestamp > pkt->ts) {
                                debug_msg("A very old packet - discarded\n");
                                discard_pkt = true;
                        } else {
                                debug_msg
                                    ("A packet for a previous frame, but might still be useful\n");
                                struct pbuf_node *curr = playout_buf->last;
                                while(curr != playout_buf->frst && curr->rtp_timestamp > pkt->ts){
                                        curr = curr->prv;
                                }
                                if (curr->rtp_timestamp == pkt->ts) {
                                        /* Packet belongs to a previous existing frame... */
                                        add_coded_unit(curr, pkt);
                                } else {
                                        /* Packet belongs to a frame that is not present */
                                        discard_pkt = true;
                                }
                        }
                        if (discard_pkt) {
                                if (pkt->m) {
                                        debug_msg
                                                ("Oops... dropped packet with M bit set\n");
                                }
                                free(pkt);
                        }
                }
        }
        pbuf_validate(playout_buf);
}

static void free_cdata(struct coded_data *head)
{
        struct coded_data *tmp;

        while (head != NULL) {
                free(head->data);
                tmp = head;
                head = head->nxt;
                free(tmp);
        }
}

void pbuf_remove(struct pbuf *playout_buf, std::chrono::high_resolution_clock::time_point const & curr_time)
{
        /* Remove previously decoded frames that have passed their playout  */
        /* time from the playout buffer. Incomplete frames that have passed */
        /* their playout time are also discarded.                           */

        struct pbuf_node *curr, *temp;

        pbuf_validate(playout_buf);

        curr = playout_buf->frst;
        while (curr != NULL) {
                temp = curr->nxt;
                if (curr_time > curr->playout_time && frame_complete(curr)) {
                        if (curr == playout_buf->frst) {
                                playout_buf->frst = curr->nxt;
                        }
                        if (curr == playout_buf->last) {
                                playout_buf->last = curr->prv;
                        }
                        if (curr->nxt != NULL) {
                                curr->nxt->prv = curr->prv;
                        }
                        if (curr->prv != NULL) {
                                curr->prv->nxt = curr->nxt;
                        }
                        free_cdata(curr->cdata);
                        delete curr;
                } else {
                        /* The playout buffer is stored in order, so once  */
                        /* we see one packet that has not yet reached it's */
                        /* playout time, we can be sure none of the others */
                        /* will have done so...                            */
                        break;
                }
                curr = temp;
        }

        pbuf_validate(playout_buf);
        return;
}

static int frame_complete(struct pbuf_node *frame)
{
        /* Return non-zero if the list of coded_data represents a    */
        /* complete frame of video. This might have to be passed the */
        /* seqnum of the last packet in the previous frame, too?     */
        /* i dont think that would reflect correctly of weather this */
        /* frame is complete or not - however we should check for all */
        /* the packtes of a frame being present - perhaps we should  */
        /* keep a bit vector in pbuf_node? LG.  */

        return (frame->mbit == 1 || frame->completed == true);
}

int pbuf_is_empty(struct pbuf *playout_buf)
{
        if (playout_buf->frst == NULL)
                return TRUE;
        else
                return FALSE;
}

int
pbuf_decode(struct pbuf *playout_buf, std::chrono::high_resolution_clock::time_point const & curr_time,
                             decode_frame_t decode_func, void *data)
{
        /* Find the first complete frame that has reached it's playout */
        /* time, and decode it into the framebuffer. Mark the frame as */
        /* decoded, but otherwise leave it in the playout buffer.      */
        struct pbuf_node *curr;

        pbuf_validate(playout_buf);

        curr = playout_buf->frst;
        while (curr != NULL) {
                if (!curr->decoded 
                                && curr_time > curr->playout_time
                   ) {
                        if (frame_complete(curr)) {
                                struct pbuf_stats stats = { playout_buf->received_pkts_cum,
                                        playout_buf->expected_pkts_cum };
                                int ret = decode_func(curr->cdata, data, &stats);
                                curr->decoded = 1;
                                return ret;
                        } else {
                                debug_msg
                                    ("Unable to decode frame due to missing data (RTP TS=%u)\n",
                                     curr->rtp_timestamp);
                        }
                }
                curr = curr->nxt;
        }
        return 0;
}

void pbuf_set_playout_delay(struct pbuf *playout_buf, double playout_delay)
{
        playout_buf->playout_delay_us = playout_delay * 1000 * 1000;
}

