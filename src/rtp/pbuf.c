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
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * $Revision: 1.7 $
 * $Date: 2010/02/05 14:06:17 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/ptime.h"
#include "rtp/pbuf.h"
#include "rtp/decoders.h"

#define PBUF_MAGIC	0xcafebabe

extern long frame_begin[2];

struct pbuf_node {
        struct pbuf_node *nxt;
        struct pbuf_node *prv;
        uint32_t rtp_timestamp; /* RTP timestamp for the frame           */
        struct timeval arrival_time;    /* Arrival time of first packet in frame */
        struct timeval playout_time;    /* Playout time for the frame            */
        struct timeval deletion_time;   /* Time after which will be frame deleted (no matter if decoded or not) */
        struct coded_data *cdata;       /*                                       */
        int decoded;            /* Non-zero if we've decoded this frame  */
        int mbit;               /* determines if mbit of frame had been seen */
        uint32_t magic;         /* For debugging                         */
};

struct pbuf {
        struct pbuf_node *frst;
        struct pbuf_node *last;
        double playout_delay;
        double deletion_delay;
};

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

struct pbuf *pbuf_init(void)
{
        struct pbuf *playout_buf = NULL;

        playout_buf = malloc(sizeof(struct pbuf));
        if (playout_buf != NULL) {
                playout_buf->frst = NULL;
                playout_buf->last = NULL;
                /* Playout delay... should really be adaptive, based on the */
                /* jitter, but we use a (conservative) fixed 32ms delay for */
                /* now (2 video frames at 60fps).                           */
                playout_buf->deletion_delay =
                        playout_buf->playout_delay = 0.032;
        } else {
                debug_msg("Failed to allocate memory for playout buffer\n");
        }
        return playout_buf;
}

static void add_coded_unit(struct pbuf_node *node, rtp_packet * pkt)
{
        /* Add "pkt" to the frame represented by "node". The "node" has    */
        /* previously been created, and has some coded data already...     */

        /* New arrivals are added at the head of the list, which is stored */
        /* in descending order of packets as they arrive (NOT necessarily  */
        /* descending sequence number order, as the network might reorder) */

        struct coded_data *tmp;

        assert(node->rtp_timestamp == pkt->ts);
        assert(node->cdata != NULL);

        tmp = malloc(sizeof(struct coded_data));
        if (tmp != NULL) {
                tmp->seqno = pkt->seq;
                tmp->data = pkt;
                tmp->prv = NULL;
                tmp->nxt = node->cdata;
                node->cdata->prv = tmp;
                node->cdata = tmp;
                node->mbit |= pkt->m;
        } else {
                /* this is bad, out of memory, drop the packet... */
                free(pkt);
        }
}

static struct pbuf_node *create_new_pnode(rtp_packet * pkt, double playout_delay,
                double deletion_delay)
{
        struct pbuf_node *tmp;

        perf_record(UVP_CREATEPBUF, pkt->ts);

        tmp = malloc(sizeof(struct pbuf_node));
        if (tmp != NULL) {
                tmp->magic = PBUF_MAGIC;
                tmp->nxt = NULL;
                tmp->prv = NULL;
                tmp->decoded = 0;
                tmp->rtp_timestamp = pkt->ts;
                tmp->mbit = pkt->m;
                gettimeofday(&(tmp->arrival_time), NULL);
                gettimeofday(&(tmp->playout_time), NULL);
                tmp->deletion_time = tmp->playout_time;
                tv_add(&(tmp->playout_time), playout_delay);
                tv_add(&(tmp->deletion_time), deletion_delay);

                tmp->cdata = malloc(sizeof(struct coded_data));
                if (tmp->cdata != NULL) {
                        tmp->cdata->nxt = NULL;
                        tmp->cdata->prv = NULL;
                        tmp->cdata->seqno = pkt->seq;
                        tmp->cdata->data = pkt;
                } else {
                        free(pkt);
                        free(tmp);
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

        if (playout_buf->frst == NULL && playout_buf->last == NULL) {
                /* playout buffer is empty - add new frame */
                playout_buf->frst = create_new_pnode(pkt, playout_buf->playout_delay,
                                playout_buf->deletion_delay);
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
                        tmp = create_new_pnode(pkt, playout_buf->playout_delay,
                                        playout_buf->deletion_delay);
                        playout_buf->last->nxt = tmp;
                        tmp->prv = playout_buf->last;
                        playout_buf->last = tmp;
                } else {
                        /* Packet belongs to a previous frame... */
                        if (playout_buf->frst->rtp_timestamp > pkt->ts) {
                                debug_msg("A very old packet - discarded\n");
                        } else {
                                debug_msg
                                    ("A packet for a previous frame, but might still be useful\n");
                                /* Should probably insert this into the playout buffer here... */
                        }
                        if (pkt->m) {
                                debug_msg
                                    ("Oops... dropped packet with M bit set\n");
                        }
                        free(pkt);
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

void pbuf_remove(struct pbuf *playout_buf, struct timeval curr_time)
{
        /* Remove previously decoded frames that have passed their playout  */
        /* time from the playout buffer. Incomplete frames that have passed */
        /* their playout time are also discarded.                           */

        struct pbuf_node *curr, *temp;

        pbuf_validate(playout_buf);

        curr = playout_buf->frst;
        while (curr != NULL) {
                temp = curr->nxt;
                if (tv_gt(curr_time, curr->deletion_time)) {
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
                        free(curr);
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

        return (frame->mbit == 1);
}

int
pbuf_decode(struct pbuf *playout_buf, struct timeval curr_time,
                             decode_frame_t decode_func, void *data)
{
        /* Find the first complete frame that has reached it's playout */
        /* time, and decode it into the framebuffer. Mark the frame as */
        /* decoded, but otherwise leave it in the playout buffer.      */
#ifdef WIN32
        UNUSED(curr_time);
#endif
        struct pbuf_node *curr;

        pbuf_validate(playout_buf);

        curr = playout_buf->frst;
        while (curr != NULL) {
                if (!curr->decoded 
#ifndef WIN32
				&& tv_gt(curr_time, curr->playout_time)
#endif
		   ) {
                        if (frame_complete(curr)) {
                                int ret = decode_func(curr->cdata, data);
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

int
audio_pbuf_decode(struct pbuf *playout_buf, struct timeval curr_time,
                             decode_frame_t decode_func, void *data)
{
        /* Find the first complete frame that has reached it's playout */
        /* time, and decode it into the framebuffer. Mark the frame as */
        /* decoded, but otherwise leave it in the playout buffer.      */
        struct pbuf_node *curr;

        pbuf_validate(playout_buf);

        curr = playout_buf->frst;
        while (curr != NULL) {
                /* WARNING: this one differs from video - we need to push audio immediately, because we do
                 * _not_ know the granularity of audio (typically 256 B for ALSA) which is only small fractal
                 * of frame time. The current RTP library isn't currently able to keep concurrently more frames.
                 */
                UNUSED(curr_time);
                if (!curr->decoded // && tv_gt(curr_time, curr->playout_time)
                                ) {
                        if (frame_complete(curr)) {
                                int ret = decode_func(curr->cdata, data);
                                curr->decoded = 1;
                                return ret;
                        }
                }
                curr = curr->nxt;
        }
        return 0;
}

void pbuf_set_playout_delay(struct pbuf *playout_buf, double playout_delay, double deletion_delay)
{
        playout_buf->playout_delay = playout_delay;
        playout_buf->deletion_delay = deletion_delay;
}

