/*
 * FILE:     rtp.c
 * AUTHOR:   Colin Perkins   <csp@csperkins.org>
 * MODIFIED: Orion Hodson    <o.hodson@cs.ucl.ac.uk>
 *           Markus Germeier <mager@tzi.de>
 *           Bill Fenner     <fenner@research.att.com>
 *           Timur Friedman  <timur@research.att.com>
 *           Ladan Gharai    <ladan@isi.edu>
 *           Martin Benes     <martinbenesh@gmail.com>
 *           Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *           Petr Holub       <hopet@ics.muni.cz>
 *           Milos Liska      <xliska@fi.muni.cz>
 *           Jiri Matela      <matela@ics.muni.cz>
 *           Dalibor Matura   <255899@mail.muni.cz>
 *           Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * The routines in this file implement the Real-time Transport Protocol,
 * RTP, as specified in RFC1889 with current updates under discussion in
 * the IETF audio/video transport working group. Portions of the code are
 * derived from the algorithms published in that specification.
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 * Copyright (c) 1998-2001 University College London
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
 *      This product includes software developed by the Computer Science
 *      Department at University College London and by the University of
 *      Southern California Information Sciences Institute. This product also
 *      includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Department, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *    
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "memory.h"
#include "debug.h"
#include "net_udp.h"
#include "crypto/random.h"
#include "compat/drand48.h"
#include "compat/gettimeofday.h"
#include "crypto/crypt_des.h"
#include "crypto/crypt_aes.h"
#include "compat/drand48.h"
#include "tv.h"
#include "crypto/md5.h"
#include "ntp.h"
#include "rtp.h"

/*
 * Encryption stuff.
 */
#define MAX_ENCRYPTION_PAD 16

static int rijndael_initialize(struct rtp *session, u_char * hash,
                               int hash_len);
static int rijndael_decrypt(struct rtp *session, unsigned char *data,
                            unsigned int size, unsigned char *initVec);
static int rijndael_encrypt(struct rtp *session, unsigned char *data,
                            unsigned int size, unsigned char *initVec);

static int des_initialize(struct rtp *session, u_char * hash, int hash_len);
static int des_decrypt(struct rtp *session, unsigned char *data,
                       unsigned int size, unsigned char *initVec);
static int des_encrypt(struct rtp *session, unsigned char *data,
                       unsigned int size, unsigned char *initVec);
static void rtp_process_data(struct rtp *session, uint32_t curr_rtp_ts,
               uint8_t *buffer, rtp_packet *packet, int buflen);

#define MAX_DROPOUT    3000
#define MAX_MISORDER   100
#define MIN_SEQUENTIAL 2

/*
 * Definitions for the RTP/RTCP packets on the wire...
 */

#define RTP_SEQ_MOD        0x10000
#define RTP_MAX_SDES_LEN   256

#define RTP_LOWER_LAYER_OVERHEAD 28     /* IPv4 + UDP */

#define RTCP_SR   200
#define RTCP_RR   201
#define RTCP_SDES 202
#define RTCP_BYE  203
#define RTCP_APP  204
#define RTCP_RX   205

typedef struct {
#ifdef WORDS_BIGENDIAN
        unsigned short version:2;       /* packet type            */
        unsigned short p:1;     /* padding flag           */
        unsigned short count:5; /* varies by payload type */
        unsigned short pt:8;    /* payload type           */
#else
        unsigned short count:5; /* varies by payload type */
        unsigned short p:1;     /* padding flag           */
        unsigned short version:2;       /* packet type            */
        unsigned short pt:8;    /* payload type           */
#endif
        uint16_t length;        /* packet length          */
} rtcp_common;

typedef struct {
        rtcp_common common;
        union {
                struct {
                        rtcp_sr sr;
                        rtcp_rr rr[1];  /* variable-length list */
                } sr;
                struct {
                        uint32_t ssrc;  /* source this RTCP packet is coming from */
                        rtcp_rr rr[1];  /* variable-length list */
                } rr;
                struct {
                        uint32_t ssrc;  /* source this RTCP packet is coming from */
                        rtcp_rr rr[1];  /* variable-length list */
                        rtcp_rx rx[1];
                } rx;
                struct rtcp_sdes_t {
                        uint32_t ssrc;
                        rtcp_sdes_item item[1]; /* list of SDES */
                } sdes;
                struct {
                        uint32_t ssrc[1];       /* list of sources */
                        /* can't express the trailing text... */
                } bye;
                struct {
                        uint32_t ssrc;
                        uint8_t name[4];
                        uint8_t data[1];
                } app;
        } r;
} rtcp_t;

typedef struct _rtcp_rr_wrapper {
        struct _rtcp_rr_wrapper *next;
        struct _rtcp_rr_wrapper *prev;
        uint32_t reporter_ssrc;
        rtcp_rr *rr;
        rtcp_rx *rx;
        struct timeval *ts;     /* Arrival time of this RR */
} rtcp_rr_wrapper;

/*
 * The RTP database contains source-specific information needed 
 * to make it all work. 
 */

uint32_t tmp_sec;
uint32_t tmp_frac;

typedef struct _source {
        struct _source *next;
        struct _source *prev;
        uint32_t ssrc;
        char *sdes_cname;
        char *sdes_name;
        char *sdes_email;
        char *sdes_phone;
        char *sdes_loc;
        char *sdes_tool;
        char *sdes_note;
        char *sdes_priv;
        rtcp_sr *sr;
        uint32_t last_sr_sec;
        uint32_t last_sr_frac;
        struct timeval last_active;
        int should_advertise_sdes;      /* TRUE if this source is a CSRC which we need to advertise SDES for */
        int sender;
        int got_bye;            /* TRUE if we've received an RTCP bye from this source */
        uint32_t base_seq;
        uint16_t max_seq;
        uint32_t bad_seq;
        uint32_t cycles;
        int received;
        int received_prior;
        int expected_prior;
        int probation;
        uint32_t jitter;
        uint32_t transit;
        uint32_t magic;         /* For debugging... */
} source;

/* The size of the hash table used to hold the source database. */
/* Should be large enough that we're unlikely to get collisions */
/* when sources are added, but not too large that we waste too  */
/* much memory. Sedgewick ("Algorithms", 2nd Ed, Addison-Wesley */
/* 1988) suggests that this should be around 1/10th the number  */
/* of entries that we expect to have in the database and should */
/* be a prime number. Everything continues to work if this is   */
/* too low, it just goes slower... for now we assume around 100 */
/* participants is a sensible limit so we set this to 11.       */
#define RTP_DB_SIZE	11

/*
 *  Options for an RTP session are stored in the "options" struct.
 */

typedef struct {
        int promiscuous_mode;
        int wait_for_rtcp;
        int filter_my_packets;
        int reuse_bufs;
} options;

/*
 * Encryption function types
 */
typedef int (*rtp_encrypt_func) (struct rtp *, unsigned char *data,
                                 unsigned int size, unsigned char *initvec);
typedef int (*rtp_decrypt_func) (struct rtp *, unsigned char *data,
                                 unsigned int size, unsigned char *initvec);

/*
 * The "struct rtp" defines an RTP session.
 */

struct rtp {
        socket_udp *rtp_socket;
        socket_udp *rtcp_socket;
        char *addr;
        uint16_t rx_port;
        uint16_t tx_port;
        int ttl;
        uint32_t my_ssrc;
        int last_advertised_csrc;
        source *db[RTP_DB_SIZE];
        rtcp_rr_wrapper rr[RTP_DB_SIZE][RTP_DB_SIZE];   /* Indexed by [hash(reporter)][hash(reportee)] */
        options *opt;
        uint8_t *userdata;
        int invalid_rtp_count;
        int invalid_rtcp_count;
        int bye_count;
        int csrc_count;
        int ssrc_count;
        int ssrc_count_prev;    /* ssrc_count at the time we last recalculated our RTCP interval */
        int sender_count;
        int initial_rtcp;
        int sending_bye;        /* TRUE if we're in the process of sending a BYE packet */
        double avg_rtcp_size;
        int we_sent;
        double rtcp_bw;         /* RTCP bandwidth fraction, in octets per second. */
        struct timeval last_update;
        struct timeval last_rtp_send_time;
        struct timeval last_rtcp_send_time;
        struct timeval next_rtcp_send_time;
        double rtcp_interval;
        int sdes_count_pri;
        int sdes_count_sec;
        int sdes_count_ter;
        uint16_t rtp_seq;
        uint32_t rtp_pcount;
        uint64_t rtp_bcount;
        int tfrc_on;            /* indicates TFRC congestion control */
        /* tfrc sender variables */
        uint32_t cmp_rtt;       /* rtt as computed by the sender */
        uint16_t new_rtt;       /* flag change in value of the RTT */
        /* tfrc recevier variables */
        uint32_t rcv_rtt;       /* rtt receiver extracts from rtp packets */

        char *encryption_algorithm;
        int encryption_enabled;
        rtp_encrypt_func encrypt_func;
        rtp_decrypt_func decrypt_func;
        int encryption_pad_length;
        union {
                struct {
                        keyInstance keyInstEncrypt;
                        keyInstance keyInstDecrypt;
                        cipherInstance cipherInst;
                } rijndael;
                struct {
                        char *encryption_key;
                } des;
        } crypto_state;
        rtp_callback callback;
        struct msghdr *mhdr;
        uint32_t magic;         /* For debugging...  */
};

static inline int filter_event(struct rtp *session, uint32_t ssrc)
{
        return session->opt->filter_my_packets
            && (ssrc == rtp_my_ssrc(session));
}

static uint32_t next_csrc(struct rtp *session)
{
        /* This returns each source marked "should_advertise_sdes" in turn. */
        int chain, cc;
        source *s;

        cc = 0;
        for (chain = 0; chain < RTP_DB_SIZE; chain++) {
                /* Check that the linked lists making up the chains in */
                /* the hash table are correctly linked together...     */
                for (s = session->db[chain]; s != NULL; s = s->next) {
                        if (s->should_advertise_sdes) {
                                if (cc == session->last_advertised_csrc) {
                                        session->last_advertised_csrc++;
                                        if (session->last_advertised_csrc ==
                                            session->csrc_count) {
                                                session->last_advertised_csrc =
                                                    0;
                                        }
                                        return s->ssrc;
                                } else {
                                        cc++;
                                }
                        }
                }
        }
        /* We should never get here... */
        abort();
}

static inline int ssrc_hash(uint32_t ssrc)
{
        /* Hash from an ssrc to a position in the source database.   */
        /* Assumes that ssrc values are uniformly distributed, which */
        /* should be true but probably isn't (Rosenberg has reported */
        /* that many implementations generate ssrc values which are  */
        /* not uniformly distributed over the space, and the H.323   */
        /* spec requires that they are non-uniformly distributed).   */
        /* This routine is written as a function rather than inline  */
        /* code to allow it to be made smart in future: probably we  */
        /* should run MD5 on the ssrc and derive a hash value from   */
        /* that, to ensure it's more uniformly distributed?          */
        return ssrc % RTP_DB_SIZE;
}

static void insert_rr(struct rtp *session, uint32_t reporter_ssrc, rtcp_rr * rr,
                      rtcp_rx * rx)
{
        /* Insert the reception report into the receiver report      */
        /* database. This database is a two dimensional table of     */
        /* rr_wrappers indexed by hashes of reporter_ssrc and        */
        /* reportee_src.  The rr_wrappers in the database are        */
        /* sentinels to reduce conditions in list operations.        */
        /* The ts is used to determine when to timeout this rr.      */

        rtcp_rr_wrapper *cur, *start;

        start = &session->rr[ssrc_hash(reporter_ssrc)][ssrc_hash(rr->ssrc)];
        cur = start->next;

        while (cur != start) {
                if (cur->reporter_ssrc == reporter_ssrc
                    && cur->rr->ssrc == rr->ssrc) {
                        /* Replace existing entry in the database  */
                        free(cur->rr);
                        free(cur->ts);
                        if (cur->rx)
                                free(cur->rx);
                        cur->rr = rr;
                        cur->rx = rx;
                        cur->ts = malloc(sizeof(struct timeval));
                        gettimeofday(cur->ts, NULL);
                        return;
                }
                cur = cur->next;
        }

        /* No entry in the database so create one now. */
        cur = (rtcp_rr_wrapper *) malloc(sizeof(rtcp_rr_wrapper));
        cur->reporter_ssrc = reporter_ssrc;
        cur->rr = rr;
        cur->rx = rx;
        cur->ts = malloc(sizeof(struct timeval));
        gettimeofday(cur->ts, NULL);
        /* Fix links */
        cur->next = start->next;
        cur->next->prev = cur;
        cur->prev = start;
        cur->prev->next = cur;

        debug_msg("Created new rr entry for 0x%08lx from source 0x%08lx\n",
                  rr->ssrc, reporter_ssrc);
        return;
}

static void remove_rr(struct rtp *session, uint32_t ssrc)
{
        /* Remove any RRs from "s" which refer to "ssrc" as either   */
        /* reporter or reportee.                                     */
        rtcp_rr_wrapper *start, *cur, *tmp;
        int i;

        /* Remove rows, i.e. ssrc == reporter_ssrc                   */
        for (i = 0; i < RTP_DB_SIZE; i++) {
                start = &session->rr[ssrc_hash(ssrc)][i];
                cur = start->next;
                while (cur != start) {
                        if (cur->reporter_ssrc == ssrc) {
                                tmp = cur;
                                cur = cur->prev;
                                tmp->prev->next = tmp->next;
                                tmp->next->prev = tmp->prev;
                                free(tmp->ts);
                                free(tmp->rr);
                                free(tmp);
                        }
                        cur = cur->next;
                }
        }

        /* Remove columns, i.e.  ssrc == reporter_ssrc */
        for (i = 0; i < RTP_DB_SIZE; i++) {
                start = &session->rr[i][ssrc_hash(ssrc)];
                cur = start->next;
                while (cur != start) {
                        if (cur->rr->ssrc == ssrc) {
                                tmp = cur;
                                cur = cur->prev;
                                tmp->prev->next = tmp->next;
                                tmp->next->prev = tmp->prev;
                                free(tmp->ts);
                                free(tmp->rr);
                                free(tmp);
                        }
                        cur = cur->next;
                }
        }
}

static void timeout_rr(struct rtp *session, struct timeval *curr_ts)
{
        /* Timeout any reception reports which have been in the database for more than 3 */
        /* times the RTCP reporting interval without refresh.                            */
        rtcp_rr_wrapper *start, *cur, *tmp;
        rtp_event event;
        int i, j;

        for (i = 0; i < RTP_DB_SIZE; i++) {
                for (j = 0; j < RTP_DB_SIZE; j++) {
                        start = &session->rr[i][j];
                        cur = start->next;
                        while (cur != start) {
                                if (tv_diff(*curr_ts, *(cur->ts)) >
                                    (session->rtcp_interval * 3)) {
                                        /* Signal the application... */
                                        if (!filter_event
                                            (session, cur->reporter_ssrc)) {
                                                event.ssrc = cur->reporter_ssrc;
                                                event.type = RR_TIMEOUT;
                                                event.data = cur->rr;
                                                session->callback(session,
                                                                  &event);
                                        }
                                        /* Delete this reception report... */
                                        tmp = cur;
                                        cur = cur->prev;
                                        tmp->prev->next = tmp->next;
                                        tmp->next->prev = tmp->prev;
                                        free(tmp->ts);
                                        free(tmp->rr);
                                        free(tmp);
                                }
                                cur = cur->next;
                        }
                }
        }
}

static const rtcp_rr *get_rr(struct rtp *session, uint32_t reporter_ssrc,
                             uint32_t reportee_ssrc)
{
        rtcp_rr_wrapper *cur, *start;

        start =
            &session->rr[ssrc_hash(reporter_ssrc)][ssrc_hash(reportee_ssrc)];
        cur = start->next;
        while (cur != start) {
                if (cur->reporter_ssrc == reporter_ssrc &&
                    cur->rr->ssrc == reportee_ssrc) {
                        return cur->rr;
                }
                cur = cur->next;
        }
        return NULL;
}

static inline void check_source(source * s)
{
#ifdef DEBUG
        assert(s != NULL);
        assert(s->magic == 0xc001feed);
#else
        UNUSED(s);
#endif
}

static inline void check_database(struct rtp *session)
{
        /* This routine performs a sanity check on the database. */
        /* This should not call any of the other routines which  */
        /* manipulate the database, to avoid common failures.    */
#ifdef DEBUG
        source *s;
        int source_count;
        int chain;

        assert(session != NULL);
        assert(session->magic == 0xfeedface);
        /* Check that we have a database entry for our ssrc... */
        /* We only do this check if ssrc_count > 0 since it is */
        /* performed during initialisation whilst creating the */
        /* source entry for my_ssrc.                           */
        if (session->ssrc_count > 0) {
                for (s = session->db[ssrc_hash(session->my_ssrc)]; s != NULL;
                     s = s->next) {
                        if (s->ssrc == session->my_ssrc) {
                                break;
                        }
                }
                assert(s != NULL);
        }

        source_count = 0;
        for (chain = 0; chain < RTP_DB_SIZE; chain++) {
                /* Check that the linked lists making up the chains in */
                /* the hash table are correctly linked together...     */
                for (s = session->db[chain]; s != NULL; s = s->next) {
                        check_source(s);
                        source_count++;
                        if (s->prev == NULL) {
                                assert(s == session->db[chain]);
                        } else {
                                assert(s->prev->next == s);
                        }
                        if (s->next != NULL) {
                                assert(s->next->prev == s);
                        }
                        /* Check that the SR is for this source... */
                        if (s->sr != NULL) {
                                assert(s->sr->ssrc == s->ssrc);
                        }
                }
        }
        /* Check that the number of entries in the hash table  */
        /* matches session->ssrc_count                         */
        assert(source_count == session->ssrc_count);
#else
        UNUSED(session);
#endif
}

static inline source *get_source(struct rtp *session, uint32_t ssrc)
{
        source *s;

        check_database(session);
        for (s = session->db[ssrc_hash(ssrc)]; s != NULL; s = s->next) {
                if (s->ssrc == ssrc) {
                        check_source(s);
                        return s;
                }
        }
        return NULL;
}

static source *really_create_source(struct rtp *session, uint32_t ssrc,
                                    int probation, source * s)
{
        /* Create a new source entry, and add it to the database.    */
        /* The database is a hash table, using the separate chaining */
        /* algorithm.                                                */
        rtp_event event;
        int h;

        check_database(session);
        /* This is a new source, we have to create it... */
        h = ssrc_hash(ssrc);
        s = (source *) malloc(sizeof(source));
        memset(s, 0, sizeof(source));
        s->magic = 0xc001feed;
        s->next = session->db[h];
        s->ssrc = ssrc;
        if (probation) {
                /* This is a probationary source, which only counts as */
                /* valid once several consecutive packets are received */
                s->probation = -1;
        } else {
                s->probation = 0;
        }

        gettimeofday(&(s->last_active), NULL);
        /* Now, add it to the database... */
        if (session->db[h] != NULL) {
                session->db[h]->prev = s;
        }
        session->db[ssrc_hash(ssrc)] = s;
        session->ssrc_count++;
        check_database(session);

        debug_msg("Created database entry 0x%08x (%d sources)\n", ssrc,
                  session->ssrc_count);
        if (ssrc != session->my_ssrc) {
                /* Do not send during rtp_init since application cannot map the address */
                /* of the rtp session to anything since rtp_init has not returned yet.  */
                if (!filter_event(session, ssrc)) {
                        event.ssrc = ssrc;
                        event.type = SOURCE_CREATED;
                        event.data = NULL;
                        session->callback(session, &event);
                }
        }

        return s;
}

static inline source *create_source(struct rtp *session, uint32_t ssrc,
                                    int probation)
{
        source *s = get_source(session, ssrc);
        if (s == NULL) {
                return really_create_source(session, ssrc, probation, s);
        }
        /* Source is already in the database... Mark it as */
        /* active and exit (this is the common case...)    */
        gettimeofday(&(s->last_active), NULL);
        return s;
}

static void delete_source(struct rtp *session, uint32_t ssrc)
{
        /* Remove a source from the RTP database... */
        source *s = get_source(session, ssrc);
        int h = ssrc_hash(ssrc);
        rtp_event event;
        struct timeval event_ts;

        assert(s != NULL);      /* Deleting a source which doesn't exist is an error... */

        gettimeofday(&event_ts, NULL);

        check_source(s);
        check_database(session);
        if (session->db[h] == s) {
                /* It's the first entry in this chain... */
                session->db[h] = s->next;
                if (s->next != NULL) {
                        s->next->prev = NULL;
                }
        } else {
                assert(s->prev != NULL);        /* Else it would be the first in the chain... */
                s->prev->next = s->next;
                if (s->next != NULL) {
                        s->next->prev = s->prev;
                }
        }
        /* Free the memory allocated to a source... */
        if (s->sdes_cname != NULL)
                free(s->sdes_cname);
        if (s->sdes_name != NULL)
                free(s->sdes_name);
        if (s->sdes_email != NULL)
                free(s->sdes_email);
        if (s->sdes_phone != NULL)
                free(s->sdes_phone);
        if (s->sdes_loc != NULL)
                free(s->sdes_loc);
        if (s->sdes_tool != NULL)
                free(s->sdes_tool);
        if (s->sdes_note != NULL)
                free(s->sdes_note);
        if (s->sdes_priv != NULL)
                free(s->sdes_priv);
        if (s->sr != NULL)
                free(s->sr);

        remove_rr(session, ssrc);

        /* Reduce our SSRC count, and perform reverse reconsideration on the RTCP */
        /* reporting interval (draft-ietf-avt-rtp-new-05.txt, section 6.3.4). To  */
        /* make the transmission rate of RTCP packets more adaptive to changes in */
        /* group membership, the following "reverse reconsideration" algorithm    */
        /* SHOULD be executed when a BYE packet is received that reduces members  */
        /* to a value less than pmembers:                                         */
        /* o  The value for tn is updated according to the following formula:     */
        /*       tn = tc + (members/pmembers)(tn - tc)                            */
        /* o  The value for tp is updated according the following formula:        */
        /*       tp = tc - (members/pmembers)(tc - tp).                           */
        /* o  The next RTCP packet is rescheduled for transmission at time tn,    */
        /*    which is now earlier.                                               */
        /* o  The value of pmembers is set equal to members.                      */
        session->ssrc_count--;
        if (session->ssrc_count < session->ssrc_count_prev) {
                gettimeofday(&(session->next_rtcp_send_time), NULL);
                gettimeofday(&(session->last_rtcp_send_time), NULL);
                tv_add(&(session->next_rtcp_send_time),
                       (session->ssrc_count / session->ssrc_count_prev)
                       * tv_diff(session->next_rtcp_send_time, event_ts));
                tv_add(&(session->last_rtcp_send_time),
                       -((session->ssrc_count / session->ssrc_count_prev)
                         * tv_diff(event_ts, session->last_rtcp_send_time)));
                session->ssrc_count_prev = session->ssrc_count;
        }

        /* Reduce our csrc count... */
        if (s->should_advertise_sdes == TRUE) {
                session->csrc_count--;
        }
        if (session->last_advertised_csrc == session->csrc_count) {
                session->last_advertised_csrc = 0;
        }

        /* Signal to the application that this source is dead... */
        if (!filter_event(session, ssrc)) {
                event.ssrc = ssrc;
                event.type = SOURCE_DELETED;
                event.data = NULL;
                session->callback(session, &event);
        }
        free(s);
        check_database(session);
}

static inline void init_seq(source * s, uint16_t seq)
{
        /* Taken from draft-ietf-avt-rtp-new-01.txt */
        check_source(s);
        s->base_seq = seq;
        s->max_seq = seq;
        s->bad_seq = RTP_SEQ_MOD + 1;
        s->cycles = 0;
        s->received = 0;
        s->received_prior = 0;
        s->expected_prior = 0;
}

static int update_seq(source * s, uint16_t seq)
{
        /* Taken from draft-ietf-avt-rtp-new-01.txt */
        uint16_t udelta = seq - s->max_seq;

        /*
         * Source is not valid until MIN_SEQUENTIAL packets with
         * sequential sequence numbers have been received.
         */
        check_source(s);
        if (s->probation) {
                /* packet is in sequence */
                if (seq == s->max_seq + 1) {
                        s->probation--;
                        s->max_seq = seq;
                        if (s->probation == 0) {
                                init_seq(s, seq);
                                s->received++;
                                return 1;
                        }
                } else {
                        s->probation = MIN_SEQUENTIAL - 1;
                        s->max_seq = seq;
                }
                return 0;
        } else if (udelta < MAX_DROPOUT) {
                /* in order, with permissible gap */
                if (seq < s->max_seq) {
                        /*
                         * Sequence number wrapped - count another 64K cycle.
                         */
                        s->cycles += RTP_SEQ_MOD;
                }
                s->max_seq = seq;
        } else if (udelta <= RTP_SEQ_MOD - MAX_MISORDER) {
                /* the sequence number made a very large jump */
                if (seq == s->bad_seq) {
                        /*
                         * Two sequential packets -- assume that the other side
                         * restarted without telling us so just re-sync
                         * (i.e., pretend this was the first packet).
                         */
                        init_seq(s, seq);
                } else {
                        s->bad_seq = (seq + 1) & (RTP_SEQ_MOD - 1);
                        return 0;
                }
        } else {
                /* duplicate or reordered packet */
        }
        s->received++;
        return 1;
}

static double rtcp_interval(struct rtp *session)
{
        /* Minimum average time between RTCP packets from this site (in   */
        /* seconds).  This time prevents the reports from `clumping' when */
        /* sessions are small and the law of large numbers isn't helping  */
        /* to smooth out the traffic.  It also keeps the report interval  */
        /* from becoming ridiculously small during transient outages like */
        /* a network partition.                                           */
        double const RTCP_MIN_TIME = 5.0;
        /* Fraction of the RTCP bandwidth to be shared among active       */
        /* senders.  (This fraction was chosen so that in a typical       */
        /* session with one or two active senders, the computed report    */
        /* time would be roughly equal to the minimum report time so that */
        /* we don't unnecessarily slow down receiver reports.) The        */
        /* receiver fraction must be 1 - the sender fraction.             */
        double const RTCP_SENDER_BW_FRACTION = 0.25;
        double const RTCP_RCVR_BW_FRACTION = (1 - RTCP_SENDER_BW_FRACTION);
        /* To compensate for "unconditional reconsideration" converging   */
        /* to a value below the intended average.                         */
        double const COMPENSATION = 2.71828 - 1.5;

        double t;               /* interval */
        double rtcp_min_time = RTCP_MIN_TIME;
        int n;                  /* no. of members for computation */
        double rtcp_bw = session->rtcp_bw;

        /* Very first call at application start-up uses half the min      */
        /* delay for quicker notification while still allowing some time  */
        /* before reporting for randomization and to learn about other    */
        /* sources so the report interval will converge to the correct    */
        /* interval more quickly.                                         */
        if (session->initial_rtcp) {
                rtcp_min_time /= 2;
        }

        /* If there were active senders, give them at least a minimum     */
        /* share of the RTCP bandwidth.  Otherwise all participants share */
        /* the RTCP bandwidth equally.                                    */
        if (session->sending_bye) {
                n = session->bye_count;
        } else {
                n = session->ssrc_count;
        }
        if (session->sender_count > 0
            && session->sender_count < n * RTCP_SENDER_BW_FRACTION) {
                if (session->we_sent) {
                        rtcp_bw *= RTCP_SENDER_BW_FRACTION;
                        n = session->sender_count;
                } else {
                        rtcp_bw *= RTCP_RCVR_BW_FRACTION;
                        n -= session->sender_count;
                }
        }

        /* The effective number of sites times the average packet size is */
        /* the total number of octets sent when each site sends a report. */
        /* Dividing this by the effective bandwidth gives the time        */
        /* interval over which those packets must be sent in order to     */
        /* meet the bandwidth target, with a minimum enforced.  In that   */
        /* time interval we send one report so this time is also our      */
        /* average time between reports.                                  */
        t = session->avg_rtcp_size * n / rtcp_bw;
        if (t < rtcp_min_time) {
                t = rtcp_min_time;
        }
        session->rtcp_interval = t;

        /* To avoid traffic bursts from unintended synchronization with   */
        /* other sites, we then pick our actual next report interval as a */
        /* random number uniformly distributed between 0.5*t and 1.5*t.   */
        return (t * (drand48() + 0.5)) / COMPENSATION;
}

#define MAXCNAMELEN	255

static char *get_cname(socket_udp * s)
{
        /* Set the CNAME. This is "user@hostname" or just "hostname" if the username cannot be found. */
        const char *hname;
        char *uname;
        char *cname;
#ifndef WIN32
        struct passwd *pwent;
#else
        char *name;
        DWORD namelen;
#endif

        cname = (char *)malloc(MAXCNAMELEN + 1);
        cname[0] = '\0';

        /* First, fill in the username... */
#ifdef WIN32
        name = NULL;
        namelen = 0;
        GetUserName(NULL, &namelen);
        if (namelen > 0) {
                name = (char *)malloc(namelen + 1);
                GetUserName(name, &namelen);
        } else {
                uname = getenv("USER");
                if (uname != NULL) {
                        name = strdup(uname);
                }
        }
        if (name != NULL) {
                strncpy(cname, name, MAXCNAMELEN - 1);
                strcat(cname, "@");
                free(name);
        }
#else
        uname = NULL;
        pwent = getpwuid(getuid());
        if (pwent != NULL) {
                uname = pwent->pw_name;
        }
        if (uname != NULL) {
                strncpy(cname, uname, MAXCNAMELEN - 1);
                strcat(cname, "@");
        }
#endif

        /* Now the hostname. Must be dotted-quad IP address. */
        hname = udp_host_addr(s);
        if (hname == NULL) {
                /* If we can't get our IP address we use the loopback address... */
                /* This is horrible, but it stops the code from failing.         */
                hname = "127.0.0.1";
        }
        strncpy(cname + strlen(cname), hname, MAXCNAMELEN - strlen(cname));
        return cname;
}

static void init_opt(struct rtp *session)
{
        /* Default option settings. */
        rtp_set_option(session, RTP_OPT_PROMISC, FALSE);
        rtp_set_option(session, RTP_OPT_WEAK_VALIDATION, FALSE);
        rtp_set_option(session, RTP_OPT_FILTER_MY_PACKETS, FALSE);
        rtp_set_option(session, RTP_OPT_REUSE_PACKET_BUFS, FALSE);
}

static void init_rng(const char *s)
{
        static uint32_t seed;
        if (seed == 0) {
                pid_t p = getpid();

#ifdef WIN32
                int32_t i, n;
#endif                          /* WIN32 */

                if(s) {
                        while (*s) {
                                seed += (uint32_t) * s++;
                                seed = seed * 31 + 1;
                        }
                } else {
                        seed = clock();
                }
                seed = 1 + seed * 31 + (uint32_t) p;
                srand48(seed);
                /* At time of writing we use srand48 -> srand on Win32
                   which is only 16 bit. lrand48 -> rand which is only
                   15 bits, step a int way through table seq */
#ifdef WIN32
                n = (seed >> 16) & 0xffff;
                for (i = 0; i < n; i++) {
                        seed = lrand48();
                }
#endif                          /* WIN32 */
        }
}

/* See rtp_init_if(); calling rtp_init() is just like calling
 * rtp_init_if() with a NULL interface argument.
 */

/**
 * rtp_init:
 * @addr: IP destination of this session (unicast or multicast),
 * as an ASCII string.  May be a host name, which will be looked up,
 * or may be an IPv4 dotted quad or IPv6 literal adddress.
 * @rx_port: The port to which to bind the UDP socket
 * @tx_port: The port to which to send UDP packets
 * @ttl: The TTL with which to send multicasts
 * @rtcp_bw: The total bandwidth (in units of bytes per second) that is
 * allocated to RTCP.
 * @callback: See section on #rtp_callback.
 * @userdata: Opaque data associated with the session.  See
 * rtp_get_userdata().
 *
 *
 * Returns: An opaque session identifier to be used in future calls to
 * the RTP library functions, or NULL on failure.
 */
struct rtp *rtp_init(const char *addr,
                     uint16_t rx_port, uint16_t tx_port,
                     int ttl, double rtcp_bw,
                     int tfrc_on, rtp_callback callback, uint8_t * userdata,
                     bool use_ipv6)
{
        return rtp_init_if(addr, NULL, rx_port, tx_port, ttl, rtcp_bw, tfrc_on,
                           callback, userdata, use_ipv6);
}

/**
 * rtp_init_if:
 * @addr: IP destination of this session (unicast or multicast),
 * as an ASCII string.  May be a host name, which will be looked up,
 * or may be an IPv4 dotted quad or IPv6 literal adddress.
 * @iface: If the destination of the session is multicast,
 * the optional interface to bind to.  May be NULL, in which case
 * the default multicast interface as determined by the system
 * will be used.
 * @rx_port: The port to which to bind the UDP socket
 * @tx_port: The port to which to send UDP packets
 * @ttl: The TTL with which to send multicasts
 * @rtcp_bw: The total bandwidth (in units of ___) that is
 * allocated to RTCP.
 * @callback: See section on #rtp_callback.
 * @userdata: Opaque data associated with the session.  See
 * rtp_get_userdata().
 *
 * Creates and initializes an RTP session.
 *
 * Returns: An opaque session identifier to be used in future calls to
 * the RTP library functions, or NULL on failure.
 */
struct rtp *rtp_init_if(const char *addr, char *iface,
                        uint16_t rx_port, uint16_t tx_port,
                        int ttl, double rtcp_bw,
                        int tfrc_on, rtp_callback callback, uint8_t * userdata,
                        bool use_ipv6)
{
        struct rtp *session;
        int i, j;
        char *cname;

        if (ttl < 0) {
                debug_msg("ttl must be greater than zero\n");
                return NULL;
        }
        if (rx_port % 2) {
                debug_msg("rx_port must be even\n");
                return NULL;
        }
        if (tx_port % 2) {
                debug_msg("tx_port must be even\n");
                return NULL;
        }

        session = (struct rtp *)malloc(sizeof(struct rtp));
        memset(session, 0, sizeof(struct rtp));

        session->magic = 0xfeedface;
        session->opt = (options *) malloc(sizeof(options));
        session->userdata = userdata;
        session->addr = strdup(addr);
        session->rx_port = rx_port;
        session->tx_port = tx_port;
        session->ttl = min(ttl, 127);
        session->rtp_socket = udp_init_if(addr, iface, rx_port, tx_port, ttl, use_ipv6);
        session->rtcp_socket =
            udp_init_if(addr, iface, (uint16_t) (rx_port + (rx_port ? 1 : 0)),
                        (uint16_t) (tx_port + 1), ttl, use_ipv6);

        init_opt(session);

        if (session->rtp_socket == NULL || session->rtcp_socket == NULL) {
                printf("Unable to open network\n");
                free(session);
                return NULL;
        }

        init_rng(udp_host_addr(session->rtp_socket));

        session->my_ssrc = (uint32_t) lrand48();
        session->callback = callback;
        session->invalid_rtp_count = 0;
        session->invalid_rtcp_count = 0;
        session->bye_count = 0;
        session->csrc_count = 0;
        session->ssrc_count = 0;
        session->ssrc_count_prev = 0;
        session->sender_count = 0;
        session->initial_rtcp = TRUE;
        session->sending_bye = FALSE;
        session->avg_rtcp_size = -1;    /* Sentinal value: reception of first packet starts initial value... */
        session->we_sent = FALSE;
        session->rtcp_bw = rtcp_bw;
        session->sdes_count_pri = 0;
        session->sdes_count_sec = 0;
        session->sdes_count_ter = 0;
        session->rtp_seq = (uint16_t) lrand48();
        session->rtp_pcount = 0;
        session->mhdr = NULL;
        session->tfrc_on = tfrc_on;
        session->rtp_bcount = 0;
        gettimeofday(&(session->last_update), NULL);
        gettimeofday(&(session->last_rtcp_send_time), NULL);
        gettimeofday(&(session->next_rtcp_send_time), NULL);
        session->encryption_enabled = 0;
        session->encryption_algorithm = NULL;

        /* Calculate when we're supposed to send our first RTCP packet... */
        tv_add(&(session->next_rtcp_send_time), rtcp_interval(session));

        /* Initialise the source database... */
        for (i = 0; i < RTP_DB_SIZE; i++) {
                session->db[i] = NULL;
        }
        session->last_advertised_csrc = 0;

        /* Initialize sentinels in rr table */
        for (i = 0; i < RTP_DB_SIZE; i++) {
                for (j = 0; j < RTP_DB_SIZE; j++) {
                        session->rr[i][j].next = &session->rr[i][j];
                        session->rr[i][j].prev = &session->rr[i][j];
                }
        }

        /* Create a database entry for ourselves... */
        create_source(session, session->my_ssrc, FALSE);
        cname = get_cname(session->rtp_socket);
        rtp_set_sdes(session, session->my_ssrc, RTCP_SDES_CNAME, cname,
                     strlen(cname));
        free(cname);            /* cname is copied by rtp_set_sdes()... */

        return session;
}

/**
 * rtp_set_my_ssrc:
 * @session: the RTP session 
 * @ssrc: the SSRC to be used by the RTP session
 * 
 * This function coerces the local SSRC identifer to be ssrc.  For
 * this function to succeed it must be called immediately after
 * rtp_init or rtp_init_if.  The intended purpose of this
 * function is to co-ordinate SSRC's between layered sessions, it
 * should not be used otherwise.
 *
 * Returns: TRUE on success, FALSE otherwise.  
 */
int rtp_set_my_ssrc(struct rtp *session, uint32_t ssrc)
{
        source *s;
        uint32_t h;

        if (session->ssrc_count != 1 && session->sender_count != 0) {
                return FALSE;
        }
        /* Remove existing source */
        h = ssrc_hash(session->my_ssrc);
        s = session->db[h];
        session->db[h] = NULL;
        /* Fill in new ssrc       */
        session->my_ssrc = ssrc;
        s->ssrc = ssrc;
        h = ssrc_hash(ssrc);
        /* Put source back        */
        session->db[h] = s;
        return TRUE;
}

/**
 * rtp_set_option:
 * @session: The RTP session.
 * @optname: The option name, see #rtp_option.
 * @optval: The value to set.
 *
 * Sets the value of a session option.  See #rtp_option for
 * documentation on the options and their legal values.
 *
 * Returns: TRUE on success, else FALSE.
 */
int rtp_set_option(struct rtp *session, rtp_option optname, int optval)
{
        switch (optname) {
        case RTP_OPT_WEAK_VALIDATION:
                session->opt->wait_for_rtcp = optval;
                break;
        case RTP_OPT_PROMISC:
                session->opt->promiscuous_mode = optval;
                break;
        case RTP_OPT_FILTER_MY_PACKETS:
                session->opt->filter_my_packets = optval;
                break;
        case RTP_OPT_REUSE_PACKET_BUFS:
                session->opt->reuse_bufs = optval;
                break;
        default:
                debug_msg
                    ("Ignoring unknown option (%d) in call to rtp_set_option().\n",
                     optname);
                return FALSE;
        }
        return TRUE;
}

/**
 * rtp_get_option:
 * @session: The RTP session.
 * @optname: The option name, see #rtp_option.
 * @optval: The return value.
 *
 * Retrieves the value of a session option.  See #rtp_option for
 * documentation on the options and their legal values.
 *
 * Returns: TRUE and the value of the option in optval on success, else FALSE.
 */
int rtp_get_option(struct rtp *session, rtp_option optname, int *optval)
{
        switch (optname) {
        case RTP_OPT_WEAK_VALIDATION:
                *optval = session->opt->wait_for_rtcp;
                break;
        case RTP_OPT_PROMISC:
                *optval = session->opt->promiscuous_mode;
                break;
        case RTP_OPT_FILTER_MY_PACKETS:
                *optval = session->opt->filter_my_packets;
                break;
        case RTP_OPT_REUSE_PACKET_BUFS:
                *optval = session->opt->reuse_bufs;
                break;
        default:
                *optval = 0;
                debug_msg
                    ("Ignoring unknown option (%d) in call to rtp_get_option().\n",
                     optname);
                return FALSE;
        }
        return TRUE;
}

/**
 * rtp_get_userdata:
 * @session: The RTP session.
 *
 * This function returns the userdata pointer that was passed to the
 * rtp_init() or rtp_init_if() function when creating this session.
 *
 * Returns: pointer to userdata.
 */
uint8_t *rtp_get_userdata(struct rtp * session)
{
        check_database(session);
        return session->userdata;
}

/**
 * rtp_my_ssrc:
 * @session: The RTP Session.
 *
 * Returns: The SSRC we are currently using in this session. Note that our
 * SSRC can change at any time (due to collisions) so applications must not
 * store the value returned, but rather should call this function each time 
 * they need it.
 */
uint32_t rtp_my_ssrc(struct rtp * session)
{
        check_database(session);
        return session->my_ssrc;
}

static int validate_rtp2(rtp_packet * packet, int len, int vlen)
{
        /* Check for valid payload types..... 72-76 are RTCP payload type numbers, with */
        /* the high bit missing so we report that someone is running on the wrong port. */
        if (packet->pt >= 72 && packet->pt <= 76) {
                debug_msg("rtp_header_validation: payload-type invalid");
                if (packet->m) {
                        debug_msg(" (RTCP packet on RTP port?)");
                }
                debug_msg("\n");
                return FALSE;
        }
        /* Check that the length of the packet is sensible... */
        if (len < (vlen + (4 * packet->cc))) {
                debug_msg
                    ("rtp_header_validation: packet length is smaller than the header\n");
                return FALSE;
        }
        /* Check that the amount of padding specified is sensible. */
        /* Note: have to include the size of any extension header! */
        if (packet->p) {
                int payload_len = len - vlen - (packet->cc * 4);
                if (packet->x) {
                        /* extension header and data */
                        payload_len -= 4 * (1 + packet->extn_len);
                }
                if (packet->data[packet->data_len - 1] > payload_len) {
                        debug_msg
                            ("rtp_header_validation: padding greater than payload length\n");
                        return FALSE;
                }
                if (packet->data[packet->data_len - 1] < 1) {
                        debug_msg("rtp_header_validation: padding zero\n");
                        return FALSE;
                }
        }
        return TRUE;
}

static inline int
validate_rtp(struct rtp *session, rtp_packet * packet, int len, int vlen)
{
        /* This function checks the header info to make sure that the packet */
        /* is valid. We return TRUE if the packet is valid, FALSE otherwise. */
        /* See Appendix A.1 of the RTP specification.                        */

        /* We only accept RTPv2 packets... */
        if (packet->v != 2) {
                debug_msg("rtp_header_validation: v != 2\n");
                return FALSE;
        }

        if (!session->opt->wait_for_rtcp) {
                /* We prefer speed over accuracy... */
                return TRUE;
        }
        return validate_rtp2(packet, len, vlen);
}

static void compute_loss_intervals(struct rtp *session, rtp_packet * packet)
{
        UNUSED(session);
        UNUSED(packet);

}

static void
process_rtp(struct rtp *session, uint32_t curr_rtp_ts, rtp_packet * packet,
            source * s)
{
        int i, d, transit;
        rtp_event event;

        if (packet->cc > 0) {
                for (i = 0; i < packet->cc; i++) {
                        create_source(session, packet->csrc[i], FALSE);
                }
        }
        /* Update the source database... */
        if (s->sender == FALSE) {
                s->sender = TRUE;
                session->sender_count++;
        }

        if (session->tfrc_on)
                compute_loss_intervals(session, packet);

        transit = curr_rtp_ts - packet->ts;
        d = transit - s->transit;
        s->transit = transit;
        if (d < 0) {
                d = -d;
        }
        s->jitter += d - ((s->jitter + 8) / 16);

        /* Callback to the application to process the packet... */
        if (!filter_event(session, packet->ssrc)) {
                event.ssrc = packet->ssrc;
                event.type = RX_RTP;

                //              printf("This packet is going to have size %d\n",sizeof(packet));
                event.data = (void *)packet;    /* The callback function MUST free this! */
                session->callback(session, &event);
        }
}

void rtp_set_recv_iov(struct rtp *session, struct msghdr *m)
{
        session->mhdr = m;
}

int rtp_recv_push_data(struct rtp *session,
                char *data, int buflen, uint32_t curr_rtp_ts)
{
        rtp_packet *packet = NULL;
        uint8_t *buffer = NULL;

        if (!session->opt->reuse_bufs || (packet == NULL)) {
                packet = (rtp_packet *) malloc(RTP_MAX_PACKET_LEN);
                buffer = ((uint8_t *) packet) + RTP_PACKET_HEADER_SIZE;
        }

        memcpy(buffer, data, buflen);

        rtp_process_data(session, curr_rtp_ts, buffer, packet, buflen);

        return buflen;
}

static void rtp_recv_data(struct rtp *session, uint32_t curr_rtp_ts)
{
        int buflen;
        rtp_packet *packet = NULL;
        uint8_t *buffer = NULL;

        if (!session->opt->reuse_bufs || (packet == NULL)) {
                packet = (rtp_packet *) malloc(RTP_MAX_PACKET_LEN);
                buffer = ((uint8_t *) packet) + RTP_PACKET_HEADER_SIZE;
        }

        buflen =
            udp_recv(session->rtp_socket, (char *)buffer,
                     RTP_MAX_PACKET_LEN - RTP_PACKET_HEADER_SIZE);

        rtp_process_data(session, curr_rtp_ts, buffer, packet, buflen);
}

static void rtp_process_data(struct rtp *session, uint32_t curr_rtp_ts,
               uint8_t *buffer, rtp_packet *packet, int buflen)
{
        /* This routine preprocesses an incoming RTP packet, deciding whether to process it. */
        uint8_t *buffer_vlen = NULL;
        int vlen = 12;          /* vlen = 12 | 16 | 20 */
        source *s;

        if (buflen > 0) {
                if (session->encryption_enabled) {
                        uint8_t initVec[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                        (session->decrypt_func) (session, buffer, buflen,
                                                 initVec);
                }

                /* figure out header lenght based on tfrc_on */
                /* might as well extract rtt and send_ts     */
                vlen = 12;
                if (session->tfrc_on) {
                        vlen += 4;
                        packet->send_ts = ntohl(packet->send_ts);
                        /* rtt is present in RTP packet - XXX */
                        if (packet->pt & 64) {
                                /* rtt is present in RTP packet - XXX */
                                vlen += 4;
                                packet->rtt = ntohl(packet->rtt);
                                //printf ("\n%8d %8d %8d", packet->send_ts, ntohl(packet->ts), packet->rtt );
                        }
                }
                buffer_vlen = buffer + vlen;

                /* Convert header fields to host byte order... */
                packet->seq = ntohs(packet->seq);
                packet->ts = ntohl(packet->ts);
                packet->ssrc = ntohl(packet->ssrc);
                /* Setup internal pointers, etc... */
                if (packet->cc) {
                        int i;
                        packet->csrc = (uint32_t *) (buffer_vlen);
                        for (i = 0; i < packet->cc; i++) {
                                packet->csrc[i] = ntohl(packet->csrc[i]);
                        }
                } else {
                        packet->csrc = NULL;
                }
                if (packet->x) {
                        packet->extn = buffer_vlen + (packet->cc * 4);
                        packet->extn_len =
                            (packet->extn[2] << 8) | packet->extn[3];
                        packet->extn_type =
                            (packet->extn[0] << 8) | packet->extn[1];
                } else {
                        packet->extn = NULL;
                        packet->extn_len = 0;
                        packet->extn_type = 0;
                }
                packet->data = (char *)(buffer_vlen + (packet->cc * 4));
                packet->data_len = buflen - (packet->cc * 4) - vlen;
                if (packet->extn != NULL) {
                        packet->data += ((packet->extn_len + 1) * 4);
                        packet->data_len -= ((packet->extn_len + 1) * 4);
                }
                if (validate_rtp(session, packet, buflen, vlen)) {
                        if (session->opt->wait_for_rtcp) {
                                s = create_source(session, packet->ssrc, TRUE);
                        } else {
                                s = get_source(session, packet->ssrc);
                        }
                        if (session->opt->promiscuous_mode) {
                                if (s == NULL) {
                                        create_source(session, packet->ssrc,
                                                      FALSE);
                                        s = get_source(session, packet->ssrc);
                                }
                                process_rtp(session, curr_rtp_ts, packet, s);
                                return; /* We don't free "packet", that's done by the callback function... */
                        }
                        if (s != NULL) {
                                if (s->probation == -1) {
                                        s->probation = MIN_SEQUENTIAL;
                                        s->max_seq = packet->seq - 1;
                                }
                                if (update_seq(s, packet->seq)) {
                                        process_rtp(session, curr_rtp_ts,
                                                    packet, s);
                                        return; /* we don't free "packet", that's done by the callback function... */
                                } else {
                                        /* This source is still on probation... */
                                        debug_msg
                                            ("RTP packet from probationary source ignored...\n");
                                }
                        } else {
                                /* debug_msg("RTP packet from unknown source ignored\n"); */
                        }
                } else {
                        session->invalid_rtp_count++;
                        debug_msg("Invalid RTP packet discarded\n");
                }

                if (!session->opt->reuse_bufs) {
                        free(packet);
                }
        }
}

static int validate_rtcp(uint8_t * packet, int len)
{
        /* Validity check for a compound RTCP packet. This function returns */
        /* TRUE if the packet is okay, FALSE if the validity check fails.   */
        /*                                                                  */
        /* The following checks can be applied to RTCP packets [RFC1889]:   */
        /* o RTP version field must equal 2.                                */
        /* o The payload type field of the first RTCP packet in a compound  */
        /*   packet must be equal to SR or RR.                              */
        /* o The padding bit (P) should be zero for the first packet of a   */
        /*   compound RTCP packet because only the last should possibly     */
        /*   need padding.                                                  */
        /* o The length fields of the individual RTCP packets must total to */
        /*   the overall length of the compound RTCP packet as received.    */

        rtcp_t *pkt = (rtcp_t *) packet;
        rtcp_t *end = (rtcp_t *) (((char *)pkt) + len);
        rtcp_t *r = pkt;
        int l = 0;
        int pc = 1;
        int p = 0;
        int is_okay = TRUE;

        /* All RTCP packets must be compound packets (RFC1889, section 6.1) */
        if (((ntohs(pkt->common.length) + 1) * 4) == len) {
                debug_msg("Bogus RTCP packet: not a compound packet\n");
                is_okay = FALSE;
        }

        /* The RTCP packet must not be larger than the surrounding UDP packet */
        if (((ntohs(pkt->common.length) + 1) * 4) > len) {
                debug_msg
                    ("Bogus RTCP packet: length exceeds length of container UDP packet\n");
                return FALSE;
        }

        /* Check the RTCP version, payload type and padding of the first in  */
        /* the compund RTCP packet...                                        */
        if (pkt->common.version != 2) {
                debug_msg
                    ("Bogus RTCP packet: version number != 2 in the first sub-packet\n");
                is_okay = FALSE;
        }
        if (pkt->common.p != 0) {
                debug_msg
                    ("Bogus RTCP packet: padding bit is set on first packet in compound\n");
                is_okay = FALSE;
        }
        if ((pkt->common.pt != RTCP_SR) && (pkt->common.pt != RTCP_RR)) {
                if (pkt->common.pt == RTCP_SDES) {
                        debug_msg
                            ("Bogus RTCP packet: compound packet starts with SDES not SR or RR\n");
                } else if (pkt->common.pt == RTCP_APP) {
                        debug_msg
                            ("Bogus RTCP packet: compound packet starts with APP not SR or RR\n");
                } else if (pkt->common.pt == RTCP_BYE) {
                        debug_msg
                            ("Bogus RTCP packet: compound packet starts with BYE not SR or RR\n");
                } else {
                        debug_msg
                            ("Bogus RTCP packet: compound packet starts with packet type %d not SR or RR\n",
                             pkt->common.pt);
                }
                is_okay = FALSE;
        }

        /* Check all following parts of the compund RTCP packet. The RTP version */
        /* number must be 2, and the padding bit must be zero on all apart from  */
        /* the last packet. Try to validate the format of each sub-packet.       */
        do {
                if (p == 1) {
                        debug_msg
                            ("Bogus RTCP packet: padding bit set in sub-packet %d which is not final sub-packet\n",
                             pc);
                        is_okay = FALSE;
                }
                if (r->common.p) {
                        p = 1;
                }
                if (r->common.version != 2) {
                        debug_msg
                            ("Bogus RTCP packet: version number != 2 in sub-packet %d\n",
                             pc);
                        is_okay = FALSE;
                }

                if (pkt->common.pt == RTCP_SR) {
                        if (((ntohs(pkt->common.length) + 1) * 4) <
                            (28 + (pkt->common.count * 24))) {
                                debug_msg
                                    ("Bogus RTCP packet: SR packet is too short (length=%d)\n",
                                     ntohs(pkt->common.length));
                                is_okay = FALSE;
                        }
                } else if (pkt->common.pt == RTCP_RR) {
                        if (((ntohs(pkt->common.length) + 1) * 4) <
                            (8 + (pkt->common.count * 24))) {
                                debug_msg
                                    ("Bogus RTCP packet: RR packet is too short (length=%d)\n",
                                     ntohs(pkt->common.length));
                                is_okay = FALSE;
                        }
                } else if (pkt->common.pt == RTCP_SDES) {
                        /* Parse and validate the SDES packet */
                        int count = pkt->common.count;
                        struct rtcp_sdes_t *sd = &pkt->r.sdes;
                        rtcp_sdes_item *rsp;
                        rtcp_sdes_item *rspn;
                        rtcp_sdes_item *sdes_end =
                            (rtcp_sdes_item *) ((uint32_t *) pkt +
                                                pkt->common.length + 1);

                        while (--count >= 0) {
                                rsp = &sd->item[0];
                                if ((char *)rsp > (char *)end) {
                                        debug_msg
                                            ("Bogus RTCP packet: SDES longer than UDP packet: no terminating null item?\n");
                                        is_okay = FALSE;
                                        break;
                                }
                                if (rsp >= sdes_end) {
                                        debug_msg
                                            ("Bogus RTCP packet: too many SDES items for packet length\n");
                                        is_okay = FALSE;
                                        break;
                                }
                                for (; rsp->type; rsp = rspn) {
                                        rspn =
                                            (rtcp_sdes_item *) ((char *)rsp +
                                                                rsp->length +
                                                                2);
                                        if (rspn == sdes_end) {
                                                break;
                                        }
                                }
                                sd = (struct rtcp_sdes_t *)((uint32_t *) sd +
                                                            (((char *)rsp -
                                                              (char *)sd) >> 2)
                                                            + 1);
                        }
                        if (count >= 0) {
                                debug_msg
                                    ("Bogus RTCP packet: malformed SDES packet\n");
                        }
                } else if (pkt->common.pt == RTCP_APP) {
                        /* No way to validate these? */
                } else if (pkt->common.pt == RTCP_BYE) {
                        /* No way to validate these? */
                }
                l += (ntohs(r->common.length) + 1) * 4;
                r = (rtcp_t *) (((uint32_t *) r) + ntohs(r->common.length) + 1);
                pc++;           /* count of sub-packets, for debugging... */
        } while (r < end);

        /* Check that the length of the packets matches the length of the UDP */
        /* packet in which they were received...                              */
        if (l != len) {
                debug_msg
                    ("Bogus RTCP packet: RTCP packet length does not match UDP packet length (%d != %d)\n",
                     l, len);
                is_okay = FALSE;
        }
        if (r != end) {
                debug_msg
                    ("Bogus RTCP packet: RTCP packet length does not match UDP packet length (%p != %p)\n",
                     r, end);
                is_okay = FALSE;
        }

        if (!is_okay) {
                debug_dump(packet, len);
        }

        return is_okay;
}

static void process_report_blocks(struct rtp *session, rtcp_t * packet,
                                  uint32_t ssrc, rtcp_rr * rrp, rtcp_rx * rrx)
{
        int i;
        rtp_event event;
        rtcp_rr *rr;
        rtcp_rx *rx = NULL;

        /* ...process RRs... */
        if (packet->common.count == 0) {
                if (!filter_event(session, ssrc)) {
                        event.ssrc = ssrc;
                        event.type = RX_RR_EMPTY;
                        event.data = NULL;
                        session->callback(session, &event);
                }
        } else {
                for (i = 0; i < packet->common.count; i++, rrp++) {
                        rr = (rtcp_rr *) malloc(sizeof(rtcp_rr));
                        rr->ssrc = ntohl(rrp->ssrc);
                        rr->fract_lost = rrp->fract_lost;       /* Endian conversion handled in the */
                        rr->total_lost = rrp->total_lost;       /* definition of the rtcp_rr type.  */
                        rr->last_seq = ntohl(rrp->last_seq);
                        rr->jitter = ntohl(rrp->jitter);
                        rr->lsr = ntohl(rrp->lsr);
                        rr->dlsr = ntohl(rrp->dlsr);

                        if (rrx) {
                                rx = (rtcp_rx *) malloc(sizeof(rtcp_rx));
                                rx->ts = ntohl(rrx->ts);
                                rx->tdelay = ntohl(rrx->tdelay);
                                rx->x_recv = ntohl(rrx->x_recv);
                                rx->p = ntohl(rrx->p);
                        }

                        /* Create a database entry for this SSRC, if one doesn't already exist... */
                        create_source(session, rr->ssrc, FALSE);

                        /* Store the RR for later use... */
                        insert_rr(session, ssrc, rr, rx);

                        /* Call the event handler... */
                        if (!filter_event(session, ssrc)) {
                                event.ssrc = ssrc;
                                event.type = RX_RR;
                                event.data = (void *)rr;
                                session->callback(session, &event);
                        }
                }
        }
}

static void process_rtcp_sr(struct rtp *session, rtcp_t * packet)
{
        uint32_t ssrc;
        rtp_event event;
        rtcp_sr *sr;
        source *s;

        ssrc = ntohl(packet->r.sr.sr.ssrc);
        s = create_source(session, ssrc, FALSE);
        if (s == NULL) {
                debug_msg("Source 0x%08x invalid, skipping...\n", ssrc);
                return;
        }

        /* Mark as an active sender, if we get a sender report... */
        if (s->sender == FALSE) {
                s->sender = TRUE;
                session->sender_count++;
        }

        /* Process the SR... */
        sr = (rtcp_sr *) malloc(sizeof(rtcp_sr));
        sr->ssrc = ssrc;
        sr->ntp_sec = ntohl(packet->r.sr.sr.ntp_sec);
        sr->ntp_frac = ntohl(packet->r.sr.sr.ntp_frac);
        sr->rtp_ts = ntohl(packet->r.sr.sr.rtp_ts);
        sr->sender_pcount = ntohl(packet->r.sr.sr.sender_pcount);
        sr->sender_bcount = ntohl(packet->r.sr.sr.sender_bcount);

        /* Store the SR for later retrieval... */
        if (s->sr != NULL) {
                free(s->sr);
        }
        s->sr = sr;
        ntp64_time(&s->last_sr_sec, &s->last_sr_frac);
        s->last_sr_sec = tmp_sec;
        s->last_sr_frac = tmp_frac;

        /* Call the event handler... */
        if (!filter_event(session, ssrc)) {
                event.ssrc = ssrc;
                event.type = RX_SR;
                event.data = (void *)sr;
                session->callback(session, &event);
        }

        process_report_blocks(session, packet, ssrc, packet->r.sr.rr, NULL);

        if (((packet->common.count * 6) + 1) <
            (ntohs(packet->common.length) - 5)) {
                debug_msg("Profile specific SR extension ignored\n");
        }
}

static void process_rtcp_rr(struct rtp *session, rtcp_t * packet)
{
        uint32_t ssrc;
        source *s;

        ssrc = ntohl(packet->r.rr.ssrc);
        s = create_source(session, ssrc, FALSE);
        if (s == NULL) {
                debug_msg("Source 0x%08x invalid, skipping...\n", ssrc);
                return;
        }

        process_report_blocks(session, packet, ssrc, packet->r.rr.rr, NULL);

        if (((packet->common.count * 6) + 1) < ntohs(packet->common.length)) {
                debug_msg("Profile specific RR extension ignored\n");
        }
}

static void process_rtcp_rx(struct rtp *session, rtcp_t * packet)
{
        uint32_t ssrc;
        source *s;

        ssrc = ntohl(packet->r.rx.ssrc);
        s = create_source(session, ssrc, FALSE);
        if (s == NULL) {
                debug_msg("Source 0x%08x invalid, skipping...\n", ssrc);
                return;
        }

        process_report_blocks(session, packet, ssrc, packet->r.rx.rr,
                              packet->r.rx.rx);

        if (((packet->common.count * 6) + 1) < ntohs(packet->common.length)) {
                debug_msg("Profile specific RR/RX extension ignored\n");
        }
}

static void process_rtcp_sdes(struct rtp *session, rtcp_t * packet)
{
        int count = packet->common.count;
        struct rtcp_sdes_t *sd = &packet->r.sdes;
        rtcp_sdes_item *rsp;
        rtcp_sdes_item *rspn;
        rtcp_sdes_item *end =
            (rtcp_sdes_item *) ((uint32_t *) packet + packet->common.length +
                                1);
        source *s;
        rtp_event event;

        while (--count >= 0) {
                rsp = &sd->item[0];
                if (rsp >= end) {
                        break;
                }
                sd->ssrc = ntohl(sd->ssrc);
                s = create_source(session, sd->ssrc, FALSE);
                if (s == NULL) {
                        debug_msg
                            ("Can't get valid source entry for 0x%08x, skipping...\n",
                             sd->ssrc);
                } else {
                        for (; rsp->type; rsp = rspn) {
                                rspn =
                                    (rtcp_sdes_item *) ((char *)rsp +
                                                        rsp->length + 2);
                                if (rspn >= end) {
                                        rsp = rspn;
                                        break;
                                }
                                if (rtp_set_sdes
                                    (session, sd->ssrc, rsp->type, rsp->data,
                                     rsp->length)) {
                                        if (!filter_event(session, sd->ssrc)) {
                                                event.ssrc = sd->ssrc;
                                                event.type = RX_SDES;
                                                event.data = (void *)rsp;
                                                session->callback(session,
                                                                  &event);
                                        }
                                } else {
                                        debug_msg
                                            ("Invalid sdes item for source 0x%08x, skipping...\n",
                                             sd->ssrc);
                                }
                        }
                }
                sd = (struct rtcp_sdes_t *)((uint32_t *) sd +
                                            (((char *)rsp - (char *)sd) >> 2) +
                                            1);
        }
        if (count >= 0) {
                debug_msg("Invalid RTCP SDES packet, some items ignored.\n");
        }
}

static void process_rtcp_bye(struct rtp *session, rtcp_t * packet)
{
        int i;
        uint32_t ssrc;
        rtp_event event;
        source *s;

        for (i = 0; i < packet->common.count; i++) {
                ssrc = ntohl(packet->r.bye.ssrc[i]);
                /* This is kind-of strange, since we create a source we are about to delete. */
                /* This is done to ensure that the source mentioned in the event which is    */
                /* passed to the user of the RTP library is valid, and simplify client code. */
                create_source(session, ssrc, FALSE);
                /* Call the event handler... */
                if (!filter_event(session, ssrc)) {
                        event.ssrc = ssrc;
                        event.type = RX_BYE;
                        event.data = NULL;
                        session->callback(session, &event);
                }
                /* Mark the source as ready for deletion. Sources are not deleted immediately */
                /* since some packets may be delayed and arrive after the BYE...              */
                s = get_source(session, ssrc);
                s->got_bye = TRUE;
                check_source(s);
                session->bye_count++;
        }
}

static void process_rtcp_app(struct rtp *session, rtcp_t * packet)
{
        uint32_t ssrc;
        rtp_event event;
        rtcp_app *app;
        source *s;
        int data_len;

        /* Update the database for this source. */
        ssrc = ntohl(packet->r.app.ssrc);
        create_source(session, ssrc, FALSE);
        s = get_source(session, ssrc);
        if (s == NULL) {
                /* This should only occur in the event of database malfunction. */
                debug_msg("Source 0x%08x invalid, skipping...\n", ssrc);
                return;
        }
        check_source(s);

        /* Copy the entire packet, converting the header (only) into host byte order. */
        app = (rtcp_app *) malloc(RTP_MAX_PACKET_LEN);
        app->version = packet->common.version;
        app->p = packet->common.p;
        app->subtype = packet->common.count;
        app->pt = packet->common.pt;
        app->length = ntohs(packet->common.length);
        app->ssrc = ssrc;
        app->name[0] = packet->r.app.name[0];
        app->name[1] = packet->r.app.name[1];
        app->name[2] = packet->r.app.name[2];
        app->name[3] = packet->r.app.name[3];
        data_len = (app->length - 2) * 4;
        memcpy(app->data, packet->r.app.data, data_len);

        /* Callback to the application to process the app packet... */
        if (!filter_event(session, ssrc)) {
                event.ssrc = ssrc;
                event.type = RX_APP;
                event.data = (void *)app;       /* The callback function MUST free this! */
                session->callback(session, &event);
        }
}

static
uint32_t compute_rtt(struct rtp *session, rtcp_rx * rrx)
{
        UNUSED(rrx);
        UNUSED(session);
        return 0;
}

static void rtp_process_ctrl(struct rtp *session, uint8_t * buffer, int buflen)
{
        /* This routine processes incoming RTCP packets */
        rtp_event event;
        rtcp_t *packet;
        uint8_t initVec[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int first;
        uint32_t packet_ssrc = rtp_my_ssrc(session);

        if (buflen > 0) {
                if (session->encryption_enabled) {
                        /* Decrypt the packet... */
                        (session->decrypt_func) (session, buffer, buflen,
                                                 initVec);
                        buffer += 4;    /* Skip the random prefix... */
                        buflen -= 4;
                }
                if (validate_rtcp(buffer, buflen)) {
                        first = TRUE;
                        packet = (rtcp_t *) buffer;
                        while (packet < (rtcp_t *) (buffer + buflen)) {
                                switch (packet->common.pt) {
                                case RTCP_SR:
                                        if (first
                                            && !filter_event(session,
                                                             ntohl(packet->r.sr.
                                                                   sr.ssrc))) {
                                                event.ssrc =
                                                    ntohl(packet->r.sr.sr.ssrc);
                                                event.type = RX_RTCP_START;
                                                event.data = &buflen;
                                                packet_ssrc = event.ssrc;
                                                session->callback(session,
                                                                  &event);
                                        }
                                        process_rtcp_sr(session, packet);
                                        break;
                                case RTCP_RR:
                                        if (first
                                            && !filter_event(session,
                                                             ntohl(packet->r.rr.
                                                                   ssrc))) {
                                                event.ssrc =
                                                    ntohl(packet->r.rr.ssrc);
                                                event.type = RX_RTCP_START;
                                                event.data = &buflen;
                                                packet_ssrc = event.ssrc;
                                                session->callback(session,
                                                                  &event);
                                        }
                                        process_rtcp_rr(session, packet);
                                        break;
                                case RTCP_RX:
                                        /* am not sending up a RX_RTCP_START... */
                                        process_rtcp_rx(session, packet);
                                        if (session->tfrc_on) {
                                                /* send up TFRC stuff ... */
                                                event.ssrc =
                                                    ntohl(packet->r.rx.ssrc);
                                                event.type = RX_TFRC_RX;
                                                event.data = &buflen;
                                                packet_ssrc = event.ssrc;
                                                session->callback(session,
                                                                  &event);
                                                /* compute the rtt time */
                                                compute_rtt(session,
                                                            packet->r.rx.rx);
                                        } else
                                                debug_msg
                                                    ("RTCP_RX received without congestion control?");
                                        break;
                                case RTCP_SDES:
                                        if (first
                                            && !filter_event(session,
                                                             ntohl(packet->r.
                                                                   sdes.
                                                                   ssrc))) {
                                                event.ssrc =
                                                    ntohl(packet->r.sdes.ssrc);
                                                event.type = RX_RTCP_START;
                                                event.data = &buflen;
                                                packet_ssrc = event.ssrc;
                                                session->callback(session,
                                                                  &event);
                                        }
                                        process_rtcp_sdes(session, packet);
                                        break;
                                case RTCP_BYE:
                                        if (first
                                            && !filter_event(session,
                                                             ntohl(packet->r.
                                                                   bye.
                                                                   ssrc[0]))) {
                                                event.ssrc =
                                                    ntohl(packet->r.bye.
                                                          ssrc[0]);
                                                event.type = RX_RTCP_START;
                                                event.data = &buflen;
                                                packet_ssrc = event.ssrc;
                                                session->callback(session,
                                                                  &event);
                                        }
                                        process_rtcp_bye(session, packet);
                                        break;
                                case RTCP_APP:
                                        if (first
                                            && !filter_event(session,
                                                             ntohl(packet->r.
                                                                   app.ssrc))) {
                                                event.ssrc =
                                                    ntohl(packet->r.app.ssrc);
                                                event.type = RX_RTCP_START;
                                                event.data = &buflen;
                                                session->callback(session,
                                                                  &event);
                                        }
                                        process_rtcp_app(session, packet);
                                        break;
                                default:
                                        debug_msg
                                            ("RTCP packet with unknown type (%d) ignored.\n",
                                             packet->common.pt);
                                        break;
                                }
                                packet =
                                    (rtcp_t *) ((char *)packet +
                                                (4 *
                                                 (ntohs(packet->common.length) +
                                                  1)));
                                first = FALSE;
                        }
                        if (session->avg_rtcp_size < 0) {
                                /* This is the first RTCP packet we've received, set our initial estimate */
                                /* of the average  packet size to be the size of this packet.             */
                                session->avg_rtcp_size =
                                    buflen + RTP_LOWER_LAYER_OVERHEAD;
                        } else {
                                /* Update our estimate of the average RTCP packet size. The constants are */
                                /* 1/16 and 15/16 (section 6.3.3 of draft-ietf-avt-rtp-new-02.txt).       */
                                session->avg_rtcp_size =
                                    (0.0625 *
                                     (buflen + RTP_LOWER_LAYER_OVERHEAD)) +
                                    (0.9375 * session->avg_rtcp_size);
                        }
                        /* Signal that we've finished processing this packet */
                        if (!filter_event(session, packet_ssrc)) {
                                event.ssrc = packet_ssrc;
                                event.type = RX_RTCP_FINISH;
                                event.data = NULL;
                                session->callback(session, &event);
                        }
                } else {
                        debug_msg("Invalid RTCP packet discarded\n");
                        session->invalid_rtcp_count++;
                }
        }
}

/**
 * rtp_recv:
 * @session: the session pointer (returned by rtp_init())
 * @timeout: the amount of time that rtcp_recv() is allowed to block
 * @curr_rtp_ts: the current time expressed in units of the media
 * timestamp.
 *
 * Receive RTP packets and dispatch them.
 *
 * Returns: TRUE if data received, FALSE if the timeout occurred.
 */
int rtp_recv(struct rtp *session, struct timeval *timeout, uint32_t curr_rtp_ts)
{
        check_database(session);
        udp_fd_zero();
        udp_fd_set(session->rtp_socket);
        udp_fd_set(session->rtcp_socket);
        if (udp_select(timeout) > 0) {
                if (udp_fd_isset(session->rtp_socket)) {
                        rtp_recv_data(session, curr_rtp_ts);
                }
                if (udp_fd_isset(session->rtcp_socket)) {
                        uint8_t buffer[RTP_MAX_PACKET_LEN];
                        int buflen;
                        buflen =
                            udp_recv(session->rtcp_socket, (char *)buffer,
                                     RTP_MAX_PACKET_LEN);
                        ntp64_time(&tmp_sec, &tmp_frac);
                        rtp_process_ctrl(session, buffer, buflen);
                }
                check_database(session);
                return TRUE;
        }
        check_database(session);
        return FALSE;
}

/**
 * reentrant variant of the above
 */
int rtp_recv_r(struct rtp *session, struct timeval *timeout, uint32_t curr_rtp_ts)
{
        struct udp_fd_r fd;
        
        check_database(session);
        udp_fd_zero_r(&fd);
        udp_fd_set_r(session->rtp_socket, &fd);
        udp_fd_set_r(session->rtcp_socket, &fd);
        if (udp_select_r(timeout, &fd) > 0) {
                if (udp_fd_isset_r(session->rtp_socket, &fd)) {
                        rtp_recv_data(session, curr_rtp_ts);
                }
                if (udp_fd_isset_r(session->rtcp_socket, &fd)) {
                        uint8_t buffer[RTP_MAX_PACKET_LEN];
                        int buflen;
                        buflen =
                            udp_recv(session->rtcp_socket, (char *)buffer,
                                     RTP_MAX_PACKET_LEN);
                        ntp64_time(&tmp_sec, &tmp_frac);
                        rtp_process_ctrl(session, buffer, buflen);
                }
                check_database(session);
                return TRUE;
        }
        check_database(session);
        return FALSE;
}


/**
 * rtp_recv_poll_r:
 * The meaning is as above with except that this function polls for first
 * nonempty stream and returns data.
 *
 * @param sessions null-terminated list of rtp sessions.
 * @param timeout timeout
 * @param cur_rtp_ts list null-terminated of timestamps for each session
 */
int rtp_recv_poll_r(struct rtp **sessions, struct timeval *timeout, uint32_t curr_rtp_ts)
{
        struct rtp **current;
        struct udp_fd_r fd;

        udp_fd_zero_r(&fd);

        for(current = sessions; *current != NULL; ++current) {
                check_database(*current);
                udp_fd_set_r((*current)->rtp_socket, &fd);
                udp_fd_set_r((*current)->rtcp_socket, &fd);
        }
        if (udp_select_r(timeout, &fd) > 0) {
                for(current = sessions; *current != NULL; ++current) {
                        if (udp_fd_isset_r((*current)->rtp_socket, &fd)) {
                                rtp_recv_data(*current, curr_rtp_ts);
                        }
                        if (udp_fd_isset_r((*current)->rtcp_socket, &fd)) {
                                uint8_t buffer[RTP_MAX_PACKET_LEN];
                                int buflen;
                                buflen =
                                    udp_recv((*current)->rtcp_socket, (char *)buffer,
                                             RTP_MAX_PACKET_LEN);
                                ntp64_time(&tmp_sec, &tmp_frac);
                                rtp_process_ctrl(*current, buffer, buflen);
                        }
                        check_database(*current);
                }
                return TRUE;
        }
        //check_database(session);
        return FALSE;
}

/**
 * rtp_add_csrc:
 * @session: the session pointer (returned by rtp_init()) 
 * @csrc: Constributing SSRC identifier
 * 
 * Adds @csrc to list of contributing sources used in SDES items.
 * Used by mixers and transcoders.
 * 
 * Return value: TRUE.
 **/
int rtp_add_csrc(struct rtp *session, uint32_t csrc)
{
        /* Mark csrc as something for which we should advertise RTCP SDES items, */
        /* in addition to our own SDES.                                          */
        source *s;

        check_database(session);
        s = get_source(session, csrc);
        if (s == NULL) {
                s = create_source(session, csrc, FALSE);
                debug_msg("Created source 0x%08x as CSRC\n", csrc);
        }
        check_source(s);
        if (!s->should_advertise_sdes) {
                s->should_advertise_sdes = TRUE;
                session->csrc_count++;
                debug_msg("Added CSRC 0x%08lx as CSRC %d\n", csrc,
                          session->csrc_count);
        }
        return TRUE;
}

/**
 * rtp_del_csrc:
 * @session: the session pointer (returned by rtp_init()) 
 * @csrc: Constributing SSRC identifier
 * 
 * Removes @csrc from list of contributing sources used in SDES items.
 * Used by mixers and transcoders.
 * 
 * Return value: TRUE on success, FALSE if @csrc is not a valid source.
 **/
int rtp_del_csrc(struct rtp *session, uint32_t csrc)
{
        source *s;

        check_database(session);
        s = get_source(session, csrc);
        if (s == NULL) {
                debug_msg("Invalid source 0x%08x\n", csrc);
                return FALSE;
        }
        check_source(s);
        s->should_advertise_sdes = FALSE;
        session->csrc_count--;
        if (session->last_advertised_csrc >= session->csrc_count) {
                session->last_advertised_csrc = 0;
        }
        return TRUE;
}

/**
 * rtp_set_sdes:
 * @session: the session pointer (returned by rtp_init()) 
 * @ssrc: the SSRC identifier of a participant
 * @type: the SDES type represented by @value
 * @value: the SDES description
 * @length: the length of the description
 * 
 * Sets session description information associated with participant
 * @ssrc.  Under normal circumstances applications always use the
 * @ssrc of the local participant, this SDES information is
 * transmitted in receiver reports.  Setting SDES information for
 * other participants affects the local SDES entries, but are not
 * transmitted onto the network.
 * 
 * Return value: Returns TRUE if participant exists, FALSE otherwise.
 **/
int rtp_set_sdes(struct rtp *session, uint32_t ssrc, rtcp_sdes_type type,
                 const char *value, int length)
{
        source *s;
        char *v;

        check_database(session);

        s = get_source(session, ssrc);
        if (s == NULL) {
                debug_msg("Invalid source 0x%08x\n", ssrc);
                return FALSE;
        }
        check_source(s);

        v = (char *)malloc(length + 1);
        memset(v, '\0', length + 1);
        memcpy(v, value, length);

        switch (type) {
        case RTCP_SDES_CNAME:
                if (s->sdes_cname)
                        free(s->sdes_cname);
                s->sdes_cname = v;
                break;
        case RTCP_SDES_NAME:
                if (s->sdes_name)
                        free(s->sdes_name);
                s->sdes_name = v;
                break;
        case RTCP_SDES_EMAIL:
                if (s->sdes_email)
                        free(s->sdes_email);
                s->sdes_email = v;
                break;
        case RTCP_SDES_PHONE:
                if (s->sdes_phone)
                        free(s->sdes_phone);
                s->sdes_phone = v;
                break;
        case RTCP_SDES_LOC:
                if (s->sdes_loc)
                        free(s->sdes_loc);
                s->sdes_loc = v;
                break;
        case RTCP_SDES_TOOL:
                if (s->sdes_tool)
                        free(s->sdes_tool);
                s->sdes_tool = v;
                break;
        case RTCP_SDES_NOTE:
                if (s->sdes_note)
                        free(s->sdes_note);
                s->sdes_note = v;
                break;
        case RTCP_SDES_PRIV:
                if (s->sdes_priv)
                        free(s->sdes_priv);
                s->sdes_priv = v;
                break;
        default:
                debug_msg("Unknown SDES item (type=%d, value=%s)\n", type, v);
                free(v);
                check_database(session);
                return FALSE;
        }
        check_database(session);
        return TRUE;
}

/**
 * rtp_get_sdes:
 * @session: the session pointer (returned by rtp_init()) 
 * @ssrc: the SSRC identifier of a participant
 * @type: the SDES information to retrieve
 * 
 * Recovers session description (SDES) information on participant
 * identified with @ssrc.  The SDES information associated with a
 * source is updated when receiver reports are received.  There are
 * several different types of SDES information, e.g. username,
 * location, phone, email.  These are enumerated by #rtcp_sdes_type.
 * 
 * Return value: pointer to string containing SDES description if
 * received, NULL otherwise.  
 */
const char *rtp_get_sdes(struct rtp *session, uint32_t ssrc,
                         rtcp_sdes_type type)
{
        source *s;

        check_database(session);

        s = get_source(session, ssrc);
        if (s == NULL) {
                debug_msg("Invalid source 0x%08x\n", ssrc);
                return NULL;
        }
        check_source(s);

        switch (type) {
        case RTCP_SDES_CNAME:
                return s->sdes_cname;
        case RTCP_SDES_NAME:
                return s->sdes_name;
        case RTCP_SDES_EMAIL:
                return s->sdes_email;
        case RTCP_SDES_PHONE:
                return s->sdes_phone;
        case RTCP_SDES_LOC:
                return s->sdes_loc;
        case RTCP_SDES_TOOL:
                return s->sdes_tool;
        case RTCP_SDES_NOTE:
                return s->sdes_note;
        case RTCP_SDES_PRIV:
                return s->sdes_priv;
        default:
                /* This includes RTCP_SDES_PRIV and RTCP_SDES_END */
                debug_msg("Unknown SDES item (type=%d)\n", type);
        }
        return NULL;
}

/**
 * rtp_get_sr:
 * @session: the session pointer (returned by rtp_init()) 
 * @ssrc: identifier of source
 * 
 * Retrieve the latest sender report made by sender with @ssrc identifier.
 * 
 * Return value: A pointer to an rtcp_sr structure on success, NULL
 * otherwise.  The pointer must not be freed.
 **/
const rtcp_sr *rtp_get_sr(struct rtp *session, uint32_t ssrc)
{
        /* Return the last SR received from this ssrc. The */
        /* caller MUST NOT free the memory returned to it. */
        source *s;

        check_database(session);

        s = get_source(session, ssrc);
        if (s == NULL) {
                return NULL;
        }
        check_source(s);
        return s->sr;
}

/**
 * rtp_get_rr:
 * @session: the session pointer (returned by rtp_init())
 * @reporter: participant originating receiver report
 * @reportee: participant included in receiver report
 * 
 * Retrieve the latest receiver report on @reportee made by @reporter.
 * Provides an indication of other receivers reception service.
 * 
 * Return value: A pointer to a rtcp_rr structure on success, NULL
 * otherwise.  The pointer must not be freed.
 **/
const rtcp_rr *rtp_get_rr(struct rtp *session, uint32_t reporter,
                          uint32_t reportee)
{
        check_database(session);
        return get_rr(session, reporter, reportee);
}

/**
 * rtp_send_data:
 * @session: the session pointer (returned by rtp_init())
 * @rtp_ts: The timestamp reflects the sampling instant of the first octet of the RTP data to be sent.  The timestamp is expressed in media units.
 * @pt: The payload type identifying the format of the data.
 * @m: Marker bit, interpretation defined by media profile of payload.
 * @cc: Number of contributing sources (excluding local participant)
 * @csrc: Array of SSRC identifiers for contributing sources.
 * @data: The RTP data to be sent.
 * @data_len: The size @data in bytes.
 * @extn: Extension data (if present).
 * @extn_len: size of @extn in bytes.
 * @extn_type: extension type indicator.
 * 
 * Send an RTP packet.  Most media applications will only set the
 * @session, @rtp_ts, @pt, @m, @data, @data_len arguments.
 *
 * Mixers and translators typically set additional contributing sources 
 * arguments (@cc, @csrc).
 *
 * Extensions fields (@extn, @extn_len, @extn_type) are for including
 * application specific information.  When the widest amount of
 * inter-operability is required these fields should be avoided as
 * some applications discard packets with extensions they do not
 * recognize.
 * 
 * Return value: Number of bytes transmitted.
 **/
int rtp_send_data(struct rtp *session, uint32_t rtp_ts, char pt, int m,
                  int cc, uint32_t * csrc,
                  char *data, int data_len,
                  char *extn, uint16_t extn_len, uint16_t extn_type)
{
        return rtp_send_data_hdr(session, rtp_ts, pt, m, cc, csrc, NULL, 0,
                                 data, data_len, extn, extn_len, extn_type);
}

int
rtp_send_data_hdr(struct rtp *session,
                  uint32_t rtp_ts, char pt, int m,
                  int cc, uint32_t csrc[],
                  char *phdr, int phdr_len,
                  char *data, int data_len,
                  char *extn, uint16_t extn_len, uint16_t extn_type)
{
        int vlen, buffer_len, i, rc, pad, pad_len;
        uint8_t *buffer = NULL;
        rtp_packet *packet = NULL;
        uint8_t initVec[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
#ifdef WIN32
        WSABUF send_vector[3];
#else
        struct iovec send_vector[3];
#endif
        int send_vector_len;

        check_database(session);

        assert((data == NULL && data_len == 0)
               || (data != NULL && data_len > 0));

        vlen = 12;
        if (session->tfrc_on) {
                vlen += 4;
                if (session->new_rtt)
                        vlen += 4;
        }
        buffer_len = vlen + (4 * cc);

        if (extn != NULL) {
                buffer_len += (extn_len + 1) * 4;
        }

        /* Do we need to pad this packet to a multiple of 64 bits? */
        /* This is only needed if encryption is enabled, since DES */
        /* only works on multiples of 64 bits. We just calculate   */
        /* the amount of padding to add here, so we can reserve    */
        /* space - the actual padding is added later.              */
#ifdef NDEF
        /* FIXME: This is broken, due to scatter send [csp]        *//* FIXME */
        if ((session->encryption_enabled) &&
            ((buffer_len % session->encryption_pad_length) != 0)) {
                pad = TRUE;
                pad_len =
                    session->encryption_pad_length -
                    (buffer_len % session->encryption_pad_length);
                buffer_len += pad_len;
                assert((buffer_len % session->encryption_pad_length) == 0);
        } else {
                pad = FALSE;
                pad_len = 0;
        }
#endif
        pad = FALSE;            /* FIXME */
        pad_len = 0;

        /* Allocate memory for the packet... */
        if (buffer == NULL) {
                assert(buffer_len < RTP_MAX_PACKET_LEN);
                /* we dont always need 20 (12|16) but this seems to work. LG */
                buffer = (uint8_t *) malloc(20 + RTP_PACKET_HEADER_SIZE);
                packet = (rtp_packet *) buffer;
        }
#ifdef WIN32
        send_vector[0].buf = buffer + RTP_PACKET_HEADER_SIZE;
        send_vector[0].len = buffer_len;
#else
        send_vector[0].iov_base = buffer + RTP_PACKET_HEADER_SIZE;
        send_vector[0].iov_len = buffer_len;
#endif
        send_vector_len = 1;

        /* These are internal pointers into the buffer... */
#ifdef NDEF
        packet->csrc = (uint32_t *) (buffer + RTP_PACKET_HEADER_SIZE + vlen);
        packet->extn =
            (uint8_t *) (buffer + RTP_PACKET_HEADER_SIZE + vlen + (4 * cc));
        packet->data =
            (uint8_t *) (buffer + RTP_PACKET_HEADER_SIZE + vlen + (4 * cc));
        if (extn != NULL) {
                packet->data += (extn_len + 1) * 4;
        }
#endif
        /* ...and the actual packet header... */
        packet->v = 2;
        packet->p = pad;
        packet->x = (extn != NULL);
        packet->cc = cc;
        packet->m = m;
        packet->pt = pt;
        packet->seq = htons(session->rtp_seq++);
        packet->ts = htonl(rtp_ts);
        packet->ssrc = htonl(session->my_ssrc);

        /* ... do tfrc stuff... */
        if (session->tfrc_on) {
                packet->send_ts = htonl(get_local_mediatime());
                if (session->new_rtt) {
                        packet->rtt = htonl(session->cmp_rtt);
                        /* hopefully this will set the 7th bit */
                        packet->pt = packet->pt | 64;
                } else
                        packet->pt = packet->pt & 63;   /* this should clear the 7th bit */
        }

        /* ...now the CSRC list... */
        for (i = 0; i < cc; i++) {
                packet->csrc[i] = htonl(csrc[i]);
        }
        /* ...a header extension? */
        if (extn != NULL) {
                /* We don't use the packet->extn_type field here, that's for receive only... */
                uint16_t *base = (uint16_t *) packet->extn;
                base[0] = htons(extn_type);
                base[1] = htons(extn_len);
                memcpy(packet->extn + 4, extn, extn_len * 4);
        }
        /* ...the payload header... */
        if (phdr != NULL) {
#ifdef WIN32
                send_vector[send_vector_len].buf = phdr;
                send_vector[send_vector_len].len = phdr_len;
#else
                send_vector[send_vector_len].iov_base = phdr;
                send_vector[send_vector_len].iov_len = phdr_len;
#endif
                send_vector_len++;
        }
        /* ...and the media data... */
        if (data_len > 0) {
#ifdef WIN32
                send_vector[send_vector_len].buf = data;
                send_vector[send_vector_len].len = data_len;
#else
                send_vector[send_vector_len].iov_base = data;
                send_vector[send_vector_len].iov_len = data_len;
#endif
                send_vector_len++;
        }
#ifdef NDEF                     /* FIXME */
        /* ...and any padding... */
        if (pad) {
                for (i = 0; i < pad_len; i++) {
                        buffer[buffer_len + RTP_PACKET_HEADER_SIZE - pad_len +
                               i] = 0;
                }
                buffer[buffer_len + RTP_PACKET_HEADER_SIZE - 1] = (char)pad_len;
        }
#endif

        /* Finally, encrypt if desired... */
        if (session->encryption_enabled) {
                assert((buffer_len % session->encryption_pad_length) == 0);
                (session->encrypt_func) (session,
                                         buffer + RTP_PACKET_HEADER_SIZE,
                                         buffer_len, initVec);
        }

        rc = udp_sendv(session->rtp_socket, send_vector, send_vector_len);
        if (rc == -1) {
                perror("sending RTP packet");
        }

        free(buffer);

        /* Update the RTCP statistics... */
        session->we_sent = TRUE;
        session->rtp_pcount += 1;
        session->rtp_bcount += buffer_len;
        gettimeofday(&session->last_rtp_send_time, NULL);

        check_database(session);
        return rc;
}

static int format_report_blocks(rtcp_rr * rrp, int remaining_length,
                                struct rtp *session)
{
        int nblocks = 0;
        int h;
        source *s;
        uint32_t now_sec;
        uint32_t now_frac;

        for (h = 0; h < RTP_DB_SIZE; h++) {
                for (s = session->db[h]; s != NULL; s = s->next) {
                        check_source(s);
                        if ((nblocks == 31) || (remaining_length < 24)) {
                                break;  /* Insufficient space for more report blocks... */
                        }
                        if (s->sender) {
                                /* Much of this is taken from A.3 of draft-ietf-avt-rtp-new-01.txt */
                                int extended_max = s->cycles + s->max_seq;
                                int expected = extended_max - s->base_seq + 1;
                                int lost = expected - s->received;
                                int expected_interval =
                                    expected - s->expected_prior;
                                int received_interval =
                                    s->received - s->received_prior;
                                int lost_interval =
                                    expected_interval - received_interval;
                                int fraction;
                                uint32_t lsr;
                                uint32_t dlsr;

                                //printf("lost_interval %d\n", lost_interval);
                                s->expected_prior = expected;
                                s->received_prior = s->received;
                                if (expected_interval == 0
                                    || lost_interval <= 0) {
                                        fraction = 0;
                                } else {
                                        fraction =
                                            (lost_interval << 8) /
                                            expected_interval;
                                }

                                if (s->sr == NULL) {
                                        lsr = 0;
                                        dlsr = 0;
                                } else {
                                        ntp64_time(&now_sec, &now_frac);
                                        lsr =
                                            ntp64_to_ntp32(s->sr->ntp_sec,
                                                           s->sr->ntp_frac);
                                        dlsr =
                                            ntp64_to_ntp32(now_sec,
                                                           now_frac) -
                                            ntp64_to_ntp32(s->last_sr_sec,
                                                           s->last_sr_frac);
                                }
                                rrp->ssrc = htonl(s->ssrc);
                                rrp->fract_lost = fraction;
                                rrp->total_lost = lost & 0x00ffffff;
                                rrp->last_seq = htonl(extended_max);
                                rrp->jitter = htonl(s->jitter / 16);
                                rrp->lsr = htonl(lsr);
                                rrp->dlsr = htonl(dlsr);
                                rrp++;
                                remaining_length -= 24;
                                nblocks++;
                                s->sender = FALSE;
                                session->sender_count--;
                                if (session->sender_count == 0) {
                                        break;  /* No point continuing, since we've reported on all senders... */
                                }
                        }
                }
        }
        return nblocks;
}

static uint8_t *format_rtcp_sr(uint8_t * buffer, int buflen,
                               struct rtp *session, uint32_t rtp_ts)
{
        /* Write an RTCP SR into buffer, returning a pointer to */
        /* the next byte after the header we have just written. */
        rtcp_t *packet = (rtcp_t *) buffer;
        int remaining_length;
        uint32_t ntp_sec, ntp_frac;

        assert(buflen >= 28);   /* ...else there isn't space for the header and sender report */

        packet->common.version = 2;
        packet->common.p = 0;
        packet->common.count = 0;
        packet->common.pt = RTCP_SR;
        packet->common.length = htons(1);

        ntp64_time(&ntp_sec, &ntp_frac);

        packet->r.sr.sr.ssrc = htonl(rtp_my_ssrc(session));
        packet->r.sr.sr.ntp_sec = htonl(ntp_sec);
        packet->r.sr.sr.ntp_frac = htonl(ntp_frac);
        packet->r.sr.sr.rtp_ts = htonl(rtp_ts);
        packet->r.sr.sr.sender_pcount = htonl(session->rtp_pcount);
        packet->r.sr.sr.sender_bcount = htonl(session->rtp_bcount);

        /* Add report blocks, until we either run out of senders */
        /* to report upon or we run out of space in the buffer.  */
        remaining_length = buflen - 28;
        packet->common.count =
            format_report_blocks(packet->r.sr.rr, remaining_length, session);
        packet->common.length =
            htons((uint16_t) (6 + (packet->common.count * 6)));
        return buffer + 28 + (24 * packet->common.count);
}

static uint8_t *format_rtcp_rr(uint8_t * buffer, int buflen,
                               struct rtp *session)
{
        /* Write an RTCP RR into buffer, returning a pointer to */
        /* the next byte after the header we have just written. */
        rtcp_t *packet = (rtcp_t *) buffer;
        int remaining_length;

        assert(buflen >= 8);    /* ...else there isn't space for the header */

        packet->common.version = 2;
        packet->common.p = 0;
        packet->common.count = 0;
        packet->common.pt = RTCP_RR;
        packet->common.length = htons(1);
        packet->r.rr.ssrc = htonl(session->my_ssrc);

        /* Add report blocks, until we either run out of senders */
        /* to report upon or we run out of space in the buffer.  */
        remaining_length = buflen - 8;
        packet->common.count =
            format_report_blocks(packet->r.rr.rr, remaining_length, session);
        packet->common.length =
            htons((uint16_t) (1 + (packet->common.count * 6)));
        return buffer + 8 + (24 * packet->common.count);
}

static int add_sdes_item(uint8_t * buf, int buflen, int type, const char *val)
{
        /* Fill out an SDES item. It is assumed that the item is a NULL    */
        /* terminated string.                                              */
        rtcp_sdes_item *shdr = (rtcp_sdes_item *) buf;
        int namelen;

        if (val == NULL) {
                debug_msg("Cannot format SDES item. type=%d val=%xp\n", type,
                          val);
                return 0;
        }
        shdr->type = type;
        namelen = strlen(val);
        shdr->length = namelen;
        strncpy(shdr->data, val, buflen - 2);   /* The "-2" accounts for the other shdr fields */
        return namelen + 2;
}

static uint8_t *format_rtcp_sdes(uint8_t * buffer, int buflen, uint32_t ssrc,
                                 struct rtp *session)
{
        /* From draft-ietf-avt-profile-new-00:                             */
        /* "Applications may use any of the SDES items described in the    */
        /* RTP specification. While CNAME information is sent every        */
        /* reporting interval, other items should be sent only every third */
        /* reporting interval, with NAME sent seven out of eight times     */
        /* within that slot and the remaining SDES items cyclically taking */
        /* up the eighth slot, as defined in Section 6.2.2 of the RTP      */
        /* specification. In other words, NAME is sent in RTCP packets 1,  */
        /* 4, 7, 10, 13, 16, 19, while, say, EMAIL is used in RTCP packet  */
        /* 22".                                                            */
        uint8_t *packet = buffer;
        rtcp_common *common = (rtcp_common *) buffer;
        const char *item;
        size_t remaining_len;
        int pad;

        assert(buflen > (int)sizeof(rtcp_common));

        common->version = 2;
        common->p = 0;
        common->count = 1;
        common->pt = RTCP_SDES;
        common->length = 0;
        packet += sizeof(rtcp_common);

        *((uint32_t *) packet) = htonl(ssrc);
        packet += 4;

        remaining_len = buflen - (packet - buffer);
        item = rtp_get_sdes(session, ssrc, RTCP_SDES_CNAME);
        if ((item != NULL) && ((strlen(item) + (size_t) 2) <= remaining_len)) {
                packet +=
                    add_sdes_item(packet, remaining_len, RTCP_SDES_CNAME, item);
        }

        remaining_len = buflen - (packet - buffer);
        item = rtp_get_sdes(session, ssrc, RTCP_SDES_NOTE);
        if ((item != NULL) && ((strlen(item) + (size_t) 2) <= remaining_len)) {
                packet +=
                    add_sdes_item(packet, remaining_len, RTCP_SDES_NOTE, item);
        }

        remaining_len = buflen - (packet - buffer);
        if ((session->sdes_count_pri % 3) == 0) {
                session->sdes_count_sec++;
                if ((session->sdes_count_sec % 8) == 0) {
                        /* Note that the following is supposed to fall-through the cases */
                        /* until one is found to send... The lack of break statements in */
                        /* the switch is not a bug.                                      */
                        switch (session->sdes_count_ter % 5) {
                        case 0:
                                item =
                                    rtp_get_sdes(session, ssrc, RTCP_SDES_TOOL);
                                if ((item != NULL)
                                    && ((strlen(item) + (size_t) 2) <=
                                        remaining_len)) {
                                        packet +=
                                            add_sdes_item(packet, remaining_len,
                                                          RTCP_SDES_TOOL, item);
                                        break;
                                }
                        case 1:
                                item =
                                    rtp_get_sdes(session, ssrc,
                                                 RTCP_SDES_EMAIL);
                                if ((item != NULL)
                                    && ((strlen(item) + (size_t) 2) <=
                                        remaining_len)) {
                                        packet +=
                                            add_sdes_item(packet, remaining_len,
                                                          RTCP_SDES_EMAIL,
                                                          item);
                                        break;
                                }
                        case 2:
                                item =
                                    rtp_get_sdes(session, ssrc,
                                                 RTCP_SDES_PHONE);
                                if ((item != NULL)
                                    && ((strlen(item) + (size_t) 2) <=
                                        remaining_len)) {
                                        packet +=
                                            add_sdes_item(packet, remaining_len,
                                                          RTCP_SDES_PHONE,
                                                          item);
                                        break;
                                }
                        case 3:
                                item =
                                    rtp_get_sdes(session, ssrc, RTCP_SDES_LOC);
                                if ((item != NULL)
                                    && ((strlen(item) + (size_t) 2) <=
                                        remaining_len)) {
                                        packet +=
                                            add_sdes_item(packet, remaining_len,
                                                          RTCP_SDES_LOC, item);
                                        break;
                                }
                        case 4:
                                item =
                                    rtp_get_sdes(session, ssrc, RTCP_SDES_PRIV);
                                if ((item != NULL)
                                    && ((strlen(item) + (size_t) 2) <=
                                        remaining_len)) {
                                        packet +=
                                            add_sdes_item(packet, remaining_len,
                                                          RTCP_SDES_PRIV, item);
                                        break;
                                }
                        }
                        session->sdes_count_ter++;
                } else {
                        item = rtp_get_sdes(session, ssrc, RTCP_SDES_NAME);
                        if (item != NULL) {
                                packet +=
                                    add_sdes_item(packet, remaining_len,
                                                  RTCP_SDES_NAME, item);
                        }
                }
        }
        session->sdes_count_pri++;

        /* Pad to a multiple of 4 bytes... */
        pad = 4 - ((packet - buffer) & 0x3);
        while (pad--) {
                *packet++ = RTCP_SDES_END;
        }

        common->length = htons((uint16_t) (((int)(packet - buffer) / 4) - 1));

        return packet;
}

static uint8_t *format_rtcp_app(uint8_t * buffer, int buflen, uint32_t ssrc,
                                rtcp_app * app)
{
        /* Write an RTCP APP into the outgoing packet buffer. */
        rtcp_app *packet = (rtcp_app *) buffer;
        int pkt_octets = (app->length + 1) * 4;
        int data_octets = pkt_octets - 12;

        assert(data_octets >= 0);       /* ...else not a legal APP packet.               */
        assert(buflen >= pkt_octets);   /* ...else there isn't space for the APP packet. */

        /* Copy one APP packet from "app" to "packet". */
        packet->version = RTP_VERSION;
        packet->p = app->p;
        packet->subtype = app->subtype;
        packet->pt = RTCP_APP;
        packet->length = htons(app->length);
        packet->ssrc = htonl(ssrc);
        memcpy(packet->name, app->name, 4);
        memcpy(packet->data, app->data, data_octets);

        /* Return a pointer to the byte that immediately follows the last byte written. */
        return buffer + pkt_octets;
}

static void send_rtcp(struct rtp *session, uint32_t rtp_ts,
                      rtcp_app_callback appcallback)
{
        /* Construct and send an RTCP packet. The order in which packets are packed into a */
        /* compound packet is defined by section 6.1 of draft-ietf-avt-rtp-new-03.txt and  */
        /* we follow the recommended order.                                                */
        uint8_t buffer[RTP_MAX_PACKET_LEN + MAX_ENCRYPTION_PAD];        /* The +8 is to allow for padding when encrypting */
        uint8_t *ptr = buffer;
        uint8_t *old_ptr;
        uint8_t *lpt;           /* the last packet in the compound */
        rtcp_app *app;
        uint8_t initVec[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int rc;

        check_database(session);
        /* If encryption is enabled, add a 32 bit random prefix to the packet */
        if (session->encryption_enabled) {
                *((uint32_t *) ptr) = lbl_random();
                ptr += 4;
        }

        /* The first RTCP packet in the compound packet MUST always be a report packet...  */
        if (session->we_sent) {
                ptr =
                    format_rtcp_sr(ptr, RTP_MAX_PACKET_LEN - (ptr - buffer),
                                   session, rtp_ts);
        } else {
                ptr =
                    format_rtcp_rr(ptr, RTP_MAX_PACKET_LEN - (ptr - buffer),
                                   session);
        }
        /* Add the appropriate SDES items to the packet... This should really be after the */
        /* insertion of the additional report blocks, but if we do that there are problems */
        /* with us being unable to fit the SDES packet in when we run out of buffer space  */
        /* adding RRs. The correct fix would be to calculate the length of the SDES items  */
        /* in advance and subtract this from the buffer length but this is non-trivial and */
        /* probably not worth it.                                                          */
        lpt = ptr;
        ptr =
            format_rtcp_sdes(ptr, RTP_MAX_PACKET_LEN - (ptr - buffer),
                             rtp_my_ssrc(session), session);

        /* If we have any CSRCs, we include SDES items for each of them in turn...         */
        if (session->csrc_count > 0) {
                ptr =
                    format_rtcp_sdes(ptr, RTP_MAX_PACKET_LEN - (ptr - buffer),
                                     next_csrc(session), session);
        }

        /* Following that, additional RR packets SHOULD follow if there are more than 31   */
        /* senders, such that the reports do not fit into the initial packet. We give up   */
        /* if there is insufficient space in the buffer: this is bad, since we always drop */
        /* the reports from the same sources (those at the end of the hash table).         */
        while ((session->sender_count > 0)
               && ((RTP_MAX_PACKET_LEN - (ptr - buffer)) >= 8)) {
                lpt = ptr;
                ptr =
                    format_rtcp_rr(ptr, RTP_MAX_PACKET_LEN - (ptr - buffer),
                                   session);
        }

        /* Finish with as many APP packets as the application will provide. */
        old_ptr = ptr;
        if (appcallback) {
                while ((app =
                        (*appcallback) (session, rtp_ts,
                                        RTP_MAX_PACKET_LEN - (ptr - buffer)))) {
                        lpt = ptr;
                        ptr =
                            format_rtcp_app(ptr,
                                            RTP_MAX_PACKET_LEN - (ptr - buffer),
                                            rtp_my_ssrc(session), app);
                        assert(ptr > old_ptr);
                        old_ptr = ptr;
                        assert(RTP_MAX_PACKET_LEN - (ptr - buffer) >= 0);
                }
        }

        /* And encrypt if desired... */
        if (session->encryption_enabled) {
                if (((ptr - buffer) % session->encryption_pad_length) != 0) {
                        /* Add padding to the last packet in the compound, if necessary. */
                        /* We don't have to worry about overflowing the buffer, since we */
                        /* intentionally allocated it 8 bytes longer to allow for this.  */
                        int padlen =
                            session->encryption_pad_length -
                            ((ptr - buffer) % session->encryption_pad_length);
                        int i;

                        for (i = 0; i < padlen - 1; i++) {
                                *(ptr++) = '\0';
                        }
                        *(ptr++) = (uint8_t) padlen;
                        assert(((ptr -
                                 buffer) % session->encryption_pad_length) ==
                               0);

                        ((rtcp_t *) lpt)->common.p = TRUE;
                        ((rtcp_t *) lpt)->common.length =
                            htons((int16_t) (((ptr - lpt) / 4) - 1));
                }
                (session->encrypt_func) (session, buffer, ptr - buffer,
                                         initVec);
        }
        rc = udp_send(session->rtcp_socket, (char *)buffer, ptr - buffer);
        if (rc == -1) {
                perror("sending RTCP packet");
        }

        /* Loop the data back to ourselves so local participant can */
        /* query own stats when using unicast or multicast with no  */
        /* loopback.                                                */
        rtp_process_ctrl(session, buffer, ptr - buffer);
        check_database(session);
}

/**
 * rtp_send_ctrl:
 * @session: the session pointer (returned by rtp_init())
 * @rtp_ts: the current time expressed in units of the media timestamp.
 * @appcallback: a callback to create an APP RTCP packet, if needed.
 *
 * Checks RTCP timer and sends RTCP data when nececessary.  The
 * interval between RTCP packets is randomized over an interval that
 * depends on the session bandwidth, the number of participants, and
 * whether the local participant is a sender.  This function should be
 * called at least once per second, and can be safely called more
 * frequently.  
 */
void rtp_send_ctrl(struct rtp *session, uint32_t rtp_ts,
                   rtcp_app_callback appcallback, struct timeval curr_time)
{
        /* Send an RTCP packet, if one is due... */

        check_database(session);
        if (tv_gt(curr_time, session->next_rtcp_send_time)) {
                /* The RTCP transmission timer has expired. The following */
                /* implements draft-ietf-avt-rtp-new-02.txt section 6.3.6 */
                int h;
                source *s;
                struct timeval new_send_time;
                double new_interval;

                new_interval =
                    rtcp_interval(session) / (session->csrc_count + 1);
                new_send_time = session->last_rtcp_send_time;
                tv_add(&new_send_time, new_interval);
                if (tv_gt(curr_time, new_send_time)) {
                        send_rtcp(session, rtp_ts, appcallback);
                        session->initial_rtcp = FALSE;
                        session->last_rtcp_send_time = curr_time;
                        session->next_rtcp_send_time = curr_time;
                        tv_add(&(session->next_rtcp_send_time),
                               rtcp_interval(session) / (session->csrc_count +
                                                         1));
                        /* We're starting a new RTCP reporting interval, zero out */
                        /* the per-interval statistics.                           */
                        session->sender_count = 0;
                        for (h = 0; h < RTP_DB_SIZE; h++) {
                                for (s = session->db[h]; s != NULL; s = s->next) {
                                        check_source(s);
                                        s->sender = FALSE;
                                }
                        }
                } else {
                        session->next_rtcp_send_time = new_send_time;
                }
                session->ssrc_count_prev = session->ssrc_count;
        }
        check_database(session);
}

/**
 * rtp_update:
 * @session: the session pointer (returned by rtp_init())
 *
 * Trawls through the internal data structures and performs
 * housekeeping.  This function should be called at least once per
 * second.  It uses an internal timer to limit the number of passes
 * through the data structures to once per second, it can be safely
 * called more frequently.
 */
void rtp_update(struct rtp *session, struct timeval curr_time)
{
        /* Perform housekeeping on the source database... */
        int h;
        source *s, *n;
        double delay;

        if (tv_diff(curr_time, session->last_update) < 1.0) {
                /* We only perform housekeeping once per second... */
                return;
        }
        session->last_update = curr_time;

        /* Update we_sent (section 6.3.8 of RTP spec) */
        delay = tv_diff(curr_time, session->last_rtp_send_time);
        if (delay >= 2 * rtcp_interval(session)) {
                session->we_sent = FALSE;
        }

        check_database(session);

        for (h = 0; h < RTP_DB_SIZE; h++) {
                for (s = session->db[h]; s != NULL; s = n) {
                        check_source(s);
                        n = s->next;
                        /* Expire sources which haven't been heard from for a int time.   */
                        /* Section 6.2.1 of the RTP specification details the timers used. */

                        /* How int since we last heard from this source?  */
                        delay = tv_diff(curr_time, s->last_active);

                        /* Check if we've received a BYE packet from this source.    */
                        /* If we have, and it was received more than 2 seconds ago   */
                        /* then the source is deleted. The arbitrary 2 second delay  */
                        /* is to ensure that all delayed packets are received before */
                        /* the source is timed out.                                  */
                        if (s->got_bye && (delay > 2.0)) {
                                debug_msg
                                    ("Deleting source 0x%08lx due to reception of BYE %f seconds ago...\n",
                                     s->ssrc, delay);
                                delete_source(session, s->ssrc);
                        }

                        /* Sources are marked as inactive if they haven't been heard */
                        /* from for more than 2 intervals (RTP section 6.3.5)        */
                        if ((s->ssrc != rtp_my_ssrc(session))
                            && (delay > (session->rtcp_interval * 2))) {
                                if (s->sender) {
                                        s->sender = FALSE;
                                        session->sender_count--;
                                }
                        }

                        /* If a source hasn't been heard from for more than 5 RTCP   */
                        /* reporting intervals, we delete it from our database...    */
                        if ((s->ssrc != rtp_my_ssrc(session))
                            && (delay > (session->rtcp_interval * 5))) {
                                debug_msg
                                    ("Deleting source 0x%08lx due to timeout...\n",
                                     s->ssrc);
                                delete_source(session, s->ssrc);
                        }
                }
        }

        /* Timeout those reception reports which haven't been refreshed for a int time */
        timeout_rr(session, &curr_time);
        check_database(session);
}

static void rtp_send_bye_now(struct rtp *session)
{
        /* Send a BYE packet immediately. This is an internal function,  */
        /* hidden behind the rtp_send_bye() wrapper which implements BYE */
        /* reconsideration for the application.                          */
        uint8_t buffer[RTP_MAX_PACKET_LEN + MAX_ENCRYPTION_PAD];        /* + 8 to allow for padding when encrypting */
        uint8_t *ptr = buffer;
        rtcp_common *common;
        uint8_t initVec[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

        check_database(session);
        /* If encryption is enabled, add a 32 bit random prefix to the packet */
        if (session->encryption_enabled) {
                *((uint32_t *) ptr) = lbl_random();
                ptr += 4;
        }

        ptr = format_rtcp_rr(ptr, RTP_MAX_PACKET_LEN - (ptr - buffer), session);
        common = (rtcp_common *) ptr;

        common->version = 2;
        common->p = 0;
        common->count = 1;
        common->pt = RTCP_BYE;
        common->length = htons(1);
        ptr += sizeof(rtcp_common);

        *((uint32_t *) ptr) = htonl(session->my_ssrc);
        ptr += 4;

        if (session->encryption_enabled) {
                if (((ptr - buffer) % session->encryption_pad_length) != 0) {
                        /* Add padding to the last packet in the compound, if necessary. */
                        /* We don't have to worry about overflowing the buffer, since we */
                        /* intentionally allocated it 8 bytes longer to allow for this.  */
                        int padlen =
                            session->encryption_pad_length -
                            ((ptr - buffer) % session->encryption_pad_length);
                        int i;

                        for (i = 0; i < padlen - 1; i++) {
                                *(ptr++) = '\0';
                        }
                        *(ptr++) = (uint8_t) padlen;

                        common->p = TRUE;
                        common->length =
                            htons((int16_t)
                                  (((ptr - (uint8_t *) common) / 4) - 1));
                }
                assert(((ptr - buffer) % session->encryption_pad_length) == 0);
                (session->encrypt_func) (session, buffer, ptr - buffer,
                                         initVec);
        }
        udp_send(session->rtcp_socket, (char *)buffer, ptr - buffer);
        /* Loop the data back to ourselves so local participant can */
        /* query own stats when using unicast or multicast with no  */
        /* loopback.                                                */
        rtp_process_ctrl(session, buffer, ptr - buffer);
        check_database(session);
}

/**
 * rtp_send_bye:
 * @session: The RTP session
 *
 * Sends a BYE message on the RTP session, indicating that this
 * participant is leaving the session. The process of sending a
 * BYE may take some time, and this function will block until 
 * it is complete. During this time, RTCP events are reported 
 * to the application via the callback function (data packets 
 * are silently discarded).
 */
void rtp_send_bye(struct rtp *session)
{
        struct timeval curr_time, timeout, new_send_time;
        uint8_t buffer[RTP_MAX_PACKET_LEN];
        int buflen;
        double new_interval;

        check_database(session);

        /* "...a participant which never sent an RTP or RTCP packet MUST NOT send  */
        /* a BYE packet when they leave the group." (section 6.3.7 of RTP spec)    */
        if ((session->we_sent == FALSE) && (session->initial_rtcp == TRUE)) {
                debug_msg("Silent BYE\n");
                return;
        }

        /* If the session is small, send an immediate BYE. Otherwise, we delay and */
        /* perform BYE reconsideration as needed.                                  */
        if (session->ssrc_count < 50) {
                rtp_send_bye_now(session);
        } else {
                gettimeofday(&curr_time, NULL);
                session->sending_bye = TRUE;
                session->last_rtcp_send_time = curr_time;
                session->next_rtcp_send_time = curr_time;
                session->bye_count = 1;
                session->initial_rtcp = TRUE;
                session->we_sent = FALSE;
                session->sender_count = 0;
                session->avg_rtcp_size = 70.0 + RTP_LOWER_LAYER_OVERHEAD;       /* FIXME */
                tv_add(&session->next_rtcp_send_time,
                       rtcp_interval(session) / (session->csrc_count + 1));

                debug_msg("Preparing to send BYE...\n");
                while (1) {
                        /* Schedule us to block in udp_select() until the time we are due to send our */
                        /* BYE packet. If we receive an RTCP packet from another participant before   */
                        /* then, we are woken up to handle it...                                      */
                        timeout.tv_sec = 0;
                        timeout.tv_usec = 0;
                        tv_add(&timeout,
                               tv_diff(session->next_rtcp_send_time,
                                       curr_time));
                        udp_fd_zero();
                        udp_fd_set(session->rtcp_socket);
                        if ((udp_select(&timeout) > 0)
                            && udp_fd_isset(session->rtcp_socket)) {
                                /* We woke up because an RTCP packet was received; process it... */
                                buflen =
                                    udp_recv(session->rtcp_socket,
                                             (char *)buffer,
                                             RTP_MAX_PACKET_LEN);
                                rtp_process_ctrl(session, buffer, buflen);
                        }
                        /* Is it time to send our BYE? */
                        gettimeofday(&curr_time, NULL);
                        new_interval =
                            rtcp_interval(session) / (session->csrc_count + 1);
                        new_send_time = session->last_rtcp_send_time;
                        tv_add(&new_send_time, new_interval);
                        if (tv_gt(curr_time, new_send_time)) {
                                debug_msg("Sent BYE...\n");
                                rtp_send_bye_now(session);
                                break;
                        }
                        /* No, we reconsider... */
                        session->next_rtcp_send_time = new_send_time;
                        debug_msg("Reconsidered sending BYE... delay = %f\n",
                                  tv_diff(session->next_rtcp_send_time,
                                          curr_time));
                        /* ...and perform housekeeping in the usual manner */
                        rtp_update(session, curr_time);
                }
        }
}

/**
 * rtp_done:
 * @session: the RTP session to finish
 *
 * Free the state associated with the given RTP session. This function does 
 * not send any packets (e.g. an RTCP BYE) - an application which wishes to
 * exit in a clean manner should call rtp_send_bye() first.
 */
void rtp_done(struct rtp *session)
{
        int i;
        source *s, *n;

        check_database(session);
        /* In delete_source, check database gets called and this assumes */
        /* first added and last removed is us.                           */
        for (i = 0; i < RTP_DB_SIZE; i++) {
                s = session->db[i];
                while (s != NULL) {
                        n = s->next;
                        if (s->ssrc != session->my_ssrc) {
                                delete_source(session, session->db[i]->ssrc);
                        }
                        s = n;
                }
        }

        delete_source(session, session->my_ssrc);

        /*
         * Introduce a memory leak until we add algorithm-specific
         * cleanup functions.
         if (session->encryption_key != NULL) {
         free(session->encryption_key);
         }
         */

        udp_exit(session->rtp_socket);
        udp_exit(session->rtcp_socket);
        free(session->addr);
        free(session->opt);
        free(session);
}

/**
 * rtp_set_encryption_key:
 * @session: The RTP session.
 * @passphrase: The user-provided "pass phrase" to map to an encryption key.
 *
 * Converts the user supplied key into a form suitable for use with RTP
 * and install it as the active key. Passing in NULL as the passphrase
 * disables encryption. The passphrase is converted into a DES key as
 * specified in RFC1890, that is:
 * 
 *   - convert to canonical form
 * 
 *   - derive an MD5 hash of the canonical form
 * 
 *   - take the first 56 bits of the MD5 hash
 * 
 *   - add parity bits to form a 64 bit key
 * 
 * Note that versions of rat prior to 4.1.2 do not convert the passphrase
 * to canonical form before taking the MD5 hash, and so will
 * not be compatible for keys which are non-invarient under this step.
 *
 * Determine from the user's encryption key which encryption
 * mechanism we're using. Per the RTP RFC, if the key is of the form
 *
 *	string/key
 *
 * then "string" is the name of the encryption algorithm,  and
 * "key" is the key to be used. If no / is present, then the
 * algorithm is assumed to be (the appropriate variant of) DES.
 *
 * Returns: TRUE on success, FALSE on failure.
 */
int rtp_set_encryption_key(struct rtp *session, const char *passphrase)
{
        char *canonical_passphrase;
        u_char hash[16];
        MD5_CTX context;
        char *slash;

        check_database(session);
        if (session->encryption_algorithm != NULL) {
                free(session->encryption_algorithm);
                session->encryption_algorithm = NULL;
        }

        if (passphrase == NULL) {
                /* A NULL passphrase means disable encryption... */
                session->encryption_enabled = 0;
                check_database(session);
                return TRUE;
        }

        debug_msg("Enabling RTP/RTCP encryption\n");
        session->encryption_enabled = 1;

        /*
         * Determine which algorithm we're using.
         */

        slash = strchr(passphrase, '/');
        if (slash == 0) {
                session->encryption_algorithm = strdup("DES");
        } else {
                int l = slash - passphrase;
                session->encryption_algorithm = malloc(l + 1);
                strncpy(session->encryption_algorithm, passphrase, l);
                session->encryption_algorithm[l] = '\0';
                passphrase = slash + 1;
        }

        debug_msg("Initializing encryption, algorithm is '%s'\n",
                  session->encryption_algorithm);

        /* Step 1: convert to canonical form, comprising the following steps:  */
        /*   a) convert the input string to the ISO 10646 character set, using */
        /*      the UTF-8 encoding as specified in Annex P to ISO/IEC          */
        /*      10646-1:1993 (ASCII characters require no mapping, but ISO     */
        /*      8859-1 characters do);                                         */
        /*   b) remove leading and trailing white space characters;            */
        /*   c) replace one or more contiguous white space characters by a     */
        /*      single space (ASCII or UTF-8 0x20);                            */
        /*   d) convert all letters to lower case and replace sequences of     */
        /*      characters and non-spacing accents with a single character,    */
        /*      where possible.                                                */
        canonical_passphrase = (char *)strdup(passphrase);      /* FIXME */

        /* Step 2: derive an MD5 hash */
        MD5Init(&context);
        MD5Update(&context, (u_char *) canonical_passphrase,
                  strlen(canonical_passphrase));
        MD5Final((u_char *) hash, &context);

        /* Initialize the encryption algorithm we've received */

        if (strcmp(session->encryption_algorithm, "DES") == 0) {
                return des_initialize(session, hash, sizeof(hash));
        } else if (strcmp(session->encryption_algorithm, "Rijndael") == 0) {
                return rijndael_initialize(session, hash, sizeof(hash));
        } else {
                debug_msg("Encryption algorithm \"%s\" not found\n",
                          session->encryption_algorithm);
                return FALSE;
        }
}

static int des_initialize(struct rtp *session, u_char * hash, int hashlen)
{
        char *key;
        int i, j, k;

        UNUSED(hashlen);

        session->encryption_pad_length = 8;
        session->encrypt_func = des_encrypt;
        session->decrypt_func = des_decrypt;

        if (session->crypto_state.des.encryption_key != NULL) {
                free(session->crypto_state.des.encryption_key);
        }

        key = session->crypto_state.des.encryption_key = (char *)malloc(8);

        /* Step 3: take first 56 bits of the MD5 hash */
        key[0] = hash[0];
        key[1] = hash[0] << 7 | hash[1] >> 1;
        key[2] = hash[1] << 6 | hash[2] >> 2;
        key[3] = hash[2] << 5 | hash[3] >> 3;
        key[4] = hash[3] << 4 | hash[4] >> 4;
        key[5] = hash[4] << 3 | hash[5] >> 5;
        key[6] = hash[5] << 2 | hash[6] >> 6;
        key[7] = hash[6] << 1;

        /* Step 4: add parity bits */
        for (i = 0; i < 8; ++i) {
                k = key[i] & 0xfe;
                j = k;
                j ^= j >> 4;
                j ^= j >> 2;
                j ^= j >> 1;
                j = (j & 1) ^ 1;
                key[i] = k | j;
        }

        check_database(session);
        return TRUE;
}

static int des_encrypt(struct rtp *session, unsigned char *data,
                       unsigned int size, unsigned char *initVec)
{
        qfDES_CBC_e((unsigned char *)session->crypto_state.des.encryption_key,
                    data, size, initVec);
        return TRUE;
}

static int des_decrypt(struct rtp *session, unsigned char *data,
                       unsigned int size, unsigned char *initVec)
{
        qfDES_CBC_d((unsigned char *)session->crypto_state.des.encryption_key,
                    data, size, initVec);
        return TRUE;
}

static int rijndael_initialize(struct rtp *session, u_char * hash, int hash_len)
{
        int rc;

        session->encryption_pad_length = 16;
        session->encrypt_func = rijndael_encrypt;
        session->decrypt_func = rijndael_decrypt;

        rc = makeKey(&session->crypto_state.rijndael.keyInstEncrypt,
                     DIR_ENCRYPT, hash_len * 8, (char *)hash);
        if (rc < 0) {
                debug_msg("makeKey failed: %d\n", rc);
                return FALSE;
        }

        rc = makeKey(&session->crypto_state.rijndael.keyInstDecrypt,
                     DIR_DECRYPT, hash_len * 8, (char *)hash);
        if (rc < 0) {
                debug_msg("makeKey failed: %d\n", rc);
                return FALSE;
        }

        rc = cipherInit(&session->crypto_state.rijndael.cipherInst,
                        MODE_ECB, NULL);
        if (rc < 0) {
                debug_msg("cipherInst failed: %d\n", rc);
                return FALSE;
        }
        return TRUE;
}

static int rijndael_encrypt(struct rtp *session, unsigned char *data,
                            unsigned int size, unsigned char *initVec)
{
        int rc;

        UNUSED(initVec);

        /*
         * Try doing this in place. If it doesn't work that way,
         * we'll have to allocate a buffer and copy back.
         */
        rc = blockEncrypt(&session->crypto_state.rijndael.cipherInst,
                          &session->crypto_state.rijndael.keyInstEncrypt,
                          data, size * 8, data);
        return rc;
}

static int rijndael_decrypt(struct rtp *session, unsigned char *data,
                            unsigned int size, unsigned char *initVec)
{
        int rc;
        UNUSED(initVec);

        /*
         * Try doing this in place. If it doesn't work that way,
         * we'll have to allocate a buffer and copy back.
         */
        rc = blockDecrypt(&session->crypto_state.rijndael.cipherInst,
                          &session->crypto_state.rijndael.keyInstDecrypt,
                          data, size * 8, data);
        return rc;
}

/**
 * rtp_get_addr:
 * @session: The RTP Session.
 *
 * Returns: The session's destination address, as set when creating the
 * session with rtp_init() or rtp_init_if().
 */
char *rtp_get_addr(struct rtp *session)
{
        check_database(session);
        return session->addr;
}

/**
 * rtp_get_rx_port:
 * @session: The RTP Session.
 *
 * Returns: The UDP port to which this session is bound, as set when
 * creating the session with rtp_init() or rtp_init_if().
 */
uint16_t rtp_get_rx_port(struct rtp * session)
{
        check_database(session);
        return session->rx_port;
}

/**
 * rtp_get_tx_port:
 * @session: The RTP Session.
 *
 * Returns: The UDP port to which RTP packets are transmitted, as set
 * when creating the session with rtp_init() or rtp_init_if().
 */
uint16_t rtp_get_tx_port(struct rtp * session)
{
        check_database(session);
        return session->tx_port;
}

/**
 * rtp_get_ttl:
 * @session: The RTP Session.
 *
 * Returns: The session's TTL, as set when creating the session with
 * rtp_init() or rtp_init_if().
 */
int rtp_get_ttl(struct rtp *session)
{
        check_database(session);
        return session->ttl;
}

/**
 * rtp_set_recv_buf:
 * Sets  receiver buffer size
 * @session: The RTP Session.
 *
 * Returns: TRUE if succeeded
 *          FALSE otherwise
 */
int rtp_set_recv_buf(struct rtp *session, int bufsize)
{
        return udp_set_recv_buf(session->rtp_socket, bufsize);
}

/**
 * rtp_set_send_buf:
 * Sets sender buffer size
 * @session: The RTP Session.
 *
 * Returns: TRUE if succeeded
 *          FALSE otherwise
 */
int rtp_set_send_buf(struct rtp *session, int bufsize)
{
        return udp_set_send_buf(session->rtp_socket, bufsize);
}

/**
 * rtp_flush_recv_buf:
 * Flushes receiver buffer contents.
 * @session: The RTP Session.
 */
void rtp_flush_recv_buf(struct rtp *session)
{
        udp_flush_recv_buf(session->rtp_socket);
}

int rtp_change_dest(struct rtp *session, const char *addr)
{
        return udp_change_dest(session->rtp_socket, addr);
}

uint64_t rtp_get_bcount(struct rtp *session)
{
        return session->rtp_bcount;
}

int rtp_compute_fract_lost(struct rtp *session, uint32_t ssrc)
{
        int h;
        source *s;

        for (h = 0; h < RTP_DB_SIZE; h++) {
                for (s = session->db[h]; s != NULL; s = s->next) {
                        check_source(s);
                        if (s->ssrc == ssrc) {
                                /* Much of this is taken from A.3 of draft-ietf-avt-rtp-new-01.txt */
                                int extended_max = s->cycles + s->max_seq;
                                int expected = extended_max - s->base_seq + 1;
                                int lost = expected - s->received;
                                int expected_interval =
                                    expected - s->expected_prior;
                                int received_interval =
                                    s->received - s->received_prior;
                                int lost_interval =
                                    expected_interval - received_interval;
                                int fraction;
                                uint32_t lsr;
                                uint32_t dlsr;

                                //printf("lost_interval %d\n", lost_interval);
                                s->expected_prior = expected;
                                s->received_prior = s->received;
                                if (expected_interval == 0
                                    || lost_interval <= 0) {
                                        fraction = 0;
                                } else {
                                        fraction =
                                            (lost_interval << 8) /
                                            expected_interval;
                                }

                                return fraction;
                        }
                }
        }
        return 0;
}

