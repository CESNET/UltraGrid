/*
 * FILE:     ldgm.c
 * AUTHOR:   Martin Pulec <pulec@cesnet.cz>
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <iostream>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "ldgm.h"

#include "ldgm-coding/ldgm-session.h"
#include "ldgm-coding/ldgm-session-cpu.h"
#include "ldgm-coding/matrix-gen/matrix-generator.h"
#include "ldgm-coding/matrix-gen/ldpc-matrix.h" // LDGM_MAX_K

#define MINIMAL_VALUE 64 // reasonable minimum (seems that 32 crashes sometimes)
#define DEFAULT_K 256
#define DEFAULT_M 192
#define SEED 1
#define DEFAULT_C 5
#define MIN_C 2 // reasonable minimum
#define MAX_C 31 // from packet format

static bool file_exists(char *filename);

static bool file_exists(char *filename)
{
        struct stat sb;
        if (stat(filename, &sb)) {
                perror("stat");
                return false;
        }

        return true;
}

struct ldgm_state_encoder {
        ldgm_state_encoder(unsigned int k_, unsigned int m_, unsigned int c_) :
                k(k_),
                m(m_),
                c(c_)
        {
                seed = SEED;
                coding_session.set_params(k, m, c);

                char filename[512];

                int res;

                res = mkdir("/var/tmp/ultragrid/", 0755);
                if(res != 0) {
                        if(errno != EEXIST) {
                                perror("mkdir");
                                fprintf(stderr, "[LDGM] Unable to create data directory.\n");
                                throw 1;
                        }
                }
                snprintf(filename, 512, "/var/tmp/ultragrid/ldgm_matrix-%u-%u-%u-%u.bin", k, m, c, seed);
                if(!file_exists(filename)) {
                        int ret = generate_ldgm_matrix(filename, k, m, c, seed);
                        if(ret != 0) {
                                fprintf(stderr, "[LDGM] Unable to initialize LDGM matrix.\n");
                                throw 1;
                        }
                }

                coding_session.set_pcMatrix(filename);
        }

        char * encode(char *hdr, int hdr_len, char *frame, int size, int *out_size) {
                char *output = coding_session.encode_hdr_frame(hdr, hdr_len,
                                frame, size, out_size);
                return output;
        }

        void freeBuffer(char *buffer) {
                coding_session.free_out_buf(buffer);
        }

        virtual ~ldgm_state_encoder() {
        }

        unsigned int get_k() {
                return k;
        }

        unsigned int get_m() {
                return m;
        }

        unsigned int get_c() {
                return c;
        }

        unsigned int get_seed() {
                return seed;
        }

        private:
        LDGM_session_cpu coding_session;
        char *buffer;
        unsigned int k, m, c;
        unsigned int seed;
        char *left_matrix;
};

struct ldgm_state_decoder {
        ldgm_state_decoder(unsigned int k, unsigned int m, unsigned int c, unsigned int seed) {
                coding_session.set_params(k, m, c);
                char filename[512];

                int res;

                res = mkdir("/var/tmp/ultragrid/", 0755);
                if(res != 0) {
                        if(errno != EEXIST) {
                                perror("mkdir");
                                fprintf(stderr, "[LDGM] Unable to create data directory.\n");
                                throw 1;
                        }
                }
                snprintf(filename, 512, "/var/tmp/ultragrid/ldgm_matrix-%u-%u-%u-%u.bin", k, m, c, seed);
                if(!file_exists(filename)) {
                        int ret = generate_ldgm_matrix(filename, k, m, c, seed);
                        if(ret != 0) {
                                fprintf(stderr, "[LDGM] Unable to initialize LDGM matrix.\n");
                                throw 1;
                        }
                }

                coding_session.set_pcMatrix(filename);
        }

        void decode(const char *frame, int size, char **out, int *out_size, struct linked_list *ll) {
                char *decoded;

                decoded = coding_session.decode_frame((char *) frame, size, out_size, ll_to_map(ll));

                *out = decoded;
        }

        virtual ~ldgm_state_decoder() {
        }

        private:
        LDGM_session_cpu coding_session;
        char *buffer;
};


//////////////////////////////////
// ENCODER
//////////////////////////////////
void usage() {
        printf("LDGM usage:\n"
                        "\t-f ldgm[:<k>:<m>[:c]]\n"
                        "\t\tk - matrix width\n"
                        "\t\tm - matrix height\n"
                        "\n\t\t\tthe bigger ratio m/k, the better correction (but also needed bandwidth)\n"
                        "\n\t\t\tk,m should be in interval [%d, %d]; c in [%d, %d]\n"
                        "\n\t\t\tk,m must be divisible by 32\n"
                        "\n\t\t\tdefault: k = %d, m = %d, c = %d\n",
                        MINIMAL_VALUE, LDGM_MAX_K,
                        MIN_C, MAX_C,
                        DEFAULT_K, DEFAULT_M, DEFAULT_C
                        );
}
void *ldgm_encoder_init(char *cfg)
{
        unsigned int k = DEFAULT_K,
                     m = DEFAULT_M,
                     c = DEFAULT_C;

        struct ldgm_state_encoder *s = NULL;

        if(cfg && strlen(cfg) > 0) {
                if(strcasecmp(cfg, "help") == 0) {
                        usage();
                        return NULL;
                }
                char *save_ptr = NULL;
                char *item;
                item = strtok_r(cfg, ":", &save_ptr);
                if(item) {
                        k = atoi(item);
                }
                item = strtok_r(NULL, ":", &save_ptr);
                if(item) {
                        m = atoi(item);
                        item = strtok_r(NULL, ":", &save_ptr);
                        if(item) {
                                c = DEFAULT_C;
                        }
                } else {
                        k = DEFAULT_K;
                        fprintf(stderr, "[LDGM] Was set k value but not m. Using default values.\n");
                }
        }

        if(c < MIN_C || c > MAX_C) {
                fprintf(stderr, "[LDGM] C value shoud be inside interval [%d, %d].\n", MIN_C, MAX_C);
                c = DEFAULT_C;
        }

        if(k > LDGM_MAX_K) {
                fprintf(stderr, "[LDGM] K value exceeds maximal value %d.\n", LDGM_MAX_K);
                k = DEFAULT_K;
                m = DEFAULT_M;
        }

        if(k < MINIMAL_VALUE || m < MINIMAL_VALUE) {
                fprintf(stderr, "[LDGM] Either k or m is lower than minimal value %d.\n", MINIMAL_VALUE);
                k = DEFAULT_K;
                m = DEFAULT_M;
        }

        if(k % 32 != 0 || m % 32 != 0) {
                fprintf(stderr, "[LDGM] Either k or m is not divisible by 32.\n");
                k = DEFAULT_K;
                m = DEFAULT_M;
        }

        printf("[LDGM] Using values k = %u, m = %u, c = %u.\n", k, m, c);

        try {
                s = new ldgm_state_encoder(k, m, c);
        } catch(...) {
        }

        return (void *) s;
}

unsigned int ldgm_encoder_get_k(void *state)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;
        return s->get_k();
}

unsigned int ldgm_encoder_get_m(void *state)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;
        return s->get_m();
}

unsigned int ldgm_encoder_get_c(void *state)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;
        return s->get_c();
}

unsigned int ldgm_encoder_get_seed(void *state)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;
        return s->get_seed();
}

void ldgm_encoder_encode(void *state, const char *hdr, int hdr_len, const char *in, int in_len, char **out, int *len)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;
        char *output_buffer;

        output_buffer = s->encode(const_cast<char *>(hdr), hdr_len, const_cast<char *>(in), in_len, len);

        *out = output_buffer;
}

void ldgm_encoder_free_buffer(void *state, char *buffer)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;

        s->freeBuffer(buffer);
}

void ldgm_encoder_destroy(void *state)
{
        struct ldgm_state_encoder *s = (struct ldgm_state_encoder *) state;

        delete s;
}


//////////////////////////////////
// DECODER
//////////////////////////////////
void * ldgm_decoder_init(unsigned int k, unsigned int m, unsigned int c, unsigned int seed) {
        struct ldgm_state_decoder *s = NULL;
       
        try {
                s = new ldgm_state_decoder(k, m, c, seed);
        } catch (...) {
        }

        return s;
}

void ldgm_decoder_decode(void *state, const char *in, int in_len, char **out, int *len, struct linked_list  *ll)
{
        struct ldgm_state_decoder *s = (struct ldgm_state_decoder *) state;

        s->decode(in, in_len, out, len, ll);
}


void ldgm_decoder_destroy(void *state)
{
        struct ldgm_state_decoder *s = (struct ldgm_state_decoder *) state;

        delete s;
}

