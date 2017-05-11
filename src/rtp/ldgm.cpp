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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>

#include <limits>

#include "host.h"

#include "ldgm.h"
#include "lib_common.h"

#include "ldgm/src/ldgm-session.h"
#include "ldgm/src/ldgm-session-cpu.h"
#include "ldgm/src/ldgm-session-gpu.h"
#include "ldgm/matrix-gen/matrix-generator.h"
#include "ldgm/matrix-gen/ldpc-matrix.h" // LDGM_MAX_K

#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "transmit.h"
#include "video.h"

using namespace std;

typedef enum {
        STD1500 = 1500,
        JUMBO9000 = 9000,
} packet_type_t;

typedef struct {
        packet_type_t packet_type;
        int frame_size;
        double loss;
        // result
        int k, m, c;

} configuration_t;


#define PCT2 2.0
#define PCT5 5.0
#define PCT10 10.0
typedef double loss_t;

loss_t losses[] = {
        std::numeric_limits<double>::min(),
                PCT2,
                PCT5,
                PCT10,
};

#define JPEG60_SIZE (144 * 1000)
#define JPEG80_SIZE (177 * 1000)
#define JPEG90_SIZE (217 * 1000)

#define UNCOMPRESSED_SIZE (1920 * 1080 * 2)

const configuration_t suggested_configurations[] = {
        // JPEG 60
        { STD1500, JPEG60_SIZE, PCT2, 750, 120, 5 },
        { STD1500, JPEG60_SIZE, PCT5, 1500, 450, 6 },
        { STD1500, JPEG60_SIZE, PCT10, 1000, 500, 7 },
        // JPEG 80
        { STD1500, JPEG80_SIZE, PCT2, 1500, 240, 5 },
        { STD1500, JPEG80_SIZE, PCT5, 1250, 375, 6 },
        { STD1500, JPEG80_SIZE, PCT10, 1500, 750, 8 },
        // JPEG 90
        { STD1500, JPEG90_SIZE, PCT2, 1500, 240, 6 },
        { STD1500, JPEG90_SIZE, PCT5, 1500, 450, 6 },
        { STD1500, JPEG90_SIZE, PCT10, 1500, 750, 8 },

        // uncompressed
        { JUMBO9000, UNCOMPRESSED_SIZE, PCT2, 1500, 180, 5 },
        { JUMBO9000, UNCOMPRESSED_SIZE, PCT5, 1000, 300, 6 },
        { JUMBO9000, UNCOMPRESSED_SIZE, PCT10, 1000, 500, 7 },

        { STD1500, UNCOMPRESSED_SIZE, PCT2, 1500, 250, 5 },
        { STD1500, UNCOMPRESSED_SIZE, PCT5, 1500, 650, 6 },
        { STD1500, UNCOMPRESSED_SIZE, PCT10, 1500, 1500, 8 },
};

#define MINIMAL_VALUE 64 ///< reasonable minimum (seems that 32 crashes sometimes)
                         ///< @todo Check why doesn't lower values work out
#define DEFAULT_K 256
#define DEFAULT_M 192
#define DEFAULT_C 5
#define MIN_C 2 // reasonable minimum
#define MAX_C 63 // from packet format
#define MAX_K (1<<13) - 1

static bool file_exists(char *filename);
static void usage(void);

static bool file_exists(char *filename)
{
        struct stat sb;
        if (stat(filename, &sb)) {
                perror("stat");
                return false;
        }

        return true;
}

void ldgm::set_params(unsigned int k, unsigned int m, unsigned int c, unsigned int seed)
{
        m_k = k;
        m_m = m;
        m_c = c;
        m_seed = seed;

        m_coding_session->set_params(k, m, c);

        char filename[512];
        char path[256];

        int res;

#ifdef WIN32
        TCHAR tmpPath[MAX_PATH];
        UINT ret = GetTempPath(MAX_PATH, tmpPath);
        if(ret == 0 || ret > MAX_PATH) {
                fprintf(stderr, "Unable to get temporary directory name.\n");
                throw 1;
        }
        snprintf(path, 256, "%s\\ultragrid\\", tmpPath);
#else
        snprintf(path, 256, "/var/tmp/ultragrid-%d/", (int) getuid());
#endif

        res = platform_mkdir(path);
        if(res != 0) {
                if(errno != EEXIST) {
                        perror("mkdir");
                        fprintf(stderr, "[LDGM] Unable to create data directory.\n");
                        throw 1;
                }
        }
        snprintf(filename, 512, "%s/ldgm_matrix-%u-%u-%u-%u.bin", path, k, m, c, seed);
        if(!file_exists(filename)) {
                int ret = generate_ldgm_matrix(filename, k, m, c, seed, 0);
                if(ret != 0) {
                        fprintf(stderr, "[LDGM] Unable to initialize LDGM matrix.\n");
                        throw 1;
                }
        }

        m_coding_session->set_pcMatrix(filename);
}

ADD_TO_PARAM(ldgm_device, "ldgm-device", "* ldgm-device={CPU|GPU}\n"
                "  specify whether use CPU or GPU for LDGM\n");

void ldgm::init(unsigned int k, unsigned int m, unsigned int c, unsigned int seed)
{
        bool ldgm_device_gpu = false;

        if (get_commandline_param("ldgm-device")) {
            ldgm_device_gpu = strcasecmp(get_commandline_param("ldgm-device"), "GPU") == 0;
        }

        if (ldgm_device_gpu) {
                LDGM_session_gpu *(*loader)();
                loader = reinterpret_cast<LDGM_session_gpu *(*)()>(
                                const_cast<void *>(
                                        load_library("ldgm_gpu",
                                                LIBRARY_CLASS_UNDEFINED, LDGM_GPU_API_VERSION)));
                if (!loader) {
                        throw string("GPU accelerated LDGM support is not compiled in");
                } else {
                        m_coding_session = unique_ptr<LDGM_session>(loader());

                }
        } else {
                m_coding_session = unique_ptr<LDGM_session>(new LDGM_session_cpu());
        }

        set_params(k, m, c, seed);
}

ldgm::ldgm(unsigned int k, unsigned int m, unsigned int c, unsigned int seed)
{
        init(k, m, c, seed);
}

void ldgm::decode(const char *frame, int size, char **out, int *out_size, const map<int, int> &packets) {
        char *decoded;
        decoded = m_coding_session->decode_frame((char *) frame, size, out_size, packets);
        *out = decoded;
}

//////////////////////////////////
// ENCODER
//////////////////////////////////
static void usage() {
        printf("LDGM usage:\n"
                        "\t-f ldgm:<expected_loss>%% | ldgm[:<k>:<m>[:c]]\n"
                        "\n"
                        "\t\t<expected_loss> - expected maximal loss in percent (including '%%'-sign)\n\n"
                        "\t\t<k> - matrix width\n"
                        "\t\t<m> - matrix height\n"
                        "\t\t<c> - number of ones per column\n"
                        "\t\t\tthe bigger ratio m/k, the better correction (but also needed bandwidth)\n"
                        "\t\t\tk,m should be in interval [%d, %d]; c in [%d, %d]\n"
                        "\t\t\tdefault: k = %d, m = %d, c = %d\n"
                        "\n",
                        MINIMAL_VALUE, MAX_K,
                        MIN_C, MAX_C,
                        DEFAULT_K, DEFAULT_M, DEFAULT_C
                        );
}
ldgm::ldgm(const char *cfg)
{
        unsigned int k = DEFAULT_K,
                     m = DEFAULT_M,
                     c = DEFAULT_C;

        if(cfg && strlen(cfg) > 0) {
                if(strcasecmp(cfg, "help") == 0) {
                        usage();
                        throw 0;
                }
                char *save_ptr = NULL;
                char *item;
                char *tmp = strdup(cfg);
                item = strtok_r(tmp, ":", &save_ptr);
                if(item) {
                        k = atoi(item);
                }
                item = strtok_r(NULL, ":", &save_ptr);
                if(item) {
                        m = atoi(item);
                        item = strtok_r(NULL, ":", &save_ptr);
                        if(item) {
                                c = atoi(item);
                        }
                } else {
                        throw string("[LDGM] Was set k value but not m.");
                }
                free(tmp);
        }

        if(c < MIN_C || c > MAX_C) {
                ostringstream oss;
                oss << "[LDGM] C value shoud be inside interval [" << MIN_C << ", " << MAX_C << "]";
                throw oss.str();
        }

        if(k > MAX_K) {
                ostringstream oss;
                oss << "[LDGM] K value exceeds maximal value " <<  MAX_K;
                throw oss.str();
        }

        if(k < MINIMAL_VALUE || m < MINIMAL_VALUE) {
                ostringstream oss;
                oss << "[LDGM] Either k or m is lower than minimal value " << MINIMAL_VALUE;
                throw oss.str();
        }

        printf("[LDGM] Using values k = %u, m = %u, c = %u.\n", k, m, c);

        init(k, m, c);
}

ldgm::ldgm(int packet_size, int frame_size, double max_expected_loss)
{
        packet_type_t packet_type;
        int nearest = INT_MAX;
        loss_t loss = -1.0;
        int k, m, c;

        assert(max_expected_loss >= 0.0 && max_expected_loss <= 100.0);

        if(frame_size < 2000000 && packet_size >= (STD1500 + JUMBO9000) / 2) {
                throw string("LDGM: with frames smaller than 2M you should use standard Ethernet frames.");
        }

        if(packet_size < (STD1500 + JUMBO9000) / 2) {
                packet_type = STD1500;
        } else {
                packet_type = JUMBO9000;
        }

        for(unsigned int i = 1; i < sizeof(losses) / sizeof(loss_t); ++i) {
                if(max_expected_loss >= losses[i - 1] && max_expected_loss <= losses[i]) {
                        loss = losses[i];
                        break;
                }
        }

        if(loss == -1.0) {
                ostringstream oss;
                oss <<  "LDGM: Cannot provide predefined settings for correction of loss of " << setprecision(2)
                        << max_expected_loss << endl << 
                        "LDGM: You have to try and set LDGM parameters manually. You can inform us if you need better protection.";
                throw oss.str();
        }

        printf("LDGM: Choosing maximal loss %2.2f percent.\n", loss);

        for(unsigned int i = 0; i < sizeof(suggested_configurations) / sizeof(configuration_t); ++i) {
                if(suggested_configurations[i].packet_type == packet_type &&
                                suggested_configurations[i].loss == loss) {
                        if(abs(suggested_configurations[i].frame_size - frame_size) < abs(nearest - frame_size)) {
                                nearest = suggested_configurations[i].frame_size;
                                k = suggested_configurations[i].k;
                                m = suggested_configurations[i].m;
                                c = suggested_configurations[i].c;
                        }
                }
        }

        if (nearest == INT_MAX) {
                throw string("LDGM: Could not find matching configuration for your request.\n"
                                "Please, set LDGM parameters manually.");
        }

        double difference_from_frame_size = abs(nearest - frame_size) / (double) frame_size;
        if(difference_from_frame_size > 0.2) {
                fprintf(stderr, "LDGM: Chosen frame size setting is %.2f percent %s than your frame size.\n",
                                difference_from_frame_size * 100.0, (nearest - frame_size > 0 ? "higher" : "lower"));
                fprintf(stderr, "LDGM: This is the most approching one to your values.\n");
                fprintf(stderr, "You may wish to set the parameters manually.\n");
        }

        if(difference_from_frame_size > 0.5) {
                throw string("");
        }

        init(k, m, c);
}

shared_ptr<video_frame> ldgm::encode(shared_ptr<video_frame> tx_frame)
{
        // We need to have copy of coding session shared pointer in order to exit
        // gracefully even when destructed LDGM state first with some frame still
        // existing.
        std::shared_ptr<LDGM_session> coding_session = this->m_coding_session;
        shared_ptr<video_frame> out(vf_alloc_desc(video_desc_from_frame(tx_frame.get())),
                        [coding_session](struct video_frame *frame) {
                                for (unsigned int i = 0; i < frame->tile_count; ++i) {
                                        coding_session->free_out_buf(frame->tiles[i].data);
                                }
                                vf_free(frame);
                        });

        for (unsigned int i = 0; i < tx_frame->tile_count; ++i) {
                video_payload_hdr_t video_hdr;
                format_video_header(tx_frame.get(), i, 0, video_hdr);

                int out_size;
                char *output = m_coding_session->encode_hdr_frame((char *) video_hdr, sizeof(video_hdr),
                                tx_frame->tiles[i].data, tx_frame->tiles[i].data_len, &out_size);

                out->tiles[i].data = output;
                out->tiles[i].data_len = out_size;
        }

        out->fec_params.type = FEC_LDGM;
        out->fec_params.k = m_k;
        out->fec_params.m = m_m;
        out->fec_params.c = m_c;
        out->fec_params.seed = m_seed;
        out->fec_params.symbol_size = m_coding_session->get_packet_size();

        return out;
}

