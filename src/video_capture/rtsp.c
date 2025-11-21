/*
 * AUTHOR:   Gerard Castillo <gerard.castillo@i2cat.net>,
 *           Martin German <martin.german@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2015-2025 CESNET
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
 */
/**
 * @file
 * Example stream to test can be generated with:
 *
 *     docker run --rm -it --network=host bluenviron/mediamtx
 *     ffmpeg -re -f lavfi -i smptebars=s=1920x1080 -vcodec libx264 -tune zerolatency -f rtsp rtsp://localhost:8554/mystream
 *     ffmpeg -re -f lavfi -i smptebars=s=1280x720 -vcodec mjpeg -huffman 0 -f rtsp rtsp://localhost:8554/mystream
 *     test also with testsrc (bigger frames -> fragments), also -pix_fmt yuv444p (implied by testsrc)
 */

#include <assert.h>                // for assert
#include <errno.h>                 // for ETIMEDOUT
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>                // for uint8_t, uint32_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>                  // for timespec
#ifndef _WIN32
#include <unistd.h>                // for unlink
#endif // defined _WIN32

#define WANT_PTHREAD_NULL
#include "audio/types.h"
#include "config.h"                // for PACKAGE_BUGREPORT
#include "compat/aligned_malloc.h" // for alignde_free, aligned_alloc
#include "compat/misc.h"           // for PTHREAD_NULL
#include "compat/strings.h"        // for strncasecmp
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "rtp/rtpenc_h264.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpdec_jpeg.h"
#include "rtp/rtpdec_state.h"
#include "rtsp/rtsp_utils.h"
#include "utils/color_out.h"       // for color_printf, TBOLD
#include "utils/fs.h"              // for get_temp_file
#include "utils/macros.h"          // for MIN, STR_LEN
#include "utils/sdp.h"             // for get_video_codec_from_pt_rtpmap
#include "utils/text.h" // base64_decode
#include "video_decompress.h"

#include "pdb.h"
#include "rtp/pbuf.h"

#include "video.h"
#include "video_codec.h"
#include "video_capture.h"

#include <curl/curl.h>

#define KEEPALIVE_INTERVAL_S 5
#define MAGIC to_fourcc('R', 'T', 'S', 'c')
#define MOD_NAME  "[rtsp] "
#define VERSION_STR  "V1.0"

//TODO set lower initial video recv buffer size (to find the minimal?)
#define DEFAULT_VIDEO_FRAME_WIDTH 1920
#define DEFAULT_VIDEO_FRAME_HEIGHT 1080
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((0.1*DEFAULT_VIDEO_FRAME_WIDTH*DEFAULT_VIDEO_FRAME_HEIGHT)*110/100) //command line net.core setup: sysctl -w net.core.rmem_max=9123840

// compat
#ifndef CURL_WRITEFUNC_ERROR
#define CURL_WRITEFUNC_ERROR 0xFFFFFFFF
#endif

enum {
    DEFAULT_RTSP_PORT = 554,
};

static const char *long_to_str(long l) {
    _Thread_local static char buf[100];
    snprintf(buf, sizeof buf, "%ld", l);
    return buf;
}

static const char *pointer_to_str(void *p) {
    _Thread_local static char buf[100];
    snprintf(buf, sizeof buf, "%p", p);
    return buf;
}

static const char *cstr_identity(const char *c) {
    return c;
}

#define get_s(X) \
        _Generic((X), \
            int: long_to_str, \
            long: long_to_str, \
            char *: cstr_identity, \
            const char *: cstr_identity, \
            default: pointer_to_str)(X)

/* error handling macros */
#define my_curl_easy_setopt_ex(l, A, B, C, action_fail) \
    { \
        log_msg(l, MOD_NAME "Setting " #B " to %s\n", get_s(C)); \
        const CURLcode res = curl_easy_setopt((A), (B), (C)); \
        if (res != CURLE_OK) { \
            log_msg(LOG_LEVEL_ERROR, MOD_NAME "curl_easy_setopt(%s, %s, %s) failed: %s (%d)\n", #A, #B, #C, curl_easy_strerror(res), res); \
            printf("[rtsp error] could not configure rtsp capture properly, \n\t\tplease check your parameters. \nExiting...\n\n"); \
            action_fail; \
        } \
    }

#define my_curl_easy_perform_ex(l, A, action_fail) \
    { \
        log_msg(l, "Performing cURL operation(s)\n"); \
        const CURLcode res = curl_easy_perform((A)); \
        if (res != CURLE_OK) { \
            log_msg(LOG_LEVEL_ERROR, MOD_NAME "[%s] curl_easy_perform(%s) failed: %s (%d)\n", __func__, #A, curl_easy_strerror(res), res); \
            printf("[rtsp error] could not configure rtsp capture properly, \n\t\tplease check your parameters. \nExiting...\n\n"); \
            action_fail; \
        } \
    }

#define my_curl_easy_setopt(A, B, C, action_fail) \
        my_curl_easy_setopt_ex(LOG_LEVEL_VERBOSE, A, B, C, action_fail)
#define my_curl_easy_perform(A, action_fail) \
        my_curl_easy_perform_ex(LOG_LEVEL_VERBOSE, A, action_fail)

struct rtsp_state;
struct audio_rtsp_state;
struct video_rtsp_state;

/* send RTSP GET_PARAMETERS request */
static bool
rtsp_get_parameters(CURL *curl, const char *uri);

/* send RTSP OPTIONS request */
static bool
rtsp_options(CURL *curl, const char *uri);

/* send RTSP DESCRIBE request and write sdp response to a file */
static bool
rtsp_describe(CURL *curl, const char *uri, FILE *sdp_fp);

/* send RTSP SETUP request */
static bool
rtsp_setup(CURL *curl, const char *uri, const char *transport);

/* send RTSP PLAY request */
static bool
rtsp_play(CURL *curl, const char *uri, const char *range);

/* send RTSP TEARDOWN request */
static void
rtsp_teardown(CURL *curl, const char *uri);

static int
get_nals(FILE *sdp_file, codec_t codec, char *nals, int *width, int *height);

static bool setup_codecs_and_controls_from_sdp(FILE              *sdp_file,
                                               struct rtsp_state *rtspState);
static bool
init_rtsp(struct rtsp_state *s);

static int
init_decompressor(struct video_rtsp_state *sr, struct video_desc desc);

static void *
vidcap_rtsp_thread(void *args);

void
rtsp_keepalive(void *state);

static void vidcap_rtsp_done(void *state);

static const uint8_t start_sequence[] = { 0, 0, 0, 1 };

/**
 * @struct rtsp_state
 */
struct video_rtsp_state {
    char codec[SHORT_STR];

    struct video_desc desc;
    struct video_frame *out_frame;

    //struct std_frame_received *rx_data;
    bool decompress;

    struct state_decompress *sd;
    struct video_desc decompress_desc;

    int port;
    char *control;

    struct rtp *device;
    struct pdb *participants;
    double rtcp_bw;
    int ttl;
    char *mcast_if;
    int required_connections;

    pthread_t vrtsp_thread_id; //the worker_id

    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;

    struct decode_data_rtsp decode_data;
};

struct audio_rtsp_state {
    struct audio_frame audio;
    int play_audio_frame;

    char codec[SHORT_STR];

    struct timeval last_audio_time;
    unsigned int grab_audio:1;

    int port;

    char *control;

    struct rtp *device;
    struct pdb *participants;
    double rtcp_bw;
    int ttl;
    char *mcast_if;
    int required_connections;

    pthread_t artsp_thread_id; //the worker_id
    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;

    int pt;
};

struct rtsp_state {
    uint32_t magic;
    CURL *curl;
    char uri[1024];
    char base_url[1024]; ///< for control URLs with relative path; '/' included
    rtsp_types_t avType;
    const char *addr;
    char *sdp;

    volatile bool should_exit;
    struct audio_rtsp_state artsp_state;
    struct video_rtsp_state vrtsp_state;

    pthread_t keep_alive_rtsp_thread_id; //the worker_id
    pthread_mutex_t lock;
    pthread_cond_t keepalive_cv;

    bool setup_completed;
    _Bool sps_pps_emitted; ///< emit SPS/PPS once first to reduce decoding errors
};

static void
show_help(bool full) {
    color_printf(TBOLD("RTSP client") " usage:\n");
    color_printf("\t" TBOLD(TRED("-t rtsp:<uri>") "[:decompress]"));
    if (full) {
        color_printf(TBOLD("[:rtp_rx_port=<port>]"));
    }
    color_printf("\n\t" TBOLD("-t rtsp:[full]help") "\n");
    color_printf("\nOptions:\n");
    color_printf("\t " TBOLD("<uri>") " - RTSP server URI\n");
    printf("\t " TBOLD("decompress") " - decompress the stream "
            "(default: disabled)\n");
    if (full) {
        printf("\t " TBOLD("<port>") " - video RTP receiver port number\n");
    }
    color_printf("\nExamples:\n");
    color_printf("\t" TBOLD("uv -t rtsp://192.168.0.30/mystream") "\n");
    color_printf("\t" TBOLD("uv -t rtsp://[fe80::30]/mystream") "\n");
    color_printf("\t\t- capture stream on implicit port (554)\n");
    color_printf("\t"
        TBOLD("uv -t rtsp://192.168.0.20:8554/mystream") "\n");
    color_printf("\t\t- capture stream on port 8554 (optionally with "
        "authentization)\n");
    color_printf("\t"
        TBOLD("uv -t rtsp://user:pass@[fe80::30]/mystream") "\n");
    color_printf("\t\t- capture stream on default port with authentization\n");
    color_printf(
        "\t" TBOLD("uv -t rtsp://192.168.0.20/mystream:decompress") "\n");
    color_printf("\t\t- same as first case but decompress the stream "
        "(to allow to use a different compression)\n");
    color_printf("\n");

    color_printf(
        "Supported audio codecs: none (support is currently broken/WIP)\n");
    color_printf("Supported video codecs: " TBOLD("H.264") ", " TBOLD(
        "H.265") ", " TBOLD("JPEG") "\n");
    color_printf("\n");
}

static void *
keep_alive_thread(void *arg){
    struct rtsp_state *s = (struct rtsp_state *) arg;

    while (1) {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        struct timespec timeout = { .tv_sec = tp.tv_sec + KEEPALIVE_INTERVAL_S, .tv_nsec = tp.tv_usec * 1000 };
        pthread_mutex_lock(&s->lock);
        int rc = 0;
        while (!s->should_exit && rc != ETIMEDOUT) {
            rc = pthread_cond_timedwait(&s->keepalive_cv, &s->lock, &timeout);
        }
        if (s->should_exit) {
            pthread_mutex_unlock(&s->lock);
            break;
        }
        pthread_mutex_unlock(&s->lock);

        // actual keepalive
        MSG(DEBUG, "GET PARAMETERS %s:\n", s->uri);
        if (!rtsp_get_parameters(s->curl, s->uri)) {
            s->should_exit = true;
            exit_uv(1);
        }
    }
    return NULL;
}

static int
decode_frame_by_pt(struct coded_data *cdata, void *decode_data,
                   struct pbuf_stats *stats)
{    UNUSED(stats);
    rtp_packet *pckt = NULL;
    pckt = cdata->data;
    struct decode_data_rtsp *d = decode_data;
    if (pckt->pt != d->video_pt) {
        error_msg("Wrong Payload type: %u\n", pckt->pt);
        return 0;
    }
    return d->decode(cdata, decode_data);
}

static void
set_desc_width_height_if_changed(struct video_desc        *desc,
                                 const struct video_frame *frame)
{
        if (frame->tiles[0].width != 0 &&
            desc->width != frame->tiles[0].width &&
            desc->height != frame->tiles[0].height) {
                MSG(VERBOSE, "Setting the stream size to %ux%u\n",
                    frame->tiles[0].width, frame->tiles[0].height);
                desc->width  = frame->tiles[0].width;
                desc->height = frame->tiles[0].height;
        }
}

static void *
vidcap_rtsp_thread(void *arg) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) arg;

    time_ns_t start_time = get_time_in_ns();

    struct video_frame *frame = vf_alloc_desc(s->vrtsp_state.desc);

    while (!s->should_exit) {
        time_ns_t curr_time = get_time_in_ns();
        uint32_t timestamp = (curr_time - start_time) / (100*1000) * 9; // at 90000 Hz

        rtp_update(s->vrtsp_state.device, curr_time);

        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 10000;

        if (!rtp_recv_r(s->vrtsp_state.device, &timeout, timestamp)) {
            pdb_iter_t it;
            struct pdb_e *cp = pdb_iter_init(s->vrtsp_state.participants, &it);

            while (cp != NULL) {
                s->vrtsp_state.decode_data.frame = frame;
                if (pbuf_decode(cp->playout_buffer, curr_time,
                            decode_frame_by_pt, &s->vrtsp_state.decode_data))
                {
                    pthread_mutex_lock(&s->vrtsp_state.lock);
                    while (s->vrtsp_state.out_frame != NULL && !s->should_exit) {
                        s->vrtsp_state.worker_waiting = true;
                        pthread_cond_wait(&s->vrtsp_state.worker_cv, &s->vrtsp_state.lock);
                        s->vrtsp_state.worker_waiting = false;
                    }
                    if (s->vrtsp_state.out_frame == NULL) {
                        s->vrtsp_state.out_frame = frame;
                        set_desc_width_height_if_changed(&s->vrtsp_state.desc,
                                                         frame);
                        frame = vf_alloc_desc(s->vrtsp_state.desc); // alloc new
                        if (s->vrtsp_state.boss_waiting)
                            pthread_cond_signal(&s->vrtsp_state.boss_cv);
                        pthread_mutex_unlock(&s->vrtsp_state.lock);
                    } else {
                        pthread_mutex_unlock(&s->vrtsp_state.lock);
                    }
                }
                pbuf_remove(cp->playout_buffer, curr_time);
                cp = pdb_iter_next(&it);
            }

            pdb_iter_done(&it);
        }
    }
    vf_free(frame);
    return NULL;
}

static struct video_frame *
vidcap_rtsp_grab(void *state, struct audio_frame **audio) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) state;

    *audio = NULL;

    if (!s->sps_pps_emitted) {
        s->sps_pps_emitted = 1;
        return get_sps_pps_frame(&s->vrtsp_state.desc,
                                 &s->vrtsp_state.decode_data);
    }

    if(pthread_mutex_trylock(&s->vrtsp_state.lock)==0){
        {
            while (s->vrtsp_state.out_frame == NULL && !s->should_exit) {
                struct timeval  tp;
                gettimeofday(&tp, NULL);
                struct timespec timeout = { .tv_sec = tp.tv_sec, .tv_nsec = (tp.tv_usec + 100*1000) * 1000 };
                if (timeout.tv_nsec >= 1000L*1000*1000) {
                    timeout.tv_nsec -= 1000L*1000*1000;
                    timeout.tv_sec += 1;
                }
                s->vrtsp_state.boss_waiting = true;
                if (pthread_cond_timedwait(&s->vrtsp_state.boss_cv, &s->vrtsp_state.lock, &timeout) == ETIMEDOUT) {
                    pthread_mutex_unlock(&s->vrtsp_state.lock);
                    return NULL;
                }
                s->vrtsp_state.boss_waiting = false;
            }

            if (s->should_exit) {
                pthread_mutex_unlock(&s->vrtsp_state.lock);
                return NULL;
            }

            struct video_frame *frame = s->vrtsp_state.out_frame;
            s->vrtsp_state.out_frame = NULL;
            pthread_mutex_unlock(&s->vrtsp_state.lock);
            pthread_cond_signal(&s->vrtsp_state.worker_cv);

            if (frame->tiles[0].width == 0) {
                    MSG(WARNING,
                        "Dropped zero-sized frame - the size was not published "
                        "in RTSP, waiting for first SPS...\n");
                    vf_free(frame);
                    return NULL;
            }

            if (s->vrtsp_state.decompress) {
                struct video_desc curr_desc = video_desc_from_frame(frame);
                if (!video_desc_eq(s->vrtsp_state.decompress_desc, curr_desc)) {
                    decompress_done(s->vrtsp_state.sd);
                    if (init_decompressor(&s->vrtsp_state, curr_desc) == 0) {
                        return NULL;
                    }
                    s->vrtsp_state.decompress_desc = curr_desc;
                }

                struct video_desc out_desc = s->vrtsp_state.decompress_desc;
                out_desc.color_spec = UYVY;
                struct video_frame *decompressed = vf_alloc_desc_data(out_desc);

                decompress_frame(s->vrtsp_state.sd, (unsigned char *) decompressed->tiles[0].data,
                    (unsigned char *) frame->tiles[0].data,
                    frame->tiles[0].data_len, 0, NULL, NULL);
                vf_free(frame);
                frame = decompressed;
            }
            frame->callbacks.dispose = vf_free;
            return frame;
        }
    } else {
        return NULL;
    }
}

/**
 * @brief check URI validity + append port if not given
 *
 * If port is not given in the URI, 554 (default) is appended (after
 * authority, before the path if there is any).
 *
 * Resulting URI is written to output.
 */
static bool
check_uri(size_t uri_len, char *uri)
{
    const char *rtsp_uri_pref = "rtsp://";
    if (strcmp(uri, rtsp_uri_pref) == 0) {
        MSG(ERROR, "No URI given!\n");
        return false;
    }
    char *authority = uri + strlen(rtsp_uri_pref);
    char *host = authority;
    if (strchr(authority, '@') != NULL) { // skip userinfo
        host = strchr(authority, '@') + 1;
    }
    if (strchr(host, ':') == NULL) { // add port 554
        char *path = NULL;
        if (strchr(host, '/') != NULL) { // store path
            path = strdup(strchr(host, '/') + 1);
            *strchr(host, '/') = '\0';
        }
        snprintf(uri + strlen(uri), uri_len - strlen(uri), ":%d",
                 DEFAULT_RTSP_PORT);
        if (path != NULL) {
            snprintf(uri + strlen(uri), uri_len - strlen(uri), "/%s",
                     path);
            free(path);
        }
    } else {
        char *port = strchr(host, ':') + 1;
        char *endptr = NULL;
        strtol(port, &endptr, 10);
        if (endptr == port) {
            MSG(ERROR, "Non-numeric port \"%s\" (wrong option?)\n", port);
            return false;
        }
        if (strchr(port, ':') != NULL) {
            MSG(WARNING, "Colon in URI path - possibly wrong option?\n");
        }
    }
    MSG(INFO, "Using URI %s\n", uri);
    return true;
}

#define FAIL_SHOW_HELP \
                    vidcap_rtsp_done(s); \
                    show_help(false); \
                    return VIDCAP_INIT_FAIL;

static int
vidcap_rtsp_init(struct vidcap_params *params, void **state) {
    char fmt[STR_LEN];
    snprintf(fmt, sizeof fmt,  "%s", vidcap_params_get_fmt(params));
    if (strcmp(fmt, "help") == 0 || strcmp(fmt, "fullhelp") == 0) {
        show_help(strcmp(fmt, "fullhelp") == 0);
        return VIDCAP_INIT_NOERR;
    }

    if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
        log_msg(LOG_LEVEL_ERROR, "Audio is not entirely implemented in RTSP. "
                "Please contact " PACKAGE_BUGREPORT " if you wish to use it.\n");
        return VIDCAP_INIT_FAIL;
    }

    struct rtsp_state *s = (struct rtsp_state *) calloc(1, sizeof(struct rtsp_state));
    if (s == NULL) {
        return VIDCAP_INIT_FAIL;
    }
    s->artsp_state.artsp_thread_id = PTHREAD_NULL;
    s->vrtsp_state.vrtsp_thread_id = PTHREAD_NULL;

    //TODO now static codec assignment, to be dynamic as a function of supported codecs
    s->vrtsp_state.codec[0] = '\0';
    s->artsp_state.codec[0] = '\0';
    s->vrtsp_state.control = strdup("");
    s->artsp_state.control = strdup("");

    char *save_ptr = NULL;
    s->magic = MAGIC;
    s->avType = rtsp_type_none;

    s->addr = "127.0.0.1";
    s->vrtsp_state.device = NULL;
    s->vrtsp_state.rtcp_bw = 5 * 1024 * 1024; /* FIXME */
    s->vrtsp_state.ttl = 255;

    s->vrtsp_state.mcast_if = NULL;
    s->vrtsp_state.required_connections = 1;

    s->vrtsp_state.participants = pdb_init("rtsp", 0);

    s->vrtsp_state.decode_data.offset_len = 0;

    s->curl = NULL;

    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->keepalive_cv, NULL);
    pthread_mutex_init(&s->vrtsp_state.lock, NULL);
    pthread_cond_init(&s->vrtsp_state.boss_cv, NULL);
    pthread_cond_init(&s->vrtsp_state.worker_cv, NULL);

    s->sps_pps_emitted = 1; // default when not H264/HEVC

    char *tmp = fmt;
    char *item = NULL;
    strcpy(s->uri, "rtsp://");

    s->vrtsp_state.desc.tile_count = 1;

    bool in_uri = true;
    while ((item = strtok_r(tmp, ":", &save_ptr))) {
        tmp = NULL;
        bool option_given = true;
        if (strstr(item, "rtp_rx_port=") == item) {
            s->vrtsp_state.port = atoi(strchr(item, '=') + 1);
        } else if (strcmp(item, "decompress") == 0) {
            s->vrtsp_state.decompress = true;
        } else if (strstr(item, "size=")) {
            MSG(WARNING, "size= parameter is not used! Will be removed in "
                "future!\n");
        } else {
            option_given = false;
            if (in_uri) {
                if (strcmp(item, "rtsp") == 0) { // rtsp:
                    continue;
                }
                if (strstr(item, "//") == item) { // rtsp://
                    item += 2;
                }
                if (strcmp(s->uri, "rtsp://") != 0) {
                    strncat(s->uri, ":", sizeof s->uri - strlen(s->uri) - 1);
                }
                strncat(s->uri, item, sizeof s->uri - strlen(s->uri) - 1);
            } else {
                MSG(ERROR, "Unknown option: %s\n", item);
                FAIL_SHOW_HELP
            }
        }
        if (option_given) {
            in_uri = false;
        }
    }

    //re-check parameters
    if (!check_uri(sizeof s->uri, s->uri)) {
        FAIL_SHOW_HELP
    }
    snprintf(s->base_url, sizeof s->base_url, "%s/", s->uri); // default

    s->vrtsp_state.device = rtp_init_if("localhost", s->vrtsp_state.mcast_if, s->vrtsp_state.port, 0, s->vrtsp_state.ttl, s->vrtsp_state.rtcp_bw,
        0, rtp_recv_callback, (uint8_t *) s->vrtsp_state.participants, 0, false);
    if (s->vrtsp_state.device == NULL) {
        log_msg(LOG_LEVEL_ERROR, "[rtsp] Cannot initialize RTP device!\n");
        vidcap_rtsp_done(s);
        return VIDCAP_INIT_FAIL;
    }
    if (!rtp_set_option(s->vrtsp_state.device, RTP_OPT_WEAK_VALIDATION, 1)) {
        error_msg("[rtsp] RTP INIT failed - set option\n");
        return VIDCAP_INIT_FAIL;
    }
    if (!rtp_set_sdes(s->vrtsp_state.device, rtp_my_ssrc(s->vrtsp_state.device),
                RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING))) {
        error_msg("[rtsp] RTP INIT failed - set sdes\n");
        return VIDCAP_INIT_FAIL;
    }

    int ret = rtp_set_recv_buf(s->vrtsp_state.device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
    if (!ret) {
        error_msg("[rtsp] RTP INIT failed - set recv buf \nset command: sudo sysctl -w net.core.rmem_max=9123840\n");
        return VIDCAP_INIT_FAIL;
    }

    if (!rtp_set_send_buf(s->vrtsp_state.device, 1024 * 56)) {
        error_msg("[rtsp] RTP INIT failed - set send buf\n");
        return VIDCAP_INIT_FAIL;
    }
    ret=pdb_add(s->vrtsp_state.participants, rtp_my_ssrc(s->vrtsp_state.device));

    verbose_msg("[rtsp] rtp receiver init done\n");

    if (s->vrtsp_state.port == 0) {
        s->vrtsp_state.port = rtp_get_udp_rx_port(s->vrtsp_state.device);
        assert(s->vrtsp_state.port != 0);
    }

    verbose_msg(MOD_NAME "selected flags:\n");
    verbose_msg(MOD_NAME "\t  uri: %s\n",s->uri);
    verbose_msg(MOD_NAME "\t  port: %d\n", s->vrtsp_state.port);
    verbose_msg(MOD_NAME "\t  decompress: %d\n\n",s->vrtsp_state.decompress);

    if (!init_rtsp(s)) {
        vidcap_rtsp_done(s);
        return VIDCAP_INIT_FAIL;
    }

    //TODO fps should be autodetected, now reset and controlled at vidcap_grab function
    s->vrtsp_state.desc.fps = 30;
    s->vrtsp_state.desc.interlacing = PROGRESSIVE;

    s->should_exit = false;

    s->vrtsp_state.boss_waiting = false;
    s->vrtsp_state.worker_waiting = false;

    if (s->vrtsp_state.decompress) {
        if (init_decompressor(&s->vrtsp_state, s->vrtsp_state.desc) == 0) {
            vidcap_rtsp_done(s);
            return VIDCAP_INIT_FAIL;
        }
    }

    pthread_create(&s->vrtsp_state.vrtsp_thread_id, NULL, vidcap_rtsp_thread, s);
    pthread_create(&s->keep_alive_rtsp_thread_id, NULL, keep_alive_thread, s);

    verbose_msg("[rtsp] rtsp capture init done\n");

    *state = s;
    return VIDCAP_INIT_OK;
}

static CURL *init_curl() {
    CURL *curl;
    /* initialize curl */
    CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
    if (res != CURLE_OK) {
        fprintf(stderr, "[rtsp] curl_global_init(%s) failed: %s (%d)\n",
            "CURL_GLOBAL_ALL", curl_easy_strerror(res), res);
        return NULL;
    }
    curl_version_info_data *data = curl_version_info(CURLVERSION_NOW);
    fprintf(stderr, "[rtsp]    cURL V%s loaded\n", data->version);

    /* initialize this curl session */
    curl = curl_easy_init();
    if (curl == NULL) {
        curl_global_cleanup();
        fprintf(stderr, "[rtsp] curl_easy_init() failed\n");
        return NULL;
    }
    my_curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L, );
    return curl;
}

static size_t print_rtsp_header(char *buffer, size_t size, size_t nitems, void *userdata) {
    int aggregate_size = size * nitems;
    struct rtsp_state *s = (struct rtsp_state *) userdata;
    assert(s->magic == MAGIC);
    bool error_occured = false;
    if (strncmp(buffer, "RTSP/1.0 ", MIN(strlen("RTSP/1.0 "), (size_t) aggregate_size)) == 0) {
        int code = atoi(buffer + strlen("RTSP/1.0 "));
        error_occured = code != 200;
    }
    if (log_level >= LOG_LEVEL_DEBUG || error_occured) {
        log_msg(error_occured ? LOG_LEVEL_ERROR : log_level,
                MOD_NAME "%.*s", aggregate_size, buffer);
    }
    return error_occured ? CURL_WRITEFUNC_ERROR : nitems;
}

/// currently only searches for Content-Base or Content-Location header
static size_t
process_rtsp_describe_header(char *buffer, size_t size, size_t nitems,
                             void *userdata)
{
        const size_t ret = print_rtsp_header(buffer, size, nitems, userdata);
        struct rtsp_state *s        = userdata;
        char              *save_ptr = NULL;
        // doc for CURLOPT_HEADERFUNCTION is unclear if buffer is
        // NULL-terminated -  one place says so, another not, so do it for sure
        char dup[CURL_MAX_HTTP_HEADER];
        memcpy(dup, buffer, MIN(nitems * size, sizeof dup));
        dup[MIN(sizeof dup - 1, nitems * size)] = '\0';

        char *item = strtok_r(dup, " ", &save_ptr);
        if (item == NULL) {
                return ret;
        }
        if ((strcasecmp(item, "Content-Base:") != 0 &&
             strcasecmp(item, "Content-Location:") != 0)) {
                return ret;
        }
        item = strtok_r(NULL, " ", &save_ptr);
        if (item == NULL) {
                return ret;
        }
        snprintf(s->base_url, sizeof s->base_url - 1, "%s", item);
        char *end = (s->base_url + strlen(s->base_url)) - 1;
        // trim \r,\n
        while (end >= s->base_url && (*end == '\r' || *end == '\n')) {
                *end = '\0';
                --end;
        }
        // append '/' if needed
        if (end >= s->base_url && *end != '/') {
                *end++ = '/';
                *end++ = '\0';
        }
        MSG(VERBOSE, "Using base URL from headers: %s\n", s->base_url);
        return ret;
}

/**
 * Initializes rtsp state and internal parameters
 */
static bool
init_rtsp(struct rtsp_state *s) {
    /* initialize curl */
    s->curl = init_curl();

    if (!s->curl) {
        return false;
    }

    const char *range = "0.000-";
    MSG(DEBUG, "request %s\n", VERSION_STR);
    MSG(DEBUG, "    Project web site: http://code.google.com/p/rtsprequest/\n");
    MSG(DEBUG, "    Requires cURL V7.20 or greater\n\n");
    char Atransport[256] = "";
    char Vtransport[256] = "";
    int port = s->vrtsp_state.port;
    FILE *sdp_file = tmpfile();
    const char *sdp_file_name = NULL;
    if (sdp_file == NULL) {
        sdp_file = get_temp_file(&sdp_file_name);
        if (sdp_file == NULL) {
            perror("Creating SDP file");
            goto error;
        }
    }

    snprintf(Vtransport, sizeof Vtransport, "RTP/AVP;unicast;client_port=%d-%d", port, port + 1);

    //THIS AUDIO PORTS ARE AS DEFAULT UG AUDIO PORTS BUT AREN'T RELATED...
    snprintf(Atransport, sizeof Atransport, "RTP/AVP;unicast;client_port=%d-%d", port+2, port + 3);

    my_curl_easy_setopt(s->curl, CURLOPT_NOSIGNAL, 1L, goto error); //This tells curl not to use any functions that install signal handlers or cause signals to be sent to your process.
    //my_curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, 1);
    my_curl_easy_setopt(s->curl, CURLOPT_VERBOSE,
                        log_level >= LOG_LEVEL_DEBUG ? 1L : 0L, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_NOPROGRESS, 1L, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_HEADERDATA, s, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_HEADERFUNCTION, print_rtsp_header, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_URL, s->uri, goto error);

    //TODO TO CHECK CONFIGURING ERRORS
    //CURLOPT_ERRORBUFFER
    //http://curl.haxx.se/libcurl/c/curl_easy_perform.html

    /* request server options */
    if (!rtsp_options(s->curl, s->uri)) {
        goto error;
    }

    /* request session description and write response to sdp file */
    if (!rtsp_describe(s->curl, s->uri, sdp_file)) {
        goto error;
    }

    if (log_level >= LOG_LEVEL_VERBOSE) {
        fprintf(stderr, MOD_NAME "SDP:\n" MOD_NAME);
        int ch = 0;
        while ((ch = getc(sdp_file)) != EOF) {
            putc(ch, stderr);
            if (ch == '\n') {
                fprintf(stderr, MOD_NAME);
            }
        }
        rewind(sdp_file);
        fprintf(stderr, "\n\n");
    }

    if (!setup_codecs_and_controls_from_sdp(sdp_file, s)) {
        goto error;
    }
    if (s->vrtsp_state.decode_data.video_pt > -1) {
        s->vrtsp_state.desc.color_spec = get_video_codec_from_pt_rtpmap(
            s->vrtsp_state.decode_data.video_pt, s->vrtsp_state.codec);
        if (s->vrtsp_state.desc.color_spec == VC_NONE) {
            goto error;
        }
        const char *uri = s->vrtsp_state.control;
        verbose_msg(MOD_NAME " V URI = %s\n", uri);
        if (!rtsp_setup(s->curl, uri, Vtransport)) {
            goto error;
        }
    }
    if (strcmp(s->artsp_state.codec, "PCMU") == 0){
        const char *uri = s->vrtsp_state.control;
        verbose_msg(MOD_NAME " A URI = %s\n", uri);
        if (!rtsp_setup(s->curl, uri, Atransport)) {
            goto error;
        }
    }
    if (s->vrtsp_state.desc.color_spec == H264 ||
        s->vrtsp_state.desc.color_spec == H265) {
        s->vrtsp_state.decode_data.decode = decode_frame_h2645;
        /* get start nal size attribute from sdp file */
        const int len_nals  = get_nals(sdp_file,
            s->vrtsp_state.desc.color_spec,
            (char *) s->vrtsp_state.decode_data.h264.offset_buffer,
            (int *) &s->vrtsp_state.desc.width,
            (int *) &s->vrtsp_state.desc.height);
        s->vrtsp_state.decode_data.offset_len = len_nals;
        s->sps_pps_emitted = 0; // emit metadata with first grab
        MSG(VERBOSE, "playing %s video from server (size: WxH = %d x %d)...\n",
            get_codec_name(s->vrtsp_state.desc.color_spec),
            s->vrtsp_state.desc.width, s->vrtsp_state.desc.height);

    } else if (s->vrtsp_state.desc.color_spec == JPEG) {
        s->vrtsp_state.decode_data.decode = decode_frame_jpeg;
    } else {
        MSG(ERROR, "Video codec %s not yet supported by UG.\n",
            get_codec_name(s->vrtsp_state.desc.color_spec));
        goto error;
    }

    if (!rtsp_play(s->curl, s->uri, range)) {
        goto error;
    }
    s->setup_completed = true;

    fclose(sdp_file);
    return true;

error:
    if(sdp_file)
            fclose(sdp_file);
    if (sdp_file_name != NULL) {
        unlink(sdp_file_name);
    }
    return false;
}

static bool
setup_codecs_and_controls_from_sdp(FILE *sdp_file, struct rtsp_state *rtspState)
{
        rtspState->artsp_state.pt =
            rtspState->vrtsp_state.decode_data.video_pt = -1;

        char line[STR_LEN];

        enum {
                MEDIA_NONE,
                MEDIA_AUDIO,
                MEDIA_VIDEO,
        } media = MEDIA_NONE;

        int advertised_pt = -1;

        while (fgets(line, sizeof line, sdp_file) != NULL) {
                char buf[2001];
                int pt = -1;
                // m=video 0 RTP/AVP 96
                if (sscanf(line, "m=%2000s %*d RTP/AVP %d", buf, &pt) == 2) {
                        advertised_pt = pt;
                        if (strcmp(buf, "audio") == 0) {
                                media = MEDIA_AUDIO;
                        } else if (strcmp(buf, "video") == 0) {
                                media = MEDIA_VIDEO;
                        } else {
                                media = MEDIA_NONE;
                                MSG(VERBOSE, "Unknown media: %s\n", buf);
                                continue;
                        }
                        if ((media == MEDIA_AUDIO &&
                             rtspState->artsp_state.pt != -1) ||
                            rtspState->vrtsp_state.decode_data.video_pt != -1) {
                                MSG(WARNING, "Multiple media of same type, "
                                             "using last one...");
                        }
                        *(media == MEDIA_AUDIO
                              ? &rtspState->artsp_state.pt
                              : &rtspState->vrtsp_state.decode_data.video_pt) = pt;
                        continue;
                }

                if (media == MEDIA_NONE) {
                        continue; // either on session level or unknown media
                }

                if (sscanf(line, "a=control:%2000s", buf) == 1) {
                        const char *rtsp_scheme = "rtsp://";
                        if (strncasecmp(buf, rtsp_scheme,
                                        strlen(rtsp_scheme)) != 0) {
                                char relative_url[sizeof buf];
                                strcpy(relative_url, buf);
                                snprintf(buf, sizeof buf, "%s%s",
                                         rtspState->base_url, relative_url);
                        }
                        *(media == MEDIA_AUDIO
                              ? &rtspState->artsp_state.control
                              : &rtspState->vrtsp_state.control) = strdup(buf);
                        continue;
                }
                /// a=rtpmap:96 H264/90000
                if (sscanf(line, "a=rtpmap:%d %2000[^/]", &pt, buf) == 2) {
                        char *codec = media == MEDIA_AUDIO
                              ? rtspState->artsp_state.codec
                              : rtspState->vrtsp_state.codec;
                        if (pt != advertised_pt) {
                                MSG(WARNING,
                                    "media packet type %d doesn't match "
                                    "media advertised PT %d!\n",
                                    pt, advertised_pt);
                                snprintf(codec, SHORT_STR, "?");
                        }
                        assert(strlen(buf) + strlen(codec) < SHORT_STR);
                        snprintf(codec + strlen(codec),
                                 SHORT_STR - strlen(codec), "%s", buf);
                }
        }

        verbose_msg(MOD_NAME "AUDIO TRACK = %s FOR CODEC = %s PT = %d\n",
                    rtspState->artsp_state.control,
                    rtspState->artsp_state.codec,
                    rtspState->artsp_state.pt);
        verbose_msg(MOD_NAME "VIDEO TRACK = %s FOR CODEC = %s PT = %d\n",
                    rtspState->vrtsp_state.control,
                    rtspState->vrtsp_state.codec,
                    rtspState->vrtsp_state.decode_data.video_pt);
        rewind(sdp_file);
        return true;
}

/**
 * Initializes decompressor if required by decompress flag
 */
static int
init_decompressor(struct video_rtsp_state *sr, struct video_desc desc) {
    if (decompress_init_multi(H264, (struct pixfmt_desc) { 0 }, UYVY, &sr->sd, 1)) {
        decompress_reconfigure(sr->sd, desc, 16, 8, 0,
            vc_get_linesize(desc.width, UYVY), UYVY);
    } else
        return 0;
    return 1;
}

/**
 * send RTSP GET PARAMS request
 */
static bool
rtsp_get_parameters(CURL *curl, const char *uri) {
    my_curl_easy_setopt_ex(LOG_LEVEL_DEBUG, curl, CURLOPT_RTSP_STREAM_URI,
        uri, return false);
    my_curl_easy_setopt_ex(LOG_LEVEL_DEBUG, curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_GET_PARAMETER, return false);
    my_curl_easy_perform_ex(LOG_LEVEL_DEBUG, curl, return false);
    return true;
}

static void
rtsp_set_user_pass(CURL *curl, char *user_pass)
{
        char *save_ptr = NULL;
        char *user     = strtok_r(user_pass, ":", &save_ptr);
        assert(user != NULL);
        my_curl_easy_setopt(curl, CURLOPT_USERNAME, user, return);
        char *pass = strtok_r(NULL, ":", &save_ptr);
        if (pass == NULL)  {
            return;
        }
        my_curl_easy_setopt(curl, CURLOPT_PASSWORD, pass, return);
}

/**
 * send RTSP OPTIONS request
 */
static bool
rtsp_options(CURL *curl, const char *uri) {
    char control[1501] = "";

    verbose_msg("\n[rtsp] OPTIONS %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return false);

    sscanf(uri, "rtsp://%1500s", control);

    if (strchr(control, '@') != NULL) {
        *strchr(control, '@') = '\0';
        rtsp_set_user_pass(curl, control);
    }

    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long) CURL_RTSPREQ_OPTIONS,
                        return false);

    my_curl_easy_perform(curl, return false);
    return true;
}

/**
 * send RTSP DESCRIBE request and write sdp response to a file
 */
static bool
rtsp_describe(CURL *curl, const char *uri, FILE *sdp_fp) {
    verbose_msg("\n[rtsp] DESCRIBE %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, sdp_fp, return false);
    my_curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION,
                        process_rtsp_describe_header, return false);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_DESCRIBE, return false);

    my_curl_easy_perform(curl, return false);

    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, stdout, return false);
    my_curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, print_rtsp_header,
                        return false);
    rewind(sdp_fp);
    return true;
}

/**
 * send RTSP SETUP request
 */
static bool
rtsp_setup(CURL *curl, const char *uri, const char *transport) {
    verbose_msg("\n[rtsp] SETUP %s\n", uri);
    verbose_msg(MOD_NAME "\t TRANSPORT %s\n", transport);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return false);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_TRANSPORT, transport, return false);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long) CURL_RTSPREQ_SETUP,
                        return false);

    my_curl_easy_perform(curl, return false);
    return true;
}

/**
 * send RTSP PLAY request
 */
static bool
rtsp_play(CURL *curl, const char *uri, const char *range) {
    UNUSED(range);
    verbose_msg("\n[rtsp] PLAY %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return false);
    //my_curl_easy_setopt(curl, CURLOPT_RANGE, range);      //range not set because we want (right now) no limit range for streaming duration
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long) CURL_RTSPREQ_PLAY,
                        return false);
    my_curl_easy_perform(curl, return false);
    return true;
}

/**
 * send RTSP TEARDOWN request
 */
static void
rtsp_teardown(CURL *curl, const char *uri) {
    verbose_msg("\n[rtsp] TEARDOWN %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_TEARDOWN, return);

    my_curl_easy_perform(curl, return);
}

static void vidcap_rtsp_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
    *deleter = free;
    *available_cards = NULL;
    *count = 0;
}

static void
vidcap_rtsp_done(void *state) {
    struct rtsp_state *s = (struct rtsp_state *) state;

    pthread_mutex_lock(&s->lock);
    pthread_mutex_lock(&s->vrtsp_state.lock);
    s->should_exit = true;
    pthread_mutex_unlock(&s->vrtsp_state.lock);
    pthread_mutex_unlock(&s->lock);

    pthread_cond_signal(&s->keepalive_cv);
    pthread_cond_signal(&s->vrtsp_state.worker_cv);

    if (!pthread_equal(s->vrtsp_state.vrtsp_thread_id, PTHREAD_NULL)) {
        pthread_join(s->vrtsp_state.vrtsp_thread_id, NULL);
    }
    if (!pthread_equal(s->keep_alive_rtsp_thread_id, PTHREAD_NULL)) {
        pthread_join(s->keep_alive_rtsp_thread_id, NULL);
    }

    if(s->vrtsp_state.sd)
        decompress_done(s->vrtsp_state.sd);

    if (s->vrtsp_state.device) {
        rtp_done(s->vrtsp_state.device);
    }

    vf_free(s->vrtsp_state.out_frame);
    free(s->vrtsp_state.control);
    free(s->artsp_state.control);

    if (s->curl != NULL) {
        if (s->setup_completed) {
            rtsp_teardown(s->curl, s->uri);
        }

        curl_easy_cleanup(s->curl);
        curl_global_cleanup();
        s->curl = NULL;
    }

    pthread_mutex_destroy(&s->lock);
    pthread_cond_destroy(&s->keepalive_cv);
    pthread_mutex_destroy(&s->vrtsp_state.lock);
    pthread_cond_destroy(&s->vrtsp_state.boss_cv);
    pthread_cond_destroy(&s->vrtsp_state.worker_cv);

    free(s);
}

/**
 * scan sdp file for media control attributes to generate coded frame required params (WxH and offset)
 */
static int
get_nals(FILE *sdp_file, codec_t codec, char *nals, int *width, int *height) {
    int len_nals = 0;
    char s[1500];
    char *sprop = NULL;

    while (1) {
        if (sprop == NULL) { // fetch new line
                if (fgets(s, sizeof s, sdp_file) == NULL) {
                    break;
                }
                sprop = s;
        }
        sprop = strstr(sprop, "sprop-");
        if (sprop == NULL) {
            continue;
        }
        char *sprop_name = sprop;
        char *sprop_val = strchr(sprop, '=') + 1;
        *strchr(sprop, '=') = '\0'; // end sprop_name
        char *term = strchr(sprop_val, ';');
        if (term) {
            *term = '\0';
            sprop = term + 1;
        } else {
            sprop = sprop_val;
        }
        if (strcmp(sprop_name, "sprop-max-don-diff") == 0) {
            bug_msg(LOG_LEVEL_ERROR, "sprop-max-don-diff not implemented. ");
            continue;
        }
        if (strcmp(sprop_name, "sprop-parameter-sets") != 0 && // H.264
            strcmp(sprop_name, "sprop-vps") != 0 &&            // HEVC
            strcmp(sprop_name, "sprop-sps") != 0 &&
            strcmp(sprop_name, "sprop-pps") != 0) {
                MSG(VERBOSE, "Skipping unsupported %s\n", sprop_name);
                continue; // do not process unknown sprops
        }

        char *nal = 0;
        char *endptr = NULL;
        while ((nal = strtok_r(sprop_val, ",", &endptr))) {
            sprop_val = NULL;
            unsigned int length = 0;
            //convert base64 to binary
            unsigned char *nal_decoded = base64_decode(nal, &length);
            if (length == 0) {
                free(nal_decoded);
                continue;
            }

            memcpy(nals+len_nals, start_sequence, sizeof(start_sequence));
            len_nals += sizeof(start_sequence);
            memcpy(nals + len_nals, nal_decoded, length);
            len_nals += length;
            free(nal_decoded);

            char *nalInfo = nals + len_nals - length;
            uint8_t type = NALU_HDR_GET_TYPE(nalInfo, codec == H265);
            MSG(DEBUG, "%s %s (%d) (base64): %s\n", sprop_name,
                get_nalu_name(type, codec == H265), (int) type, nal);
            if (type == NAL_H264_SPS){
                width_height_from_h264_sps(width, height, (unsigned char *) (nals+(len_nals - length)), length);
            }
            if (type == NAL_HEVC_SPS){
                width_height_from_hevc_sps(width, height, (unsigned char *) (nals+(len_nals - length)), length);
            }
        }
    }

    rewind(sdp_file);
    return len_nals;
}

static const struct video_capture_info vidcap_rtsp_info = {
        vidcap_rtsp_probe,
        vidcap_rtsp_init,
        vidcap_rtsp_done,
        vidcap_rtsp_grab,
        MOD_NAME,
};

REGISTER_MODULE(rtsp, &vidcap_rtsp_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=4: */
