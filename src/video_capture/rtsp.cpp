/*
 * AUTHOR:   Gerard Castillo <gerard.castillo@i2cat.net>,
 *           Martin German <martin.german@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <glib.h>

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/rtpdec_h264.h"
#include "rtsp/rtsp_utils.h"

#include "video_decompress.h"

#include "pdb.h"
#include "rtp/pbuf.h"

#include "video.h"
#include "video_codec.h"
#include "video_capture.h"

#include <curl/curl.h>
#include <chrono>

#define KEEPALIVE_INTERVAL_S 5
#define MOD_NAME  "[rtsp] "
#define VERSION_STR  "V1.0"

//TODO set lower initial video recv buffer size (to find the minimal?)
#define DEFAULT_VIDEO_FRAME_WIDTH 1920
#define DEFAULT_VIDEO_FRAME_HEIGHT 1080
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((0.1*DEFAULT_VIDEO_FRAME_WIDTH*DEFAULT_VIDEO_FRAME_HEIGHT)*110/100) //command line net.core setup: sysctl -w net.core.rmem_max=9123840


/* error handling macros */
#define my_curl_easy_setopt(A, B, C, action_fail) \
    if ((res = curl_easy_setopt((A), (B), (C))) != CURLE_OK){ \
        fprintf(stderr, "[rtsp error] curl_easy_setopt(%s, %s, %s) failed: %d\n", #A, #B, #C, res); \
        printf("[rtsp error] could not configure rtsp capture properly, \n\t\tplease check your parameters. \nExiting...\n\n"); \
        action_fail; \
    }

#define my_curl_easy_perform(A) \
    if ((res = curl_easy_perform((A))) != CURLE_OK){ \
        fprintf(stderr, "[rtsp error] curl_easy_perform(%s) failed: %d\n", #A, res); \
        printf("[rtsp error] could not configure rtsp capture properly, \n\t\tplease check your parameters. \nExiting...\n\n"); \
        return NULL; \
    }

/* send RTSP GET_PARAMETERS request */
static int
rtsp_get_parameters(CURL *curl, const char *uri);

/* send RTSP OPTIONS request */
static int
rtsp_options(CURL *curl, const char *uri);

/* send RTSP DESCRIBE request and write sdp response to a file */
static bool
rtsp_describe(CURL *curl, const char *uri, FILE *sdp_fp);

/* send RTSP SETUP request */
static int
rtsp_setup(CURL *curl, const char *uri, const char *transport);

/* send RTSP PLAY request */
static int
rtsp_play(CURL *curl, const char *uri, const char *range);

/* send RTSP TEARDOWN request */
static int
rtsp_teardown(CURL *curl, const char *uri);

static int
get_nals(FILE *sdp_file, char *nals, int *width, int *height);

bool setup_codecs_and_controls_from_sdp(FILE *sdp_file, void *state);

static int
init_rtsp(struct rtsp_state *s);

static int
init_decompressor(struct video_rtsp_state *sr);

static void *
vidcap_rtsp_thread(void *args);

static void
show_help(void);

void getNewLine(const char* buffer, int* i, char* line);

void
rtsp_keepalive(void *state);

int decode_frame_by_pt(struct coded_data *cdata, void *decode_data, struct pbuf_stats *);

static void vidcap_rtsp_done(void *state);

static const uint8_t start_sequence[] = { 0, 0, 0, 1 };

/**
 * @struct rtsp_state
 */
struct video_rtsp_state {
    const char *codec;

    struct timeval t0, t;
    int frames;
    struct video_frame *frame;
    struct tile *tile;

    //struct std_frame_received *rx_data;
    bool new_frame;
    bool decompress;
    bool grab;

    struct state_decompress *sd;
    struct video_desc des;
    char * out_frame;

    int port;
    float fps;
    char *control;

    struct rtp *device;
    struct pdb *participants;
    struct pdb_e *cp;
    double rtcp_bw;
    int ttl;
    char *mcast_if;
    struct timeval curr_time;
    struct timeval timeout;
    struct timeval start_time;
    int required_connections;
    uint32_t timestamp;

    pthread_t vrtsp_thread_id; //the worker_id

    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;

    unsigned int h264_offset_len;
    unsigned char *h264_offset_buffer;

    int pt;
};

struct audio_rtsp_state {
    struct audio_frame audio;
    int play_audio_frame;

    const char *codec;

    struct timeval last_audio_time;
    unsigned int grab_audio:1;

    int port;
    float fps;

    char *control;

    struct rtp *device;
    struct pdb *participants;
    struct pdb_e *cp;
    double rtcp_bw;
    int ttl;
    char *mcast_if;
    struct timeval curr_time;
    struct timeval timeout;
    struct timeval prev_time;
    struct timeval start_time;
    int required_connections;
    uint32_t timestamp;

    pthread_t artsp_thread_id; //the worker_id
    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;
};

struct rtsp_state {
    CURL *curl;
    char uri[1024];
    rtps_types_t avType;
    const char *addr;
    char *sdp;

    volatile bool should_exit;
    struct audio_rtsp_state artsp_state;
    struct video_rtsp_state vrtsp_state;

    pthread_t keep_alive_rtsp_thread_id; //the worker_id
    pthread_mutex_t lock;
    pthread_cond_t keepalive_cv;
};

static void
show_help() {
    printf("[rtsp] usage:\n");
    printf("\t-t rtsp:<uri>[:rtp_rx_port=<port>][:decompress][size=<width>x<height>]\n");
    printf("\t\t <uri> - RTSP server URI\n");
    printf("\t\t <port> - receiver port number \n");
    printf(
        "\t\t decompress - decompress the stream (default: disabled)\n\n");
}

static void *
keep_alive_thread(void *arg){
    struct rtsp_state *s = (struct rtsp_state *) arg;

    pthread_mutex_lock(&s->lock);
    while (1) {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        struct timespec timeout = { .tv_sec = tp.tv_sec + KEEPALIVE_INTERVAL_S, .tv_nsec = tp.tv_usec * 1000 };
        pthread_cond_timedwait(&s->keepalive_cv, &s->lock, &timeout);
        if (s->should_exit) {
            pthread_mutex_unlock(&s->vrtsp_state.lock);
            break;
        }
        pthread_mutex_unlock(&s->vrtsp_state.lock);

        // actuall keepalive
        if (rtsp_get_parameters(s->curl, s->uri) == 0) {
            s->should_exit = TRUE;
            exit_uv(1);
        }
    }
    return NULL;
}

int decode_frame_by_pt(struct coded_data *cdata, void *decode_data, struct pbuf_stats *) {
    rtp_packet *pckt = NULL;
    pckt = cdata->data;
    struct decode_data_h264 *d = (struct decode_data_h264 *) decode_data;
    if (pckt->pt == d->video_pt) {
        return decode_frame_h264(cdata,decode_data);
    } else {
        error_msg("Wrong Payload type: %u\n", pckt->pt);
        return FALSE;
    }
}

static void *
vidcap_rtsp_thread(void *arg) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) arg;

    gettimeofday(&s->vrtsp_state.start_time, NULL);

    while (!s->should_exit) {
    	usleep(10);
        auto curr_time_hr = std::chrono::high_resolution_clock::now();
        gettimeofday(&s->vrtsp_state.curr_time, NULL);
        s->vrtsp_state.timestamp = tv_diff(s->vrtsp_state.curr_time, s->vrtsp_state.start_time) * 90000;

        rtp_update(s->vrtsp_state.device, s->vrtsp_state.curr_time);

        s->vrtsp_state.timeout.tv_sec = 0;
        s->vrtsp_state.timeout.tv_usec = 10000;

        if (!rtp_recv_r(s->vrtsp_state.device, &s->vrtsp_state.timeout, s->vrtsp_state.timestamp)) {
            pdb_iter_t it;
            s->vrtsp_state.cp = pdb_iter_init(s->vrtsp_state.participants, &it);

            while (s->vrtsp_state.cp != NULL) {
                if (pthread_mutex_trylock(&s->vrtsp_state.lock) == 0) {
                    {
                        if(s->vrtsp_state.grab){

                            while (s->vrtsp_state.new_frame && !s->should_exit) {
                                s->vrtsp_state.worker_waiting = true;
                                pthread_cond_wait(&s->vrtsp_state.worker_cv, &s->vrtsp_state.lock);
                                s->vrtsp_state.worker_waiting = false;
                            }
                            struct decode_data_h264 d;
                            d.frame = s->vrtsp_state.frame;
                            d.offset_len = s->vrtsp_state.h264_offset_len;
                            d.video_pt = s->vrtsp_state.pt;
                            if (pbuf_decode(s->vrtsp_state.cp->playout_buffer, curr_time_hr,
                                decode_frame_by_pt, &d))
                            {
                                 s->vrtsp_state.new_frame = true;
                                 if (s->vrtsp_state.boss_waiting)
                                     pthread_cond_signal(&s->vrtsp_state.boss_cv);
                            }
                        }
                    }
                    pthread_mutex_unlock(&s->vrtsp_state.lock);
                }
                pbuf_remove(s->vrtsp_state.cp->playout_buffer, curr_time_hr);
                s->vrtsp_state.cp = pdb_iter_next(&it);
            }

            pdb_iter_done(&it);
        }
    }
    return NULL;
}

static struct video_frame *
vidcap_rtsp_grab(void *state, struct audio_frame **audio) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) state;

    *audio = NULL;

    if(pthread_mutex_trylock(&s->vrtsp_state.lock)==0){
        {
            s->vrtsp_state.grab = true;

            while (!s->vrtsp_state.new_frame && !s->should_exit) {
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

            gettimeofday(&s->vrtsp_state.curr_time, NULL);

            if(s->vrtsp_state.h264_offset_len>0 && s->vrtsp_state.frame->frame_type == INTRA){
                    memcpy(s->vrtsp_state.frame->tiles[0].data, s->vrtsp_state.h264_offset_buffer, s->vrtsp_state.h264_offset_len);
            }

            if (s->vrtsp_state.decompress) {
                if(s->vrtsp_state.des.width != s->vrtsp_state.tile->width || s->vrtsp_state.des.height != s->vrtsp_state.tile->height){
                    s->vrtsp_state.des.width = s->vrtsp_state.tile->width;
                    s->vrtsp_state.des.height = s->vrtsp_state.tile->height;
                    decompress_done(s->vrtsp_state.sd);
                    s->vrtsp_state.frame->color_spec = H264;
                    if (init_decompressor(&s->vrtsp_state) == 0) {
                        pthread_mutex_unlock(&s->vrtsp_state.lock);
                        return NULL;
                    }
                    s->vrtsp_state.frame->color_spec = UYVY;
                }

                decompress_frame(s->vrtsp_state.sd, (unsigned char *) s->vrtsp_state.out_frame,
                    (unsigned char *) s->vrtsp_state.frame->tiles[0].data,
                    s->vrtsp_state.tile->data_len, 0, nullptr, nullptr);
                s->vrtsp_state.frame->tiles[0].data = s->vrtsp_state.out_frame;               //TODO memcpy?
                s->vrtsp_state.frame->tiles[0].data_len = vc_get_linesize(s->vrtsp_state.des.width, UYVY)
                            * s->vrtsp_state.des.height;                           //TODO reconfigurable?
            }
            s->vrtsp_state.new_frame = false;

            if (s->vrtsp_state.worker_waiting) {
                pthread_cond_signal(&s->vrtsp_state.worker_cv);
            }
        }

        pthread_mutex_unlock(&s->vrtsp_state.lock);

        gettimeofday(&s->vrtsp_state.t, NULL);
        double seconds = tv_diff(s->vrtsp_state.t, s->vrtsp_state.t0);
        if (seconds >= 5) {
            float fps = s->vrtsp_state.frames / seconds;
            fprintf(stderr, "[rtsp capture] %d frames in %g seconds = %g FPS\n",
                s->vrtsp_state.frames, seconds, fps);
            s->vrtsp_state.t0 = s->vrtsp_state.t;
            s->vrtsp_state.frames = 0;
            //TODO: Threshold of ¿1fps? in order to update fps parameter. Now a higher fps is fixed to 30fps...
            //if (fps > s->fps + 1 || fps < s->fps - 1) {
            //      debug_msg(
            //          "\n[rtsp] updating fps from rtsp server stream... now = %f , before = %f\n",fps,s->fps);
            //      s->frame->fps = fps;
            //      s->fps = fps;
            //  }
        }
        s->vrtsp_state.frames++;
        s->vrtsp_state.grab = false;
    } else {
        return NULL;
    }

    return s->vrtsp_state.frame;
}

#define INIT_FAIL(msg) log_msg(LOG_LEVEL_ERROR, MOD_NAME msg); \
                    free(tmp); \
                    vidcap_rtsp_done(s); \
                    show_help(); \
                    return VIDCAP_INIT_FAIL

static int
vidcap_rtsp_init(struct vidcap_params *params, void **state) {

    log_msg(LOG_LEVEL_WARNING, "RTSP capture module is most likely broken, "
            "please contact " PACKAGE_BUGREPORT " if you wish to use it.\n");

    if (vidcap_params_get_fmt(params)
        && strcmp(vidcap_params_get_fmt(params), "help") == 0)
    {
        show_help();
        return VIDCAP_INIT_NOERR;
    }

    struct rtsp_state *s = (struct rtsp_state *) calloc(1, sizeof(struct rtsp_state));
    if (s == NULL) {
        return VIDCAP_INIT_FAIL;
    }

    //TODO now static codec assignment, to be dynamic as a function of supported codecs
    s->vrtsp_state.codec = "";
    s->artsp_state.codec = "";
    s->artsp_state.control = strdup("");
    s->artsp_state.control = strdup("");

    int len = -1;
    char *save_ptr = NULL;
    s->avType = none;  //-1 none, 0 a&v, 1 v, 2 a

    gettimeofday(&s->vrtsp_state.t0, NULL);
    s->vrtsp_state.frames = 0;
    s->vrtsp_state.grab = FALSE;

    s->addr = "127.0.0.1";
    s->vrtsp_state.device = NULL;
    s->vrtsp_state.rtcp_bw = 5 * 1024 * 1024; /* FIXME */
    s->vrtsp_state.ttl = 255;

    s->vrtsp_state.mcast_if = NULL;
    s->vrtsp_state.required_connections = 1;

    s->vrtsp_state.timeout.tv_sec = 0;
    s->vrtsp_state.timeout.tv_usec = 10000;

    s->vrtsp_state.participants = pdb_init(0);

    s->vrtsp_state.new_frame = FALSE;

    s->vrtsp_state.frame = vf_alloc(1);
    s->vrtsp_state.h264_offset_buffer = (unsigned char *) malloc(2048);
    s->vrtsp_state.h264_offset_len = 0;

    s->curl = NULL;
    char *fmt = NULL;

    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->keepalive_cv, NULL);
    pthread_mutex_init(&s->vrtsp_state.lock, NULL);
    pthread_cond_init(&s->vrtsp_state.boss_cv, NULL);
    pthread_cond_init(&s->vrtsp_state.worker_cv, NULL);

    char *tmp, *item;
    fmt = strdup(vidcap_params_get_fmt(params));
    tmp = fmt;
    strcpy(s->uri, "rtsp://");

    s->vrtsp_state.tile = vf_get_tile(s->vrtsp_state.frame, 0);
    s->vrtsp_state.tile->width = DEFAULT_VIDEO_FRAME_WIDTH/2;
    s->vrtsp_state.tile->height = DEFAULT_VIDEO_FRAME_HEIGHT/2;

    bool in_uri = true;
    while ((item = strtok_r(fmt, ":", &save_ptr))) {
        fmt = NULL;
        bool option_given = true;
        if (strstr(item, "rtp_rx_port=") == item) {
            s->vrtsp_state.port = atoi(strchr(item, '=') + 1);
        } else if (strcmp(item, "decompress") == 0) {
            s->vrtsp_state.decompress = TRUE;
        } else if (strstr(item, "size=")) {
            assert(strchr(item, 'x') != NULL);
            item = strchr(item, '=') + 1;
            s->vrtsp_state.tile->width = atoi(item);
            s->vrtsp_state.tile->height = atoi(strchr(item, 'x') + 1);
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
                INIT_FAIL("Unknown option\n");
            }
        }
        if (option_given) {
            in_uri = false;
        }
    }
    free(tmp);
    tmp = NULL;

    //re-check parameters
    if (strcmp(s->uri, "rtsp://") == 0) {
        INIT_FAIL("No URI given!\n");
    }

    s->vrtsp_state.device = rtp_init_if("localhost", s->vrtsp_state.mcast_if, s->vrtsp_state.port, 0, s->vrtsp_state.ttl, s->vrtsp_state.rtcp_bw,
        0, rtp_recv_callback, (uint8_t *) s->vrtsp_state.participants, 0, false);
    if (s->vrtsp_state.device == NULL) {
        log_msg(LOG_LEVEL_ERROR, "[rtsp] Cannot intialize RTP device!\n");
        vidcap_rtsp_done(s);
        return VIDCAP_INIT_FAIL;
    }
    if (!rtp_set_option(s->vrtsp_state.device, RTP_OPT_WEAK_VALIDATION, 1)) {
        debug_msg("[rtsp] RTP INIT - set option\n");
        return VIDCAP_INIT_FAIL;
    }
    if (!rtp_set_sdes(s->vrtsp_state.device, rtp_my_ssrc(s->vrtsp_state.device),
                RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING))) {
        debug_msg("[rtsp] RTP INIT - set sdes\n");
        return VIDCAP_INIT_FAIL;
    }

    int ret = rtp_set_recv_buf(s->vrtsp_state.device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
    if (!ret) {
        debug_msg("[rtsp] RTP INIT - set recv buf \nset command: sudo sysctl -w net.core.rmem_max=9123840\n");
        return VIDCAP_INIT_FAIL;
    }

    if (!rtp_set_send_buf(s->vrtsp_state.device, 1024 * 56)) {
        debug_msg("[rtsp] RTP INIT - set send buf\n");
        return VIDCAP_INIT_FAIL;
    }
    ret=pdb_add(s->vrtsp_state.participants, rtp_my_ssrc(s->vrtsp_state.device));

    debug_msg("[rtsp] rtp receiver init done\n");

    if (s->vrtsp_state.port == 0) {
        s->vrtsp_state.port = rtp_get_udp_rx_port(s->vrtsp_state.device);
        assert(s->vrtsp_state.port != 0);
    }

    debug_msg("[rtsp] selected flags:\n");
    debug_msg("\t  uri: %s\n",s->uri);
    debug_msg("\t  port: %d\n", s->vrtsp_state.port);
    debug_msg("\t  decompress: %d\n\n",s->vrtsp_state.decompress);

    len = init_rtsp(s);

    if(len < 0){
        vidcap_rtsp_done(s);
        return VIDCAP_INIT_FAIL;
    }else{
        s->vrtsp_state.h264_offset_len = len;
    }

    s->vrtsp_state.tile->data = (char *) malloc(4 * s->vrtsp_state.tile->width * s->vrtsp_state.tile->height);
    s->vrtsp_state.tile->data_len = 0;

    s->vrtsp_state.frame->frame_type = BFRAME;

    //TODO fps should be autodetected, now reset and controlled at vidcap_grab function
    s->vrtsp_state.frame->fps = 30;
    s->vrtsp_state.fps = 30;
    s->vrtsp_state.frame->interlacing = PROGRESSIVE;

    s->vrtsp_state.frame->tiles[0].data = (char *) calloc(1, s->vrtsp_state.tile->width * s->vrtsp_state.tile->height);

    s->should_exit = FALSE;

    s->vrtsp_state.boss_waiting = false;
    s->vrtsp_state.worker_waiting = false;

    if (s->vrtsp_state.decompress) {
        s->vrtsp_state.frame->color_spec = H264;
        if (init_decompressor(&s->vrtsp_state) == 0) {
            vidcap_rtsp_done(s);
            return VIDCAP_INIT_FAIL;
        }
        s->vrtsp_state.frame->color_spec = UYVY;
    }

    pthread_create(&s->vrtsp_state.vrtsp_thread_id, NULL, vidcap_rtsp_thread, s);
    pthread_create(&s->keep_alive_rtsp_thread_id, NULL, keep_alive_thread, s);

    debug_msg("[rtsp] rtsp capture init done\n");

    *state = s;
    return VIDCAP_INIT_OK;
}

static CURL *init_curl() {
    CURL *curl;
    /* initialize curl */
    CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
    if (res != CURLE_OK) {
        fprintf(stderr, "[rtsp] curl_global_init(%s) failed: %d\n",
            "CURL_GLOBAL_ALL", res);
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
    return curl;
}

/**
 * Initializes rtsp state and internal parameters
 */
static int
init_rtsp(struct rtsp_state *s) {
    /* initialize curl */
    s->curl = init_curl();

    if (!s->curl) {
        return -1;
    }

    const char *range = "0.000-";
    int len_nals = -1;
    debug_msg("\n[rtsp] request %s\n", VERSION_STR);
    debug_msg("    Project web site: http://code.google.com/p/rtsprequest/\n");
    debug_msg("    Requires cURL V7.20 or greater\n\n");
    char Atransport[256];
    char Vtransport[256];
    memset(Atransport, 0, 256);
    memset(Vtransport, 0, 256);
    int port = s->vrtsp_state.port;
    CURLcode res;
    FILE *sdp_file = tmpfile();
    if (sdp_file == NULL) {
        sdp_file = fopen("rtsp.sdp", "w+");
        if (sdp_file == NULL) {
            perror("Creating SDP file");
            goto error;
        }
    }

    sprintf(Vtransport, "RTP/AVP;unicast;client_port=%d-%d", port, port + 1);

    //THIS AUDIO PORTS ARE AS DEFAULT UG AUDIO PORTS BUT AREN'T RELATED...
    sprintf(Atransport, "RTP/AVP;unicast;client_port=%d-%d", port+2, port + 3);

    my_curl_easy_setopt(s->curl, CURLOPT_NOSIGNAL, 1, goto error); //This tells curl not to use any functions that install signal handlers or cause signals to be sent to your process.
    //my_curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, 1);
    my_curl_easy_setopt(s->curl, CURLOPT_VERBOSE, 0L, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_NOPROGRESS, 1L, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_WRITEHEADER, stdout, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_URL, s->uri, goto error);

    //TODO TO CHECK CONFIGURING ERRORS
    //CURLOPT_ERRORBUFFER
    //http://curl.haxx.se/libcurl/c/curl_easy_perform.html

    /* request server options */
    if(rtsp_options(s->curl, s->uri)==0){
        goto error;
    }

    /* request session description and write response to sdp file */
    if (!rtsp_describe(s->curl, s->uri, sdp_file)) {
        goto error;
    }

    if (log_level >= LOG_LEVEL_VERBOSE) {
        fprintf(stderr, "SDP:\n");
        while (!feof(sdp_file)) {
            putc(getc(sdp_file), stderr);
        }
        rewind(sdp_file);
        fprintf(stderr, "\n\n");
    }

    if (!setup_codecs_and_controls_from_sdp(sdp_file, s)) {
        goto error;
    }
    if (strcmp(s->vrtsp_state.codec, "H264") == 0){
        s->vrtsp_state.frame->color_spec = H264;
        char uri[strlen(s->uri) + 1 + strlen(s->vrtsp_state.control) + 1];
        strcpy(uri, s->uri);
        strcat(uri, "/");
        strcat(uri, s->vrtsp_state.control);
        debug_msg("\n V URI = %s\n", uri);
        if (rtsp_setup(s->curl, uri, Vtransport) == 0) {
            goto error;
        }
    }
    if (strcmp(s->artsp_state.codec, "PCMU") == 0){
        char uri[strlen(s->uri) + 1 + strlen(s->artsp_state.control) + 1];
        strcpy(uri, s->uri);
        strcat(uri, "/");
        strcat(uri, s->artsp_state.control);
        debug_msg("\n A URI = %s\n", uri);
        if (rtsp_setup(s->curl, uri, Atransport) == 0) {
            goto error;
        }
    }
    if (strlen(s->artsp_state.codec) == 0 && strlen(s->vrtsp_state.codec) == 0){
        goto error;
    }
    else{
        if(rtsp_play(s->curl, s->uri, range)==0){
            goto error;
        }
    }

    /* get start nal size attribute from sdp file */
    len_nals = get_nals(sdp_file, (char *) s->vrtsp_state.h264_offset_buffer, (int *) &s->vrtsp_state.tile->width, (int *) &s->vrtsp_state.tile->height);

    debug_msg("[rtsp] playing video from server (size: WxH = %d x %d)...\n",s->vrtsp_state.tile->width,s->vrtsp_state.tile->height);

    fclose(sdp_file);
    return len_nals;

error:
    fclose(sdp_file);
    return -1;
}

#define LEN 10

bool setup_codecs_and_controls_from_sdp(FILE *sdp_file, void *state) {
    struct rtsp_state *rtspState;
    rtspState = (struct rtsp_state *) state;

    int n=0;
    char *line = (char*) malloc(1024);
    char* tmpBuff;
    int countT = 0;
    int countC = 0;
    char codecs[2][LEN] = { 0 };
    char tracks[2][LEN] = { 0 };

    fseek(sdp_file, 0, SEEK_END);
    long fileSize = ftell(sdp_file);
    if (fileSize < 0) {
            perror("RTSP ftell");
            free(line);
            return false;
    }
    rewind(sdp_file);

    char* buffer = (char*) malloc(fileSize+1);
    unsigned long readResult = fread(buffer, sizeof(char), fileSize, sdp_file);
    if (ferror(sdp_file)){
        perror(MOD_NAME "SDP file read failed");
        free(line);
        free(buffer);
        return false;
    }
    buffer[readResult] = '\0';

    while (buffer[n] != '\0'){
        getNewLine(buffer,&n,line);
        sscanf(line, " a = control: %*s");
        tmpBuff = strstr(line, "track");
        if(tmpBuff!=NULL){
            if ((unsigned) countT < sizeof tracks / sizeof tracks[0]) {
                //debug_msg("track = %s\n",tmpBuff);
                strncpy(tracks[countT],tmpBuff,MIN(strlen(tmpBuff)-2, sizeof tracks[countT] - 1));
                tracks[countT][MIN(strlen(tmpBuff)-2, sizeof tracks[countT] - 1)] = '\0';
                countT++;
            } else {
                log_msg(LOG_LEVEL_WARNING, "skipping track = %s\n",tmpBuff);
            }
        }
        tmpBuff=NULL;
        int pt = 0;
        sscanf(line, " a=rtpmap:%d %*s", &pt);
        tmpBuff = strstr(line, "H264");
        if(tmpBuff!=NULL){
            if ((unsigned) countC < sizeof codecs / sizeof codecs[0]) {
                //debug_msg("codec = %s\n",tmpBuff);
                strncpy(codecs[countC],tmpBuff,4);
                codecs[countC][4] = '\0';
                countC++;
                if (pt == 0) {
                    log_msg(LOG_LEVEL_ERROR, MOD_NAME "Missing video PT for H.264!\n");
                    return false;
                }
                rtspState->vrtsp_state.pt = pt;
            } else {
                log_msg(LOG_LEVEL_WARNING, "skipping codec = %s\n",tmpBuff);
            }
        }
        tmpBuff=NULL;
        sscanf(line, " a=rtpmap:97 %*s");
        tmpBuff = strstr(line, "PCMU");
        if(tmpBuff!=NULL){
            if ((unsigned) countC < sizeof codecs / sizeof codecs[0]) {
                //debug_msg("codec = %s\n",tmpBuff);
                strncpy(codecs[countC],tmpBuff,4);
                codecs[countC][4] = '\0';
                countC++;
            } else {
                log_msg(LOG_LEVEL_WARNING, "skipping codec = %s\n",tmpBuff);
            }
        }
        tmpBuff=NULL;

        if(countT > 1 && countC > 1) break;
    }
    debug_msg("\nTRACK = %s FOR CODEC = %s",tracks[0],codecs[0]);
    debug_msg("\nTRACK = %s FOR CODEC = %s\n",tracks[1],codecs[1]);

    for(int p=0;p<2;p++){
        if(strncmp(codecs[p],"H264",4)==0){
            rtspState->vrtsp_state.codec = "H264";
            rtspState->vrtsp_state.control = strdup(tracks[p]);

        }if(strncmp(codecs[p],"PCMU",4)==0){
            rtspState->artsp_state.codec = "PCMU";
            rtspState->artsp_state.control = strdup(tracks[p]);
        }
    }
    free(line);
    free(buffer);
    rewind(sdp_file);
    return true;
}

void getNewLine(const char* buffer, int* i, char* line){
    int j=0;
    while(buffer[*i] != '\n' && buffer[*i] != '\0'){
        j++;
        (*i)++;
    }
    if(buffer[*i] == '\n'){
        j++;
        (*i)++;
    }

    if(j>0){
        memcpy(line,buffer+(*i)-j,j);
        if (line[j - 1] == '\r') {
            line[j - 1] = '\0';
        }
    }
    line[j] = '\0';
}
/**
 * Initializes decompressor if required by decompress flag
 */
static int
init_decompressor(struct video_rtsp_state *sr) {
    if (decompress_init_multi(H264, VIDEO_CODEC_NONE, UYVY, &sr->sd, 1)) {
        sr->des.width = sr->tile->width;
        sr->des.height = sr->tile->height;
        sr->des.color_spec = sr->frame->color_spec;
        sr->des.tile_count = 0;
        sr->des.interlacing = PROGRESSIVE;

        decompress_reconfigure(sr->sd, sr->des, 16, 8, 0,
            vc_get_linesize(sr->des.width, UYVY), UYVY);
    } else
        return 0;
    sr->out_frame = (char *) malloc(sr->tile->width * sr->tile->height * 4);
    return 1;
}

/**
 * send RTSP GET PARAMS request
 */
static int
rtsp_get_parameters(CURL *curl, const char *uri) {
    CURLcode res = CURLE_OK;
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return -1);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_GET_PARAMETER, return -1);
    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP GET PARAMETERS] curl_easy_perform failed\n");
        error_msg("[RTSP GET PARAMETERS] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        return 0;
    }else{
        return 1;
    }
}

/**
 * send RTSP OPTIONS request
 */
static int
rtsp_options(CURL *curl, const char *uri) {
    char control[1500] = "",
         user[1500] = "",
         pass[1500] = "",
         *strtoken;

    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] OPTIONS %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return -1);

    sscanf(uri, "rtsp://%1500s", control);
    strtoken = strtok(control, ":");
    assert(strtoken != NULL);
    strncpy(user, strtoken, sizeof user - 1);
    strtoken = strtok(NULL, "@");
    if (strtoken != NULL) {
        strncpy(pass, strtoken, sizeof pass - 1);
        my_curl_easy_setopt(curl, CURLOPT_USERNAME, user, return -1);
        my_curl_easy_setopt(curl, CURLOPT_PASSWORD, pass, return -1);
    }

    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long )CURL_RTSPREQ_OPTIONS, return -1);

    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP OPTIONS] curl_easy_perform failed\n");
        error_msg("[RTSP OPTIONS] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        return 0;
    }else{
        return 1;
    }
}

/**
 * send RTSP DESCRIBE request and write sdp response to a file
 */
static bool
rtsp_describe(CURL *curl, const char *uri, FILE *sdp_fp) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] DESCRIBE %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, sdp_fp, goto error);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_DESCRIBE, goto error);

    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP DESCRIBE] curl_easy_perform failed\n");
        error_msg("[RTSP DESCRIBE] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        goto error;
    }

    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, stdout, goto error);
    rewind(sdp_fp);
    return true;
error:
    return false;
}

/**
 * send RTSP SETUP request
 */
static int
rtsp_setup(CURL *curl, const char *uri, const char *transport) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] SETUP %s\n", uri);
    debug_msg("\t TRANSPORT %s\n", transport);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return -1);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_TRANSPORT, transport, return -1);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long )CURL_RTSPREQ_SETUP, return -1);

    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP SETUP] curl_easy_perform failed\n");
        error_msg("[RTSP SETUP] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        return 0;
    }else{
        return 1;
    }
}

/**
 * send RTSP PLAY request
 */
static int
rtsp_play(CURL *curl, const char *uri, const char * /* range */) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] PLAY %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri, return -1);
    //my_curl_easy_setopt(curl, CURLOPT_RANGE, range);      //range not set because we want (right now) no limit range for streaming duration
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long )CURL_RTSPREQ_PLAY, return -1);

    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP PLAY] curl_easy_perform failed\n");
        error_msg("[RTSP PLAY] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        return 0;
    }else{
        return 1;
    }
}

/**
 * send RTSP TEARDOWN request
 */
static int
rtsp_teardown(CURL *curl, const char *uri) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] TEARDOWN %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_TEARDOWN, return -1);

    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP TEARD DOWN] curl_easy_perform failed\n");
        error_msg("[RTSP TEARD DOWN] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        return 0;
    }else{
        return 1;
    }
}

static struct vidcap_type *
vidcap_rtsp_probe(bool verbose, void (**deleter)(void *)) {
    UNUSED(verbose);
    *deleter = free;
    struct vidcap_type *vt;

    vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
    if (vt != NULL) {
        vt->name = "rtsp";
        vt->description = "Video capture from RTSP remote server";
    }
    return vt;
}

static void
vidcap_rtsp_done(void *state) {
    struct rtsp_state *s = (struct rtsp_state *) state;

    pthread_mutex_lock(&s->lock);
    s->should_exit = TRUE;
    pthread_cond_signal(&s->keepalive_cv);
    pthread_mutex_unlock(&s->lock);

    pthread_join(s->vrtsp_state.vrtsp_thread_id, NULL);
    pthread_join(s->keep_alive_rtsp_thread_id, NULL);

    if(s->vrtsp_state.sd)
        decompress_done(s->vrtsp_state.sd);

    if (s->vrtsp_state.device != nullptr) {
        rtp_done(s->vrtsp_state.device);
    }

    free(s->vrtsp_state.tile->data);
    if(s->vrtsp_state.h264_offset_buffer!=NULL) free(s->vrtsp_state.h264_offset_buffer);
    if(s->vrtsp_state.frame!=NULL) free(s->vrtsp_state.frame);
    free(s->vrtsp_state.control);
    free(s->artsp_state.control);

    rtsp_teardown(s->curl, s->uri);

    curl_easy_cleanup(s->curl);
    curl_global_cleanup();
    s->curl = NULL;

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
get_nals(FILE *sdp_file, char *nals, int *width, int *height) {

    uint8_t nalInfo;
    uint8_t type;
    uint8_t nri __attribute__((unused));
    int max_len = 1500, len_nals = 0;
    char *s = (char *) malloc(max_len);
    char *sprop;
    memset(s, 0, max_len);
    nals[0] = '\0';

    while (fgets(s, max_len - 2, sdp_file) != NULL) {
        sprop = strstr(s, "sprop-parameter-sets=");
        if (sprop != NULL) {
            gsize length;   //gsize is an unsigned int.
            char *nal_aux, *nal;
            memcpy(nals, start_sequence, sizeof(start_sequence));
            len_nals = sizeof(start_sequence);
            nal_aux = strstr(sprop, "=");
            nal_aux++;
            nal = strtok(nal_aux, ",;");
            if (nal == nullptr) {
                continue;
            }
            //convert base64 to hex
            guchar *nals_aux = g_base64_decode(nal, &length);
            memcpy(nals + len_nals, nals_aux, length);
            g_free(nals_aux);
            len_nals += length;

            nalInfo = (uint8_t) nals[4];
            type = nalInfo & 0x1f;
            nri = nalInfo & 0x60;

            if (type == 7){
                width_height_from_SDP(width, height , (unsigned char *) (nals+4), length);
            }

            while ((nal = strtok(NULL, ",;")) != NULL) {
                guchar *nals_aux = g_base64_decode(nal, &length);
                if (length) {
                    //convert base64 to hex
                    memcpy(nals+len_nals, start_sequence, sizeof(start_sequence));
                    len_nals += sizeof(start_sequence);
                    memcpy(nals + len_nals, nals_aux, length);
                    len_nals += length;

                    nalInfo = (uint8_t) nals[len_nals - length];
                    type = nalInfo & 0x1f;
                    nri = nalInfo & 0x60;

                    if (type == 7){
                        width_height_from_SDP(width, height , (unsigned char *) (nals+(len_nals - length)), length);
                    }
                    //assure start sequence injection between sps, pps and other nals
                    memcpy(nals+len_nals, start_sequence, sizeof(start_sequence));
                    len_nals += sizeof(start_sequence);
                }
                g_free(nals_aux);
            }
        }
    }

    free(s);
    rewind(sdp_file);
    return len_nals;
}

static const struct video_capture_info vidcap_rtsp_info = {
        vidcap_rtsp_probe,
        vidcap_rtsp_init,
        vidcap_rtsp_done,
        vidcap_rtsp_grab,
        false
};

REGISTER_MODULE(rtsp, &vidcap_rtsp_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=4: */
