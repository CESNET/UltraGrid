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
rtsp_describe(CURL *curl, const char *uri, const char *sdp_filename);

/* send RTSP SETUP request */
static int
rtsp_setup(CURL *curl, const char *uri, const char *transport);

/* send RTSP PLAY request */
static int
rtsp_play(CURL *curl, const char *uri, const char *range);

/* send RTSP TEARDOWN request */
static int
rtsp_teardown(CURL *curl, const char *uri);

/* convert url into an sdp filename */
static char *
get_sdp_filename(const char *url);

static int
get_nals(const char *sdp_filename, char *nals, int *width, int *height);

bool setup_codecs_and_controls_from_sdp(const char *sdp_filename, void *state);

static int
init_rtsp(char* rtsp_uri, int rtsp_port, void *state, char* nals);

static int
init_decompressor(void *state);

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
    const char *control;

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

    pthread_t vrtsp_thread_id; //the worker_id

    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;

    unsigned int h264_offset_len;
    unsigned char *h264_offset_buffer;
};

struct audio_rtsp_state {
    struct audio_frame audio;
    int play_audio_frame;

    const char *codec;

    struct timeval last_audio_time;
    unsigned int grab_audio:1;

    int port;
    float fps;

    const char *control;

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
    char *uri;
    rtps_types_t avType;
    const char *addr;
    char *sdp;

    volatile bool should_exit;
    struct audio_rtsp_state *artsp_state;
    struct video_rtsp_state *vrtsp_state;

    pthread_t keep_alive_rtsp_thread_id; //the worker_id
    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;
};

static void
show_help() {
    printf("[rtsp] usage:\n");
    printf("\t-t rtsp:<uri>:<port>[:<decompress>[:<width>:<height>]]\n");
    printf("\t\t <uri> RTSP server URI\n");
    printf("\t\t <port> receiver port number \n");
    printf(
        "\t\t <decompress> receiver decompress boolean [true|false] - default: false - no decompression active\n\n");
}

static void
rtsp_keepalive_video(void *state) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) state;
    struct timeval now;
    gettimeofday(&now, NULL);
    if (tv_diff(now, s->vrtsp_state->prev_time) >= 20) {
        if(rtsp_get_parameters(s->curl, s->uri)==0){
            s->should_exit = TRUE;
            exit_uv(1);
        }
        gettimeofday(&s->vrtsp_state->prev_time, NULL);
    }
}

static void *
keep_alive_thread(void *arg){
    struct rtsp_state *s;
    s = (struct rtsp_state *) arg;
    while (!s->should_exit) {
        rtsp_keepalive_video(s);
    }
    return NULL;
}

int decode_frame_by_pt(struct coded_data *cdata, void *decode_data, struct pbuf_stats *) {
    rtp_packet *pckt = NULL;
    pckt = cdata->data;

    switch(pckt->pt){
        case PT_H264:
            return decode_frame_h264(cdata,decode_data);
        default:
            error_msg("Wrong Payload type: %u\n", pckt->pt);
            return FALSE;
    }
}

static void *
vidcap_rtsp_thread(void *arg) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) arg;

    gettimeofday(&s->vrtsp_state->start_time, NULL);
    gettimeofday(&s->vrtsp_state->prev_time, NULL);

    while (!s->should_exit) {
    	usleep(10);
        auto curr_time_hr = std::chrono::high_resolution_clock::now();
        gettimeofday(&s->vrtsp_state->curr_time, NULL);
        s->vrtsp_state->timestamp = tv_diff(s->vrtsp_state->curr_time, s->vrtsp_state->start_time) * 90000;

        rtp_update(s->vrtsp_state->device, s->vrtsp_state->curr_time);

        s->vrtsp_state->timeout.tv_sec = 0;
        s->vrtsp_state->timeout.tv_usec = 10000;

        if (!rtp_recv_r(s->vrtsp_state->device, &s->vrtsp_state->timeout, s->vrtsp_state->timestamp)) {
            pdb_iter_t it;
            s->vrtsp_state->cp = pdb_iter_init(s->vrtsp_state->participants, &it);

            while (s->vrtsp_state->cp != NULL) {
                if (pthread_mutex_trylock(&s->vrtsp_state->lock) == 0) {
                    {
                        if(s->vrtsp_state->grab){

                            while (s->vrtsp_state->new_frame && !s->should_exit) {
                                s->vrtsp_state->worker_waiting = true;
                                pthread_cond_wait(&s->vrtsp_state->worker_cv, &s->vrtsp_state->lock);
                                s->vrtsp_state->worker_waiting = false;
                            }
                            struct decode_data_h264 d;
                            d.frame = s->vrtsp_state->frame;
                            d.offset_len = s->vrtsp_state->h264_offset_len;
                            if (pbuf_decode(s->vrtsp_state->cp->playout_buffer, curr_time_hr,
                                decode_frame_by_pt, &d))
                            {
                                 s->vrtsp_state->new_frame = true;
                            }
                            if (s->vrtsp_state->boss_waiting)
                                pthread_cond_signal(&s->vrtsp_state->boss_cv);
                        }
                    }
                    pthread_mutex_unlock(&s->vrtsp_state->lock);
                }
                pbuf_remove(s->vrtsp_state->cp->playout_buffer, curr_time_hr);
                s->vrtsp_state->cp = pdb_iter_next(&it);
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

    if(pthread_mutex_trylock(&s->vrtsp_state->lock)==0){
        {
            s->vrtsp_state->grab = true;

            while (!s->vrtsp_state->new_frame && !s->should_exit) {
                s->vrtsp_state->boss_waiting = true;
                pthread_cond_wait(&s->vrtsp_state->boss_cv, &s->vrtsp_state->lock);
                s->vrtsp_state->boss_waiting = false;
            }

            if (s->should_exit) {
                return NULL;
            }

            gettimeofday(&s->vrtsp_state->curr_time, NULL);

            if(s->vrtsp_state->h264_offset_len>0 && s->vrtsp_state->frame->frame_type == INTRA){
                    memcpy(s->vrtsp_state->frame->tiles[0].data, s->vrtsp_state->h264_offset_buffer, s->vrtsp_state->h264_offset_len);
            }

            if (s->vrtsp_state->decompress) {
                if(s->vrtsp_state->des.width != s->vrtsp_state->tile->width || s->vrtsp_state->des.height != s->vrtsp_state->tile->height){
                    s->vrtsp_state->des.width = s->vrtsp_state->tile->width;
                    s->vrtsp_state->des.height = s->vrtsp_state->tile->height;
                    decompress_done(s->vrtsp_state->sd);
                    s->vrtsp_state->frame->color_spec = H264;
                    if (init_decompressor(s->vrtsp_state) == 0) {
                        pthread_mutex_unlock(&s->vrtsp_state->lock);
                        return NULL;
                    }
                    s->vrtsp_state->frame->color_spec = UYVY;
                }

                decompress_frame(s->vrtsp_state->sd, (unsigned char *) s->vrtsp_state->out_frame,
                    (unsigned char *) s->vrtsp_state->frame->tiles[0].data,
                    s->vrtsp_state->tile->data_len, 0, nullptr, nullptr);
                s->vrtsp_state->frame->tiles[0].data = s->vrtsp_state->out_frame;               //TODO memcpy?
                s->vrtsp_state->frame->tiles[0].data_len = vc_get_linesize(s->vrtsp_state->des.width, UYVY)
                            * s->vrtsp_state->des.height;                           //TODO reconfigurable?
            }
            s->vrtsp_state->new_frame = false;

            if (s->vrtsp_state->worker_waiting) {
                pthread_cond_signal(&s->vrtsp_state->worker_cv);
            }
        }

        pthread_mutex_unlock(&s->vrtsp_state->lock);

        gettimeofday(&s->vrtsp_state->t, NULL);
        double seconds = tv_diff(s->vrtsp_state->t, s->vrtsp_state->t0);
        if (seconds >= 5) {
            float fps = s->vrtsp_state->frames / seconds;
            fprintf(stderr, "[rtsp capture] %d frames in %g seconds = %g FPS\n",
                s->vrtsp_state->frames, seconds, fps);
            s->vrtsp_state->t0 = s->vrtsp_state->t;
            s->vrtsp_state->frames = 0;
            //TODO: Threshold of ¿1fps? in order to update fps parameter. Now a higher fps is fixed to 30fps...
            //if (fps > s->fps + 1 || fps < s->fps - 1) {
            //      debug_msg(
            //          "\n[rtsp] updating fps from rtsp server stream... now = %f , before = %f\n",fps,s->fps);
            //      s->frame->fps = fps;
            //      s->fps = fps;
            //  }
        }
        s->vrtsp_state->frames++;
        s->vrtsp_state->grab = false;
    }

    return s->vrtsp_state->frame;
}

#define INIT_FAIL(msg) log_msg(LOG_LEVEL_ERROR, msg); \
                    free(tmp); \
                    vidcap_rtsp_done(s); \
                    show_help(); \
                    return VIDCAP_INIT_FAIL

static int
vidcap_rtsp_init(struct vidcap_params *params, void **state) {

    log_msg(LOG_LEVEL_WARNING, "RTSP capture module is most likely broken, "
            "please contact " PACKAGE_BUGREPORT " if you wish to use it.\n");

    struct rtsp_state *s;

    s = (struct rtsp_state *) calloc(1, sizeof(struct rtsp_state));
    if (!s)
        return VIDCAP_INIT_FAIL;

    s->artsp_state = (struct audio_rtsp_state *) calloc(1,sizeof(struct audio_rtsp_state));
    s->vrtsp_state = (struct video_rtsp_state *) calloc(1,sizeof(struct video_rtsp_state));

    //TODO now static codec assignment, to be dynamic as a function of supported codecs
    s->vrtsp_state->codec = "";
    s->artsp_state->codec = "";
    s->artsp_state->control = "";
    s->artsp_state->control = "";

    int len = -1;
    char *save_ptr = NULL;
    s->avType = none;  //-1 none, 0 a&v, 1 v, 2 a

    gettimeofday(&s->vrtsp_state->t0, NULL);
    s->vrtsp_state->frames = 0;
    s->vrtsp_state->grab = FALSE;

    s->addr = "127.0.0.1";
    s->vrtsp_state->device = NULL;
    s->vrtsp_state->rtcp_bw = 5 * 1024 * 1024; /* FIXME */
    s->vrtsp_state->ttl = 255;

    s->vrtsp_state->mcast_if = NULL;
    s->vrtsp_state->required_connections = 1;

    s->vrtsp_state->timeout.tv_sec = 0;
    s->vrtsp_state->timeout.tv_usec = 10000;

    s->vrtsp_state->participants = pdb_init(0);

    s->vrtsp_state->new_frame = FALSE;

    s->vrtsp_state->frame = vf_alloc(1);
    s->vrtsp_state->h264_offset_buffer = (unsigned char *) malloc(2048);
    s->vrtsp_state->h264_offset_len = 0;

    s->uri = NULL;
    s->curl = NULL;
    char *fmt = NULL;

    if (vidcap_params_get_fmt(params)
        && strcmp(vidcap_params_get_fmt(params), "help") == 0)
    {
        show_help();
        free(s);
        return VIDCAP_INIT_NOERR;
    }

    char *tmp, *item;
    fmt = strdup(vidcap_params_get_fmt(params));
    tmp = fmt;
    int i = 0;
    const size_t uri_len = 1024;
    s->uri = (char *) malloc(uri_len);
    strcpy(s->uri, "rtsp://");

    s->vrtsp_state->tile = vf_get_tile(s->vrtsp_state->frame, 0);
    s->vrtsp_state->tile->width = DEFAULT_VIDEO_FRAME_WIDTH/2;
    s->vrtsp_state->tile->height = DEFAULT_VIDEO_FRAME_HEIGHT/2;

    while ((item = strtok_r(fmt, ":", &save_ptr))) {
        switch (i) {
            case 0:
                strncat(s->uri, item, uri_len - strlen(s->uri) - 1);
                item = strtok_r(NULL, ":", &save_ptr);
                if (item == NULL) {
                    INIT_FAIL("[rtsp] Missing port number!\n");
                }
                strncat(s->uri, ":", uri_len - strlen(s->uri) - 1);
                strncat(s->uri, item, uri_len - strlen(s->uri) - 1);
                break;
            case 1:
                s->vrtsp_state->port = atoi(item);
                break;
            case 2:
                if (strcmp(item, "true") == 0) {
                    s->vrtsp_state->decompress = TRUE;
                } else if (strcmp(item, "false") == 0) {
                    s->vrtsp_state->decompress = FALSE;
                } else {
                    INIT_FAIL("\n[rtsp] Wrong format for boolean decompress flag! \n");
                }
                break;
            case 3:
                s->vrtsp_state->tile->width = atoi(item);
                break;
            case 4:
                s->vrtsp_state->tile->height = atoi(item);
                break;
        }
        fmt = NULL;
        ++i;
    }
    free(tmp);

    //re-check parameters
    if (i < 2) {
        printf("\n[rtsp] Not enough parameters!\n");
        vidcap_rtsp_done(s);
        show_help();
        return VIDCAP_INIT_FAIL;
    }

    debug_msg("[rtsp] selected flags:\n");
    debug_msg("\t  uri: %s\n",s->uri);
    debug_msg("\t  port: %d\n", s->vrtsp_state->port);
    debug_msg("\t  decompress: %d\n\n",s->vrtsp_state->decompress);

    len = init_rtsp(s->uri, s->vrtsp_state->port, s, (char *) s->vrtsp_state->h264_offset_buffer);

    if(len < 0){
        vidcap_rtsp_done(s);
        return VIDCAP_INIT_FAIL;
    }else{
        s->vrtsp_state->h264_offset_len = len;
    }

    s->vrtsp_state->tile->data = (char *) malloc(4 * s->vrtsp_state->tile->width * s->vrtsp_state->tile->height);
    s->vrtsp_state->tile->data_len = 0;

    s->vrtsp_state->frame->frame_type = BFRAME;

    //TODO fps should be autodetected, now reset and controlled at vidcap_grab function
    s->vrtsp_state->frame->fps = 30;
    s->vrtsp_state->fps = 30;
    s->vrtsp_state->frame->interlacing = PROGRESSIVE;

    s->vrtsp_state->frame->tiles[0].data = (char *) calloc(1, s->vrtsp_state->tile->width * s->vrtsp_state->tile->height);

    s->should_exit = FALSE;

    s->vrtsp_state->device = rtp_init_if("localhost", s->vrtsp_state->mcast_if, s->vrtsp_state->port, 0, s->vrtsp_state->ttl, s->vrtsp_state->rtcp_bw,
        0, rtp_recv_callback, (uint8_t *) s->vrtsp_state->participants, 0, true);

    if (s->vrtsp_state->device != NULL) {
        if (!rtp_set_option(s->vrtsp_state->device, RTP_OPT_WEAK_VALIDATION, 1)) {
            debug_msg("[rtsp] RTP INIT - set option\n");
            return VIDCAP_INIT_FAIL;
        }
        if (!rtp_set_sdes(s->vrtsp_state->device, rtp_my_ssrc(s->vrtsp_state->device),
            RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING))) {
            debug_msg("[rtsp] RTP INIT - set sdes\n");
            return VIDCAP_INIT_FAIL;
        }

        int ret = rtp_set_recv_buf(s->vrtsp_state->device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
        if (!ret) {
            debug_msg("[rtsp] RTP INIT - set recv buf \nset command: sudo sysctl -w net.core.rmem_max=9123840\n");
            return VIDCAP_INIT_FAIL;
        }

        if (!rtp_set_send_buf(s->vrtsp_state->device, 1024 * 56)) {
            debug_msg("[rtsp] RTP INIT - set send buf\n");
            return VIDCAP_INIT_FAIL;
        }
        ret=pdb_add(s->vrtsp_state->participants, rtp_my_ssrc(s->vrtsp_state->device));
    }

    debug_msg("[rtsp] rtp receiver init done\n");

    pthread_mutex_init(&s->vrtsp_state->lock, NULL);
    pthread_cond_init(&s->vrtsp_state->boss_cv, NULL);
    pthread_cond_init(&s->vrtsp_state->worker_cv, NULL);

    s->vrtsp_state->boss_waiting = false;
    s->vrtsp_state->worker_waiting = false;

    if (s->vrtsp_state->decompress) {
        s->vrtsp_state->frame->color_spec = H264;
        if (init_decompressor(s->vrtsp_state) == 0) {
            vidcap_rtsp_done(s);
            return VIDCAP_INIT_FAIL;
        }
        s->vrtsp_state->frame->color_spec = UYVY;
    }

    pthread_create(&s->vrtsp_state->vrtsp_thread_id, NULL, vidcap_rtsp_thread, s);
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
init_rtsp(char* rtsp_uri, int rtsp_port, void *state, char* nals) {
    /* initialize curl */
    struct rtsp_state *s = (struct rtsp_state *) state;

    s->curl = init_curl();

    if (!s->curl) {
        return -1;
    }

    const char *range = "0.000-";
    int len_nals = -1;
    debug_msg("\n[rtsp] request %s\n", VERSION_STR);
    debug_msg("    Project web site: http://code.google.com/p/rtsprequest/\n");
    debug_msg("    Requires cURL V7.20 or greater\n\n");
    const char *url = rtsp_uri;
    s->uri = (char *) malloc(strlen(url) + 32);
    char *sdp_filename = nullptr;
    char Atransport[256];
    char Vtransport[256];
    memset(Atransport, 0, 256);
    memset(Vtransport, 0, 256);
    int port = rtsp_port;
    CURLcode res;

    if ((sdp_filename = get_sdp_filename(url)) == nullptr) {
        LOG(LOG_LEVEL_ERROR) << "[RTSP] Cannot SDP file name from URL: " << url << "\n";
        return -1;
    }

    sprintf(Vtransport, "RTP/AVP;unicast;client_port=%d-%d", port, port + 1);

    //THIS AUDIO PORTS ARE AS DEFAULT UG AUDIO PORTS BUT AREN'T RELATED...
    sprintf(Atransport, "RTP/AVP;unicast;client_port=%d-%d", port+2, port + 3);

    my_curl_easy_setopt(s->curl, CURLOPT_NOSIGNAL, 1, goto error); //This tells curl not to use any functions that install signal handlers or cause signals to be sent to your process.
    //my_curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, 1);
    my_curl_easy_setopt(s->curl, CURLOPT_VERBOSE, 0L, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_NOPROGRESS, 1L, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_WRITEHEADER, stdout, goto error);
    my_curl_easy_setopt(s->curl, CURLOPT_URL, url, goto error);

    sprintf(s->uri, "%s", url);

    //TODO TO CHECK CONFIGURING ERRORS
    //CURLOPT_ERRORBUFFER
    //http://curl.haxx.se/libcurl/c/curl_easy_perform.html

    /* request server options */
    if(rtsp_options(s->curl, s->uri)==0){
        goto error;
    }

    /* request session description and write response to sdp file */
    if (!rtsp_describe(s->curl, s->uri, sdp_filename)) {
        goto error;
    }

    if (!setup_codecs_and_controls_from_sdp(sdp_filename, s)) {
        goto error;
    }
    if (strcmp(s->vrtsp_state->codec, "H264") == 0){
        s->vrtsp_state->frame->color_spec = H264;
        sprintf(s->uri, "%s/%s", url, s->vrtsp_state->control);
        debug_msg("\n V URI = %s\n", s->uri);
        if(rtsp_setup(s->curl, s->uri, Vtransport)==0){
            goto error;
        }
        sprintf(s->uri, "%s", url);
    }
    if (strcmp(s->artsp_state->codec, "PCMU") == 0){
        sprintf(s->uri, "%s/%s", url, s->artsp_state->control);
        debug_msg("\n A URI = %s\n", s->uri);
        if(rtsp_setup(s->curl, s->uri, Atransport)==0){
            goto error;
        }
        sprintf(s->uri, "%s", url);
    }
    if (strlen(s->artsp_state->codec) == 0 && strlen(s->vrtsp_state->codec) == 0){
        goto error;
    }
    else{
        if(rtsp_play(s->curl, s->uri, range)==0){
            goto error;
        }
    }

    /* get start nal size attribute from sdp file */
    len_nals = get_nals(sdp_filename, nals, (int *) &s->vrtsp_state->tile->width, (int *) &s->vrtsp_state->tile->height);

    debug_msg("[rtsp] playing video from server (size: WxH = %d x %d)...\n",s->vrtsp_state->tile->width,s->vrtsp_state->tile->height);

    free(sdp_filename);
    return len_nals;

error:
    free(sdp_filename);
    return -1;
}

bool setup_codecs_and_controls_from_sdp(const char *sdp_filename, void *state) {
    struct rtsp_state *rtspState;
    rtspState = (struct rtsp_state *) state;

    int n=0;
    char *line = (char*) malloc(1024);
    char* tmpBuff;
    int countT = 0;
    int countC = 0;
    const size_t len = 10;
    char* codecs[2];
    char* tracks[2];
    for(int q=0; q<2 ; q++){
        codecs[q] = (char*) malloc(len);
        tracks[q] = (char*) malloc(len);
    }

    FILE* fp;

    fp = fopen(sdp_filename, "r");
    if(fp == 0){
        debug_msg("unable to open asset %s", sdp_filename);
        free(line);
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    if (fileSize < 0) {
            perror("RTSP ftell");
            free(line);
            fclose(fp);
            return false;
    }
    rewind(fp);

    char* buffer = (char*) malloc(fileSize+1);
    unsigned long readResult = fread(buffer, sizeof(char), fileSize, fp);

    if(readResult != (unsigned long) fileSize){
        debug_msg("something bad happens, read result != file size");
        free(line);
        fclose(fp);
        free(buffer);
        return false;
    }
    buffer[fileSize] = '\0';

    while (buffer[n] != '\0'){
        getNewLine(buffer,&n,line);
        sscanf(line, " a = control: %*s");
        tmpBuff = strstr(line, "track");
        if(tmpBuff!=NULL){
            if ((unsigned) countT < sizeof tracks / sizeof tracks[0]) {
                //debug_msg("track = %s\n",tmpBuff);
                strncpy(tracks[countT],tmpBuff,MIN(strlen(tmpBuff)-2, len-1));
                tracks[countT][MIN(strlen(tmpBuff)-2, len-1)] = '\0';
                countT++;
            } else {
                log_msg(LOG_LEVEL_WARNING, "skipping track = %s\n",tmpBuff);
            }
        }
        tmpBuff=NULL;
        sscanf(line, " a=rtpmap:96 %*s");
        tmpBuff = strstr(line, "H264");
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
            rtspState->vrtsp_state->codec = "H264";
            rtspState->vrtsp_state->control = tracks[p];

        }if(strncmp(codecs[p],"PCMU",4)==0){
            rtspState->artsp_state->codec = "PCMU";
            rtspState->artsp_state->control = tracks[p];
        }
    }
    free(line);
    free(buffer);
    fclose(fp);
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
        line[j] = '\0';
    }
}
/**
 * Initializes decompressor if required by decompress flag
 */
static int
init_decompressor(void *state) {
    struct video_rtsp_state *sr;
    sr = (struct video_rtsp_state *) state;

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
rtsp_describe(CURL *curl, const char *uri, const char *sdp_filename) {
    CURLcode res = CURLE_OK;
    FILE *sdp_fp = fopen(sdp_filename, "wt");
    debug_msg("\n[rtsp] DESCRIBE %s\n", uri);
    if (sdp_fp == NULL) {
        fprintf(stderr, "Could not open '%s' for writing\n", sdp_filename);
        sdp_fp = stdout;
    } else {
        debug_msg("Writing SDP to '%s'\n", sdp_filename);
    }
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, sdp_fp, goto error);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_DESCRIBE, goto error);

    if (curl_easy_perform(curl) != CURLE_OK){
        error_msg("[RTSP DESCRIBE] curl_easy_perform failed\n");
        error_msg("[RTSP DESCRIBE] could not configure rtsp capture properly, \n\t\tplease check your parameters. \ncleaning...\n\n");
        goto error;
    }

    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, stdout, goto error);
    if (sdp_fp != stdout) {
        fclose(sdp_fp);
    }
    return true;
error:
    if (sdp_fp != stdout) {
        fclose(sdp_fp);
    }
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
/**
 * convert url into an sdp filename
 */
static char *
get_sdp_filename(const char *url) {
    const char *s = strrchr(url, '/');

    if (s == nullptr) {
        return nullptr;
    }
    s++;
    char *sdp_filename = nullptr;
    if (strlen(s) > 0) {
        sdp_filename = static_cast<char *>(malloc(strlen(s) + 4 + 1));
        sprintf(sdp_filename, "%s.sdp", s);
        debug_msg("sdp_file get: %s\n", sdp_filename);
    }
    return sdp_filename;
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

    s->should_exit = TRUE;
    pthread_join(s->vrtsp_state->vrtsp_thread_id, NULL);
    pthread_join(s->keep_alive_rtsp_thread_id, NULL);

    if(s->vrtsp_state->decompress)
        decompress_done(s->vrtsp_state->sd);

    if (s->vrtsp_state->device != nullptr) {
        rtp_done(s->vrtsp_state->device);
    }

    free(s->vrtsp_state->tile->data);
    if(s->vrtsp_state->h264_offset_buffer!=NULL) free(s->vrtsp_state->h264_offset_buffer);
    if(s->vrtsp_state->frame!=NULL) free(s->vrtsp_state->frame);
    free(s->vrtsp_state);
    free(s->artsp_state);


    rtsp_teardown(s->curl, s->uri);

    curl_easy_cleanup(s->curl);
    curl_global_cleanup();
    s->curl = NULL;

    free(s);
}

/**
 * scan sdp file for media control attributes to generate coded frame required params (WxH and offset)
 */
static int
get_nals(const char *sdp_filename, char *nals, int *width, int *height) {

    uint8_t nalInfo;
    uint8_t type;
    uint8_t nri __attribute__((unused));
    int max_len = 1500, len_nals = 0;
    char *s = (char *) malloc(max_len);
    char *sprop;
    memset(s, 0, max_len);
    FILE *sdp_fp = fopen(sdp_filename, "rt");
    nals[0] = '\0';
    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
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
        fclose(sdp_fp);
    }

    free(s);
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
