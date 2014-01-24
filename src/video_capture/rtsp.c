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

#include "debug.h"
#include "host.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/rtpdec_h264.h"

#include "video_decompress.h"
#include "video_decompress/libavcodec.h"

#include "pdb.h"
#include "rtp/pbuf.h"

#include "video.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_capture/rtsp.h"
#include "audio/audio.h"

#include <curl/curl.h>

#define VERSION_STR  "V1.0"

//TODO set lower initial video recv buffer size (to find the minimal?)
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((0.1*1920*1080)*110/100) //command line net.core setup: sysctl -w net.core.rmem_max=9123840

/* error handling macros */
#define my_curl_easy_setopt(A, B, C) \
    if ((res = curl_easy_setopt((A), (B), (C))) != CURLE_OK){ \
        fprintf(stderr, "[rtsp error] curl_easy_setopt(%s, %s, %s) failed: %d\n", #A, #B, #C, res); \
        printf("[rtsp error] could not configure rtsp capture properly, \n\t\tplease check your parameters. \nExiting...\n\n"); \
        exit(0); \
    }

#define my_curl_easy_perform(A) \
    if ((res = curl_easy_perform((A))) != CURLE_OK){ \
        fprintf(stderr, "[rtsp error] curl_easy_perform(%s) failed: %d\n", #A, res); \
        printf("[rtsp error] could not configure rtsp capture properly, \n\t\tplease check your parameters. \nExiting...\n\n"); \
        exit(0); \
    }

/* send RTSP GET_PARAMETERS request */
static void
rtsp_get_parameters(CURL *curl, const char *uri);

/* send RTSP OPTIONS request */
static void
rtsp_options(CURL *curl, const char *uri);

/* send RTSP DESCRIBE request and write sdp response to a file */
static void
rtsp_describe(CURL *curl, const char *uri, const char *sdp_filename);

/* send RTSP SETUP request */
static void
rtsp_setup(CURL *curl, const char *uri, const char *transport);

/* send RTSP PLAY request */
static void
rtsp_play(CURL *curl, const char *uri, const char *range);

/* send RTSP TEARDOWN request */
static void
rtsp_teardown(CURL *curl, const char *uri);

/* convert url into an sdp filename */
static void
get_sdp_filename(const char *url, char *sdp_filename);

static int
get_nals(const char *sdp_filename, char *nals);

static int
init_rtsp(char* rtsp_uri, int rtsp_port, void *state, char* nals);

static int
init_decompressor(void *state);

static void *
vidcap_rtsp_thread(void *args);

static void
show_help(void);

void
rtsp_keepalive(void *state);

FILE *F_video_rtsp = NULL;
/**
 * @struct rtsp_state
 */
struct video_rtsp_state {
    char *nals;
    int nals_size;
    char *data; //nals + data
    uint32_t *in_codec;

    char *codec;

    struct timeval t0, t;
    int frames;
    struct video_frame *frame;
    struct tile *tile;
    int width;
    int height;

    struct std_frame_received *rx_data;
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
};

struct audio_rtsp_state {
    struct audio_frame audio;
    int play_audio_frame;

    char *codec;

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
    char *uri;
    uint avType;
    char *addr;
    char *sdp;

    volatile bool should_exit;
    struct audio_rtsp_state *artsp_state;
    struct video_rtsp_state *vrtsp_state;
};

static void
show_help() {
    printf("[rtsp] usage:\n");
    printf("\t-t rtsp:<uri>:<port>:<width>:<height>[:<decompress>]\n");
    printf("\t\t <uri> uri server without 'rtsp://' \n");
    printf("\t\t <port> receiver port number \n");
    printf("\t\t <width> receiver width number \n");
    printf("\t\t <height> receiver height number \n");
    printf(
        "\t\t <decompress> receiver decompress boolean [true|false] - default: false - no decompression active\n\n");
}

void
rtsp_keepalive(void *state) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) state;
    struct timeval now;
    gettimeofday(&now, NULL);
    if (tv_diff(now, s->vrtsp_state->prev_time) >= 20) {
        rtsp_get_parameters(s->curl, s->uri);
        gettimeofday(&s->vrtsp_state->prev_time, NULL);
    }
}

static void *
vidcap_rtsp_thread(void *arg) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) arg;

    gettimeofday(&s->vrtsp_state->start_time, NULL);
    gettimeofday(&s->vrtsp_state->prev_time, NULL);

    while (!s->should_exit) {
        gettimeofday(&s->vrtsp_state->curr_time, NULL);
        s->vrtsp_state->timestamp = tv_diff(s->vrtsp_state->curr_time, s->vrtsp_state->start_time) * 90000;

        rtsp_keepalive(s);

        rtp_update(s->vrtsp_state->device, s->vrtsp_state->curr_time);
        //TODO no need of rtcp communication between ug and rtsp server?
        //rtp_send_ctrl(s->device, s->timestamp, 0, s->curr_time);

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
                            if (pbuf_decode(s->vrtsp_state->cp->playout_buffer, s->vrtsp_state->curr_time,
                                decode_frame_h264, s->vrtsp_state->rx_data))
                            {
                                s->vrtsp_state->new_frame = true;
                            }
                            if (s->vrtsp_state->boss_waiting)
                                pthread_cond_signal(&s->vrtsp_state->boss_cv);
                        }
                    }
                    pthread_mutex_unlock(&s->vrtsp_state->lock);
                }
                pbuf_remove(s->vrtsp_state->cp->playout_buffer, s->vrtsp_state->curr_time);
                s->vrtsp_state->cp = pdb_iter_next(&it);
            }

            pdb_iter_done(&it);
        }
    }
    return NULL;
}

struct video_frame *
vidcap_rtsp_grab(void *state, struct audio_frame **audio) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) state;

    *audio = NULL;

    if(pthread_mutex_trylock(&s->vrtsp_state->lock)==0){
        {
            s->vrtsp_state->grab = true;

            while (!s->vrtsp_state->new_frame) {
                s->vrtsp_state->boss_waiting = true;
                pthread_cond_wait(&s->vrtsp_state->boss_cv, &s->vrtsp_state->lock);
                s->vrtsp_state->boss_waiting = false;
            }

            gettimeofday(&s->vrtsp_state->curr_time, NULL);
            s->vrtsp_state->frame->h264_iframe = s->vrtsp_state->rx_data->iframe;
            s->vrtsp_state->frame->h264_iframe = s->vrtsp_state->rx_data->iframe;
            s->vrtsp_state->frame->tiles[0].data_len = s->vrtsp_state->rx_data->buffer_len;
            memcpy(s->vrtsp_state->data + s->vrtsp_state->nals_size, s->vrtsp_state->rx_data->frame_buffer,
                s->vrtsp_state->rx_data->buffer_len);
            memcpy(s->vrtsp_state->frame->tiles[0].data, s->vrtsp_state->data,
                s->vrtsp_state->rx_data->buffer_len + s->vrtsp_state->nals_size);
            s->vrtsp_state->frame->tiles[0].data_len += s->vrtsp_state->nals_size;

            if (s->vrtsp_state->decompress) {
                decompress_frame(s->vrtsp_state->sd, (unsigned char *) s->vrtsp_state->out_frame,
                    (unsigned char *) s->vrtsp_state->frame->tiles[0].data,
                    s->vrtsp_state->rx_data->buffer_len + s->vrtsp_state->nals_size, 0);
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

void *
vidcap_rtsp_init(const struct vidcap_params *params) {

    struct rtsp_state *s;

    s = calloc(1, sizeof(struct rtsp_state));
    if (!s)
        return NULL;

    s->artsp_state = calloc(1,sizeof(struct audio_rtsp_state));
    s->vrtsp_state = calloc(1,sizeof(struct video_rtsp_state));

    //TODO now static codec assignment, to be dynamic as a function of supported codecs
    s->vrtsp_state->codec = "";
    s->artsp_state->codec = "";
    s->artsp_state->control = "";
    s->artsp_state->control = "";

    char *save_ptr = NULL;
    s->avType = -1;  //-1 none, 0 a&v, 1 v, 2 a

    gettimeofday(&s->vrtsp_state->t0, NULL);
    s->vrtsp_state->frames = 0;
    s->vrtsp_state->nals = malloc(1024);
    s->vrtsp_state->grab = false;

    s->addr = "127.0.0.1";
    s->vrtsp_state->device = NULL;
    s->vrtsp_state->rtcp_bw = 5 * 1024 * 1024; /* FIXME */
    s->vrtsp_state->ttl = 255;

    s->vrtsp_state->mcast_if = NULL;
    s->vrtsp_state->required_connections = 1;

    s->vrtsp_state->timeout.tv_sec = 0;
    s->vrtsp_state->timeout.tv_usec = 10000;

    s->vrtsp_state->device = (struct rtp *) malloc(
        (s->vrtsp_state->required_connections) * sizeof(struct rtp *));
    s->vrtsp_state->participants = pdb_init();

    s->vrtsp_state->rx_data = malloc(sizeof(struct std_frame_received));
    s->vrtsp_state->new_frame = false;

    s->vrtsp_state->in_codec = malloc(sizeof(uint32_t *) * 10);

    s->uri = NULL;
    s->curl = NULL;
    char *fmt = NULL;
    char *uri_tmp1;
    char *uri_tmp2;

    if (vidcap_params_get_fmt(params)
        && strcmp(vidcap_params_get_fmt(params), "help") == 0)
    {
        show_help();
        return &vidcap_init_noerr;
    } else {
        char *tmp = NULL;
        fmt = strdup(vidcap_params_get_fmt(params));
        int i = 0;

        while ((tmp = strtok_r(fmt, ":", &save_ptr))) {
            switch (i) {
                case 0:
                    if (tmp) {
                        tmp = strtok_r(NULL, ":", &save_ptr);
                        uri_tmp1 = malloc(strlen(tmp) + 32);
                        sprintf(uri_tmp1, "%s", tmp);
                        tmp = strtok_r(NULL, ":", &save_ptr);
                        uri_tmp2 = malloc(strlen(tmp) + 32);
                        sprintf(uri_tmp2, "%s", tmp);
                        s->uri = malloc(1024 + 32);
                        sprintf(s->uri, "rtsp:%s:%s", uri_tmp1, uri_tmp2);
                    } else {
                        printf("\n[rtsp] Wrong format for uri! \n");
                        show_help();
                        exit(0);
                    }
                    break;
                case 1:
                    if (tmp) {    //TODO check if it's a number
                        s->vrtsp_state->port = atoi(tmp);
                    } else {
                        printf("\n[rtsp] Wrong format for port! \n");
                        show_help();
                        exit(0);
                    }
                    break;
                case 2:
                    if (tmp) {  //TODO check if it's a number
                        s->vrtsp_state->width = atoi(tmp);
                    } else {
                        printf("\n[rtsp] Wrong format for width! \n");
                        show_help();
                        exit(0);
                    }
                    break;
                case 3:
                    if (tmp) {  //TODO check if it's a number
                        s->vrtsp_state->height = atoi(tmp);
                        //Now checking if we have user and password parameters...
                        if (s->vrtsp_state->height == 0) {
                            int ntmp = 0;
                            ntmp = s->vrtsp_state->width;
                            s->vrtsp_state->width = s->vrtsp_state->port;
                            s->vrtsp_state->height = ntmp;
                            sprintf(s->uri, "rtsp:%s", uri_tmp1);
                            s->vrtsp_state->port = atoi(uri_tmp2);
                            if (tmp) {
                                if (strcmp(tmp, "true") == 0)
                                    s->vrtsp_state->decompress = true;
                                else if (strcmp(tmp, "false") == 0)
                                    s->vrtsp_state->decompress = false;
                                else {
                                    printf("\n[rtsp] Wrong format for boolean decompress flag! \n");
                                    show_help();
                                    exit(0);
                                }
                            } else
                                continue;
                        }
                    } else {
                        printf("\n[rtsp] Wrong format for height! \n");
                        show_help();
                        exit(0);
                    }
                    break;
                case 4:
                    if (tmp) {
                        if (strcmp(tmp, "true") == 0)
                            s->vrtsp_state->decompress = true;
                        else if (strcmp(tmp, "false") == 0)
                            s->vrtsp_state->decompress = false;
                        else {
                            printf(
                                "\n[rtsp] Wrong format for boolean decompress flag! \n");
                            show_help();
                            exit(0);
                        }
                    } else
                        continue;
                    break;
                case 5:
                    continue;
            }
            fmt = NULL;
            ++i;
        }
    }
    //re-check parameters
    if (s->vrtsp_state->height == 0) {
        int ntmp = 0;
        ntmp = s->vrtsp_state->width;
        s->vrtsp_state->width = (int) s->vrtsp_state->port;
        s->vrtsp_state->height = (int) ntmp;
        sprintf(s->uri, "rtsp:%s", uri_tmp1);
        s->vrtsp_state->port = (int) atoi(uri_tmp2);
    }

    debug_msg("[rtsp] selected flags:\n");
    debug_msg("\t  uri: %s\n",s->uri);
    debug_msg("\t  port: %d\n", s->vrtsp_state->port);
    debug_msg("\t  width: %d\n",s->vrtsp_state->width);
    debug_msg("\t  height: %d\n",s->vrtsp_state->height);
    debug_msg("\t  decompress: %d\n\n",s->vrtsp_state->decompress);

    if (uri_tmp1 != NULL)
        free(uri_tmp1);
    if (uri_tmp2 != NULL)
        free(uri_tmp2);

    s->vrtsp_state->rx_data->frame_buffer = malloc(4 * s->vrtsp_state->width * s->vrtsp_state->height);
    s->vrtsp_state->data = malloc(4 * s->vrtsp_state->width * s->vrtsp_state->height + s->vrtsp_state->nals_size);

    s->vrtsp_state->frame = vf_alloc(1);
    s->vrtsp_state->frame->isStd = TRUE;
    s->vrtsp_state->frame->h264_bframe = FALSE;
    s->vrtsp_state->frame->h264_iframe = FALSE;
    s->vrtsp_state->tile = vf_get_tile(s->vrtsp_state->frame, 0);
    vf_get_tile(s->vrtsp_state->frame, 0)->width = s->vrtsp_state->width;
    vf_get_tile(s->vrtsp_state->frame, 0)->height = s->vrtsp_state->height;
    //TODO fps should be autodetected, now reset and controlled at vidcap_grab function
    s->vrtsp_state->frame->fps = 30;
    s->vrtsp_state->fps = 30;
    s->vrtsp_state->frame->interlacing = PROGRESSIVE;

    s->vrtsp_state->frame->tiles[0].data = calloc(1, s->vrtsp_state->width * s->vrtsp_state->height);

    s->should_exit = false;

    s->vrtsp_state->device = rtp_init_if(NULL, s->vrtsp_state->mcast_if, s->vrtsp_state->port, 0, s->vrtsp_state->ttl, s->vrtsp_state->rtcp_bw,
        0, rtp_recv_callback, (void *) s->vrtsp_state->participants, 0);

    if (s->vrtsp_state->device != NULL) {
        if (!rtp_set_option(s->vrtsp_state->device, RTP_OPT_WEAK_VALIDATION, 1)) {
            debug_msg("[rtsp] RTP INIT - set option\n");
            return NULL;
        }
        if (!rtp_set_sdes(s->vrtsp_state->device, rtp_my_ssrc(s->vrtsp_state->device),
            RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING))) {
            debug_msg("[rtsp] RTP INIT - set sdes\n");
            return NULL;
        }

        int ret = rtp_set_recv_buf(s->vrtsp_state->device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
        if (!ret) {
            debug_msg("[rtsp] RTP INIT - set recv buf \nset command: sudo sysctl -w net.core.rmem_max=9123840\n");
            return NULL;
        }

        if (!rtp_set_send_buf(s->vrtsp_state->device, 1024 * 56)) {
            debug_msg("[rtsp] RTP INIT - set send buf\n");
            return NULL;
        }
        ret=pdb_add(s->vrtsp_state->participants, rtp_my_ssrc(s->vrtsp_state->device));
    }

    debug_msg("[rtsp] rtp receiver init done\n");

    pthread_mutex_init(&s->vrtsp_state->lock, NULL);
    pthread_cond_init(&s->vrtsp_state->boss_cv, NULL);
    pthread_cond_init(&s->vrtsp_state->worker_cv, NULL);

    s->vrtsp_state->boss_waiting = false;
    s->vrtsp_state->worker_waiting = false;

    s->vrtsp_state->nals_size = init_rtsp(s->uri, s->vrtsp_state->port, s, s->vrtsp_state->nals);

    if (s->vrtsp_state->nals_size >= 0)
        memcpy(s->vrtsp_state->data, s->vrtsp_state->nals, s->vrtsp_state->nals_size);
    else{
        printf("\n[rtsp] something went wrong with the sdp parser...\n");
        return NULL;
    }

    if (s->vrtsp_state->decompress) {
        if (init_decompressor(s->vrtsp_state) == 0)
            return NULL;
        s->vrtsp_state->frame->color_spec = UYVY;
    }

    pthread_create(&s->vrtsp_state->vrtsp_thread_id, NULL, vidcap_rtsp_thread, s);

    debug_msg("[rtsp] rtsp capture init done\n");

    return s;
}

/**
 * Initializes rtsp state and internal parameters
 */
static int
init_rtsp(char* rtsp_uri, int rtsp_port, void *state, char* nals) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) state;
    const char *range = "0.000-";
    int len_nals = -1;
    CURLcode res;
    debug_msg("\n[rtsp] request %s\n", VERSION_STR);
    debug_msg("    Project web site: http://code.google.com/p/rtsprequest/\n");
    debug_msg("    Requires cURL V7.20 or greater\n\n");
    const char *url = rtsp_uri;
    char *uri = malloc(strlen(url) + 32);
    char *sdp_filename = malloc(strlen(url) + 32);
    char *control = malloc(150 * sizeof(char *));
    memset(control, 0, 150 * sizeof(char *));
    char Atransport[256];
    char Vtransport[256];
    bzero(Atransport, 256);
    bzero(Vtransport, 256);
    int port = rtsp_port;

    get_sdp_filename(url, sdp_filename);

    sprintf(Vtransport, "RTP/AVP;unicast;client_port=%d-%d", port, port + 1);

    //THIS AUDIO PORTS ARE AS DEFAULT UG AUDIO PORTS BUT AREN'T RELATED...
    sprintf(Atransport, "RTP/AVP;unicast;client_port=%d-%d", port+2, port + 3);

    /* initialize curl */
    res = curl_global_init(CURL_GLOBAL_ALL);
    if (res == CURLE_OK) {
        curl_version_info_data *data = curl_version_info(CURLVERSION_NOW);
        CURL *curl;
        fprintf(stderr, "[rtsp]    cURL V%s loaded\n", data->version);

        /* initialize this curl session */
        curl = curl_easy_init();
        if (curl != NULL) {
            my_curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1); //This tells curl not to use any functions that install signal handlers or cause signals to be sent to your process.
            //my_curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, 1);
            my_curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
            my_curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
            my_curl_easy_setopt(curl, CURLOPT_WRITEHEADER, stdout);
            my_curl_easy_setopt(curl, CURLOPT_URL, url);

            sprintf(uri, "%s", url);

            s->curl = curl;
            s->uri = uri;

            //TODO TO CHECK CONFIGURING ERRORS
            //CURLOPT_ERRORBUFFER
            //http://curl.haxx.se/libcurl/c/curl_easy_perform.html

            /* request server options */
            rtsp_options(curl, uri);
            printf("sdp_file: %s\n", sdp_filename);
            /* request session description and write response to sdp file */
            rtsp_describe(curl, uri, sdp_filename);

            setup_codecs_and_controls_from_sdp(sdp_filename, s);
            if (s->vrtsp_state->codec == "H264"){
                s->vrtsp_state->frame->color_spec = H264;
                sprintf(uri, "%s/%s", url, s->vrtsp_state->control);
                debug_msg("\n V URI = %s\n",uri);
                rtsp_setup(curl, uri, Vtransport);
                sprintf(uri, "%s", url);
            }
            if (s->artsp_state->codec == "PCMU"){
                sprintf(uri, "%s/%s", url, s->artsp_state->control);
                debug_msg("\n A URI = %s\n",uri);
                rtsp_setup(curl, uri, Atransport);
                sprintf(uri, "%s", url);
            }
            if (s->artsp_state->codec == "" && s->vrtsp_state->codec == "") return -1;
            else rtsp_play(curl, uri, range);

            /* get start nal size attribute from sdp file */
            len_nals = get_nals(sdp_filename, nals);

            s->curl = curl;
            s->uri = uri;
            debug_msg("[rtsp] playing video from server...\n");

        } else {
            fprintf(stderr, "[rtsp] curl_easy_init() failed\n");
        }
        curl_global_cleanup();
    } else {
        fprintf(stderr, "[rtsp] curl_global_init(%s) failed: %d\n",
            "CURL_GLOBAL_ALL", res);
    }
    return len_nals;
}

void setup_codecs_and_controls_from_sdp(const char *sdp_filename, void *state) {
    struct rtsp_state *rtspState;
    rtspState = (struct rtsp_state *) state;

    int n=0;
    char *line = (char*) malloc(1024);
    char* tmpBuff;
    int countT = 0;
    int countC = 0;
    char* codecs[2];
    char* tracks[2];
    for(int q=0; q<2 ; q++){
        codecs[q] = (char*) malloc(10);
        tracks[q] = (char*) malloc(10);
    }

    FILE* fp;

    fp = fopen(sdp_filename, "r");
    if(fp == 0){
        printf("unable to open asset %s", sdp_filename);
        fclose(fp);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    unsigned long fileSize = ftell(fp);
    rewind(fp);

    char* buffer = (char*) malloc(fileSize+1);
    unsigned long readResult = fread(buffer, sizeof(char), fileSize, fp);

    if(readResult != fileSize){
        printf("something bad happens, read result != file size");
        return -1;
    }
    buffer[fileSize] = '\0';

    while (buffer[n] != '\0'){
        newLine(buffer,&n,line);
        sscanf(line, " a = control: %s", tmpBuff);
        tmpBuff = strstr(line, "track");
        if(tmpBuff!=NULL){
            //printf("track = %s\n",tmpBuff);
            strncpy(tracks[countT],tmpBuff,strlen(tmpBuff)-2);
            tracks[countT][strlen(tmpBuff)-2] = '\0';
            countT++;
        }
        tmpBuff='\0';
        sscanf(line, " a=rtpmap:96 %s", tmpBuff);
        tmpBuff = strstr(line, "H264");
        if(tmpBuff!=NULL){
            //printf("codec = %s\n",tmpBuff);
            strncpy(codecs[countC],tmpBuff,4);
            codecs[countC][4] = '\0';
            countC++;
        }
        tmpBuff='\0';
        sscanf(line, " a=rtpmap:97 %s", tmpBuff);
        tmpBuff = strstr(line, "PCMU");
        if(tmpBuff!=NULL){
            //printf("codec = %s\n",tmpBuff);
            strncpy(codecs[countC],tmpBuff,4);
            codecs[countC][4] = '\0';
            countC++;
        }
        tmpBuff='\0';

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
}

void newLine(const char* buffer, int* i, char* line){
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

    sr->sd = (struct state_decompress *) calloc(2,
        sizeof(struct state_decompress *));
    initialize_video_decompress();

    if (decompress_is_available(LIBAVCODEC_MAGIC)) {
        sr->sd = decompress_init(LIBAVCODEC_MAGIC);

        sr->des.width = sr->width;
        sr->des.height = sr->height;
        sr->des.color_spec = sr->frame->color_spec;
        sr->des.tile_count = 0;
        sr->des.interlacing = PROGRESSIVE;

        decompress_reconfigure(sr->sd, sr->des, 16, 8, 0,
            vc_get_linesize(sr->des.width, UYVY), UYVY);
    } else
        return 0;
    sr->out_frame = malloc(sr->width * sr->height * 4);
    return 1;
}

static void
rtsp_get_parameters(CURL *curl, const char *uri) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] GET_PARAMETERS %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_GET_PARAMETER);
    my_curl_easy_perform(curl);
}

/**
 * send RTSP OPTIONS request
 */
static void
rtsp_options(CURL *curl, const char *uri) {
    char control[1500], *user, *pass, *strtoken;
    user = malloc(1500);
    pass = malloc(1500);
    bzero(control, 1500);
    bzero(user, 1500);
    bzero(pass, 1500);

    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] OPTIONS %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);

    sscanf(uri, "rtsp://%s", control);
    strtoken = strtok(control, ":");
    memcpy(user, strtoken, strlen(strtoken));
    strtoken = strtok(NULL, "@");
    if (strtoken == NULL) {
        user = NULL;
        pass = NULL;
    } else
        memcpy(pass, strtoken, strlen(strtoken));
    if (user != NULL)
        my_curl_easy_setopt(curl, CURLOPT_USERNAME, user);
    if (pass != NULL)
        my_curl_easy_setopt(curl, CURLOPT_PASSWORD, pass);

    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long )CURL_RTSPREQ_OPTIONS);
    my_curl_easy_perform(curl);
}

/**
 * send RTSP DESCRIBE request and write sdp response to a file
 */
static void
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
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, sdp_fp);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_DESCRIBE);
    my_curl_easy_perform(curl);
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, stdout);
    if (sdp_fp != stdout) {
        fclose(sdp_fp);
    }
}

/**
 * send RTSP SETUP request
 */
static void
rtsp_setup(CURL *curl, const char *uri, const char *transport) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] SETUP %s\n", uri);
    debug_msg("\t TRANSPORT %s\n", transport);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_TRANSPORT, transport);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long )CURL_RTSPREQ_SETUP);
    my_curl_easy_perform(curl);
}

/**
 * send RTSP PLAY request
 */
static void
rtsp_play(CURL *curl, const char *uri, const char *range) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] PLAY %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    //my_curl_easy_setopt(curl, CURLOPT_RANGE, range);      //range not set because we want (right now) no limit range for streaming duration
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long )CURL_RTSPREQ_PLAY);
    my_curl_easy_perform(curl);
}

/**
 * send RTSP TEARDOWN request
 */
static void
rtsp_teardown(CURL *curl, const char *uri) {
    CURLcode res = CURLE_OK;
    debug_msg("\n[rtsp] TEARDOWN %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST,
        (long )CURL_RTSPREQ_TEARDOWN);
    my_curl_easy_perform(curl);
}

/**
 * convert url into an sdp filename
 */
static void
get_sdp_filename(const char *url, char *sdp_filename) {
    const char *s = strrchr(url, '/');
    debug_msg("sdp_file get: %s\n", sdp_filename);

    if (s != NULL) {
        s++;
        if (s[0] != '\0') {
            sprintf(sdp_filename, "%s.sdp", s);
        }
    }
}

struct vidcap_type *
vidcap_rtsp_probe(void) {
    struct vidcap_type *vt;

    vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
    if (vt != NULL) {
        vt->id = VIDCAP_RTSP_ID;
        vt->name = "rtsp";
        vt->description = "Video capture from RTSP remote server";
    }
    return vt;
}

void
vidcap_rtsp_done(void *state) {
    struct rtsp_state *s = state;

    s->should_exit = true;
    pthread_join(s->vrtsp_state->vrtsp_thread_id, NULL);

    free(s->vrtsp_state->rx_data->frame_buffer);
    free(s->vrtsp_state->data);

    rtsp_teardown(s->curl, s->uri);

    curl_easy_cleanup(s->curl);
    s->curl = NULL;

    if (s->vrtsp_state->decompress)
        decompress_done(s->vrtsp_state->sd);
    rtp_done(s->vrtsp_state->device);

    free(s);
}

/**
 * scan sdp file for media control attribute to generate nal starting bytes
 */
static int
get_nals(const char *sdp_filename, char *nals) {
    int max_len = 1500, len_nals = 0;
    char *s = malloc(max_len);
    char *sprop;
    bzero(s, max_len);
    FILE *sdp_fp = fopen(sdp_filename, "rt");
    nals[0] = '\0';
    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
            sprop = strstr(s, "sprop-parameter-sets=");
            if (sprop != NULL) {
                gsize length;   //gsize is an unsigned int.
                char *nal_aux, *nals_aux, *nal;
                nals_aux = malloc(max_len);
                nals[0] = 0x00;
                nals[1] = 0x00;
                nals[2] = 0x00;
                nals[3] = 0x01;
                len_nals = 4;
                nal_aux = strstr(sprop, "=");
                nal_aux++;
                nal = strtok(nal_aux, ",;");
                //convert base64 to hex
                nals_aux = g_base64_decode(nal, &length);
                memcpy(nals + len_nals, nals_aux, length);
                len_nals += length;

                while ((nal = strtok(NULL, ",;")) != NULL) {
                    nals_aux = g_base64_decode(nal, &length);
                    if (length) {
                        //convert base64 to hex
                        nals[len_nals] = 0x00;
                        nals[len_nals + 1] = 0x00;
                        nals[len_nals + 2] = 0x00;
                        nals[len_nals + 3] = 0x01;
                        len_nals += 4;
                        memcpy(nals + len_nals, nals_aux, length);
                        len_nals += length;
                    } //end if (length) {
                } //end while ((nal = strtok(NULL, ",;")) != NULL){
            } //end if (sprop != NULL) {
        } //end while (fgets(s, max_len - 2, sdp_fp) != NULL) {
        fclose(sdp_fp);
    } //end if (sdp_fp != NULL) {

    free(s);
    return len_nals;
}
