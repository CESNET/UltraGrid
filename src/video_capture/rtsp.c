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

/* scan sdp file for media control attribute */
static void
get_media_control_attribute(const char *sdp_filename, char *control);

/* scan sdp file for incoming codec */
static int
set_codec_attribute_from_incoming_media(const char *sdp_filename, void *state);

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
struct rtsp_state {
    char *nals;
    int nals_size;
    char *data; //nals + data
    uint32_t *in_codec;

    struct timeval t0, t;
    int frames;
    struct video_frame *frame;
    struct tile *tile;
	struct audio_frame audio;
    int width;
    int height;

    struct std_frame_received *rx_data;
    bool new_frame;
    bool decompress;
    bool grab;

    struct rtp *device;
    struct pdb *participants;
    struct pdb_e *cp;
    double rtcp_bw;
    int ttl;
    char *addr;
    char *mcast_if;
    struct timeval curr_time;
    struct timeval timeout;
    struct timeval prev_time;
    struct timeval start_time;
    int required_connections;
    uint32_t timestamp;

	int play_audio_frame;

	struct timeval last_audio_time;
	unsigned int grab_audio:1;

    pthread_t rtsp_thread_id; //the worker_id
    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;

    volatile bool should_exit;

    struct state_decompress *sd;
    struct video_desc des;
    char * out_frame;

    CURL *curl;
    char *uri;
    int port;
    float fps;
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
    if (tv_diff(now, s->prev_time) >= 20) {
        rtsp_get_parameters(s->curl, s->uri);
        gettimeofday(&s->prev_time, NULL);
    }
}

static void *
vidcap_rtsp_thread(void *arg) {
    struct rtsp_state *s;
    s = (struct rtsp_state *) arg;

    gettimeofday(&s->start_time, NULL);
    gettimeofday(&s->prev_time, NULL);

    while (!s->should_exit) {
        gettimeofday(&s->curr_time, NULL);
        s->timestamp = tv_diff(s->curr_time, s->start_time) * 90000;

        rtsp_keepalive(s);

        rtp_update(s->device, s->curr_time);
        //TODO no need of rtcp communication between ug and rtsp server?
        //rtp_send_ctrl(s->device, s->timestamp, 0, s->curr_time);

        s->timeout.tv_sec = 0;
        s->timeout.tv_usec = 10000;

        if (!rtp_recv_r(s->device, &s->timeout, s->timestamp)) {
            pdb_iter_t it;
            s->cp = pdb_iter_init(s->participants, &it);

            while (s->cp != NULL) {
                if (pthread_mutex_trylock(&s->lock) == 0) {
                    {
                        if(s->grab){

                            while (s->new_frame && !s->should_exit) {
                                s->worker_waiting = true;
                                pthread_cond_wait(&s->worker_cv, &s->lock);
                                s->worker_waiting = false;
                            }

                            if (pbuf_decode(s->cp->playout_buffer, s->curr_time,
                                decode_frame_h264, s->rx_data))
                            {
                                s->new_frame = true;
                            }
                            if (s->boss_waiting)
                                pthread_cond_signal(&s->boss_cv);
                        }
                    }
                    pthread_mutex_unlock(&s->lock);
                }
                pbuf_remove(s->cp->playout_buffer, s->curr_time);
                s->cp = pdb_iter_next(&it);
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

    if(pthread_mutex_trylock(&s->lock)==0){
        {
            s->grab = true;

            while (!s->new_frame) {
                s->boss_waiting = true;
                pthread_cond_wait(&s->boss_cv, &s->lock);
                s->boss_waiting = false;
            }

            gettimeofday(&s->curr_time, NULL);
            s->frame->h264_iframe = s->rx_data->iframe;
            s->frame->h264_iframe = s->rx_data->iframe;
            s->frame->tiles[0].data_len = s->rx_data->buffer_len;
            memcpy(s->data + s->nals_size, s->rx_data->frame_buffer,
                s->rx_data->buffer_len);
            memcpy(s->frame->tiles[0].data, s->data,
                s->rx_data->buffer_len + s->nals_size);
            s->frame->tiles[0].data_len += s->nals_size;

            if (s->decompress) {
                decompress_frame(s->sd, (unsigned char *) s->out_frame,
                    (unsigned char *) s->frame->tiles[0].data,
                    s->rx_data->buffer_len + s->nals_size, 0);
                s->frame->tiles[0].data = s->out_frame;               //TODO memcpy?
                s->frame->tiles[0].data_len = vc_get_linesize(s->des.width, UYVY)
                    * s->des.height;                           //TODO reconfigurable?
            }
            s->new_frame = false;

            if (s->worker_waiting) {
                pthread_cond_signal(&s->worker_cv);
            }
        }
        pthread_mutex_unlock(&s->lock);

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);
        if (seconds >= 5) {
            float fps = s->frames / seconds;
            fprintf(stderr, "[rtsp capture] %d frames in %g seconds = %g FPS\n",
                s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
            //TODO: Threshold of ¿1fps? in order to update fps parameter. Now a higher fps is fixed to 30fps...
            //if (fps > s->fps + 1 || fps < s->fps - 1) {
            //      debug_msg(
            //          "\n[rtsp] updating fps from rtsp server stream... now = %f , before = %f\n",fps,s->fps);
            //      s->frame->fps = fps;
            //      s->fps = fps;
            //  }
        }
        s->frames++;
        s->grab = false;
    }

    return s->frame;
}

void *
vidcap_rtsp_init(const struct vidcap_params *params) {

    struct rtsp_state *s;

    s = calloc(1, sizeof(struct rtsp_state));
    if (!s)
        return NULL;

    char *save_ptr = NULL;

    gettimeofday(&s->t0, NULL);
    s->frames = 0;
    s->nals = malloc(1024);
    s->grab = false;

    s->addr = "127.0.0.1";
    s->device = NULL;
    s->rtcp_bw = 5 * 1024 * 1024; /* FIXME */
    s->ttl = 255;

    s->mcast_if = NULL;
    s->required_connections = 1;

    s->timeout.tv_sec = 0;
    s->timeout.tv_usec = 10000;

    s->device = (struct rtp *) malloc(
        (s->required_connections) * sizeof(struct rtp *));
    s->participants = pdb_init();

    s->rx_data = malloc(sizeof(struct std_frame_received));
    s->new_frame = false;

    s->in_codec = malloc(sizeof(uint32_t *) * 10);

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
                        s->port = atoi(tmp);
                    } else {
                        printf("\n[rtsp] Wrong format for port! \n");
                        show_help();
                        exit(0);
                    }
                    break;
                case 2:
                    if (tmp) {  //TODO check if it's a number
                        s->width = atoi(tmp);
                    } else {
                        printf("\n[rtsp] Wrong format for width! \n");
                        show_help();
                        exit(0);
                    }
                    break;
                case 3:
                    if (tmp) {  //TODO check if it's a number
                        s->height = atoi(tmp);
                        //Now checking if we have user and password parameters...
                        if (s->height == 0) {
                            int ntmp = 0;
                            ntmp = s->width;
                            s->width = s->port;
                            s->height = ntmp;
                            sprintf(s->uri, "rtsp:%s", uri_tmp1);
                            s->port = atoi(uri_tmp2);
                            if (tmp) {
                                if (strcmp(tmp, "true") == 0)
                                    s->decompress = true;
                                else if (strcmp(tmp, "false") == 0)
                                    s->decompress = false;
                                else {
                                    printf(
                                        "\n[rtsp] Wrong format for boolean decompress flag! \n");
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
                            s->decompress = true;
                        else if (strcmp(tmp, "false") == 0)
                            s->decompress = false;
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
    if (s->height == 0) {
        int ntmp = 0;
        ntmp = s->width;
        s->width = (int) s->port;
        s->height = (int) ntmp;
        sprintf(s->uri, "rtsp:%s", uri_tmp1);
        s->port = (int) atoi(uri_tmp2);
    }

    debug_msg("[rtsp] selected flags:\n");
    debug_msg("\t  uri: %s\n",s->uri);
    debug_msg("\t  port: %d\n", s->port);
    debug_msg("\t  width: %d\n",s->width);
    debug_msg("\t  height: %d\n",s->height);
    debug_msg("\t  decompress: %d\n\n",s->decompress);

    if (uri_tmp1 != NULL)
        free(uri_tmp1);
    if (uri_tmp2 != NULL)
        free(uri_tmp2);

    s->rx_data->frame_buffer = malloc(4 * s->width * s->height);
    s->data = malloc(4 * s->width * s->height + s->nals_size);

    s->frame = vf_alloc(1);
    s->frame->isStd = TRUE;
    s->frame->h264_bframe = FALSE;
    s->frame->h264_iframe = FALSE;
    s->tile = vf_get_tile(s->frame, 0);
    vf_get_tile(s->frame, 0)->width = s->width;
    vf_get_tile(s->frame, 0)->height = s->height;
    //TODO fps should be autodetected, now reset and controlled at vidcap_grab function
    s->frame->fps = 30;
    s->fps = 30;
    s->frame->interlacing = PROGRESSIVE;

    s->frame->tiles[0].data = calloc(1, s->width * s->height);

    s->should_exit = false;

    s->device = rtp_init_if(NULL, s->mcast_if, s->port, 0, s->ttl, s->rtcp_bw,
        0, rtp_recv_callback, (void *) s->participants, 0, false);

    if (s->device != NULL) {
        if (!rtp_set_option(s->device, RTP_OPT_WEAK_VALIDATION, 1)) {
            debug_msg("[rtsp] RTP INIT - set option\n");
            return NULL;
        }
        if (!rtp_set_sdes(s->device, rtp_my_ssrc(s->device),
                RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING))) {
            debug_msg("[rtsp] RTP INIT - set sdes\n");
            return NULL;
        }

        int ret = rtp_set_recv_buf(s->device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
        if (!ret) {
            debug_msg("[rtsp] RTP INIT - set recv buf \nset command: sudo sysctl -w net.core.rmem_max=9123840\n");
            return NULL;
        }

        if (!rtp_set_send_buf(s->device, 1024 * 56)) {
            debug_msg("[rtsp] RTP INIT - set send buf\n");
            return NULL;
        }
        ret=pdb_add(s->participants, rtp_my_ssrc(s->device));
    }

    debug_msg("[rtsp] rtp receiver init done\n");

    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->boss_cv, NULL);
    pthread_cond_init(&s->worker_cv, NULL);

    s->boss_waiting = false;
    s->worker_waiting = false;

    s->nals_size = init_rtsp(s->uri, s->port, s, s->nals);

    if (s->nals_size >= 0)
        memcpy(s->data, s->nals, s->nals_size);
    else
        return NULL;

    if (s->decompress) {
        if (init_decompressor(s) == 0)
            return NULL;
        s->frame->color_spec = UYVY;
    }

    pthread_create(&s->rtsp_thread_id, NULL, vidcap_rtsp_thread, s);

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
    char transport[256];
    bzero(transport, 256);
    int port = rtsp_port;
    get_sdp_filename(url, sdp_filename);
    sprintf(transport, "RTP/AVP;unicast;client_port=%d-%d", port, port + 1);

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
            debug_msg("sdp_file: %s\n", sdp_filename);
            /* request session description and write response to sdp file */
            rtsp_describe(curl, uri, sdp_filename);
            debug_msg("sdp_file!!!!: %s\n", sdp_filename);
            /* get media control attribute from sdp file */
            get_media_control_attribute(sdp_filename, control);

            /* set incoming media codec attribute from sdp file */
            if (set_codec_attribute_from_incoming_media(sdp_filename, s) == 0)
                return -1;

            /* setup media stream */
            sprintf(uri, "%s/%s", url, control);
            rtsp_setup(curl, uri, transport);

            /* start playing media stream */
            sprintf(uri, "%s", url);
            rtsp_play(curl, uri, range);

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

/**
 * Initializes decompressor if required by decompress flag
 */
static int
init_decompressor(void *state) {
    struct rtsp_state *sr;
    sr = (struct rtsp_state *) state;

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

/**
 * scan sdp file for media control attribute
 */
static void
get_media_control_attribute(const char *sdp_filename, char *control) {
    int max_len = 1256;
    char *s = malloc(max_len);

    char *track = malloc(max_len);
    char *track_ant = malloc(max_len);

    FILE *sdp_fp = fopen(sdp_filename, "rt");
    //control[0] = '\0';
    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
            sscanf(s, " a = control: %s", track_ant);
            if (strcmp(track_ant, "") != 0) {
                track = strstr(track_ant, "track");
                if (track != NULL)
                    break;
            }
        }

        fclose(sdp_fp);
    }
    free(s);
    memcpy(control, track, strlen(track));
}

/**
 * scan sdp file for incoming codec and set it
 */
static int
set_codec_attribute_from_incoming_media(const char *sdp_filename, void *state) {
    struct rtsp_state *sr;
    sr = (struct rtsp_state *) state;

    int max_len = 1256;
    char *pt = malloc(4 * sizeof(char *));
    char *codec = malloc(32 * sizeof(char *));
    char *s = malloc(max_len);
    FILE *sdp_fp = fopen(sdp_filename, "rt");

    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
            sscanf(s, " m = video 0 RTP/AVP %s", pt);
        }
        fclose(sdp_fp);
    }
    free(s);

    s = malloc(max_len);
    sdp_fp = fopen(sdp_filename, "rt");

    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
            sscanf(s, " a=rtpmap:96 %s", codec);
        }
        fclose(sdp_fp);
    }
    free(s);

    char *save_ptr = NULL;
    char *tmp;
    tmp = strtok_r(codec, "/", &save_ptr);
    if (!tmp) {
        fprintf(stderr, "[rtsp] no codec atribute found into sdp file...\n");
        return 0;
    }
    sprintf((char *) sr->in_codec, "%s", tmp);

    if (memcmp(sr->in_codec, "H264", 4) == 0)
        sr->frame->color_spec = H264;
    else if (memcmp(sr->in_codec, "VP8", 3) == 0)
        sr->frame->color_spec = VP8;
    else
        sr->frame->color_spec = RGBA;
    return 1;
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
    pthread_join(s->rtsp_thread_id, NULL);

    free(s->rx_data->frame_buffer);
    free(s->data);

    rtsp_teardown(s->curl, s->uri);

    curl_easy_cleanup(s->curl);
    s->curl = NULL;

    if (s->decompress)
        decompress_done(s->sd);
    rtp_done(s->device);

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
