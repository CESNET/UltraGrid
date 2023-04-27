/*
 * FILE:    utils/sdp.c
 * AUTHORS: Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2018-2023 CESNET, z. s. p. o.
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
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 * @todo
 * * consider using serverStop() to stop the thread - likely doesn't work now
 * * createResponseForRequest() should be probably static (in case that other
 *   modules want also to use EmbeddableWebServer)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef WIN32
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "audio/types.h"
#include "debug.h"
#include "rtp/rtp_types.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/fs.h"
#include "utils/misc.h"
#include "utils/net.h"
#include "utils/sdp.h"
#ifdef SDP_HTTP
#define EWS_DISABLE_SNPRINTF_COMPAT
#include "EmbeddableWebServer.h"
#endif // SDP_HTTP

#define MOD_NAME "[SDP] "
#define SDP_FILE "ug.sdp"

enum {
    DEFAULT_SDP_HTTP_PORT = 8554,
    MAX_STREAMS = 2,
    STR_LENGTH = 2048,
};

bool autorun;
int requested_http_port = DEFAULT_SDP_HTTP_PORT;
static bool want_sdp_audio = false;
static bool want_sdp_video = false;
static char sdp_receiver[1024];
static char sdp_filename[MAX_PATH_SIZE];

static struct sdp *sdp_state = NULL;

struct stream_info {
    char media_info[STR_LENGTH];
    char rtpmap[STR_LENGTH];
    char fmtp[STR_LENGTH];
};

struct sdp {
    bool audio_set;
    bool video_set;
#ifdef SDP_HTTP
    struct Server http_server;
#endif // defined SDP_HTTP
    pthread_t http_server_thr;
    int ip_version;
    char version[STR_LENGTH];
    char origin[STR_LENGTH];
    char session_name[STR_LENGTH];
    char connection[STR_LENGTH];
    char times[STR_LENGTH];
    struct stream_info stream[MAX_STREAMS];
    int stream_count; //between 1 and MAX_STREAMS
    char *sdp_dump;
    void (*address_callback)(void *udata, const char *address);
    void *address_callback_udata;
};

static bool gen_sdp(void);
static struct sdp *new_sdp(bool ipv6, const char *receiver);
static bool sdp_run_http_server(struct sdp *sdp, int port);
static void sdp_stop_http_server(struct sdp *sdp);
static void clean_sdp(struct sdp *sdp);

static struct sdp *new_sdp(bool ipv6, const char *receiver) {
    struct sdp *sdp;
    sdp = calloc(1, sizeof(struct sdp));
    assert(sdp != NULL);
    sdp->ip_version = ipv6 ? 6 : 4;
    const char *ip_loopback;
    if (sdp->ip_version == 6) {
        ip_loopback = "::1";
    } else {
        ip_loopback = "127.0.0.1";
    }
    char hostname[256];
    const char *connection_address = ip_loopback;
    const char *origin_address = ip_loopback;
    struct sockaddr_storage addrs[20];
    size_t len = sizeof addrs;
    if (get_local_addresses(addrs, &len, sdp->ip_version)) {
        for (int i = 0; i < (int)(len / sizeof addrs[0]); ++i) {
            if (!is_addr_linklocal((struct sockaddr *) &addrs[i]) && !is_addr_loopback((struct sockaddr *) &addrs[i])) {
                bool ipv6 = addrs[i].ss_family == AF_INET6;
                size_t sa_len = ipv6 ? sizeof(struct sockaddr_in6) : sizeof(struct sockaddr_in);
                getnameinfo((struct sockaddr *) &addrs[i], sa_len, hostname, sizeof(hostname), NULL, 0, NI_NUMERICHOST);
                origin_address = hostname;
                break;
            }
        }
    }
    if (is_addr_multicast(receiver)) {
        connection_address = receiver;
    }
    strncpy(sdp->version, "v=0\n", STR_LENGTH - 1);
    snprintf(sdp->origin, STR_LENGTH, "o=- 0 0 IN IP%d %s\n", sdp->ip_version, origin_address);
    strncpy(sdp->session_name, "s=Ultragrid streams\n", STR_LENGTH - 1);
    snprintf(sdp->connection, STR_LENGTH, "c=IN IP%d %s\n", sdp->ip_version, connection_address);
    strncpy(sdp->times, "t=0 0\n", STR_LENGTH - 1);

#ifdef SDP_HTTP
    serverInit(&sdp->http_server);
#endif // defined SDP_HTTP

    return sdp;
}

static int new_stream(struct sdp *sdp){
    if (sdp->stream_count < MAX_STREAMS){
        sdp->stream_count++;
        return sdp->stream_count - 1;
    }
    return -1;
}

static void cleanup() {
#ifdef SDP_HTTP
    sdp_stop_http_server(sdp_state);
#endif // defined SDP_HTTP
    clean_sdp(sdp_state);
}

static void start() {
    // either SDP properties not set or not all streams already configured
    if (!sdp_state || want_sdp_audio != sdp_state->audio_set || want_sdp_video != sdp_state->video_set) {
        return;
    }
    if (!gen_sdp()) {
        log_msg(LOG_LEVEL_ERROR, "[SDP] File creation failed\n");
        return;
    }
#ifdef SDP_HTTP
    if (!sdp_run_http_server(sdp_state, requested_http_port)) {
        log_msg(LOG_LEVEL_ERROR, "[SDP] Server run failed!\n");
    }
#else
    log_msg(LOG_LEVEL_WARNING, "[SDP] HTTP support not enabled - skipping server creation!\n");
#endif

    atexit(cleanup);
}

/**
 * @retval  0 ok
 * @retval -1 too much streams
 * @retval -2 unsupported codec
 */
int sdp_add_audio(bool ipv6, int port, int sample_rate, int channels, audio_codec_t codec)
{
    if (!sdp_state) {
        sdp_state = new_sdp(ipv6, sdp_receiver);
        if (!sdp_state) {
                assert(0 && "[SDP] SDP creation failed\n");
        }
    }
    int index = new_stream(sdp_state);
    if (index < 0) {
        return -1;
    }
    int pt = PT_DynRTP_Type97; // default

    if (sample_rate == 8000 && channels == 1 && (codec == AC_ALAW || codec == AC_MULAW)) {
	pt = codec == AC_MULAW ? PT_ITU_T_G711_PCMU : PT_ITU_T_G711_PCMA;
    }
    snprintf(sdp_state->stream[index].media_info, sizeof sdp_state->stream[index].media_info, "m=audio %d RTP/AVP %d\n", port, pt);
    if (pt == PT_DynRTP_Type97) { // we need rtpmap for our dynamic packet type
	const char *audio_codec = NULL;
        int ts_rate = sample_rate; // equals for PCMA/PCMU
	switch (codec) {
	    case AC_ALAW:
		audio_codec = "PCMA";
		break;
	    case AC_MULAW:
		audio_codec = "PCMU";
		break;
	    case AC_OPUS:
		audio_codec = "opus";
                ts_rate = 48000; // RFC 7587 specifies always 48 kHz for Opus
		break;
            default:
                log_msg(LOG_LEVEL_ERROR, "[SDP] Currently only PCMA, PCMU and Opus audio codecs are supported!\n");
                return -2;
	}

	snprintf(sdp_state->stream[index].rtpmap, STR_LENGTH, "a=rtpmap:%d %s/%i/%i\n", PT_DynRTP_Type97, audio_codec, ts_rate, channels);
    }
    sdp_state->audio_set = true;
    start();

    return 0;
}

/**
 * @retval  0 ok
 * @retval -1 too much streams
 * @retval -2 unsupported codec
 */
int sdp_add_video(bool ipv6, int port, codec_t codec, address_callback_t addr_callback, void *addr_callback_udata)
{
    if (codec != H264 && codec != JPEG && codec != MJPG) {
        return -2;
    }
    if (!sdp_state) {
        sdp_state = new_sdp(ipv6, sdp_receiver);
        if (!sdp_state) {
                assert(0 && "[SDP] SDP creation failed\n");
        }
    }
    sdp_state->address_callback = addr_callback;
    sdp_state->address_callback_udata = addr_callback_udata;

    int index = new_stream(sdp_state);
    if (index < 0) {
        return -1;
    }
    snprintf(sdp_state->stream[index].media_info, STR_LENGTH, "m=video %d RTP/AVP %d\n", port, codec == H264 ? PT_DynRTP_Type96 : PT_JPEG);
    if (codec == H264) {
        snprintf(sdp_state->stream[index].rtpmap, STR_LENGTH, "a=rtpmap:%d H264/90000\n", PT_DynRTP_Type96);
    }

    sdp_state->video_set = true;
    start();
    return 0;
}

static void strappend(char **dst, size_t *dst_alloc_len, const char *src)
{
    if (*dst_alloc_len < strlen(*dst) + strlen(src) + 1) {
        *dst_alloc_len = strlen(*dst) + strlen(src) + 1;
        *dst = realloc(*dst, *dst_alloc_len);
    }
    strncat(*dst, src, *dst_alloc_len - strlen(*dst) - 1);
}

static bool gen_sdp() {
    size_t len = 1;
    char *buf = calloc(1, 1);
    strappend(&buf, &len, sdp_state->version);
    strappend(&buf, &len, sdp_state->origin);
    strappend(&buf, &len, sdp_state->session_name);
    strappend(&buf, &len, sdp_state->connection);
    strappend(&buf, &len, sdp_state->times);
    for (int i = 0; i < sdp_state->stream_count; ++i) {
        strappend(&buf, &len, sdp_state->stream[i].media_info);
        strappend(&buf, &len, sdp_state->stream[i].rtpmap);
    }
    strappend(&buf, &len, "\n");

    printf("Printed version:\n%s", buf);

    sdp_state->sdp_dump = buf;
    if (strcmp(sdp_filename, "no") == 0) {
        return true;
    }

    if (strlen(sdp_filename) == 0) {
        strncpy(sdp_filename, get_temp_dir(), sizeof sdp_filename - 1);
        strncat(sdp_filename, SDP_FILE, sizeof sdp_filename - strlen(sdp_filename) - 1);
    }
    FILE *fOut = fopen(sdp_filename, "w");
    if (fOut == NULL) {
        log_msg(LOG_LEVEL_ERROR, "Unable to write SDP file %s: %s\n", sdp_filename, ug_strerror(errno));
    } else {
        if (fprintf(fOut, "%s", buf) != (int) strlen(buf)) {
            perror("fprintf");
        } else {
            printf("[SDP] File %s created.\n", sdp_filename);
        }
        fclose(fOut);
    }

    return true;
}

void clean_sdp(struct sdp *sdp){
    if (!sdp) {
            return;
    }
    free(sdp->sdp_dump);
#ifdef SDP_HTTP
    serverDeInit(&sdp->http_server);
#endif // defined SDP_HTTP
    free(sdp);
}


// --------------------------------------------------------------------
// HTTP server stuff
// --------------------------------------------------------------------
#ifdef SDP_HTTP

#define ROBOTS_TXT "User-agent: *\nDisallow: /\n"
#define SECURITY_TXT "Contact: http://www.ultragrid.cz/contact\n"

struct Response* createResponseForRequest(const struct Request* request, struct Connection* connection) {
    struct sdp *sdp = connection->server->tag;

    if (autorun && sdp->address_callback){
        sdp->address_callback(sdp->address_callback_udata, connection->remoteHost);
    }

    log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Requested %s.\n", request->pathDecoded);

    if (strcasecmp(request->pathDecoded, "/robots.txt") == 0 ||
            strcasecmp(request->pathDecoded, "/.well-known/security.txt") == 0 ||
            strcasecmp(request->pathDecoded, "/security.txt") == 0) {
        struct Response* response = responseAlloc(200, "OK", "text/plain", 0);
        heapStringSetToCString(&response->body, strcasecmp(request->pathDecoded, "/robots.txt") == 0 ? ROBOTS_TXT : SECURITY_TXT);
        return response;
    }

    log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Returning the SDP.\n");
    const char *sdp_content = sdp->sdp_dump;
    struct Response* response = responseAlloc(200, "OK", "application/sdp", 0);
    heapStringSetToCString(&response->body, sdp_content);
    return response;
}

static uint16_t portInHostOrder;

static THREAD_RETURN_TYPE STDCALL_ON_WIN32 acceptConnectionsThread(void* param) {
    struct sockaddr_storage ss = { 0 };
    struct sdp *sdp = ((struct Server *) param)->tag;
    ss.ss_family = sdp->ip_version == 4 ? AF_INET : AF_INET6;
    size_t sa_len = sdp->ip_version == 4 ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    if (sdp->ip_version == 4) {
        struct sockaddr_in *sin = (struct sockaddr_in *) &ss;
        sin->sin_addr.s_addr = htonl(INADDR_ANY);
        sin->sin_port = htons(portInHostOrder);
    } else {
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *) &ss;
        sin6->sin6_addr = in6addr_any;
        sin6->sin6_port = htons(portInHostOrder);
    }
    acceptConnectionsUntilStopped(param, (struct sockaddr *) &ss, sa_len);
    log_msg(LOG_LEVEL_WARNING, "Warning: HTTP/SDP thread has exited.\n");
    return (THREAD_RETURN_TYPE) 0;
}

static void print_http_path(struct sdp *sdp) {
    struct sockaddr_storage addrs[20];
    size_t len = sizeof addrs;
    if (get_local_addresses(addrs, &len, sdp->ip_version)) {
        bool found_public_ip = false;
        for (size_t i = 0; i < len / sizeof addrs[0]; ++i) {
            if (!is_addr_loopback((struct sockaddr *) &addrs[i]) && !is_addr_linklocal((struct sockaddr *) &addrs[i])) {
                found_public_ip = true;
            }
        }
        for (size_t i = 0; i < len / sizeof addrs[0]; ++i) {
            if (!found_public_ip || (!is_addr_loopback((struct sockaddr *) &addrs[i]) && !is_addr_linklocal((struct sockaddr *) &addrs[i]))) {
                char hostname[256];
                bool ipv6 = addrs[i].ss_family == AF_INET6;
                size_t sa_len = ipv6 ? sizeof(struct sockaddr_in6) : sizeof(struct sockaddr_in);
                getnameinfo((struct sockaddr *) &addrs[i], sa_len, hostname, sizeof(hostname), NULL, 0, NI_NUMERICHOST);
                log_msg(LOG_LEVEL_NOTICE, "Receiver can play SDP with URL http://%s%s%s:%u/%s\n", ipv6 ? "[" : "", hostname, ipv6 ? "]" : "", portInHostOrder, SDP_FILE);
            }
        }
    }
}

static bool sdp_run_http_server(struct sdp *sdp, int port)
{
    assert(port >= 0 && port < 65536);
    assert(sdp->sdp_dump != NULL);

    portInHostOrder = port;
    sdp->http_server.tag = sdp;
    pthread_create(&sdp->http_server_thr, NULL, &acceptConnectionsThread, &sdp->http_server);
    // some resource will definitely leak but it shouldn't be a problem
    print_http_path(sdp);
    return true;
}

void sdp_stop_http_server(struct sdp *sdp)
{
    ///@todo use "serverStop(&sdp->http_server);" instead
    sdp->http_server.shouldRun = false;
    pthread_cancel(sdp->http_server_thr);

    pthread_join(sdp->http_server_thr, NULL);
}
#endif // defined SDP_HTTP

void sdp_set_properties(const char *receiver, bool has_sdp_video, bool has_sdp_audio)
{
    strncpy(sdp_receiver, receiver, sizeof sdp_receiver - 1);
    want_sdp_audio = has_sdp_audio;
    want_sdp_video = has_sdp_video;

    start();
}

int sdp_set_options(const char *opts) {
    if (strcmp(opts, "help") == 0) {
        color_printf("Usage:\n");
        color_printf("\t" TBOLD("uv " TRED("--protocol sdp") "[:autorun][:file=<name>|no][:port=<http_port>]") "\n");
        color_printf("where:\n");
        color_printf("\t" TBOLD("autorun") " - automatically send to the address that requested the SDP over HTTP without giving an address (use with caution!)\n");
        return 1;
    }

    char opts_c[strlen(opts) + 1];
    char *tmp = opts_c;
    strcpy(opts_c, opts);
    char *item, *save_ptr;
    while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
        if (strstr(item, "port=") == item) {
            requested_http_port = atoi(strchr(item, '=') + 1);
        } else if (strstr(item, "file=") == item) {
            strncpy(sdp_filename, strchr(item, '=') + 1, sizeof sdp_filename - 1);
        } else if (strstr(item, "autorun") == item) {
            autorun = true;
        } else {
            log_msg(LOG_LEVEL_ERROR, "[SDP] Wrong option: %s\n", item);
            return -1;
        }
        tmp = NULL;
    }
    return 0;
}


/* vim: set expandtab sw=4 : */
