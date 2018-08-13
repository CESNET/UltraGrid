/*
 * FILE:    sdp.c
 * AUTHORS: Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2018 CESNET, z. s. p. o.
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
 * @todo
 * * this file structure is now a bit messy
 * * exit correctly HTTP thread (but it is a bit tricky because it waits on accept())
 * * allow use of different HTTP ports than 8080
 * * IPv6 support (?)
 * * createResponseForRequest() should be probably static (in case that other
 *   modules want also to use EmbeddableWebServer
 * * HTTP server should work even if the SDP file cannot be written
 * * MSW support (HTTP server)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "debug.h"
#include "utils/net.h"
#include "utils/sdp.h"
#ifdef SDP_HTTP
#include "EmbeddableWebServer.h"
#endif // SDP_HTTP

#define SDP_FILE "ug.sdp"

//TODO could be a vector of many streams
static struct sdp *sdp_global;

struct sdp *new_sdp(enum rtp_standard std, int port){
    struct sdp *sdp;
    sdp = NULL;
    sdp = calloc(1, sizeof(struct sdp));
    sdp->stream_count = 0;
    sdp->std_rtp = std;
    sdp->port = port;
    switch (std){
        case 0: //H264
            set_version(sdp);
            set_origin(sdp);
            set_session_name(sdp);
            set_connection(sdp);
            set_times(sdp);
            set_stream(sdp);
            return sdp;
            break;
        default://UNKNOWN
            free(sdp);
            return NULL;
    }
}

static void strappend(char **dst, size_t *dst_alloc_len, const char *src)
{
    if (*dst_alloc_len < strlen(*dst) + strlen(src) + 1) {
        *dst_alloc_len = strlen(*dst) + strlen(src) + 1;
        *dst = realloc(*dst, *dst_alloc_len);
    }
    strcat(*dst, src);
}

bool gen_sdp(struct sdp *sdp){
    size_t len = 1;
    char *buf = calloc(1, 1);
    strappend(&buf, &len, sdp->version);
    strappend(&buf, &len, sdp->origin);
    strappend(&buf, &len, sdp->session_name);
    strappend(&buf, &len, sdp->connection);
    strappend(&buf, &len, sdp->times);
    for (int i = 0; i < sdp->stream_count; ++i) {
        strappend(&buf, &len, sdp->stream[i].media_info);
        strappend(&buf, &len, sdp->stream[i].rtpmap);
    }
    strappend(&buf, &len, "\n\n");
    sdp->sdp_dump = buf;

    FILE *fOut = fopen (SDP_FILE, "w+");
    if (fOut == NULL) {
        log_msg(LOG_LEVEL_ERROR, "Unable to write SDP file\n");
    } else {
        if (fprintf(fOut, "%s", buf) != (int) strlen(buf)) {
            perror("fprintf");
        } else {
            printf("[SDP] File " SDP_FILE " created.\n");
        }
        fclose(fOut);
    }

    printf("Printed version:\n%s", buf);
    return true;
}

void set_version(struct sdp *sdp){
    sdp->version = malloc(strLength);
    sdp->version = "v=0\n";
}
void get_version(struct sdp *sdp);

void set_origin(struct sdp *sdp){
    sdp->origin = malloc(strLength);
    sdp->origin = "o=- 0 0 IN IP4 127.0.0.1\n";
}
void get_origin(struct sdp *sdp);

void set_session_name(struct sdp *sdp){
    sdp->session_name = malloc(strLength);
    sdp->session_name = "s=Ultragrid streams\n";
}
void get_session_name(struct sdp *sdp);

void set_connection(struct sdp *sdp){
    sdp->connection =malloc(strLength);
    sdp->connection = "c=IN IP4 127.0.0.1\n";
}
void get_connection(struct sdp *sdp);

void set_times(struct sdp *sdp){
    sdp->times = malloc(strLength);
    sdp->times = "t=0 0\n";
}
void get_times(struct sdp *sdp);

void set_stream(struct sdp *sdp){
    if(new_stream(sdp)){
    }
    else{
        printf("[SDP] stream NOT added -> error: maximum stream definition reached\n");
    }

}
void get_stream(struct sdp *sdp, int index);


bool new_stream(struct sdp *sdp){
    //struct stream_info *stream;
    if(sdp->stream_count < MAX_STREAMS){
        sdp->stream_count++;
        set_stream_media_info(sdp, sdp->stream_count - 1);
        set_stream_rtpmap(sdp, sdp->stream_count - 1);

        return true;
    }
    return true;
}

char *set_stream_media_info(struct sdp *sdp, int index){
    debug_msg("[SDP] SETTING MEDIA INFO    \n\n");

    sprintf(sdp->stream[index].media_info,"m=video %d RTP/AVP 96\n",sdp->port);

    return sdp->stream[index].media_info;
}

char *set_stream_rtpmap(struct sdp *sdp, int index){
    debug_msg("[SDP] SETTING RTPMAP INFO    \n\n");
    strcpy(sdp->stream[index].rtpmap, "a=rtpmap:96 H264/90000\n");

    return sdp->stream[index].rtpmap;
}

void clean_sdp(struct sdp *sdp){
    if (!sdp) {
            return;
    }
    free(sdp->sdp_dump);
    free(sdp);
}


// --------------------------------------------------------------------
// HTTP server stuff
// --------------------------------------------------------------------
#ifdef SDP_HTTP
struct Response* createResponseForRequest(const struct Request* request, struct Connection* connection) {
    UNUSED(connection);
    if (strlen(request->pathDecoded) > 1 && request->pathDecoded[0] == '/' && strcmp(request->pathDecoded + 1, SDP_FILE) == 0) {
	return responseAllocWithFile(SDP_FILE, "application/sdp");
    }
    return responseAlloc404NotFoundHTML(request->pathDecoded);
}

static const uint16_t portInHostOrder = 8080;

static THREAD_RETURN_TYPE STDCALL_ON_WIN32 acceptConnectionsThread(void* param) {
    acceptConnectionsUntilStoppedFromEverywhereIPv4(param, portInHostOrder);
    log_msg(LOG_LEVEL_WARNING, "Warning: HTTP/SDP thread has exited.\n");
    return (THREAD_RETURN_TYPE) 0;
}

static void print_http_path() {
    struct sockaddr_in addrs[20];
    size_t len = sizeof addrs;
    if (get_local_ipv4_addresses(addrs, &len)) {
        for (size_t i = 0; i < len / sizeof addrs[0]; ++i) {
            if (!is_addr_loopback((struct sockaddr *) &addrs[i])) {
                char hostname[256];
                getnameinfo((struct sockaddr *) &addrs[i], sizeof(struct sockaddr_in), hostname, sizeof(hostname), NULL, 0, NI_NUMERICHOST);
                log_msg(LOG_LEVEL_NOTICE, "Receiver can play SDP with URL http://%s:%u/%s\n", hostname, portInHostOrder, SDP_FILE);
            }
        }
    }
}

bool sdp_run_http_server(struct sdp *sdp)
{
    struct Server *http_server = calloc(1, sizeof(struct Server));
    sdp_global = sdp;
    serverInit(http_server);
    pthread_t http_server_thr;
    pthread_create(&http_server_thr, NULL, &acceptConnectionsThread, http_server);
    pthread_detach(http_server_thr);
    // some resource will definitely leak but it shouldn't be a problem
    print_http_path();
    return true;
}
#endif // SDP_HTTP

/* vim: set expandtab sw=4 : */
