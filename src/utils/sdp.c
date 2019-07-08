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
 * * exit correctly HTTP thread (but it is a bit tricky because it waits on accept())
 * * createResponseForRequest() should be probably static (in case that other
 *   modules want also to use EmbeddableWebServer)
 * * HTTP server should work even if the SDP file cannot be written
 * * at least some Windows compatibility functions should be perhaps deleted from
 *   EmbeddableWebServer, eg. pthread_* which we have from winpthreads, either.
 *   This can also be potentially dangerous.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
// config_win32.h must not be included if using EWS because EWS has
// some incomatible implementations of POSIX functions
#endif

#ifdef WIN32
#define _WIN32_WINNT 0x0600
#include <stdint.h>
#include <winsock2.h>
#endif

#include "audio/types.h"
#include "debug.h"
#include "rtp/rtp_types.h"
#include "types.h"
#include "utils/fs.h"
#include "utils/net.h"
#include "utils/sdp.h"
#ifdef SDP_HTTP
#define EWS_DISABLE_SNPRINTF_COMPAT
#include "EmbeddableWebServer.h"
#endif // SDP_HTTP

#define SDP_FILE "ug.sdp"

#define MAX_STREAMS 2
#define STR_LENGTH 2048

struct stream_info {
    char media_info[STR_LENGTH];
    char rtpmap[STR_LENGTH];
    char fmtp[STR_LENGTH];
};

struct sdp {
    int ip_version;
    char version[STR_LENGTH];
    char origin[STR_LENGTH];
    char session_name[STR_LENGTH];
    char connection[STR_LENGTH];
    char times[STR_LENGTH];
    struct stream_info stream[MAX_STREAMS];
    int stream_count; //between 1 and MAX_STREAMS
    char *sdp_dump;
};

// this is needed for HTTP server
static struct sdp *sdp_global;

struct sdp *new_sdp(int ip_version, const char *receiver) {
    assert(ip_version == 4 || ip_version == 6);
    struct sdp *sdp;
    sdp = calloc(1, sizeof(struct sdp));
    assert(sdp != NULL);
    sdp->ip_version = ip_version;
    const char *ip_loopback;
    if (ip_version == 4) {
        ip_loopback = "127.0.0.1";
    } else {
        ip_loopback = "::1";
    }
    char hostname[256];
    const char *connection_address = ip_loopback;
    const char *origin_address = ip_loopback;
    struct sockaddr_storage addrs[20];
    size_t len = sizeof addrs;
    if (get_local_addresses(addrs, &len, ip_version)) {
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
    snprintf(sdp->origin, STR_LENGTH, "o=- 0 0 IN IP%d %s\n", ip_version, origin_address);
    strncpy(sdp->session_name, "s=Ultragrid streams\n", STR_LENGTH - 1);
    snprintf(sdp->connection, STR_LENGTH, "c=IN IP%d %s\n", ip_version, connection_address);
    strncpy(sdp->times, "t=0 0\n", STR_LENGTH - 1);

    return sdp;
}

static int new_stream(struct sdp *sdp){
    if (sdp->stream_count < MAX_STREAMS){
        sdp->stream_count++;
        return sdp->stream_count - 1;
    }
    return -1;
}

/**
 * @retval  0 ok
 * @retval -1 too much streams
 * @retval -2 unsupported codec
 */
int sdp_add_audio(struct sdp *sdp, int port, int sample_rate, int channels, audio_codec_t codec)
{
    int index = new_stream(sdp);
    if (index < 0) {
        return -1;
    }
    int pt = PT_DynRTP_Type97; // default

    if (sample_rate == 8000 && channels == 1 && (codec == AC_ALAW || codec == AC_MULAW)) {
	pt = codec == AC_MULAW ? PT_ITU_T_G711_PCMU : PT_ITU_T_G711_PCMA;
    }
    sprintf(sdp->stream[index].media_info, "m=audio %d RTP/AVP %d\n", port, pt);
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
		audio_codec = "OPUS";
                ts_rate = 48000; // RFC 7587 specifies always 48 kHz for OPUS
		break;
            default:
                log_msg(LOG_LEVEL_ERROR, "[SDP] Currently only PCMA, PCMU and OPUS audio codecs are supported!\n");
                return -2;
	}

	snprintf(sdp->stream[index].rtpmap, STR_LENGTH, "a=rtpmap:%d %s/%i/%i\n", PT_DynRTP_Type97, audio_codec, ts_rate, channels);
    }

    return 0;
}

/**
 * @retval  0 ok
 * @retval -1 too much streams
 * @retval -2 unsupported codec
 */
int sdp_add_video(struct sdp *sdp, int port, codec_t codec)
{
    if (codec != H264 && codec != JPEG && codec != MJPG) {
        return -2;
    }

    int index = new_stream(sdp);
    if (index < 0) {
        return -1;
    }
    snprintf(sdp->stream[index].media_info, STR_LENGTH, "m=video %d RTP/AVP %d\n", port, codec == H264 ? PT_H264 : PT_JPEG);
    if (codec == H264) {
        snprintf(sdp->stream[index].rtpmap, STR_LENGTH, "a=rtpmap:%d H264/90000\n", PT_H264);
    }
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
    strappend(&buf, &len, "\n");
    sdp->sdp_dump = buf;

    char *sdp_file_name = alloca(strlen(SDP_FILE) + strlen(get_temp_dir()) + 1);
    strcpy(sdp_file_name, get_temp_dir());
    strcat(sdp_file_name, SDP_FILE);
    FILE *fOut = fopen(sdp_file_name, "w");
    if (fOut == NULL) {
        log_msg(LOG_LEVEL_ERROR, "Unable to write SDP file\n");
    } else {
        if (fprintf(fOut, "%s", buf) != (int) strlen(buf)) {
            perror("fprintf");
        } else {
            printf("[SDP] File %s created.\n", sdp_file_name);
        }
        fclose(fOut);
    }

    printf("Printed version:\n%s", buf);
    return true;
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
        char *sdp_file_name = alloca(strlen(SDP_FILE) + strlen(get_temp_dir()) + 1);
        strcpy(sdp_file_name, get_temp_dir());
        strcat(sdp_file_name, SDP_FILE);
	return responseAllocWithFile(sdp_file_name, "application/sdp");
    }
    return responseAlloc404NotFoundHTML(request->pathDecoded);
}

static uint16_t portInHostOrder;

static THREAD_RETURN_TYPE STDCALL_ON_WIN32 acceptConnectionsThread(void* param) {
    struct sockaddr_storage ss = { 0 };
    ss.ss_family = sdp_global->ip_version == 4 ? AF_INET : AF_INET6;
    size_t sa_len = sdp_global->ip_version == 6 ? sizeof(struct sockaddr_in6) : sizeof(struct sockaddr_in);
    if (sdp_global->ip_version == 4) {
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

static void print_http_path() {
    struct sockaddr_storage addrs[20];
    size_t len = sizeof addrs;
    if (get_local_addresses(addrs, &len, sdp_global->ip_version)) {
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

bool sdp_run_http_server(struct sdp *sdp, int port)
{
    assert(port >= 0 && port < 65536);
    portInHostOrder = port;
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
