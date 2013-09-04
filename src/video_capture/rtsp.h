/* AUTHORS:   Gerard Castillo <gerard.castillo@i2cat.net>,
 *            Martin German <martin.german@i2cat.net>
 *
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2011, Jim Hollinger
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Jim Hollinger nor the names of its contributors
 *     may be used to endorse or promote products derived from this
 *     software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef _RTSP_H_
#define _RTSP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <glib.h>

#if defined (WIN32)
#  include <conio.h>  /* _getch() */
#else
#  include <termios.h>
#  include <unistd.h>

/*
   static int _getch(void)
   {
   struct termios oldt, newt;
   int ch;
   tcgetattr( STDIN_FILENO, &oldt );
   newt = oldt;
   newt.c_lflag &= ~( ICANON | ECHO );
   tcsetattr( STDIN_FILENO, TCSANOW, &newt );
   ch = getchar();
   tcsetattr( STDIN_FILENO, TCSANOW, &oldt );
   return ch;
   }
*/
#endif

#include <curl/curl.h>

#define VERSION_STR  "V1.0"

#define VIDCAP_RTSP_ID	0x45b3d828  //md5 hash of VIDCAP_RTSP_ID string == a208d26f519a2664a48781c845b3d828

//TODO set lower initial video recv buffer size (to find the minimal)
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((4*1920*1080)*110/100) //command line net.core setup: sysctl -w net.core.rmem_max=9123840

//#define MAX_SUBSTREAMS 1

struct recieved_data{
    uint32_t buffer_len;//[MAX_SUBSTREAMS];
    //uint32_t buffer_num;//[MAX_SUBSTREAMS];
    char *frame_buffer;//[MAX_SUBSTREAMS];
};

struct rtsp_state *s_global;

/* error handling macros */
#define my_curl_easy_setopt(A, B, C) \
    if ((res = curl_easy_setopt((A), (B), (C))) != CURLE_OK) \
fprintf(stderr, "curl_easy_setopt(%s, %s, %s) failed: %d\n", \
#A, #B, #C, res);

#define my_curl_easy_perform(A) \
    if ((res = curl_easy_perform((A))) != CURLE_OK) \
fprintf(stderr, "curl_easy_perform(%s) failed: %d\n", #A, res);



/* send RTSP GET_PARAMETERS request */
void rtsp_get_parameters(CURL *curl, const char *uri);

/* send RTSP OPTIONS request */
void rtsp_options(CURL *curl, const char *uri);

/* send RTSP DESCRIBE request and write sdp response to a file */
void rtsp_describe(CURL *curl, const char *uri, const char *sdp_filename);

/* send RTSP SETUP request */
void rtsp_setup(CURL *curl, const char *uri, const char *transport);

/* send RTSP PLAY request */
void rtsp_play(CURL *curl, const char *uri, const char *range);

/* send RTSP TEARDOWN request */
void rtsp_teardown(CURL *curl, const char *uri);

/* convert url into an sdp filename */
void get_sdp_filename(const char *url, char *sdp_filename);

/* scan sdp file for media control attribute */
void get_media_control_attribute(const char *sdp_filename, char *control);

/* scan sdp file for incoming codec */
void set_codec_attribute_from_incoming_media(const char *sdp_filename, void *state);

int get_nals(const char *sdp_filename, unsigned char *nals);

/* sigaction handler that forces teardown on current session */
void sigaction_handler();

int init_rtsp(char* rtsp_uri, int rtsp_port,void *state, unsigned char* nals);

int init_decompressor(void *state);

/* sigaction handler that forces teardown on current session */
void sigaction_handler();

struct vidcap_type	*vidcap_rtsp_probe(void);
void			*vidcap_rtsp_init(const struct vidcap_params *params);
void			 vidcap_rtsp_done(void *state);
struct video_frame	*vidcap_rtsp_grab(void *state, struct audio_frame **audio);

#endif

#ifdef __cplusplus
} // END extern "C"
#endif
