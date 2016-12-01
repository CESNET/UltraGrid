/**
 * @file   host.h
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file contains common (global) variables and functions.
 */
/*
 * Copyright (c) 2005-2014 CESNET z.s.p.o.
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
 *
 */
#ifndef __host_h
#define __host_h

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#define EXIT_FAIL_USAGE        2
#define EXIT_FAIL_UI           3
#define EXIT_FAIL_DISPLAY      4
#define EXIT_FAIL_CAPTURE      5
#define EXIT_FAIL_DECODER      6
#define EXIT_FAIL_TRANSMIT     7
#define EXIT_FAIL_COMPRESS     8
#define EXIT_FAIL_CONTROL_SOCK 9
#define EXIT_FAIL_NETWORK      10
#define EXIT_FAIL_AUDIO        11

#ifdef __cplusplus
extern "C" {
#endif

struct module;
struct video_frame;
struct vidcap_params;

extern int uv_argc;
extern char **uv_argv;

extern volatile bool should_exit;

void exit_uv(int status);

#define DEFAULT_AUDIO_CAPTURE_CHANNELS 1
extern unsigned int audio_capture_channels;
extern unsigned int audio_capture_bps;         // user-specified bps, if zero, module should choose
                                               // best bps by itself
extern unsigned int audio_capture_sample_rate; // user-specified sample rate, if zero, module should
                                               // choose best value by itself (usually 48000)

#define MAX_CUDA_DEVICES 4
extern unsigned int cuda_devices[];
extern unsigned int cuda_devices_count;

#define LOG_LEVEL_QUIET   0
#define LOG_LEVEL_FATAL   1
#define LOG_LEVEL_ERROR   2
#define LOG_LEVEL_WARNING 3
#define LOG_LEVEL_NOTICE  4
#define LOG_LEVEL_INFO    5
#define LOG_LEVEL_VERBOSE 6
#define LOG_LEVEL_DEBUG   7
#define LOG_LEVEL_MAX LOG_LEVEL_DEBUG
extern volatile int log_level;
extern bool color_term;

extern bool ldgm_device_gpu;

extern const char *window_title;

#define MODE_SENDER   (1<<0)
#define MODE_RECEIVER (1<<1)

// if not NULL, data should be exported
extern char *export_dir;
extern char *sage_network_device;

// Both of following varables are non-negative. It indicates amount of milliseconds that
// audio or video should be delayed. This shall be used for AV sync control.
extern volatile int audio_offset;
extern volatile int video_offset;

#define RATE_UNLIMITED 0
#define RATE_AUTO -1
#define compute_packet_rate(bitrate, mtu) (1000ll * 1000 * 1000 * mtu * 8 / bitrate)

bool common_preinit(int argc, char *argv[]);

/**
 * @param use_vidcap Use user suplied video capture device to elaborate input format.
 *                   This is used to adjust compression bitrate to correspond detected
 *                   input video format.
 */
void print_capabilities(struct module *root, bool use_vidcap);

void print_version(void);

const char *get_commandline_param(const char *key);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <unordered_map>
#include <string>
extern std::unordered_map<std::string, std::string> commandline_params;
#endif

#endif
