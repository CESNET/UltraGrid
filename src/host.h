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
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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

#ifndef __cplusplus
#include <stdbool.h>
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
/**This variable represents a pseudostate and may be returned when initialization
 * of module was successful but no state was created (eg. when driver had displayed help). */
#define INIT_NOERR          ((void*)1) // NOLINT (cppcoreguidelines-pro-type-cstyle-cast)
#define OPT_PARAM (('O' << 8) | 'P')

#define BUG_MSG "Please report a bug to " PACKAGE_BUGREPORT " if you reach here."

#ifdef __cplusplus
extern "C" {
#endif

struct module;
struct video_frame;
struct vidcap_params;

extern int uv_argc;
extern char **uv_argv;

void handle_error(int status);
void exit_uv(int status);

#define DEFAULT_AUDIO_CAPTURE_CHANNELS 1
extern unsigned int audio_capture_channels;    ///< user-specified chan. count, if zero, module should choose
                                               ///< best/native or DEFAULT_AUDIO_CAPTURE_CHANNELS
extern unsigned int audio_capture_bps;         // user-specified bps, if zero, module should choose
                                               // best bps by itself
extern unsigned int audio_capture_sample_rate; // user-specified sample rate, if zero, module should
                                               // choose best value by itself (usually 48000)

#define MAX_CUDA_DEVICES 4
extern unsigned int cuda_devices[];
extern unsigned int cuda_devices_count;
extern bool cuda_devices_explicit; ///< --cuda-device/-D specified expilcitly

#define MODE_SENDER   (1U<<0U)
#define MODE_RECEIVER (1U<<1U)

typedef void (*mainloop_t)(void *);
/// mainloop to be run if display mainloop isn't run
extern mainloop_t mainloop;
extern void *mainloop_udata;

extern int glfw_init_count;
extern char pixfmt_conv_pref[]; // defined in video_codec.c
extern char *sage_network_device;

// Both of following varables are non-negative. It indicates amount of milliseconds that
// audio or video should be delayed. This shall be used for AV sync control. For
// getting/setting you can use get_av_delay()/set_av_delay(). All is in milliseconds.
extern volatile int audio_offset;
extern volatile int video_offset;
int get_audio_delay(void);
void set_audio_delay(int val);

#define RATE_UNLIMITED                0
#define RATE_AUTO                   (-1) ///< spread packets evenly across frame time (currently 3/4)
#define RATE_DYNAMIC                (-2) ///< same as @ref RATE_AUTO but occasional excess frame allowed
#define RATE_MIN                RATE_DYNAMIC
#define RATE_DEFAULT                (-3) ///< imaginary value, must not be passed to trasmit module
#define RATE_FLAG_FIXED_RATE (1ll<<62ll) ///< use the bitrate as fixed, not capped

struct init_data;
struct init_data *common_preinit(int argc, char *argv[]);
void common_cleanup(struct init_data *init_data);

// root module management
void init_root_module(struct module *root_mod);
void register_should_exit_callback(struct module *mod, void (*callback)(void *), void *udata);
void exit_uv(int status);
int get_exit_status(struct module *root_mod);

void print_capabilities(const char *cfg);

const char *get_version_details(void);
void print_version(void);
void print_configuration(void);

const char *get_commandline_param(const char *key);
void set_commandline_param(const char *key, const char *val);

bool tok_in_argv(char **argv, const char *tok);
int set_audio_capture_format(const char *optarg);
int set_pixfmt_conv_policy(const char *optarg);
bool parse_params(const char *optarg, bool preinit);
void print_pixel_formats(void);
void print_video_codecs(void);
void register_param(const char *param, const char *doc);

bool register_mainloop(mainloop_t, void *);

bool running_in_debugger();

void crash_signal_handler(int sig);
void hang_signal_handler(int sig);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <unordered_map>
extern std::unordered_map<std::string, std::string> commandline_params;
#endif

#define MERGE_(a,b)  a##b
#define LABEL_(a) MERGE_(unique_name_, a)
#define UNIQUE_LABEL LABEL_(__COUNTER__)

/**
 * Introduces new parameter. Without calling that, parameter from command-line
 * would be rejected.
 *
 * @param param parameter name
 * @param doc   documentation - string
 */
#define ADD_TO_PARAM(param, doc) ADD_TO_PARAM_SALT(UNIQUE_LABEL, param, doc)
#define ADD_TO_PARAM_SALT(salt, param, doc) static void MERGE_(add_to_param_doc_, salt)(void)  __attribute__((constructor));\
\
static void MERGE_(add_to_param_doc_, salt)(void) \
{\
        register_param(param, doc);\
}\
struct NOT_DEFINED_STRUCT_THAT_SWALLOWS_SEMICOLON

#endif
