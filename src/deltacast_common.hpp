/**
 * @file   deltacast_common.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2025 CESNET, zájmové sdružení právnických osob
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
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

#ifndef DELTACAST_COMMON_HPP
#define DELTACAST_COMMON_HPP

#define MAX_DELTA_CH 7 ///< maximal supported channel index

#include <string>
#include <unordered_map>
#ifdef __APPLE__
#include <VideoMasterHD/VideoMasterHD_Core.h>
#ifdef ENUMBASE_DV
#include <VideoMasterHD/VideoMasterHD_Dv.h>
#else
#include <VideoMasterHD/VideoMasterHD_Dvi.h>
#endif
#include <VideoMasterHD/VideoMasterHD_Sdi.h>
#include <VideoMasterHD_Audio/VideoMasterHD_Sdi_Audio.h>
#else
#include <VideoMasterHD_Core.h>
#ifdef ENUMBASE_DV
#include <VideoMasterHD_Dv.h>
#else
#include <VideoMasterHD_Dvi.h>
#endif
#include <VideoMasterHD_Sdi.h>
#include <VideoMasterHD_Sdi_Audio.h>
#endif

#ifdef _WIN32
#define PRIu_ULONG "lu"
#define PRIX_ULONG "lX"
#else
#include <cinttypes>
#define PRIu_ULONG PRIu32
#define PRIX_ULONG PRIX32
#endif

#include "types.h"

#if defined VHD_DV_SP_INPUT_CS // (VideoMasterHD 6.14)
#define DELTA_DVI_DEPRECATED 1
#endif

#ifdef __APPLE__
#include <VideoMasterHD/VideoMasterHD_Ip_Board.h>
#else
#include <VideoMasterHD_Ip_Board.h>
#endif
#if defined VHD_IP_FILTER_UDP_PORT_DEST && !defined VHD_IS_6_19
        #ifdef VHD_CORE_BP_BYPASS_RELAY_0
                // enum membber until 6.20, macro since 6.21
                #define VHD_MIN_6_21 1
        #endif
        #if !defined VHD_MIN_6_21 && !defined VHD_IS_6_20 // 6.19 or 6.20
                #warning cannot determine if VideoMaster is 6.19 or 6.20 - \
                        assuming 6.20. Pass -DVHD_IS_6_19 (or 6_20) to enforce \
                        specific version.
        #endif
        #define VHD_MIN_6_20 1
#endif

// compat
#ifdef DELTA_DVI_DEPRECATED
#define VHD_BOARDTYPE_DVI VHD_BOARDTYPE_DVI_DEPRECATED
#define VHD_BOARDTYPE_HDKEY VHD_BOARDTYPE_HDKEY_DEPRECATED
#endif
// Following items have been actually deprecated in 6.20. But 6.20 doesn't
// bring any new define and thus it is undistinguishable from 6.19. As a
// consequence, it won't compile with 6.19.
#if defined VHD_MIN_6_20
#define VHD_BOARDTYPE_SD VHD_BOARDTYPE_SD_DEPRECATED
#define VHD_BOARDTYPE_SDKEY VHD_BOARDTYPE_SDKEY_DEPRECATED
#define VHD_CHNTYPE_SDSDI VHD_CHNTYPE_SDSDI_DEPRECATED
#ifndef VHD_CORE_BP_BYPASS_RELAY_0 // not defined in 6.20
// VHD_BOARDTYPE_HDMI is not deprecated anymore in 6.21
#define VHD_BOARDTYPE_HDMI VHD_BOARDTYPE_HDMI_DEPRECATED
#endif
#endif

struct deltacast_frame_mode_t {
	int              mode;
	const char  *     name;
	unsigned int     width;
	unsigned int     height;
	double           fps;
	enum interlacing_t interlacing;
	unsigned long int iface;
	VHD_CLOCKDIVISOR clock_system; // VHD_CLOCKDIV_1 or VHD_CLOCKDIV_1001
};

const static struct deltacast_frame_mode_t deltacast_frame_modes[] = {
        {VHD_VIDEOSTD_S274M_1080p_25Hz, "SMPTE 274M 1080p 25 Hz",
                1920u, 1080u, 25.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 29.97 Hz",
                1920u, 1080u, 29.97, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 30 Hz",
                1920u, 1080u, 30.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080i_50Hz, "SMPTE 274M 1080i 50 Hz",
                1920u, 1080u, 25.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 59.94 Hz",
                1920u, 1080u, 29.97, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 60 Hz",
                1920u, 1080u, 30.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S296M_720p_50Hz, "SMPTE 296M 720p 50 Hz",
                1280u, 720u, 50.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 59.94 Hz",
                1280u, 720u, 59.94, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 60 Hz",
                1280u, 720u, 60.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S259M_PAL, "SMPTE 259M PAL",
                720u, 576u, 25.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S259M_NTSC, "SMPTE 259M NTSC",
                720u, 487u, 29.97, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 23.98 Hz",
                1920u, 1080u, 23.98, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 24 Hz",
                1920u, 1080u, 24.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 59.94 Hz",
                1920u, 1080u, 59.94, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 60 Hz",
                1920u, 1080u, 60.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 50 Hz",
                1920u, 1080u, 50.0, PROGRESSIVE, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 23.98 Hz",
                1920u, 1080u, 23.98, SEGMENTED_FRAME, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 24 Hz",
                1920u, 1080u, 24.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080psf_25Hz, "SMPTE 274M 1080psf 25 Hz",
                1920u, 1080u, 25.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 29.97 Hz",
                1920u, 1080u, 29.97, SEGMENTED_FRAME, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 30 Hz",
                1920u, 1080u, 30.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO, VHD_CLOCKDIV_1},
        // UHD modes
        {VHD_VIDEOSTD_3840x2160p_24Hz, "3840x2160 24 Hz",
                3840u, 2160u, 24.0, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_3840x2160p_24Hz, "3840x2160 23.98 Hz",
                3840u, 2160u, 23.98, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_3840x2160p_25Hz, "3840x2160 25 Hz",
                3840u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_3840x2160p_30Hz, "3840x2160 30 Hz",
                3840u, 2160u, 30.0, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_3840x2160p_30Hz, "3840x2160 29.97 Hz",
                3840u, 2160u, 29.97, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_3840x2160p_50Hz, "3840x2160 50 Hz",
                3840u, 2160u, 50.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_3840x2160p_60Hz, "3840x2160 60 Hz",
                3840u, 2160u, 60.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_3840x2160p_60Hz, "3840x2160 59.94 Hz",
                3840u, 2160u, 59.94, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_4096x2160p_24Hz, "4096x2160 24 Hz",
                4096u, 2160u, 24.0, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_4096x2160p_24Hz, "4096x2160 23.98 1Hz",
                4096u, 2160u, 23.98, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_4096x2160p_25Hz, "4096x2160 25 Hz",
                4096u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_4096x2160p_48Hz, "4096x2160 48 Hz",
                4096u, 2160u, 48.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_4096x2160p_48Hz, "4096x2160 47.96 Hz",
                4096u, 2160u, 47.96, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1001},
        {VHD_VIDEOSTD_4096x2160p_50Hz, "4096x2160 50 Hz",
                4096u, 2160u, 50.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_4096x2160p_60Hz, "4096x2160 60 Hz",
                4096u, 2160u, 60.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1},
        {VHD_VIDEOSTD_4096x2160p_60Hz, "4096x2160 59.97 Hz",
                4096u, 2160u, 59.97, PROGRESSIVE, VHD_INTERFACE_4X3G_A, VHD_CLOCKDIV_1001},
};

const static int deltacast_frame_modes_count = sizeof(deltacast_frame_modes)/sizeof(deltacast_frame_mode_t);

static std::unordered_map<ULONG, std::string> board_type_map = {
        { VHD_BOARDTYPE_HD, "HD board type" },
        { VHD_BOARDTYPE_HDKEY, "HD key board type" },
        { VHD_BOARDTYPE_SD, "SD board type"},
        { VHD_BOARDTYPE_SDKEY, "SD key board type"},
        { VHD_BOARDTYPE_DVI, "DVI board type"},
        { VHD_BOARDTYPE_CODEC, "CODEC board type"},
        { VHD_BOARDTYPE_3G, "3G board type"},
        { VHD_BOARDTYPE_3GKEY, "3G key board type"},
        { VHD_BOARDTYPE_HDMI, "HDMI board type"},
};

/// some DELTA enums use continuous values for a channel (RX or TX) < 4 and
/// distinct for >= 4, so this is a simple displatcher
#define DELTA_CH_TO_VAL(ch, base0, base4) \
        ((ch) < 4 ? (base0) + (ch) : (base4) + ((ch) - 4))

/**
 * GetErrorDescription from SDK
 */
const char *delta_get_error_description(ULONG CodeError);
std::string delta_format_version(uint32_t version, bool long_out);

void print_available_delta_boards(bool full);
void delta_print_ch_layout_help(bool full);

bool delta_set_nb_channels(ULONG BrdId, HANDLE BoardHandle, ULONG RequestedRx, ULONG RequestedTx);

VHD_STREAMTYPE delta_rx_ch_to_stream_t(unsigned channel);
VHD_STREAMTYPE delta_tx_ch_to_stream_t(unsigned channel);

bool           delta_is_quad_channel_interface(ULONG Interface);
void           delta_set_loopback_state(HANDLE BoardHandle, int ChannelIndex,
                                        BOOL32 State);

#endif // defined DELTACAST_COMMON_HPP

