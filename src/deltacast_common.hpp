/**
 * @file   deltacast_common.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * ## SDK Compatibility
 *
 * VideoMaster SDK makes API changes over the time. The SDK/driver version
 * compatibility is uncertain (according to [ventuz] information, those must
 * match). DELTACAST removes support for older cards in newer driver which may
 * require the use of older SDK to run on those.
 *
 * At this moment (2025-10), the source code compat is with 6.13 (last one
 * supporting DELTACAST DVI devices) but also 5.19 compat is currently kept.
 *
 * [ventuz]:
 * <https://www.ventuz.com/support/help/latest/MachineConfigurationVendors.html#SupportedModelsasofVentuz6.08.00>
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

#include "module.h"
#define MAX_DELTA_CH 7 ///< maximal supported channel index

#include <string>
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

#include <cinttypes>

#ifdef __APPLE__
        #if __has_include(<VideoMasterHD/VideoMasterHD_String.h>)
                #include <VideoMasterHD/VideoMasterHD_String.h>
                #define HAVE_VHD_STRING
        #endif
#else
        #if __has_include(<VideoMasterHD_String.h>)
                #include <VideoMasterHD_String.h>
                #define HAVE_VHD_STRING
        #endif
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

// VHD_MIN_X_YZ defined below means that the SDK version is at least X.YZ
#ifdef __APPLE__
        #if __has_include(<VideoMasterHD/VideoMasterHD_Ip_Board.h>)
                #include <VideoMasterHD/VideoMasterHD_Ip_Board.h>
                #define VHD_MIN_6_00 1
        #endif
#else
        #if __has_include(<VideoMasterHD_Ip_Board.h>)
                #include <VideoMasterHD_Ip_Board.h>
                #define VHD_MIN_6_00 1
        #endif
#endif
#if defined VHD_IS_6_19
        #define VHD_MIN_6_19 1
#elif defined VHD_IP_FILTER_UDP_PORT_DEST
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
        #define VHD_MIN_6_19 1
#endif

#ifdef HAVE_VHD_STRING
        #define VHD_MIN_6_30 1
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
#else
        #define VHD_CHNTYPE_3GSDI_ASI VHD_CHNTYPE_DISABLE
        #define VHD_CHNTYPE_12GSDI_ASI VHD_CHNTYPE_DISABLE
#endif // not defined VHD_MIN_6_20

#if defined VHD_MIN_6_00
        #define VHD_BOARDTYPE_FLEX VHD_BOARDTYPE_FLEX_DEPRECATED
#else
        #define VHD_CHNTYPE_12GSDI VHD_CHNTYPE_DISABLE
#endif

enum {
        DELTA_DROP_WARN_INT_SEC = 20,
};

struct deltacast_frame_mode_t {
	unsigned int     width;
	unsigned int     height;
	double           fps;
	enum interlacing_t interlacing;
};

struct deltacast_frame_mode_t deltacast_get_mode_info(unsigned mode,
                                                       bool     want_1001);
const char *deltacast_get_mode_name(unsigned mode, bool want_1001);

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
void           delta_print_intefrace_info(ULONG Interface);
void delta_single_to_quad_links_interface(ULONG RXStatus, ULONG *pInterface,
                                          ULONG *pVideoStandard);
const char *delta_get_board_type_name(ULONG BoardType);
bool        delta_chn_type_is_sdi(ULONG ChnType);
void        delta_print_slot_stats(HANDLE StreamHandle, ULONG *SlotsDroppedLast,
                                   const char *action);
VHD_BOARDTYPE delta_get_board_type(ULONG BoardIndex);
bool          delta_board_type_is_dv(VHD_BOARDTYPE BoardType, bool include_mixed);
const char   *delta_get_model_name(ULONG BoardIndex);

#ifdef HAVE_VHD_STRING
        #define DELTA_PRINT_ERROR(error_code, error_message, ...) \
        { \
            char pLastErrorMessage[VHD_MAX_ERROR_STRING_SIZE] = ""; \
            VHD_GetLastErrorMessage(pLastErrorMessage, VHD_MAX_ERROR_STRING_SIZE); \
            MSG(ERROR, error_message " Result = 0x%08" PRIX_ULONG " (%s)\nVHD_GetLastErrorMessage -->\n%s\n", ##__VA_ARGS__, error_code, VHD_ERRORCODE_ToPrettyString(VHD_ERRORCODE(error_code)), pLastErrorMessage); \
        }
#else
        #define DELTA_PRINT_ERROR(error_code, error_message, ...) \
            MSG(ERROR, error_message " Result = 0x%08" PRIX_ULONG " (%s)\n", ##__VA_ARGS__, error_code, delta_get_error_description(error_code));
#endif

#endif // defined DELTACAST_COMMON_HPP

