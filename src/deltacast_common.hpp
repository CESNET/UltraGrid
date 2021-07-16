/**
 * @file   deltacast_common.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2020 CESNET, z. s. p. o.
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

#ifndef DELTACAST_COMMON_H
#define DELTACAST_COMMON_H

#include <cinttypes>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#ifdef HAVE_MACOSX
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

#include "debug.h"
#include "rang.hpp"
#include "types.h"

#if defined VHD_DV_SP_INPUT_CS // (VideoMasterHD 6.14)
#define DELTA_DVI_DEPRECATED 1
#endif

// compat
#ifdef DELTA_DVI_DEPRECATED
#define VHD_BOARDTYPE_DVI VHD_BOARDTYPE_DVI_DEPRECATED
#define VHD_BOARDTYPE_HDKEY VHD_BOARDTYPE_HDKEY_DEPRECATED
#endif

struct deltacast_frame_mode_t {
	int              mode;
	const char  *     name;
	unsigned int     width;
	unsigned int     height;
	double           fps;
	enum interlacing_t interlacing;
	unsigned long int iface;
};

const static struct deltacast_frame_mode_t deltacast_frame_modes[] = {
        {VHD_VIDEOSTD_S274M_1080p_25Hz, "SMPTE 274M 1080p 25 Hz",
                1920u, 1080u, 25.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 29.97 Hz",
                1920u, 1080u, 29.97, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 30 Hz",
                1920u, 1080u, 30.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080i_50Hz, "SMPTE 274M 1080i 50 Hz",
                1920u, 1080u, 25.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 59.94 Hz",
                1920u, 1080u, 29.97, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 60 Hz",
                1920u, 1080u, 30.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S296M_720p_50Hz, "SMPTE 296M 720p 50 Hz",
                1280u, 720u, 50.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 59.94 Hz",
                1280u, 720u, 59.94, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 60 Hz",
                1280u, 720u, 60.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S259M_PAL, "SMPTE 259M PAL",
                720u, 576u, 25.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S259M_NTSC, "SMPTE 259M NTSC",
                720u, 487u, 29.97, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 23.98 Hz",
                1920u, 1080u, 23.98, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 24 Hz",
                1920u, 1080u, 24.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 59.94 Hz",
                1920u, 1080u, 59.94, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 60 Hz",
                1920u, 1080u, 60.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 50 Hz",
                1920u, 1080u, 50.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 23.98 Hz",
                1920u, 1080u, 23.98, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 24 Hz",
                1920u, 1080u, 24.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_25Hz, "SMPTE 274M 1080psf 25 Hz",
                1920u, 1080u, 25.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 29.97 Hz",
                1920u, 1080u, 29.97, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 30 Hz",
                1920u, 1080u, 30.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        // UHD modes
        {VHD_VIDEOSTD_3840x2160p_24Hz, "3840x2160 24 Hz",
                3840u, 2160u, 24.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_3840x2160p_25Hz, "3840x2160 25 Hz",
                3840u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_3840x2160p_30Hz, "3840x2160 30 Hz",
                3840u, 2160u, 30.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_3840x2160p_50Hz, "3840x2160 50 Hz",
                3840u, 2160u, 50.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_3840x2160p_60Hz, "3840x2160 60 Hz",
                3840u, 2160u, 60.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_4096x2160p_24Hz, "4096x2160 24 Hz",
                4096u, 2160u, 24.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_4096x2160p_25Hz, "4096x2160 25 Hz",
                4096u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_4096x2160p_25Hz, "4096x2160 25 Hz",
                4096u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_4096x2160p_48Hz, "4096x2160 48 Hz",
                4096u, 2160u, 48.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_4096x2160p_50Hz, "4096x2160 50 Hz",
                4096u, 2160u, 50.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_4096x2160p_60Hz, "4096x2160 60 Hz",
                4096u, 2160u, 60.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
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

/**
 * GetErrorDescription from SDK
 */
static const char *delta_get_error_description(ULONG CodeError) __attribute__((unused));
static const char *delta_get_error_description(ULONG CodeError)
{
   switch (CodeError)
   {
   case VHDERR_NOERROR:                              return "No error";
   case VHDERR_FATALERROR:                           return "Fatal error occurred (should re-install)";
   case VHDERR_OPERATIONFAILED:                      return "Operation failed (undefined error)";
   case VHDERR_NOTENOUGHRESOURCE:                    return "Not enough resource to complete the operation";
   case VHDERR_NOTIMPLEMENTED:                       return "Not implemented yet";
   case VHDERR_NOTFOUND:                             return "Required element was not found";
   case VHDERR_BADARG:                               return "Bad argument value";
   case VHDERR_INVALIDPOINTER:                       return "Invalid pointer";
   case VHDERR_INVALIDHANDLE:                        return "Invalid handle";
   case VHDERR_INVALIDPROPERTY:                      return "Invalid property index";
   case VHDERR_INVALIDSTREAM:                        return "Invalid stream or invalid stream type";
   case VHDERR_RESOURCELOCKED:                       return "Resource is currently locked";
   case VHDERR_BOARDNOTPRESENT:                      return "Board is not available";
   case VHDERR_INCOHERENTBOARDSTATE:                 return "Incoherent board state or register value";
   case VHDERR_INCOHERENTDRIVERSTATE:                return "Incoherent driver state";
   case VHDERR_INCOHERENTLIBSTATE:                   return "Incoherent library state";
   case VHDERR_SETUPLOCKED:                          return "Configuration is locked";
   case VHDERR_CHANNELUSED:                          return "Requested channel is already used or doesn't exist";
   case VHDERR_STREAMUSED:                           return "Requested stream is already used";
   case VHDERR_READONLYPROPERTY:                     return "Property is read-only";
   case VHDERR_OFFLINEPROPERTY:                      return "Property is off-line only";
   case VHDERR_TXPROPERTY:                           return "Property is of TX streams";
   case VHDERR_TIMEOUT:                              return "Time-out occurred";
   case VHDERR_STREAMNOTRUNNING:                     return "Stream is not running";
   case VHDERR_BADINPUTSIGNAL:                       return "Bad input signal, or unsupported standard";
   case VHDERR_BADREFERENCESIGNAL:                   return "Bad genlock signal or unsupported standard";
   case VHDERR_FRAMELOCKED:                          return "Frame already locked";
   case VHDERR_FRAMEUNLOCKED:                        return "Frame already unlocked";
   case VHDERR_INCOMPATIBLESYSTEM:                   return "Selected video standard is incompatible with running clock system";
   case VHDERR_ANCLINEISEMPTY:                       return "ANC line is empty";
   case VHDERR_ANCLINEISFULL:                        return "ANC line is full";
   case VHDERR_BUFFERTOOSMALL:                       return "Buffer too small";
   case VHDERR_BADANC:                               return "Received ANC aren't standard";
   case VHDERR_BADCONFIG:                            return "Invalid configuration";
   case VHDERR_FIRMWAREMISMATCH:                     return "The loaded firmware is not compatible with the installed driver";
   case VHDERR_LIBRARYMISMATCH:                      return "The loaded VideomasterHD library is not compatible with the installed driver";
   case VHDERR_FAILSAFE:                             return "The fail safe firmware is loaded. You need to upgrade your firmware";
   case VHDERR_RXPROPERTY:                           return "Property is of RX streams";
   case VHDERR_ALREADYINITIALIZED:                   return "Already initialized";
   case VHDERR_NOTINITIALIZED:                       return "Not initialized";
   case VHDERR_CROSSTHREAD:                          return "Cross-thread";
   case VHDERR_INCOHERENTDATA:                       return "Incoherent data";
   case VHDERR_BADSIZE:                              return "Bad size";
   case VHDERR_WAKEUP:                               return "Wake up";
   case VHDERR_DEVICE_REMOVED:                       return "Device removed";
   case VHDERR_LTCSOURCEUNLOCKED:                    return "LTC source unlocked";
   case VHDERR_INVALIDACCESSRIGHT:                   return "Invalid access right";
   case VHDERR_INVALIDCAPABILITY:                                    return "Invalid capability index";
#ifdef DELTA_DVI_DEPRECATED
   case VHDERR_DEPRECATED:                                         return "Symbol is deprecated";
#endif
   default:                                          return "Unknown code error";
   }
}

static auto delta_format_version(uint32_t version, bool long_out) -> std::string
{
        using namespace std::string_literals;
        std::string out = std::to_string(version >> 24U) + "."s + std::to_string((version >> 16U) & 0xFFU);
        if (long_out) {
                out += std::to_string((version >> 8U) & 0xFFU) + "."s + std::to_string(version & 0xFFU);
        }
        return out;
}

static void print_available_delta_boards() {
        ULONG             Result,DllVersion,NbBoards;
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08" PRIX32 "\n",
                                Result);
                return;
        }
        std::cout << "\t\tAPI version: " << delta_format_version(DllVersion, false) << "\n";
        if (NbBoards == 0) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] No DELTA board detected, exiting...\n");
                return;
        }

        std::cout << "\n\t\tAvailable cards:\n";
        /* Query DELTA boards information */
        for (ULONG i = 0; i < NbBoards; i++)
        {
                ULONG BoardType = 0U;
                ULONG DriverVersion = 0U;
                HANDLE            BoardHandle = NULL;
                ULONG Result = VHD_OpenBoardHandle(i,&BoardHandle,NULL,0);
                if (Result != VHDERR_NOERROR) {
                        LOG(LOG_LEVEL_ERROR) << "[DELTACAST] Unable to open board " << i << ": " << delta_get_error_description(Result) << "\n";
                        continue;
                }
                Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);
                if (Result != VHDERR_NOERROR) {
                        LOG(LOG_LEVEL_ERROR) << "[DELTACAST] Unable to get board " << i << " type: " << delta_get_error_description(Result) << "\n";
                        continue;
                }
                Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_DRIVER_VERSION, &DriverVersion);
                if (Result != VHDERR_NOERROR) {
                        LOG(LOG_LEVEL_ERROR) << "[DELTACAST] Unable to get board " << i << " version: " << delta_get_error_description(Result) << "\n";
                }

                std::string board{"Unknown board type"};
                auto it = board_type_map.find(BoardType);
                if (it != board_type_map.end()) {
                        board = it->second;
                }
                std::cout << "\t\t\tBoard " << rang::style::bold << i << rang::style::reset << ": " << rang::style::bold << board << rang::style::reset << " (driver: " << delta_format_version(DriverVersion, false) << ")\n";
                if ((DllVersion >> 16U) != (DriverVersion >> 16U)) {
                        LOG(LOG_LEVEL_WARNING) << "[DELTACAST] API and driver version mismatch: " << delta_format_version(DllVersion, true) << " vs " << delta_format_version(DriverVersion, true) << "\n";
                }
                VHD_CloseBoardHandle(BoardHandle);
        }
        std::cout << "\n";
}

[[gnu::unused]] static bool delta_set_nb_channels(ULONG BrdId, HANDLE BoardHandle, ULONG RequestedRx, ULONG RequestedTx) {
        ULONG             Result;
        ULONG             NbRxOnBoard = 0;
        ULONG             NbTxOnBoard = 0;
        ULONG             NbChanOnBoard;
        BOOL32            IsBiDir = FALSE;

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_RXCHANNELS, &NbRxOnBoard);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR: Cannot get number of RX channels. Result = 0x%08" PRIX32 "\n", Result);
                return false;
        }

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_TXCHANNELS, &NbTxOnBoard);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR: Cannot get number of TX channels. Result = 0x%08" PRIX32 "\n", Result);
                return false;
        }

        if (NbRxOnBoard >= RequestedRx && NbTxOnBoard >= RequestedTx) {
                return true;
        }

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_IS_BIDIR, (ULONG*)&IsBiDir);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR: Cannot check whether board channels are bidirectional. Result = 0x%08" PRIX32 "\n", Result);
                return false;
        }

        NbChanOnBoard = NbRxOnBoard + NbTxOnBoard;

	if (IsBiDir && NbChanOnBoard >= (RequestedRx + RequestedTx)) {
                std::map<std::pair<ULONG, ULONG>, ULONG> mapping = { // key - (NbChanOnBoard, RequestedRX), value - member of VHD_BIDIRCFG_2C, VHD_BIDIRCFG_4C or VHD_BIDIRCFG_8C
                        //{{2, 0}, VHD_BIDIR_02},
                        //{{2, 1}, VHD_BIDIR_11},
                        //{{2, 2}, VHD_BIDIR_20},

                        {{4, 0}, VHD_BIDIR_04},
                        {{4, 1}, VHD_BIDIR_13},
                        {{4, 2}, VHD_BIDIR_22},
                        {{4, 3}, VHD_BIDIR_31},
                        {{4, 4}, VHD_BIDIR_40},

                        {{8, 0}, VHD_BIDIR_08},
                        {{8, 1}, VHD_BIDIR_17},
                        {{8, 2}, VHD_BIDIR_26},
                        {{8, 3}, VHD_BIDIR_35},
                        {{8, 4}, VHD_BIDIR_44},
                        {{8, 5}, VHD_BIDIR_53},
                        {{8, 6}, VHD_BIDIR_62},
                        {{8, 7}, VHD_BIDIR_71},
                        {{8, 8}, VHD_BIDIR_80}
                };
                auto it = mapping.find({NbChanOnBoard, RequestedRx});
                if (it != mapping.end()) {
                        Result = VHD_SetBiDirCfg(BrdId, it->second);
                        if (Result != VHDERR_NOERROR) {
                                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR: Cannot bidirectional set bidirectional channels. Result = 0x%08" PRIX32 "\n", Result);
                                return false;
                        } else {
                                log_msg(LOG_LEVEL_VERBOSE, "[DELTACAST] Successfully bidirectional channels.\n");
                                return true;
                        }
                }
        }

        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR: Insufficient number of channels - requested %" PRId32 " RX + %" PRId32 " TX, got %" PRId32 " RX + %" PRId32 " TX. %s. Result = 0x%08" PRIX32 "\n", RequestedRx, RequestedTx, NbRxOnBoard, NbTxOnBoard, IsBiDir ? "Bidirectional" : "Non-bidirectional", Result);
        return false;
}

#endif // defined DELTACAST_COMMON_H

