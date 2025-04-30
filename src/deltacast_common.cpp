/**
 * @file   deltacast_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2025 CESNET
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

#include "deltacast_common.hpp"

#include <cstddef>               // for NULL
#include <cstdio>                // for printf
#include <iostream>              // for basic_ostream, operator<<, cout, bas...
#include <map>                   // for map, operator!=, _Rb_tree_iterator

#include "debug.h"
#include "utils/color_out.h"

#define MOD_NAME "[DELTACAST] "

const char *
delta_get_error_description(ULONG CodeError)
{
        switch (CodeError) {
        case VHDERR_NOERROR:
                return "No error";
        case VHDERR_FATALERROR:
                return "Fatal error occurred (should re-install)";
        case VHDERR_OPERATIONFAILED:
                return "Operation failed (undefined error)";
        case VHDERR_NOTENOUGHRESOURCE:
                return "Not enough resource to complete the operation";
        case VHDERR_NOTIMPLEMENTED:
                return "Not implemented yet";
        case VHDERR_NOTFOUND:
                return "Required element was not found";
        case VHDERR_BADARG:
                return "Bad argument value";
        case VHDERR_INVALIDPOINTER:
                return "Invalid pointer";
        case VHDERR_INVALIDHANDLE:
                return "Invalid handle";
        case VHDERR_INVALIDPROPERTY:
                return "Invalid property index";
        case VHDERR_INVALIDSTREAM:
                return "Invalid stream or invalid stream type";
        case VHDERR_RESOURCELOCKED:
                return "Resource is currently locked";
        case VHDERR_BOARDNOTPRESENT:
                return "Board is not available";
        case VHDERR_INCOHERENTBOARDSTATE:
                return "Incoherent board state or register value";
        case VHDERR_INCOHERENTDRIVERSTATE:
                return "Incoherent driver state";
        case VHDERR_INCOHERENTLIBSTATE:
                return "Incoherent library state";
        case VHDERR_SETUPLOCKED:
                return "Configuration is locked";
        case VHDERR_CHANNELUSED:
                return "Requested channel is already used or doesn't exist";
        case VHDERR_STREAMUSED:
                return "Requested stream is already used";
        case VHDERR_READONLYPROPERTY:
                return "Property is read-only";
        case VHDERR_OFFLINEPROPERTY:
                return "Property is off-line only";
        case VHDERR_TXPROPERTY:
                return "Property is of TX streams";
        case VHDERR_TIMEOUT:
                return "Time-out occurred";
        case VHDERR_STREAMNOTRUNNING:
                return "Stream is not running";
        case VHDERR_BADINPUTSIGNAL:
                return "Bad input signal, or unsupported standard";
        case VHDERR_BADREFERENCESIGNAL:
                return "Bad genlock signal or unsupported standard";
        case VHDERR_FRAMELOCKED:
                return "Frame already locked";
        case VHDERR_FRAMEUNLOCKED:
                return "Frame already unlocked";
        case VHDERR_INCOMPATIBLESYSTEM:
                return "Selected video standard is incompatible with running "
                       "clock system";
        case VHDERR_ANCLINEISEMPTY:
                return "ANC line is empty";
        case VHDERR_ANCLINEISFULL:
                return "ANC line is full";
        case VHDERR_BUFFERTOOSMALL:
                return "Buffer too small";
        case VHDERR_BADANC:
                return "Received ANC aren't standard";
        case VHDERR_BADCONFIG:
                return "Invalid configuration";
        case VHDERR_FIRMWAREMISMATCH:
                return "The loaded firmware is not compatible with the "
                       "installed driver";
        case VHDERR_LIBRARYMISMATCH:
                return "The loaded VideomasterHD library is not compatible "
                       "with the installed driver";
        case VHDERR_FAILSAFE:
                return "The fail safe firmware is loaded. You need to upgrade "
                       "your firmware";
        case VHDERR_RXPROPERTY:
                return "Property is of RX streams";
        case VHDERR_ALREADYINITIALIZED:
                return "Already initialized";
        case VHDERR_NOTINITIALIZED:
                return "Not initialized";
        case VHDERR_CROSSTHREAD:
                return "Cross-thread";
        case VHDERR_INCOHERENTDATA:
                return "Incoherent data";
        case VHDERR_BADSIZE:
                return "Bad size";
        case VHDERR_WAKEUP:
                return "Wake up";
        case VHDERR_DEVICE_REMOVED:
                return "Device removed";
        case VHDERR_LTCSOURCEUNLOCKED:
                return "LTC source unlocked";
        case VHDERR_INVALIDACCESSRIGHT:
                return "Invalid access right";
        case VHDERR_INVALIDCAPABILITY:
                return "Invalid capability index";
#ifdef DELTA_DVI_DEPRECATED
        case VHDERR_DEPRECATED:
                return "Symbol is deprecated";
#endif
        default:
                return "Unknown code error";
        }
}

auto
delta_format_version(uint32_t version, bool long_out) -> std::string
{
        using namespace std::string_literals;
        std::string out = std::to_string(version >> 24U) + "."s +
                          std::to_string((version >> 16U) & 0xFFU);
        if (long_out) {
                out += std::to_string((version >> 8U) & 0xFFU) + "."s +
                       std::to_string(version & 0xFFU);
        }
        return out;
}

static void
print_avail_channels(HANDLE BoardHandle)
{
        ULONG avail_channs = 0;
        ULONG Result       = VHD_GetBoardProperty(
            BoardHandle, VHD_CORE_BP_CHN_AVAILABILITY, &avail_channs);
        if (Result != VHDERR_NOERROR) {
                LOG(LOG_LEVEL_ERROR)
                    << "[DELTACAST] Unable to available channels: "
                    << delta_get_error_description(Result) << "\n";
                return;
        }
        printf("\t\tavailable channels:");
        // RXx
        // bit0 = RX0, bit1 = RX1, bit2 = RX2, bit3 = RX3
        for (int i = 0; i < 4; ++i) {
                if (((avail_channs >> i) & 0x1) != 0U) {
                        printf(" RX%d", i);
                }
        }
        // bit8 = RX4, bit9 = RX5, bit10 = RX6, bit11 = RX7
        for (int i = 4; i < 8; ++i) {
                if (((avail_channs >> (i + 4)) & 0x1) != 0U) {
                        printf(" RX%d", i);
                }
        }

        // TXx
        // bit4 = TX0, bit5 = TX1, bit6 = TX2, bit7 = TX3
        for (int i = 0; i < 4; ++i) {
                if (((avail_channs >> (i + 4)) & 0x1) != 0U) {
                        printf(" TX%d", i);
                }
        }
        // bit12 = TX4, bit13 = TX5, bit14 = TX6, bit15 = TX7
        for (int i = 4; i < 8; ++i) {
                if (((avail_channs >> (i + 8)) & 0x1) != 0U) {
                        printf(" TX%d", i);
                }
        }
        printf("\n");
}

void
print_available_delta_boards(bool full)
{
        ULONG Result, DllVersion, NbBoards;
        Result = VHD_GetApiInfo(&DllVersion, &NbBoards);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR,
                        "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                        " information. Result = 0x%08" PRIX_ULONG "\n",
                        Result);
                return;
        }
        std::cout << "\nAPI version: "
                  << delta_format_version(DllVersion, false) << "\n";
        std::cout << "\nAvailable cards:\n";
        if (NbBoards == 0) {
                log_msg(LOG_LEVEL_ERROR,
                        "[DELTACAST] No DELTA board detected, exiting...\n");
                return;
        }

        /* Query DELTA boards information */
        for (ULONG i = 0; i < NbBoards; i++) {
                ULONG  BoardType     = 0U;
                ULONG  DriverVersion = 0U;
                HANDLE BoardHandle   = NULL;
                ULONG  Result = VHD_OpenBoardHandle(i, &BoardHandle, NULL, 0);
                if (Result != VHDERR_NOERROR) {
                        LOG(LOG_LEVEL_ERROR)
                            << "[DELTACAST] Unable to open board " << i << ": "
                            << delta_get_error_description(Result) << "\n";
                        continue;
                }
                Result = VHD_GetBoardProperty(
                    BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);
                if (Result != VHDERR_NOERROR) {
                        LOG(LOG_LEVEL_ERROR)
                            << "[DELTACAST] Unable to get board " << i
                            << " type: " << delta_get_error_description(Result)
                            << "\n";
                        continue;
                }
                Result = VHD_GetBoardProperty(
                    BoardHandle, VHD_CORE_BP_DRIVER_VERSION, &DriverVersion);
                if (Result != VHDERR_NOERROR) {
                        LOG(LOG_LEVEL_ERROR)
                            << "[DELTACAST] Unable to get board " << i
                            << " version: "
                            << delta_get_error_description(Result) << "\n";
                }

                std::string board{ "Unknown board type" };
                auto        it = board_type_map.find(BoardType);
                if (it != board_type_map.end()) {
                        board = it->second;
                }
                col() << "\tBoard " << SBOLD(i) << ": " << SBOLD(board)
                      << " (driver: "
                      << delta_format_version(DriverVersion, false) << ")\n";
                if (full) {
                        print_avail_channels(BoardHandle);

                        ULONG IsBiDir = 2;
                        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_IS_BIDIR,
                                             &IsBiDir);
                        printf("\t\tbidirectional (switchable) channels: "
                               "%s\n",
                               IsBiDir == 2      ? "ERROR"
                               : IsBiDir == TRUE ? "supported"
                                                 : "not supported");
                }
                if ((DllVersion >> 16U) != (DriverVersion >> 16U)) {
                        LOG(LOG_LEVEL_WARNING)
                            << "[DELTACAST] API and driver version mismatch: "
                            << delta_format_version(DllVersion, true) << " vs "
                            << delta_format_version(DriverVersion, true)
                            << "\n";
                }
                VHD_CloseBoardHandle(BoardHandle);
        }
        std::cout << "\n";
}

void
delta_print_ch_layout_help(bool full)
{
        color_printf(
            "\t" TBOLD("ch_layout") " - configure bidirectional channels%s\n",
            full ? ":" : ", see \":fullhelp\" for details");
        if (full) {
                color_printf("\t\tSet the layout with a number in format RxTx, "
                             "examples:\n"
                             "\t\t\t- 80 - 8 Rx and 0 TX\n"
                             "\t\t\t- 44 - 4 Rx and 4 TX\n"
                             "\t\t\t- 13 - 1 Rx and 3 TX (4 channel device)\n"
                             "\t\tIt is user responsibility to enter the valid "
                             "number (not exceeding device channels).\n");
        }
}

/// from SDK SetNbChannels()
bool
delta_set_nb_channels(ULONG BrdId, HANDLE BoardHandle, ULONG RequestedRx,
                      ULONG RequestedTx)
{
        ULONG  Result;
        ULONG  NbRxOnBoard = 0;
        ULONG  NbTxOnBoard = 0;
        ULONG  NbChanOnBoard;
        BOOL32 IsBiDir = FALSE;

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_RXCHANNELS,
                                      &NbRxOnBoard);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR,
                        "[DELTACAST] ERROR: Cannot get number of RX channels. "
                        "Result = 0x%08" PRIX_ULONG "\n",
                        Result);
                return false;
        }

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_TXCHANNELS,
                                      &NbTxOnBoard);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR,
                        "[DELTACAST] ERROR: Cannot get number of TX channels. "
                        "Result = 0x%08" PRIX_ULONG "\n",
                        Result);
                return false;
        }

        if (NbRxOnBoard >= RequestedRx && NbTxOnBoard >= RequestedTx) {
                return true;
        }

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_IS_BIDIR,
                                      (ULONG *) &IsBiDir);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR,
                        "[DELTACAST] ERROR: Cannot check whether board "
                        "channels are bidirectional. Result = 0x%08" PRIX_ULONG
                        "\n",
                        Result);
                return false;
        }

        NbChanOnBoard = NbRxOnBoard + NbTxOnBoard;

        if (!IsBiDir || NbChanOnBoard < (RequestedRx + RequestedTx)) {
                MSG(ERROR,
                        "ERROR: Insufficient number of channels - "
                        "requested %" PRIu_ULONG " RX + %" PRIu_ULONG
                        " TX, got %" PRIu_ULONG " RX + %" PRIu_ULONG
                        " TX. %s\n",
                        RequestedRx, RequestedTx, NbRxOnBoard, NbTxOnBoard,
                        IsBiDir ? "Bidirectional" : "Non-bidirectional");
                return false;
        }

        // key - (NbChanOnBoard, RequestedRX), value - member of
        // VHD_BIDIRCFG_2C, VHD_BIDIRCFG_4C or VHD_BIDIRCFG_8C
        std::map<std::pair<ULONG, ULONG>, ULONG> mapping = {
                //{{2, 0}, VHD_BIDIR_02},
                //{{2, 1}, VHD_BIDIR_11},
                //{{2, 2}, VHD_BIDIR_20},

                { { 4, 0 }, VHD_BIDIR_04 },
                { { 4, 1 }, VHD_BIDIR_13 },
                { { 4, 2 }, VHD_BIDIR_22 },
                { { 4, 3 }, VHD_BIDIR_31 },
                { { 4, 4 }, VHD_BIDIR_40 },

                { { 8, 0 }, VHD_BIDIR_08 },
                { { 8, 1 }, VHD_BIDIR_17 },
                { { 8, 2 }, VHD_BIDIR_26 },
                { { 8, 3 }, VHD_BIDIR_35 },
                { { 8, 4 }, VHD_BIDIR_44 },
                { { 8, 5 }, VHD_BIDIR_53 },
                { { 8, 6 }, VHD_BIDIR_62 },
                { { 8, 7 }, VHD_BIDIR_71 },
                { { 8, 8 }, VHD_BIDIR_80 }
        };
        auto it = mapping.find({ NbChanOnBoard, RequestedRx });
        if (it == mapping.end()) {
                MSG(ERROR, "Sufficient number of channels and board is "
                           "switchable  but cannot find a mapping! Please "
                           "report...\n");
                return false;
        }
        Result = VHD_SetBiDirCfg(BrdId, it->second);
        if (Result == VHDERR_NOERROR) {
                MSG(INFO,
                    "Set bidirectional channel configuration %d In / %d Out\n",
                    RequestedRx, NbChanOnBoard - RequestedRx);
                return true;
        }
        MSG(ERROR,
            "ERROR: Cannot set bidirectional channels. Result = "
            "0x%08" PRIX_ULONG "\n",
            Result);
        MSG(WARNING, "See also option \":ch_layout\" to switch "
                     "bidirectional channels layout.\n");
        return false;
}

/// @returns stream type corresponding channel ID or NB_VHD_STREAMTYPES if too
/// high
VHD_STREAMTYPE
delta_rx_ch_to_stream_t(unsigned channel)
{
        switch (channel) {
        case 0:
                return VHD_ST_RX0;
        case 1:
                return VHD_ST_RX1;
        case 2:
                return VHD_ST_RX2;
        case 3:
                return VHD_ST_RX3;
        case 4:
                return VHD_ST_RX4;
        case 5:
                return VHD_ST_RX5;
        case 6:
                return VHD_ST_RX6;
        case 7:
                return VHD_ST_RX7;
        }
        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Channel index %u out of bound!\n",
                channel);
        return NB_VHD_STREAMTYPES;
}

/// @copydoc delta_rx_ch_to_stream_t
VHD_STREAMTYPE
delta_tx_ch_to_stream_t(unsigned channel)
{
        switch (channel) {
        case 0:
                return VHD_ST_TX0;
        case 1:
                return VHD_ST_TX1;
        case 2:
                return VHD_ST_TX2;
        case 3:
                return VHD_ST_TX3;
        case 4:
                return VHD_ST_TX4;
        case 5:
                return VHD_ST_TX5;
        case 6:
                return VHD_ST_TX6;
        case 7:
                return VHD_ST_TX7;
        }
        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Channel index %u out of bound!\n",
                channel);
        return NB_VHD_STREAMTYPES;
}

/// @see SDK Is4KInterface()
bool
delta_is_quad_channel_interface(ULONG Interface)
{
        bool Result = FALSE;
        switch (Interface) {
        case VHD_INTERFACE_4XHD_QUADRANT:
        case VHD_INTERFACE_4X3G_A_QUADRANT:
        case VHD_INTERFACE_4X3G_B_DL_QUADRANT:
        // case VHD_INTERFACE_2X3G_B_DS_425_3:
        case VHD_INTERFACE_4X3G_A_425_5:
        case VHD_INTERFACE_4X3G_B_DL_425_5:
        // case VHD_INTERFACE_2X3G_B_DS_425_3_DUAL:
        case VHD_INTERFACE_4XHD_QUADRANT_DUAL:
        case VHD_INTERFACE_4X3G_A_QUADRANT_DUAL:
        case VHD_INTERFACE_4X3G_A_425_5_DUAL:
        case VHD_INTERFACE_4X3G_B_DL_QUADRANT_DUAL:
        case VHD_INTERFACE_4X3G_B_DL_425_5_DUAL:
                Result = TRUE;
                break;
        default:
                Result = FALSE;
                break;
        }

        return Result;
}

/// equally named fn in SDK
static VHD_CORE_BOARDPROPERTY GetPassiveLoopbackProperty(int ChannelIdx)
{
   switch (ChannelIdx)
   {
      case 0: return VHD_CORE_BP_BYPASS_RELAY_0;
      case 1: return VHD_CORE_BP_BYPASS_RELAY_1;
      case 2: return VHD_CORE_BP_BYPASS_RELAY_2;
      case 3: return VHD_CORE_BP_BYPASS_RELAY_3;
      default: return NB_VHD_CORE_BOARDPROPERTIES;
   }
}

/// @sa SDK SetLoopbackState() since VideoMaster 6.21 but simplified to the extent
/// of features suppoprted by the prior versions (passive loopback only)
void
delta_set_loopback_state(HANDLE BoardHandle, int ChannelIndex, BOOL32 State)
{
        VHD_CORE_BOARDPROPERTY prop = GetPassiveLoopbackProperty(ChannelIndex);

        if (prop == NB_VHD_CORE_BOARDPROPERTIES) {
                return;
        }
        ULONG err = VHD_SetBoardProperty(BoardHandle, prop, State);
        if (err != VHDERR_NOERROR) {
                MSG(VERBOSE, "Cannot set passive loopback for channel %d!\n",
                    ChannelIndex);
        }
}
