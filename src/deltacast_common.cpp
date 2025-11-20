/**
 * @file   deltacast_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @sa deltacast_common.hpp for common DELTACAST information
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
#include <cinttypes>             // for PRIu32
#include <iostream>              // for basic_ostream, operator<<, cout, bas...
#include <map>                   // for map, operator!=, _Rb_tree_iterator
#include <unordered_map>

#include "debug.h"
#include "types.h"
#include "utils/color_out.h"
#include "video_frame.h"         // for get_interlacing_suffix

#if !defined VHD_MIN_6_19
#define VHD_GetPCIeIdentificationString(BoardIndex, pIdString_c) \
        snprintf_ch(pIdString_c, "UNKNOWN")
#endif

#define MOD_NAME "[DELTACAST] "

VHD_BOARDTYPE
delta_get_board_type(ULONG BoardIndex)
{
        HANDLE BoardHandle = nullptr;
        ULONG  BoardType     = 0U;
        ULONG  Result =
            VHD_OpenBoardHandle(BoardIndex, &BoardHandle, nullptr, 0);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Unable to open board %d.",
                                  BoardIndex);
                return NB_VHD_BOARDTYPES;
        }
        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE,
                                      &BoardType);
        VHD_CloseBoardHandle(BoardHandle);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Unable to get board %d type.",
                                  BoardIndex);
                return NB_VHD_BOARDTYPES;
        }
        return (VHD_BOARDTYPE) BoardType;
}

const char *
delta_get_model_name(ULONG BoardIndex)
{
        thread_local char buf[128];
#if !defined VHD_MIN_6_00
#define VHD_GetBoardModel(BoardIndex) \
        delta_get_board_type_name(delta_get_board_type(BoardIndex))
#endif
        snprintf_ch(buf, "%s #%" PRIu_ULONG,
                    VHD_GetBoardModel(BoardIndex),
                    BoardIndex);
#ifdef VHD_GetBoardModel
#undef VHD_GetBoardModel
#endif
        return buf;
}

const char *
delta_get_error_description(ULONG CodeError)
{
#ifdef HAVE_VHD_STRING
        return VHD_ERRORCODE_ToPrettyString((VHD_ERRORCODE) CodeError);
#else // compat VHD < 6.30
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
#ifdef VHD_MIN_6_00
        case VHDERR_INVALIDACCESSRIGHT:
                return "Invalid access right";
#endif // defined VHD_MIN_6_00
#ifdef VHD_MIN_6_19
        case VHDERR_INVALIDCAPABILITY:
                return "Invalid capability index";
#endif // defined VHD_MIN_6_19
#ifdef DELTA_DVI_DEPRECATED
        case VHDERR_DEPRECATED:
                return "Symbol is deprecated";
#endif
        default:
                return "Unknown code error";
        }
#endif
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

// PrintChnType in SDK
static void
print_avail_channels(HANDLE BoardHandle)
{
        printf("\t  - available channels:");
#ifdef HAVE_VHD_STRING
        printf(" ");
        // PrintChnType in SDK
        ULONG ChnType, NbOfChn;

        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_RXCHANNELS, &NbOfChn);
        for (ULONG i = 0; i < NbOfChn; i++) {
                VHD_GetChannelProperty(BoardHandle, VHD_RX_CHANNEL, i,
                                       VHD_CORE_CP_TYPE, &ChnType);
                printf("RX%" PRIu_ULONG "=%s / ", i,
                       VHD_CHANNELTYPE_ToPrettyString(
                           ((VHD_CHANNELTYPE) ChnType)));
        }

        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_TXCHANNELS, &NbOfChn);
        for (ULONG i = 0; i < NbOfChn; i++) {
                VHD_GetChannelProperty(BoardHandle, VHD_TX_CHANNEL, i,
                                       VHD_CORE_CP_TYPE, &ChnType);
                printf("TX%" PRIu_ULONG "=%s / ", i,
                       VHD_CHANNELTYPE_ToPrettyString(
                           ((VHD_CHANNELTYPE) ChnType)));
        }

        printf("\b\b\b   \n");

#else // compat
        ULONG avail_channs = 0;
        ULONG Result       = VHD_GetBoardProperty(
            BoardHandle, VHD_CORE_BP_CHN_AVAILABILITY, &avail_channs);
        if (Result != VHDERR_NOERROR) {
                LOG(LOG_LEVEL_ERROR)
                    << "[DELTACAST] Unable to available channels: "
                    << delta_get_error_description(Result) << "\n";
                return;
        }
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
#endif
}

// PrintBoardInfo in SDK
static void
print_board_info(int BoardIndex, ULONG DllVersion, bool full)
{
        color_printf("\tBoard " TBOLD("%d") ": " TBOLD("%s") "\n", BoardIndex,
                     delta_get_model_name(BoardIndex));

        ULONG  DriverVersion = 0U;
        HANDLE BoardHandle   = nullptr;
        ULONG  Result =
            VHD_OpenBoardHandle(BoardIndex, &BoardHandle, nullptr, 0);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Unable to open board %d.",
                                  BoardIndex);
                return;
        }
        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_DRIVER_VERSION,
                                      &DriverVersion);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Unable to get board %d version.",
                                  BoardIndex);
        }
        if ((DllVersion >> 16U) != (DriverVersion >> 16U)) {
                MSG(WARNING, "API and driver version mismatch: %s vs %s\n",
                    delta_format_version(DllVersion, true).c_str(),
                    delta_format_version(DriverVersion, true).c_str());
        }
        if (!full) {
                VHD_CloseBoardHandle(BoardHandle);
                return;
        }

        ULONG  BoardType     = 0U;

        ULONG  SerialNumber_UL[4] = {};
        ULONG  NbOfLane, BusType, FirmwareVersion, Firmware3Version, LowProfile,
            NbRxChannels, NbTxChannels, ProductVersion = 0;
        char pIdString_c[64];

        Result = VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE,
                                      &BoardType);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Unable to get board %d type.",
                                  BoardIndex);
        }

        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_FIRMWARE_VERSION,
                             &FirmwareVersion);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);
#if defined VHD_MIN_6_21 // (not tested exactly)
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_SERIALNUMBER_PART1_LSW,
                             &SerialNumber_UL[0]);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_SERIALNUMBER_PART2,
                             &SerialNumber_UL[1]);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_SERIALNUMBER_PART3,
                             &SerialNumber_UL[2]);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_SERIALNUMBER_PART4_MSW,
                             &SerialNumber_UL[3]);
#endif
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NBOF_LANE, &NbOfLane);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_LOWPROFILE, &LowProfile);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_RXCHANNELS,
                             &NbRxChannels);
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_NB_TXCHANNELS,
                             &NbTxChannels);
#if defined VHD_MIN_6_19
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_PRODUCT_VERSION,
                             &ProductVersion);
#endif
        VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BUS_TYPE, &BusType);
        VHD_GetPCIeIdentificationString(BoardIndex, pIdString_c);

        printf("\t  - PCIe Id string : %s\n", pIdString_c);
        printf("\t  - Driver %s\n",
               delta_format_version(DriverVersion, false).c_str());
        printf("\t  - Board fpga firmware v%02" PRIX_ULONG " (%02" PRIX_ULONG
               "-%02" PRIX_ULONG "-%02" PRIX_ULONG ")\n",
               FirmwareVersion & 0xFF, (FirmwareVersion >> 24) & 0xFF,
               (FirmwareVersion >> 16) & 0xFF, (FirmwareVersion >> 8) & 0xFF);
        if (BoardType == VHD_BOARDTYPE_3G || BoardType == VHD_BOARDTYPE_3GKEY ||
            (BoardType == VHD_BOARDTYPE_HD && NbTxChannels == 4)) {
                VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_FIRMWARE3_VERSION,
                                     &Firmware3Version);
                printf("\t  - Board micro-controller firmware v%02" PRIX_ULONG
                       " (%02" PRIX_ULONG "-%02" PRIX_ULONG "-%02" PRIX_ULONG
                       ")\n",
                       Firmware3Version & 0xFF, (Firmware3Version >> 24) & 0xFF,
                       (Firmware3Version >> 16) & 0xFF,
                       (Firmware3Version >> 8) & 0xFF);
        }
#if defined VHD_MIN_6_00
        ULONG Firmware4Version = 0;
        if (BoardType == VHD_BOARDTYPE_IP) {
                VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_FIRMWARE4_VERSION,
                                     &Firmware4Version);
                printf("\t  - Board microcode firmware v%02" PRIX_ULONG
                       " (%02" PRIX_ULONG "-%02" PRIX_ULONG "-%02" PRIX_ULONG
                       ")\n",
                       Firmware4Version & 0xFF, (Firmware4Version >> 24) & 0xFF,
                       (Firmware4Version >> 16) & 0xFF,
                       (Firmware4Version >> 8) & 0xFF);
        }
#endif
        printf("\t  - Board serial# : 0x%08" PRIX_ULONG "%08" PRIX_ULONG
               "%08" PRIX_ULONG "%08" PRIX_ULONG "\n",
               SerialNumber_UL[3], SerialNumber_UL[2], SerialNumber_UL[1],
               SerialNumber_UL[0]);

        if (ProductVersion != 0) {
                printf("\t  - Board product v%04" PRIX_ULONG "\n",
                       ProductVersion);
        }

#ifdef HAVE_VHD_STRING
#define bus_type_to_str(x) VHD_BUSTYPE_ToPrettyString((VHD_BUSTYPE) x)
#else
#define bus_type_to_str(x) "unknown bus"
#endif
        printf("\t  - %s on %s", delta_get_board_type_name(BoardType),
               bus_type_to_str(BusType));
#undef bus_type_to_str
        if (NbOfLane)
                printf(" (%" PRIu_ULONG " lane%s)\n", NbOfLane,
                       (NbOfLane > 1) ? "s" : "");
        else
                printf("\n");
        printf("\t  - %s\n", LowProfile ? "Low profile" : "Full height");
        printf("\t  - %" PRIu_ULONG " In / %" PRIu_ULONG " Out\n", NbRxChannels,
               NbTxChannels);

        print_avail_channels(BoardHandle);
        const char *bidir_status = "ERROR";
        ULONG       IsBiDir      = 0;
        if (VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_IS_BIDIR, &IsBiDir) ==
            VHDERR_NOERROR) {
                bidir_status = IsBiDir == true ? "supported" : "not supported";
        }
        printf("\t  - bidirectional (switchable) channels: "
               "%s\n",
               bidir_status);
        VHD_CloseBoardHandle(BoardHandle);
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
                print_board_info(i, DllVersion, full);
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
        BOOL32 IsBiDir = false;

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
                    "Set bidirectional channel configuration %" PRIu_ULONG
                    " In / %" PRIu_ULONG " Out\n",
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

/// @see SDK Is4KInterface() + Is8KInterface
bool
delta_is_quad_channel_interface(ULONG Interface)
{
        switch (Interface) {
        // Is4KInterface
        case VHD_INTERFACE_4XHD_QUADRANT:
        case VHD_INTERFACE_4X3G_A_QUADRANT:
        case VHD_INTERFACE_4X3G_B_DL_QUADRANT:
        case VHD_INTERFACE_2X3G_B_DS_425_3:
        case VHD_INTERFACE_4X3G_A_425_5:
        case VHD_INTERFACE_4X3G_B_DL_425_5:
#if defined VHD_MIN_6_00
        case VHD_INTERFACE_2X3G_B_DS_425_3_DUAL:
        case VHD_INTERFACE_4XHD_QUADRANT_DUAL:
        case VHD_INTERFACE_4X3G_A_QUADRANT_DUAL:
        case VHD_INTERFACE_4X3G_A_425_5_DUAL:
        case VHD_INTERFACE_4X3G_B_DL_QUADRANT_DUAL:
        case VHD_INTERFACE_4X3G_B_DL_425_5_DUAL:
#endif
        // Is8KInterface
#if defined VHD_MIN_6_19
        case VHD_INTERFACE_4X6G_2081_10_QUADRANT:
        case VHD_INTERFACE_4X12G_2082_10_QUADRANT:
        case VHD_INTERFACE_4X6G_2081_12:
        case VHD_INTERFACE_4X12G_2082_12:
#endif
             return true;
        default:
             return false;
   }
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
/// of features supported by the prior versions (passive loopback only)
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

#if !defined VHD_MIN_6_13
struct deltacast_mode_info {
        unsigned int       width;
        unsigned int       height;
        int                fps;
        enum interlacing_t interlacing;
};
static struct deltacast_mode_info
deltacast_get_frame_mode(unsigned mode)
{
        switch (mode) {
        // clang-format off
        case VHD_VIDEOSTD_S274M_1080p_25Hz:   return { 1920, 1080, 25, PROGRESSIVE       };
        case VHD_VIDEOSTD_S274M_1080p_30Hz:   return { 1920, 1080, 30, PROGRESSIVE       };
        case VHD_VIDEOSTD_S274M_1080i_50Hz:   return { 1920, 1080, 25, UPPER_FIELD_FIRST };
        case VHD_VIDEOSTD_S274M_1080i_60Hz:   return { 1920, 1080, 30, UPPER_FIELD_FIRST };
        case VHD_VIDEOSTD_S296M_720p_50Hz:    return { 1280,  720, 50, PROGRESSIVE       };
        case VHD_VIDEOSTD_S296M_720p_60Hz:    return { 1280,  720, 60, PROGRESSIVE       };
        case VHD_VIDEOSTD_S259M_PAL:          return {  720,  576, 25, UPPER_FIELD_FIRST };
        case VHD_VIDEOSTD_S259M_NTSC:         return {  720,  487, 30, UPPER_FIELD_FIRST };
        case VHD_VIDEOSTD_S274M_1080p_24Hz:   return { 1920, 1080, 24, PROGRESSIVE       };
        case VHD_VIDEOSTD_S274M_1080p_60Hz:   return { 1920, 1080, 60, PROGRESSIVE       };
        case VHD_VIDEOSTD_S274M_1080p_50Hz:   return { 1920, 1080, 50, PROGRESSIVE       };
        case VHD_VIDEOSTD_S274M_1080psf_24Hz: return { 1920, 1080, 24, SEGMENTED_FRAME   };
        case VHD_VIDEOSTD_S274M_1080psf_25Hz: return { 1920, 1080, 25, SEGMENTED_FRAME   };
        case VHD_VIDEOSTD_S274M_1080psf_30Hz: return { 1920, 1080, 30, SEGMENTED_FRAME   };
        // UHD modes
        case VHD_VIDEOSTD_3840x2160p_24Hz:    return { 3840, 2160, 24, PROGRESSIVE       };
        case VHD_VIDEOSTD_3840x2160p_25Hz:    return { 3840, 2160, 25, PROGRESSIVE       };
        case VHD_VIDEOSTD_3840x2160p_30Hz:    return { 3840, 2160, 30, PROGRESSIVE       };
        case VHD_VIDEOSTD_3840x2160p_50Hz:    return { 3840, 2160, 50, PROGRESSIVE       };
        case VHD_VIDEOSTD_3840x2160p_60Hz:    return { 3840, 2160, 60, PROGRESSIVE       };
        case VHD_VIDEOSTD_4096x2160p_24Hz:    return { 4096, 2160, 24, PROGRESSIVE       };
        case VHD_VIDEOSTD_4096x2160p_25Hz:    return { 4096, 2160, 25, PROGRESSIVE       };
        case VHD_VIDEOSTD_4096x2160p_48Hz:    return { 4096, 2160, 48, PROGRESSIVE       };
        case VHD_VIDEOSTD_4096x2160p_50Hz:    return { 4096, 2160, 50, PROGRESSIVE       };
        case VHD_VIDEOSTD_4096x2160p_60Hz:    return { 4096, 2160, 60, PROGRESSIVE       };
        // clang-format on
        default:
                return {};
        };
}
// compat with old SDK
static ULONG
VHD_GetVideoCharacteristics(unsigned mode, ULONG *Width, ULONG *Height,
                            BOOL32 *Interlaced, ULONG *Framerate)
{
        const struct deltacast_mode_info info = deltacast_get_frame_mode(mode);
        if (info.width == 0) {
                return VHDERR_NOTFOUND;
        }
        *Width = info.width;
        *Height = info.height;
        *Interlaced = info.interlacing == UPPER_FIELD_FIRST ? true : false;
        *Framerate = info.fps;
        return VHDERR_NOERROR;
}
#endif // not defined VHD_MIN_6_00

/**
 * @brief returns DELTACAST mode metadata
 * @param mode      valid VHD_VIDEOSTANDARD item
 * @param want_1001 American clock system to be used
 * @return the mode information, {} if mode not found (can happen eg. if mode
 * not handled by deltacast_get_frame_mode() asked, eg. 8K modes)
 */
struct deltacast_frame_mode_t
deltacast_get_mode_info(unsigned mode, bool want_1001)
{
        // do not return "European" NTSC
        if (mode == VHD_VIDEOSTD_S259M_NTSC && !want_1001) {
                return {};
        }
        ULONG  Width      = 0;
        ULONG  Height     = 0;
        BOOL32 Interlaced = false;
        ULONG  Framerate  = 0;
        ULONG  Result     = VHD_GetVideoCharacteristics(
            (VHD_VIDEOSTANDARD) mode, &Width, &Height, &Interlaced, &Framerate);
        if (Result != VHDERR_NOERROR) {
                return {};
        }

        if (want_1001 && (Framerate == 25 || Framerate == 50)) {
                return {};
        }

        double fps = Framerate;
        if (want_1001) {
                fps = fps * 1000. / 1001.;
        }
        return { .width       = Width,
                 .height      = Height,
                 .fps         = fps,
                 .interlacing = Interlaced ? UPPER_FIELD_FIRST : PROGRESSIVE };
}

/**
 * @brief returns DELTACAST mode name
 * @copydetails deltacast_get_mode_info
 * @sa VHD_VIDEOSTANDARD_ToPrettyString (but doesn't alter for 1001 modes)
 */
const char *
deltacast_get_mode_name(unsigned mode, bool want_1001)
{
        struct deltacast_frame_mode_t info =
            deltacast_get_mode_info(mode, want_1001);
        if (info.width == 0) {
                return nullptr;
        }
        if (mode == VHD_VIDEOSTD_S259M_PAL) {
                return "SMPTE 259M PAL";
        }
        if (mode == VHD_VIDEOSTD_S259M_NTSC) {
                return "SMPTE 259M NTSC";
        }
        if (info.interlacing == UPPER_FIELD_FIRST) {
                info.interlacing = INTERLACED_MERGED;
                info.fps *= 2;
        }
        thread_local char buf[128];
        if (info.height <= 1080) {
                int std = 274;
                if (info.width == 2048) {
                        std = 2048;
                } else if (info.height == 720) {
                        std = 296;
                }
                snprintf_ch(buf, "SMPTE %dM %ux%u%s %.4g Hz", std, info.width,
                            info.height,
                            get_interlacing_suffix(info.interlacing), info.fps);
        } else {
                snprintf_ch(buf, "%ux%u %.4g Hz", info.width, info.height,
                            info.fps);
        }

        return buf;
}

// PrintVideoStandardInfo
void
delta_print_intefrace_info(ULONG Interface)
{
#ifdef HAVE_VHD_STRING
        printf("\nIncoming interface : %s\n",
               VHD_INTERFACE_ToString((VHD_INTERFACE) Interface));
#else
        (void) Interface;
#endif
}

// SingleToQuadLinksInterface
void
delta_single_to_quad_links_interface(ULONG RXStatus, ULONG *pInterface,
                                     ULONG *pVideoStandard)
{
      switch (*pVideoStandard)
      {
      case VHD_VIDEOSTD_S274M_1080p_24Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160p_24Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S274M_1080p_25Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160p_25Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S274M_1080p_30Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160p_30Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
#if defined VHD_MIN_6_19
      case VHD_VIDEOSTD_S274M_1080psf_24Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160psf_24Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S274M_1080psf_25Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160psf_25Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S274M_1080psf_30Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160psf_30Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
#endif
      case VHD_VIDEOSTD_S274M_1080p_50Hz: *pVideoStandard = VHD_VIDEOSTD_3840x2160p_50Hz;
         if (RXStatus&VHD_SDI_RXSTS_LEVELB_3G)
            *pInterface = VHD_INTERFACE_4X3G_B_DL_QUADRANT;
         else
            *pInterface = VHD_INTERFACE_4X3G_A_QUADRANT;
         break;
      case VHD_VIDEOSTD_S274M_1080p_60Hz:	*pVideoStandard = VHD_VIDEOSTD_3840x2160p_60Hz;
         if (RXStatus&VHD_SDI_RXSTS_LEVELB_3G)
            *pInterface = VHD_INTERFACE_4X3G_B_DL_QUADRANT;
         else
            *pInterface = VHD_INTERFACE_4X3G_A_QUADRANT;
         break;
      case VHD_VIDEOSTD_S2048M_2048p_24Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160p_24Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S2048M_2048p_25Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160p_25Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S2048M_2048p_30Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160p_30Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
#if defined VHD_MIN_6_19
      case VHD_VIDEOSTD_S2048M_2048psf_24Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160psf_24Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S2048M_2048psf_25Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160psf_25Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
      case VHD_VIDEOSTD_S2048M_2048psf_30Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160psf_30Hz;
         *pInterface = VHD_INTERFACE_4XHD_QUADRANT; break;
#endif
      case VHD_VIDEOSTD_S2048M_2048p_48Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160p_48Hz;
         if (RXStatus&VHD_SDI_RXSTS_LEVELB_3G)
            *pInterface = VHD_INTERFACE_4X3G_B_DL_QUADRANT;
         else
            *pInterface = VHD_INTERFACE_4X3G_A_QUADRANT;
         break;
      case VHD_VIDEOSTD_S2048M_2048p_50Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160p_50Hz;
         if (RXStatus&VHD_SDI_RXSTS_LEVELB_3G)
            *pInterface = VHD_INTERFACE_4X3G_B_DL_QUADRANT;
         else
            *pInterface = VHD_INTERFACE_4X3G_A_QUADRANT;
         break;
      case VHD_VIDEOSTD_S2048M_2048p_60Hz:	*pVideoStandard = VHD_VIDEOSTD_4096x2160p_60Hz;
         if (RXStatus&VHD_SDI_RXSTS_LEVELB_3G)
            *pInterface = VHD_INTERFACE_4X3G_B_DL_QUADRANT;
         else
            *pInterface = VHD_INTERFACE_4X3G_A_QUADRANT;
         break;
#if defined VHD_MIN_6_19
      case VHD_VIDEOSTD_3840x2160p_24Hz:
         *pVideoStandard = VHD_VIDEOSTD_7680x4320p_24Hz;
         *pInterface = VHD_INTERFACE_4X6G_2081_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_3840x2160p_25Hz:
         *pVideoStandard = VHD_VIDEOSTD_7680x4320p_25Hz;
         *pInterface = VHD_INTERFACE_4X6G_2081_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_3840x2160p_30Hz:
         *pVideoStandard = VHD_VIDEOSTD_7680x4320p_30Hz;
         *pInterface = VHD_INTERFACE_4X6G_2081_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_3840x2160p_50Hz:
         *pVideoStandard = VHD_VIDEOSTD_7680x4320p_50Hz;
         *pInterface = VHD_INTERFACE_4X12G_2082_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_3840x2160p_60Hz:
         *pVideoStandard = VHD_VIDEOSTD_7680x4320p_60Hz;
         *pInterface = VHD_INTERFACE_4X12G_2082_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_4096x2160p_24Hz:
         *pVideoStandard = VHD_VIDEOSTD_8192x4320p_24Hz;
         *pInterface = VHD_INTERFACE_4X6G_2081_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_4096x2160p_25Hz:
         *pVideoStandard = VHD_VIDEOSTD_8192x4320p_25Hz;
         *pInterface = VHD_INTERFACE_4X6G_2081_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_4096x2160p_30Hz:
         *pVideoStandard = VHD_VIDEOSTD_8192x4320p_30Hz;
         *pInterface = VHD_INTERFACE_4X6G_2081_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_4096x2160p_48Hz:
         *pVideoStandard = VHD_VIDEOSTD_8192x4320p_48Hz;
         *pInterface = VHD_INTERFACE_4X12G_2082_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_4096x2160p_50Hz:
         *pVideoStandard = VHD_VIDEOSTD_8192x4320p_50Hz;
         *pInterface = VHD_INTERFACE_4X12G_2082_10_QUADRANT;
         break;
      case VHD_VIDEOSTD_4096x2160p_60Hz:
         *pVideoStandard = VHD_VIDEOSTD_8192x4320p_60Hz;
         *pInterface = VHD_INTERFACE_4X12G_2082_10_QUADRANT;
         break;
#endif
      }
}

const char *
delta_get_board_type_name(ULONG BoardType)
{
#ifdef HAVE_VHD_STRING
        thread_local char buf[128];
        snprintf_ch(buf, "%s type",
                    VHD_BOARDTYPE_ToPrettyString((VHD_BOARDTYPE) BoardType));
        return buf;
#else
        static const std::unordered_map<ULONG, std::string> board_type_map = {
                { VHD_BOARDTYPE_HD,    "HD board type"     },
                { VHD_BOARDTYPE_HDKEY, "HD key board type" },
                { VHD_BOARDTYPE_SD,    "SD board type"     },
                { VHD_BOARDTYPE_SDKEY, "SD key board type" },
                { VHD_BOARDTYPE_DVI,   "DVI board type"    },
                { VHD_BOARDTYPE_CODEC, "CODEC board type"  },
                { VHD_BOARDTYPE_3G,    "3G board type"     },
                { VHD_BOARDTYPE_3GKEY, "3G key board type" },
                { VHD_BOARDTYPE_HDMI,  "HDMI board type"   },
        };
        auto it = board_type_map.find(BoardType);
        if (it != board_type_map.end()) {
                return it->second.c_str();
        }
        return "Unknown DELTACAST type";
#endif
}

bool
delta_chn_type_is_sdi(ULONG ChnType)
{
        return ChnType == VHD_CHNTYPE_HDSDI || ChnType == VHD_CHNTYPE_3GSDI ||
               ChnType == VHD_CHNTYPE_3GSDI_ASI ||
               ChnType == VHD_CHNTYPE_12GSDI ||
               ChnType == VHD_CHNTYPE_12GSDI_ASI;
}

/**
 * @param action           "sent" or "received"
 * @param is_final_summary print always even if no change in dropped frames
 *                         since last time
 */
void
delta_print_slot_stats(HANDLE StreamHandle, ULONG *SlotsDroppedLast,
                       const char *action)
{
        ULONG SlotsCount = 0, SlotsDropped = 0;
        /* Print some statistics */
        VHD_GetStreamProperty(StreamHandle, VHD_CORE_SP_SLOTS_DROPPED,
                              &SlotsDropped);
        if (SlotsDropped == *SlotsDroppedLast) {
                return;
        }
        VHD_GetStreamProperty(StreamHandle, VHD_CORE_SP_SLOTS_COUNT,
                              &SlotsCount);
        MSG(WARNING, "%" PRIu_ULONG " frames %s (%" PRIu_ULONG " dropped)\n",
            SlotsCount, action, SlotsDropped);
        *SlotsDroppedLast = SlotsDropped;
}

bool
delta_board_type_is_dv(VHD_BOARDTYPE         BoardType,
                       [[maybe_unused]] bool include_mixed)
{
        switch (BoardType) {
        case VHD_BOARDTYPE_HD: return false;
        case VHD_BOARDTYPE_HDKEY: return false;
        case VHD_BOARDTYPE_SD: return false;
        case VHD_BOARDTYPE_SDKEY: return false;
        case VHD_BOARDTYPE_DVI: return true;
        case VHD_BOARDTYPE_CODEC: return include_mixed; // _MIXEDINTERFACE
        case VHD_BOARDTYPE_3G: return false;
        case VHD_BOARDTYPE_3GKEY: return false;
        case VHD_BOARDTYPE_HDMI: return true;
        case VHD_BOARDTYPE_FLEX: return include_mixed;
        case VHD_BOARDTYPE_ASI: return false;
#if defined VHD_MIN_6_00
        case VHD_BOARDTYPE_IP: return false;
#endif
#if defined VHD_MIN_6_13
        case VHD_BOARDTYPE_HDMI20: return true;
        case VHD_BOARDTYPE_FLEX_DP: return include_mixed;
        case VHD_BOARDTYPE_FLEX_SDI: return false;
        case VHD_BOARDTYPE_12G: return false;
        case VHD_BOARDTYPE_FLEX_HMI: return include_mixed;
#endif
        case NB_VHD_BOARDTYPES: return false;
        }
        return false;
}
