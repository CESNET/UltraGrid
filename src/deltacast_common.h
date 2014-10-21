/**
 * @file   deltacast_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
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

#include <string>
#include <unordered_map>
#include <VideoMasterHD_Core.h>

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

static void print_available_delta_boards() {
        ULONG             Result,DllVersion,NbBoards;
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08X\n",
                                Result);
                return;
        }
        if (NbBoards == 0) {
                fprintf(stderr, "[DELTACAST] No DELTA board detected, exiting...\n");
                return;
        }

        printf("\nAvailable cards:\n");
        /* Query DELTA boards information */
        for (ULONG i = 0; i < NbBoards; i++)
        {
                ULONG BoardType;
                HANDLE            BoardHandle = NULL;
                ULONG Result = VHD_OpenBoardHandle(i,&BoardHandle,NULL,0);
                VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);
                if (Result == VHDERR_NOERROR)
                {
                        std::string board{"Unknown board type"};
                        auto it = board_type_map.find(BoardType);
                        if (it != board_type_map.end()) {
                                board = it->second;
                        }
                        printf("\t\tBoard %d: %s\n", i, board.c_str());
                        VHD_CloseBoardHandle(BoardHandle);
                }
        }
}

#endif // defined DELTACAST_COMMON_H

