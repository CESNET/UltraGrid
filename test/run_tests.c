/*
 * FILE:    run_tests.c
 * AUTHORS: Colin Perkins
 *
 * Copyright (c) 2004 University of Glasgow
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the University of
 *      Glasgow Department of Computing Science
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $Revision: 1.2 $
 * $Date: 2008/01/10 11:07:42 $
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "test_host.h"
#include "test_aes.h"
#include "test_audio_hw.h"
#include "test_bitstream.h"
#include "test_codec_dvi.h"
#include "test_codec_g711.h"
#include "test_codec_g726.h"
#include "test_codec_gsm.h"
#include "test_codec_l16.h"
#include "test_codec_l8.h"
#include "test_codec_lpc.h"
#include "test_codec_vdvi.h"
#include "test_des.h"
#include "test_md5.h"
#include "test_random.h"
#include "test_tv.h"
#include "test_net_udp.h"
#include "test_rtp.h"
#include "test_video_capture.h"
#include "test_video_display.h"

uint32_t RTT;                   /* FIXME: will be removed once the global in main.c is removed */

/* These globals should be fixed in the future as well */
uint32_t hd_size_x = 1920;
uint32_t hd_size_y = 1080;
uint32_t hd_color_bpp = 3;
uint32_t bitdepth = 10;
uint32_t progressive = 0;
uint32_t hd_video_mode;

long packet_rate = 13600;

int main()
{
        if (test_bitstream() != 0)
                return 1;
        if (test_codec_dvi() != 0)
                return 1;
        if (test_codec_g711() != 0)
                return 1;
        if (test_codec_g726() != 0)
                return 1;
        if (test_codec_gsm() != 0)
                return 1;
        if (test_codec_l16() != 0)
                return 1;
        if (test_codec_l8() != 0)
                return 1;
        if (test_codec_lpc() != 0)
                return 1;
        if (test_codec_vdvi() != 0)
                return 1;
        if (test_des() != 0)
                return 1;
        if (test_aes() != 0)
                return 1;
        if (test_md5() != 0)
                return 1;
        if (test_random() != 0)
                return 1;
        if (test_tv() != 0)
                return 1;
        if (test_net_udp() != 0)
                return 1;
        if (test_rtp() != 0)
                return 1;

#ifdef TEST_AV_HW
        if (test_audio_hw() != 0)
                return 1;
        if (test_video_capture() != 0)
                return 1;
        if (test_video_display() != 0)
                return 1;
#endif

        return 0;
}
