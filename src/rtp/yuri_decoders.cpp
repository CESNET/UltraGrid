/**
 * @file    rtp/yuri_decoders.cpp
 * @author  Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "audio/utils.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/audio_decoders.h"
#include "rtp/yuri_decoders.h"
#include "rtp/video_decoders.h"

#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "modules/ultragrid/YuriUltragrid.h"

using namespace std;

static int decode_yuri_audio_frame(struct coded_data *cdata, yuri_decoder_data *s)
{
        rtp_packet *pckt = NULL;
        uint32_t *hdr;
        int buffer_length;
        struct audio_desc network_desc;
        char *received_data = NULL;

        pckt = cdata->data;
        hdr = (uint32_t *)(void *) pckt->data;
        parse_audio_hdr(hdr, &network_desc);
        buffer_length = ntohl(hdr[2]);

        if (network_desc.codec != AC_PCM) {
                s->log[yuri::log::warning] << "Compressed audio is not yet supported!";
                return FALSE;
        }

        yuri::core::pRawAudioFrame frame = yuri::core::RawAudioFrame::create_empty(
                        yuri::ultragrid::audio_uv_to_yuri(network_desc.bps * 8), network_desc.ch_count,
                        network_desc.sample_rate, yuri::uvector<uint8_t>(buffer_length * network_desc.ch_count));
        received_data = reinterpret_cast<char *>(frame->data());
        s->yuri_frame = frame;
        if (!s->yuri_frame) {
                return FALSE;
        }

        while (cdata != NULL) {
                uint32_t *hdr;
                uint32_t substream;
                char *data;
                uint32_t data_pos;
                int len;

                pckt = cdata->data;

                hdr = (uint32_t *)(void *) pckt->data;
                data_pos = ntohl(hdr[1]);
                substream = ntohl(hdr[0]) >> 22;

                len = pckt->data_len - sizeof(audio_payload_hdr_t);
                data = (char *) hdr + sizeof(audio_payload_hdr_t);

                mux_channel(received_data + data_pos * network_desc.ch_count, data, network_desc.bps,
                                len, network_desc.ch_count, substream, 1.0);

                cdata = cdata->nxt;
        }

        return TRUE;
}

static int decode_yuri_video_frame(struct coded_data *cdata, yuri_decoder_data *s)
{
        char *received_data = NULL;

        rtp_packet *pckt = NULL;
        int buffer_length;
        uint32_t *hdr;
        struct video_desc network_desc;

        pckt = cdata->data;
        hdr = (uint32_t *)(void *) pckt->data;
        parse_video_hdr(hdr, &network_desc);
        buffer_length = ntohl(hdr[2]);

        s->yuri_frame = yuri::ultragrid::create_yuri_from_uv_desc(&network_desc, buffer_length, s->log);
        if (!s->yuri_frame) {
                return FALSE;
        }

        auto raw = dynamic_pointer_cast<yuri::core::RawVideoFrame>(s->yuri_frame);
        if (raw) {
                received_data = reinterpret_cast<char *>(PLANE_RAW_DATA(raw, 0));
        } else {
                auto compressed = dynamic_pointer_cast<yuri::core::CompressedVideoFrame>(s->yuri_frame);
                if (compressed) {
                        received_data = reinterpret_cast<char *>(compressed->begin());
                }
        }

        while (cdata != NULL) {
                char *data;
                uint32_t data_pos;
                int len;
                uint32_t *hdr;
                uint32_t substream;
                pckt = cdata->data;

                hdr = (uint32_t *)(void *) pckt->data;
                data_pos = ntohl(hdr[1]);
                substream = ntohl(hdr[0]) >> 22;

                if (substream > 0) {
                        s->log[yuri::log::warning] << "Multiple substreams for video yet supported!";
                        return FALSE;
                }

                len = pckt->data_len - sizeof(video_payload_hdr_t);
                data = (char *) hdr + sizeof(video_payload_hdr_t);

                memcpy(received_data + data_pos, (unsigned char*) data, len);

                cdata = cdata->nxt;
        }

        return TRUE;
}

int decode_yuri_frame(struct coded_data *cdata, void *decoder_data)
{
        yuri_decoder_data *s = (yuri_decoder_data *) decoder_data;
        if (cdata->data->pt == PT_VIDEO) {
                return decode_yuri_video_frame(cdata, s);
        } else if (cdata->data ->pt == PT_AUDIO) {
                return decode_yuri_audio_frame(cdata, s);
        } else {
                s->log[yuri::log::warning] << "Unsupported UltraGrid pt " << cdata->data->pt << "!";
                return FALSE;
        }
}

