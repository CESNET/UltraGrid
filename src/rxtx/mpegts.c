// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include <assert.h>  // for assert
#include <math.h>    // for fabs
#include <pthread.h> // for pthread_mutex_t
#include <stdint.h>  // for uint8_t, int64_t, uint32_t
#include <stdio.h>   // for FILE, fclose, fopen, fwrite
#include <stdlib.h>  // for abort, free, calloc, malloc
#include <string.h>  // for memcpy, strcmp

#include "../../ext-deps/libmpegts/common.h" // for TIMESTAMP_CLOCK, TS_CLOCK, TS_PACKE...
#include "../../ext-deps/libmpegts/libmpegts.h" // for LIBMPEGTS_MPEG2_AAC_1_CHANNEL, LIBM...

#include "audio/types.h"   // for AC_OPUS, AC_AAC, audio_frame2_get_c...
#include "compat/c23.h"    // for countof
#include "debug.h"         // for LOG_LEVEL_DEBUG, MSG
#include "lib_common.h"    // for REGISTER_MODULE, library_class
#include "rtp/net_udp.h"   // for socket_udp, udp_init, udp_send
#include "rxtx.h"          // for rxtx_params, RXTX_ABI_VERSION, rxtx...
#include "utils/macros.h"  // for CONST_CAST
#include "utils/pthread.h" // for ug_pthread_mutex_init

struct audio_frame2;

#define MOD_NAME "[rxtx/mpegts] "

#include "types.h"

struct rxtx_mpegts {
        uint32_t        magic;
        ts_writer_t    *writer;
        bool            init;
        socket_udp     *sock;
        int             mtu;
        pthread_mutex_t lock;

        FILE *dump_f;

        bool use_audio;
        bool use_video;

        // audio paramsters
        audio_codec_t ac;
        double af_duration;

        // video parameters
        codec_t vc;
        double  vf_duration;

        double video_duration;
        double audio_duration;
};

enum {
        PCR_PID   = 0x100,
        VIDEO_PID = 0x100,
        AUDIO_PID = 0x101,
        PMT_PID   = 0x1000
};

static void *
init(const struct rxtx_params *params)
{
        if (params->mtu == 0) {
                params->mtu = 1500;
        }
        struct rxtx_mpegts *s = calloc(1, sizeof *s);
        ug_pthread_mutex_init(&s->lock);
        s->mtu                = params->mtu / TS_PACKET_SIZE * TS_PACKET_SIZE;
        s->writer             = ts_create_writer();
        s->sock               = udp_init(params->receiver, 0, 1234, params->ttl,
                                         params->force_ip_version, false);
        s->use_audio = params->medium[TX_MEDIA_AUDIO].rxtx_mode & MODE_SENDER;
        s->use_video = params->medium[TX_MEDIA_VIDEO].rxtx_mode & MODE_SENDER;

        if (strcmp(params->protocol_opts, "dump") == 0) {
                s->dump_f = fopen("out.ts", "wb");
        }

        return s;
}

static bool
init_libmpegts(struct rxtx_mpegts *s)
{
        if ((s->use_audio && s->ac == AC_NONE) ||
            (s->use_video && s->vc == VC_NONE)) {
                MSG(WARNING, "Waiting for both audio and video present...\n");
                return false;
        }

        ts_stream_t stream[2]   = { 0 };
        int str_cnt = 0;
        if (s->use_video) {
                assert(s->vc == H264);
                stream[str_cnt].pid           = VIDEO_PID;
                stream[str_cnt].stream_format = LIBMPEGTS_VIDEO_AVC;           // = 2 [1]
                stream[str_cnt].stream_id     = LIBMPEGTS_STREAM_ID_MPEGVIDEO; // = 0xe0 [1]
                str_cnt += 1;
        }
        if (s->use_audio) {
                assert(s->ac == AC_OPUS || s->ac == AC_AAC);
                stream[str_cnt].pid = AUDIO_PID;
                stream[str_cnt].stream_format =
                    s->ac == AC_OPUS ? LIBMPEGTS_AUDIO_OPUS
                                     : LIBMPEGTS_AUDIO_ADTS; // = 2 [1]
                stream[str_cnt].stream_id =
                    LIBMPEGTS_STREAM_ID_MPEGAUDIO; // = 0xe0 [1]
                stream[str_cnt].audio_frame_size =
                    TIMESTAMP_CLOCK * s->af_duration;
                str_cnt += 1;
        }

        ts_program_t prog = { 0 }; // = &main_params.programs[0];
        prog.program_num  = 1;
        prog.pmt_pid      = PMT_PID;
        prog.pcr_pid      = PCR_PID;
        prog.num_streams  = str_cnt;
        prog.streams      = stream;

        ts_main_t main_params  = { 0 };
        main_params.lowlatency = 1;
        main_params.ts_id      = 1;
        // even for audio only must be a larger number - see
        // libmpegts/libmpegts.c:1795, if set to eg 200 kbps check_pcr returns
        // always 1 and no actual audio data get sent
        main_params.muxrate = 5000000; // 5 Mbps
        // Constant bitrate - if set to 1, it will fill to match bitrate
        main_params.cbr          = 0;
        main_params.ts_type      = TS_TYPE_GENERIC;
        main_params.num_programs = 1;
        main_params.programs     = &prog;

        int rc = ts_setup_transport_stream(s->writer, &main_params);
        if (rc != 0) {
                return false;
        }

        if (s->use_video) {
                // Setup AVC stream parameters
                rc = ts_setup_mpegvideo_stream(
                    s->writer,
                    VIDEO_PID, // PID
                    52,        // level
                    AVC_HIGH,  // profile (from avc_profile_t enum) [1]
                    5000000,   // vbv_maxrate (bits/s)
                    1000000,   // vbv_bufsize
                    0          // frame_rate (not used for AVC) [1]
                );
                if (rc != 0) {
                        return false;
                }
        }

        if (s->use_audio) {
                rc = s->ac == AC_OPUS
                         ? ts_setup_opus_stream(s->writer, AUDIO_PID,
                                                LIBMPEGTS_CHANNEL_CONFIG_MONO)
                         : ts_setup_mpeg2_aac_stream(
                               s->writer, AUDIO_PID,
                               LIBMPEGTS_MPEG2_AAC_LC_PROFILE, // FF default
                               LIBMPEGTS_MPEG2_AAC_1_CHANNEL);
                if (rc != 0) {
                        return false;
                }
        }
        return true;
}

static void
udp_send_packets(struct rxtx_mpegts *s, uint8_t *output, int output_len)
{
        if (s->dump_f != nullptr) {
                fwrite(output, output_len, 1, s->dump_f);
        }

        int len = s->mtu;
        while (output_len > 0) {
                if (output_len < len) {
                        len = output_len;
                }
                udp_send(s->sock, (char *) output, len);
                output_len -= len;
                output += len;
        }
}

static void
send_video_frame_impl(struct rxtx_mpegts *s, struct video_frame *f)
{
        if (!s->init) {
                s->vc          = f->color_spec;
                s->vf_duration = 1. / f->fps;
                if (!init_libmpegts(s)) {
                        return;
                }
                s->init = true;
        } else {
                if (s->vc != f->color_spec) {
                        MSG(ERROR, "video codec changed, reconf not implemented!\n");
                        return;
                }
                double old_fps = 1. / s->vf_duration;
                if (fabs(f->fps - old_fps) > .001) {
                        MSG(ERROR,
                            "FPS changed from %f to %f, reconf not "
                            "implemented!\n",
                            old_fps, f->fps);
                        return;
                }
        }

        ts_frame_t ts_frame = { 0 };
        ts_frame.pid        = VIDEO_PID;
        ts_frame.data       = (uint8_t *) f->tiles[0].data;
        ts_frame.size       = (int) f->tiles[0].data_len;
        // ts_frame.random_access = 1; // is keyframe
        // ts_frame.frame_type    = LIBMPEGTS_CODING_TYPE_SLICE_IDR;
        // int nal_ref_idc        = 3; // @todo
        // ts_frame.ref_pic_idc   = nal_ref_idc;

        // 90kHz clock ticks [1]
        double ts = s->video_duration + s->vf_duration;
        ts_frame.dts = ts_frame.pts = (int64_t) (ts * TIMESTAMP_CLOCK);
        MSG(DEBUG, "video TS %f s\n", ts);

        ts_frame.cpb_initial_arrival_time =
            (int64_t) (s->video_duration * TS_CLOCK);
        ts_frame.cpb_final_arrival_time =
            (s->video_duration + s->vf_duration) * TS_CLOCK;

        uint8_t *output     = nullptr;
        int      output_len = 0;
        int64_t *pcr_list   = nullptr;

        ts_write_frames(s->writer, &ts_frame, 1, &output, &output_len,
                        &pcr_list);

        MSG(DEBUG, "ts_write_frames video: %d B\n", output_len);
        udp_send_packets(s, output, output_len);

        s->video_duration += s->vf_duration;
}

static void
send_video_frame(void *state, struct video_frame *f)
{
        struct rxtx_mpegts *s = state;
        CHK_PTHR(pthread_mutex_lock(&s->lock));
        send_video_frame_impl(s, f);
        CHK_PTHR(pthread_mutex_unlock(&s->lock));
        f->callbacks.dispose(f);
}

enum {
        ADTS_HDR_SIZE_BASE = 7, // without CRC and explicit sampling rate
        ADTS_EXPLICIT_SR   = 3,
        ADTS_HDR_SZ_MAX    = ADTS_HDR_SIZE_BASE + ADTS_EXPLICIT_SR,
};
// <https://wiki.multimedia.cx/index.php/ADTS>
static int
write_adts_header(uint8_t *header, size_t raw_frame_size, unsigned sample_rate,
                  int channel_count, int profile)
{
        header[0] = 0xFF; // syncword - first 8/12 bits
        enum {
                MPEG4 = 0,
                MPEG2 = 1,
                CRC   = 0,
                NOCRC = 1,
        };
        enum {
                MASK_2B = 0x3,
                MASK_3B = 0x7,
                MASK_6B = 0x3F,
                MASK_8B = 0xFF,
        };
        size_t hdr_len = ADTS_HDR_SZ_MAX;
        // 4 lsb of syncword, 1b mpeg ver, 2b layer (always 0), 1b CRC presence
        header[1] = 0xF0 | MPEG2 << 3 | NOCRC;

        unsigned       sample_rate_idx = 15; // default - freq explicitly
        const unsigned rates[]         = {
                [0] = 96000, [1] = 88200, [2] = 64000,  [3] = 48000,
                [4] = 44100, [5] = 32000, [6] = 24000,  [7] = 22050,
                [8] = 16000, [9] = 12000, [10] = 11025, [11] = 8000,
                [12] = 7350,
                /* 13,14 reserved, 15 freq explicitly */
        };
        for (unsigned i = 0; i < countof(rates); i++) {
                if (rates[i] == sample_rate) {
                        sample_rate_idx = i;
                        hdr_len -= ADTS_EXPLICIT_SR;
                        break;
                }
        }
        assert(channel_count <= 6 || channel_count == 8);
        // for 1-6 equals channel count, 8 ch is 7; vals 8-15 reserved
        int channel_cfg = channel_count < 7 ? channel_count : 7;
        // 2b profile minus 1 (we have already subtracted), 4b sample rate idx,
        // 1b private (MBZ), msb bit of channel_cfg
        header[2] = profile << 6 | sample_rate_idx << 2 | channel_cfg >> 2;
        // 2 lsb of channel_cfg, 4 flags: originality, home, copyright
        // bit&start, 2/13 msb of frame length
        size_t frame_size = raw_frame_size + hdr_len;
        header[3] = (channel_cfg & MASK_2B) << 6 | frame_size >> 11;
        // bits 2-10 of 13b length
        header[4] = (frame_size >> 3) & MASK_8B;
        enum { VBR = 0x7ff };
        // 3 lsb of frame length, 5/11 msb of buffer fullness
        header[5] = (frame_size & MASK_3B) << 5 | VBR >> 6;
        // 6 lsb of buffer_fullness, 2 bits - nr of frames minus 1 (so 1 is 0)
        header[6] = (VBR & MASK_6B) << 2;
        // not using CRC, otherwise additional 2 bytes
        if (sample_rate_idx != 15) {
                return hdr_len;
        }
        header[7] = sample_rate >> 16;
        header[8] = (sample_rate >> 8) & MASK_8B;
        header[9] = sample_rate & MASK_8B;
        return hdr_len;
}

static uint8_t *
add_adts_header(const uint8_t *data, int *len)
{
        size_t   new_sz = *len + ADTS_HDR_SZ_MAX;
        uint8_t *ret    = malloc(new_sz);
        int hdr_len = write_adts_header((unsigned char *) ret, *len, kHz48,
                                        LIBMPEGTS_MPEG2_AAC_1_CHANNEL,
                                        LIBMPEGTS_MPEG2_AAC_LC_PROFILE);
        memcpy(ret + hdr_len, data, *len);
        *len += hdr_len;
        return ret;
}

static void
send_audio_frame_impl(struct rxtx_mpegts *s, const struct audio_frame2 *f)
{
        double duration = audio_frame2_get_duration(f);
        if (!s->init) {
                s->af_duration = duration;
                s->ac          = audio_frame2_get_codec(f);
                if (!init_libmpegts(s)) {
                        return;
                }
                s->init = true;
        } else {
                if (s->ac != audio_frame2_get_codec(f)) {
                        MSG(ERROR, "audio codec changed, reconf not implemented!\n");
                        return;
                }
                if (fabs(s->af_duration - duration) > .001) {
                        MSG(ERROR,
                            "audio frame duration changed from %f to %f, "
                            "reconf not implemented!\n",
                            s->af_duration, duration);
                        return;
                }
        }

        const uint8_t *ad = (const uint8_t *) audio_frame2_get_data(f, 0);

        ts_frame_t ts_frame = { 0 };
        ts_frame.pid        = AUDIO_PID;
        ts_frame.data       = CONST_CAST(uint8_t *, ts_frame.data, ad);
        ts_frame.size       = (int) audio_frame2_get_data_len(f, 0);
        // ts_frame.random_access = 1; // is keyframe
        // ts_frame.frame_type    = LIBMPEGTS_CODING_TYPE_SLICE_IDR;
        // int nal_ref_idc        = 3; // @todo
        // ts_frame.ref_pic_idc   = nal_ref_idc;

        if (audio_frame2_get_codec(f) == AC_AAC) { // we need to add ADTS hdr
                ts_frame.data = add_adts_header(ad, &ts_frame.size);
        }
        // fwrite(ts_frame.data, ts_frame.size, 1, s->dump_f);
        // fclose(s->dump_f); abort();

        // 90kHz clock ticks [1]
        double ts = s->audio_duration + duration;
        ts_frame.dts = ts_frame.pts = (int64_t) (ts * TIMESTAMP_CLOCK);
        MSG(DEBUG, "audio TS: %f\n", ts);

        // ts_frame.cpb_initial_arrival_time = s->audio_duration * TS_CLOCK;
        // ts_frame.cpb_final_arrival_time =
        //     ts_frame.cpb_initial_arrival_time + (TS_CLOCK * duration);

        uint8_t *output     = nullptr;
        int      output_len = 0;
        int64_t *pcr_list   = nullptr;

        ts_write_frames(s->writer, &ts_frame, 1, &output, &output_len,
                        &pcr_list);
        MSG(DEBUG, "ts_write_frames audio: %d B (in %d B)\n", output_len,
            ts_frame.size);

        udp_send_packets(s, output, output_len);

        // ts_write_frames(s->writer, &ts_frame, 0, &output, &output_len,
        // &pcr_list); MSG(DEBUG, " 2 ts_write_frames audio: %d B (in %d B)\n",
        // output_len, ts_frame.size);

        if ((const uint8_t *) ts_frame.data != ad) {
                free(ts_frame.data);
        }

        s->audio_duration += duration;
}

static void
send_audio_frame(void *state, const struct audio_frame2 *f)
{
        struct rxtx_mpegts *s = state;
        CHK_PTHR(pthread_mutex_lock(&s->lock));
        {
                send_audio_frame_impl(s, f);
        }
        CHK_PTHR(pthread_mutex_unlock(&s->lock));
}

static void
done(void *state)
{
        struct rxtx_mpegts *s = state;
        if (s->dump_f) {
                fclose(s->dump_f);
        }
        free(s);
}

static const struct rxtx_info mpegts_rxtx_info = {
        .long_name    = "MPEG transport stream",
        .create       = init,
        .done         = done,
        .ctl_property = nullptr,

        .send_audio_frame = send_audio_frame,
        .recv_audio_frame = nullptr,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_video_frame,
        .video_recv_routine = nullptr,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(mpegts, &mpegts_rxtx_info, LIBRARY_CLASS_RXTX,
                RXTX_ABI_VERSION);
