/*
 * Copyright (c) 2003-2004 University of Southern California
 * Copyright (c) 2005-2023 CESNET, z. s. p. o.
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
/**
 * @file    video_decoders.cpp
 * @author  Colin Perkins
 * @author  Ladan Gharai
 * @author  Martin Pulec <pulec@cesnet.cz>
 * @ingroup video_rtp_decoder
 *
 * ## Workflow ##
 *
 * Normal workflow through threads is following:
 * 1. decode data from in context of receiving thread with function decode_frame(),
 *    it will decode it to framebuffer, passes it to FEC thread
 * 2. fec_thread() passes frame to decompress thread
 * 3. thread running decompress_thread(), displays the frame
 *
 * ### Uncompressed video (without FEC) ###
 * In step one, the decoder is a linedecoder and framebuffer is the display framebuffer
 *
 * ### Compressed video ###
 * Data is saved to decompress buffer. The decompression itself is done by decompress_thread().
 *
 * ### video with FEC ###
 * Data is saved to FEC buffer. Decoded with fec_thread().
 *
 * ### Encrypted video (without FEC) ###
 * Prior to decoding, packet is decompressed.
 *
 * ### Encrypted video (with FEC) ###
 * After FEC decoding, the whole block is decompressed.
 *
 * ## Probing compressed video codecs ##
 * When video is received, it is probed for an internal format:
 * * compression is reconfigured to (VIDEO_CODEC_NONE, VIDEO_CODEC_NONE) -
 *   internal and out codecs are set to 0
 * * if any decoder for the codec supports probing, s->out_codec is set to
 *   VIDEO_CODEC_END (otherwise a decompression is directly configured)
 * * decompression is done as usual (display is not reconfigured and output is
 *   put to a dummy framebuffer)
 * * when decoder is able to establish the codec, it returns
 *   DECODER_GOT_CODEC, thread send a reconf message to the main threads and
 *   the reconfiguration shall begin - display codecs are reordered ideally to
 *   match the incoming stream
 * * BUGS: usually multiple reconf messages are sent, decompress priority is
 *   evaluated only when there are multiple same (in_codec,internal_codec,out_codec)
 *   tuples.
 *
 * @todo
 * This code is very very messy, it needs to be rewritten.
 *
 * @note
 * @anchor vdec_note1
 * The properties of DXTn do not exactly match - bpp is 0.5, but line (actually 4
 * lines) is (2 * width) long, so it makes troubles when using line decoder
 * and tiles. So the fallback is external decoder. The DXT compression is exceptional
 * in that, that it can be both internally and externally decompressed.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "control_socket.h"
#include "crypto/openssl_decrypt.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "pixfmt_conv.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/video_decoders.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/synchronized_queue.h"
#include "utils/thread.h"
#include "utils/timed_message.h"
#include "utils/worker.h"
#include "video.h"
#include "video_decompress.h"
#include "video_display.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

#ifdef HAVE_LIBAVCODEC_AVCODEC_H
#include <libavcodec/avcodec.h> // AV_INPUT_BUFFER_PADDING_SIZE
#endif

#define MOD_NAME "[video dec.] "

#define FRAMEBUFFER_NOT_READY(decoder) (decoder->frame == NULL && decoder->out_codec != VIDEO_CODEC_END)

using namespace std;
using namespace std::string_literals;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

struct state_video_decoder;

/**
 * Interlacing changing function protoype. The function should be able to change buffer
 * in place, that is when dst and src are the same.
 */
typedef void (*change_il_t)(char *dst, char *src, int linesize, int height, void **state);

// prototypes
static bool reconfigure_decoder(struct state_video_decoder *decoder,
                struct video_desc desc, struct pixfmt_desc comp_int_desc);
static void check_for_mode_change(struct state_video_decoder *decoder, const uint32_t *hdr);
static void wait_for_framebuffer_swap(struct state_video_decoder *decoder);
static void *fec_thread(void *args);
static void *decompress_thread(void *args);
static void cleanup(struct state_video_decoder *decoder);
static void decoder_process_message(struct module *);

static int sum_map(map<int, int> const & m) {
        int ret = 0;
        for (map<int, int>::const_iterator it = m.begin();
                        it != m.end(); ++it) {
                ret += it->second;
        }
        return ret;
}

namespace {

#ifdef HAVE_LIBAVCODEC_AVCODEC_H
constexpr int PADDING = AV_INPUT_BUFFER_PADDING_SIZE;
#else
constexpr int PADDING = 0;
#endif
/**
 * Enumerates 2 possibilities how to decode arriving data.
 */
enum decoder_type_t {
        UNSET,
        LINE_DECODER,    ///< This is simple decoder, that decodes incoming data per line (pixelformats only).
        EXTERNAL_DECODER /**< real decompress, received buffer is opaque and has to be decompressed as
                          * a whole. */
};

/**
 * This structure holds data needed to use a linedecoder.
 */
struct line_decoder {
        int                  base_offset;  ///< from the beginning of buffer. Nonzero if decoding from mutiple tiles.
        long                 conv_num;     ///< in->out bpp conv numerator
        long                 conv_den;     ///< in->out bpp conv denominator
        int                  shifts[3];    ///< requested red,green and blue shift (in bits)
        decoder_t            decode_line;  ///< actual decoding function
        unsigned int         dst_linesize; ///< destination linesize
        unsigned int         dst_pitch;    ///< framebuffer pitch - it can be larger if SDL resolution is larger than data
        unsigned int         src_linesize; ///< source linesize
};

struct reported_statistics_cumul {
        ~reported_statistics_cumul() {
                print();
        }
        long long int last_buffer_number = -1; ///< last received buffer ID
        chrono::steady_clock::time_point t_last = chrono::steady_clock::now();
        unsigned long int displayed = 0, dropped = 0, corrupted = 0, missing = 0;
        atomic_ulong fec_ok = 0, fec_corrected = 0, fec_nok = 0;
        void print() {
                ostringstream fec;
                if (fec_ok + fec_nok + fec_corrected > 0) {
                        fec << " FEC noerr/OK/NOK: " << SBOLD(fec_ok) << "/" << SBOLD(fec_corrected) << "/" << SBOLD(fec_nok);
                }
                unsigned long total = displayed + dropped + missing;
                LOG(LOG_LEVEL_INFO) << SUNDERLINE("Video dec stats") << " (cumulative): "
                        << SBOLD(total) << " total / "
                        << SBOLD(displayed) << " disp / "
                        << SBOLD(dropped) << " drop / "
                        << SBOLD(corrupted) << " corr / "
                        << SBOLD(missing) << " miss"
                        << fec.str() << "\n";
                if (total > 3000 && dropped * 50 >= total) { // more than 2% frames were dropped
                        log_msg_once(LOG_LEVEL_WARNING, to_fourcc('D', 'R', 'P', 'S'), MOD_NAME "Dropped %lu of %lu frames. This may be due "
                                        "to network jitter, try adding \"--param decoder-drop-policy=blocking\" if the problem persists.\n", dropped, total);
                }
        }
        void update(int buffer_number) {
                if (last_buffer_number != -1) {
                        long long int diff = buffer_number -
                                ((last_buffer_number + 1) & ((1U<<BUFNUM_BITS) - 1));
                        diff = (diff + (1U<<BUFNUM_BITS)) % (1U<<BUFNUM_BITS);
                        if (diff < (1U<<BUFNUM_BITS) / 2) {
                                missing += diff;
                        } else { // frames may have been reordered, add arbitrary 1
                                missing += 1;
                        }
                }
                last_buffer_number = buffer_number;
                auto now = chrono::steady_clock::now();
                if (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - t_last).count() > CUMULATIVE_REPORTS_INTERVAL) {
                        print();
                        t_last = now;
                }
        }
};

// message definitions
struct frame_msg {
        inline frame_msg(struct control_state *c, struct reported_statistics_cumul &sr) : control(c), recv_frame(nullptr),
                                nofec_frame(nullptr),
                             received_pkts_cum(0), expected_pkts_cum(0),
                             stats(sr)
        {}
        inline ~frame_msg() {
                if (recv_frame) {
                        int received_bytes = 0;
                        for (unsigned int i = 0; i < recv_frame->tile_count; ++i) {
                                received_bytes += sum_map(pckt_list[i]);
                        }
                        int expected_bytes = vf_get_data_len(recv_frame);
                        if (recv_frame->fec_params.type != FEC_NONE) {
                                if (is_corrupted) {
                                        stats.fec_nok += 1;
                                } else {
                                        if (received_bytes == expected_bytes) {
                                                stats.fec_ok += 1;
                                        } else {
                                                stats.fec_corrected += 1;
                                        }
                                }
                        }
                        stats.corrupted += is_corrupted;
                        stats.displayed += is_displayed;
                        stats.dropped += !is_displayed;
                }
                vf_free(recv_frame);
                vf_free(nofec_frame);
        }
        struct control_state *control;
        vector <uint32_t> buffer_num;
        struct video_frame *recv_frame; ///< received frame with FEC and/or compression
        struct video_frame *nofec_frame; ///< frame without FEC
        unique_ptr<map<int, int>[]> pckt_list;
        unsigned long long int received_pkts_cum, expected_pkts_cum;
        struct reported_statistics_cumul &stats;
        bool is_corrupted = false;
        bool is_displayed = false;
};

struct main_msg_reconfigure {
        inline main_msg_reconfigure(struct video_desc d,
                        unique_ptr<frame_msg> &&f,
                        bool force = false,
                        struct pixfmt_desc pd = {}) :
                desc(d),
                last_frame(std::move(f)),
                force(force),
                compress_internal_prop(pd) {}

        struct video_desc desc;
        unique_ptr<frame_msg> last_frame;
        bool force;
        struct pixfmt_desc compress_internal_prop;
};
}

/**
 * @brief Decoder state
 */
struct state_video_decoder
{
        state_video_decoder(struct module *parent) {
                module_init_default(&mod);
                mod.cls = MODULE_CLASS_DECODER;
                mod.priv_data = this;
                mod.new_message = decoder_process_message;
                module_register(&mod, parent);
                control = (struct control_state *) get_module(get_root_module(parent), "control");
        }
        ~state_video_decoder() {
                module_done(&mod);
        }
        struct module mod;
        struct control_state *control = {};

        thread decompress_thread_id,
                  fec_thread_id;
        struct video_desc received_vid_desc = {}; ///< description of the network video
        struct pixfmt_desc received_int_desc = {}; ///< compression int desc
        struct video_desc display_desc = {};      ///< description of the mode that display is currently configured to

        struct video_frame *frame = NULL; ///< @todo rewrite this more reasonably

        struct display   *display = NULL; ///< assigned display device
        /// @{
        vector<codec_t>   native_codecs; ///< list of display native codecs
        enum interlacing_t *disp_supported_il = NULL; ///< display supported interlacing mode
        size_t            disp_supported_il_cnt = 0; ///< count of @ref disp_supported_il
        /// @}

        unsigned int      max_substreams = 0; ///< maximal number of expected substreams
        change_il_t       change_il = NULL;      ///< function to change interlacing, if needed. Otherwise NULL.
        vector<void *>    change_il_state;

        mutex lock;

        enum decoder_type_t decoder_type = {};  ///< how will the video data be decoded
        struct line_decoder *line_decoder = NULL; ///< if the video is uncompressed and only pixelformat change
                                           ///< is neeeded, use this structure
        vector<struct state_decompress *> decompress_state; ///< state of the decompress (for every substream)
        bool accepts_corrupted_frame = false;     ///< whether we should pass corrupted frame to decompress
        bool buffer_swapped = true; /**< variable indicating that display buffer
                              * has been processed and we can write to a new one */
        condition_variable buffer_swapped_cv; ///< condition variable associated with @ref buffer_swapped

        synchronized_queue<unique_ptr<frame_msg>, 1> decompress_queue;

        codec_t           out_codec = VIDEO_CODEC_NONE;
        int               pitch = 0;

        synchronized_queue<unique_ptr<frame_msg>, 1> fec_queue;

        enum video_mode   video_mode = {} ;  ///< video mode set for this decoder
        bool          merged_fb = false; ///< flag if the display device driver requires tiled video or not

        timed_message<LOG_LEVEL_WARNING> slow_msg; ///< shows warning ony in certain interval

        synchronized_queue<main_msg_reconfigure *, -1> msg_queue;

        const struct openssl_decrypt_info *dec_funcs = NULL; ///< decrypt state
        struct openssl_decrypt      *decrypt = NULL; ///< decrypt state

        struct reported_statistics_cumul stats = {}; ///< stats to be reported through control socket
};

/**
 * This function blocks until video frame is displayed and decoder::frame
 * can be filled with new data. Until this point, the video frame is not considered
 * valid.
 */
static void wait_for_framebuffer_swap(struct state_video_decoder *decoder) {
        unique_lock<mutex> lk(decoder->lock);
        decoder->buffer_swapped_cv.wait(lk, [decoder]{return decoder->buffer_swapped;});
}

#define ENCRYPTED_ERR "Receiving encrypted video data but " \
        "no decryption key entered!\n"
#define NOT_ENCRYPTED_ERR "Receiving unencrypted video data " \
        "while expecting encrypted.\n"

static void *fec_thread(void *args) {
        set_thread_name(__func__);
        struct state_video_decoder *decoder =
                (struct state_video_decoder *) args;

        fec *fec_state = NULL;
        struct fec_desc desc(FEC_NONE);

        while(1) {
                unique_ptr<frame_msg> data = decoder->fec_queue.pop();

                if (!data->recv_frame) { // poisoned
                        decoder->decompress_queue.push(std::move(data));
                        break; // exit from loop
                }

                struct video_frame *frame = decoder->frame;
                struct tile *tile = NULL;

                if (data->recv_frame->fec_params.type != FEC_NONE) {
                        if(!fec_state || desc.k != data->recv_frame->fec_params.k ||
                                        desc.m != data->recv_frame->fec_params.m ||
                                        desc.c != data->recv_frame->fec_params.c ||
                                        desc.seed != data->recv_frame->fec_params.seed
                          ) {
                                delete fec_state;
                                desc = data->recv_frame->fec_params;
                                fec_state = fec::create_from_desc(desc);
                                if(fec_state == NULL) {
                                        log_msg(LOG_LEVEL_FATAL, "[decoder] Unable to initialize FEC.\n");
                                        exit_uv(1);
                                        goto cleanup;
                                }
                        }
                }

                data->nofec_frame = vf_alloc(data->recv_frame->tile_count);
                data->nofec_frame->ssrc = data->recv_frame->ssrc;
                data->nofec_frame->timestamp = data->recv_frame->timestamp;

                if (data->recv_frame->fec_params.type != FEC_NONE) {
                        bool buffer_swapped = false;
                        for (int pos = 0; pos < get_video_mode_tiles_x(decoder->video_mode)
                                        * get_video_mode_tiles_y(decoder->video_mode); ++pos) {
                                char *fec_out_buffer = NULL;
                                int fec_out_len = 0;

                                if (data->recv_frame->tiles[pos].data_len != (unsigned int) sum_map(data->pckt_list[pos])) {
                                        debug_msg("Frame incomplete - substream %d, buffer %d: expected %u bytes, got %u.\n", pos,
                                                        (unsigned int) data->buffer_num[pos],
                                                        data->recv_frame->tiles[pos].data_len,
                                                        (unsigned int) sum_map(data->pckt_list[pos]));
                                }

                                bool ret = fec_state->decode(data->recv_frame->tiles[pos].data,
                                                data->recv_frame->tiles[pos].data_len,
                                                &fec_out_buffer, &fec_out_len, data->pckt_list[pos]);

                                if (ret == false) {
                                        data->is_corrupted = true;
                                        verbose_msg("[decoder] FEC: unable to reconstruct data.\n");
                                        if (fec_out_len < (int) sizeof(video_payload_hdr_t)) {
                                                goto cleanup;
                                        }
                                        if (decoder->decoder_type == EXTERNAL_DECODER && !decoder->accepts_corrupted_frame) {
                                                goto cleanup;
                                        }
                                }

                                video_payload_hdr_t video_hdr;
                                memcpy(&video_hdr, fec_out_buffer,
                                                sizeof(video_payload_hdr_t));
                                fec_out_buffer += sizeof(video_payload_hdr_t);
                                fec_out_len -= sizeof(video_payload_hdr_t);

                                struct video_desc network_desc;
                                parse_video_hdr(video_hdr, &network_desc);
                                if (!video_desc_eq_excl_param(decoder->received_vid_desc,
                                                        network_desc, PARAM_TILE_COUNT)) {
                                        decoder->msg_queue.push(new main_msg_reconfigure(network_desc, std::move(data)));
                                        goto cleanup;
                                }

                                if (FRAMEBUFFER_NOT_READY(decoder)) {
                                        goto cleanup;
                                }

                                if(decoder->decoder_type == EXTERNAL_DECODER) {
                                        data->nofec_frame->tiles[pos].data_len = fec_out_len;
                                        data->nofec_frame->tiles[pos].data = fec_out_buffer;
                                } else { // linedecoder
                                        if (!buffer_swapped) {
                                                buffer_swapped = true;
                                                wait_for_framebuffer_swap(decoder);
                                                unique_lock<mutex> lk(decoder->lock);
                                                decoder->buffer_swapped = false;
                                        }

                                        int divisor;

                                        if (!decoder->merged_fb) {
                                                divisor = decoder->max_substreams;
                                        } else {
                                                divisor = 1;
                                        }

                                        tile = vf_get_tile(frame, pos % divisor);

                                        struct line_decoder *line_decoder =
                                                &decoder->line_decoder[pos];

                                        int data_pos = 0;
                                        char *src = fec_out_buffer;
                                        char *dst = tile->data + line_decoder->base_offset;
                                        while(data_pos < (int) fec_out_len) {
                                                line_decoder->decode_line((unsigned char*)dst, (unsigned char *) src, line_decoder->dst_linesize,
                                                                line_decoder->shifts[0],
                                                                line_decoder->shifts[1],
                                                                line_decoder->shifts[2]);
                                                src += line_decoder->src_linesize;
                                                dst += vc_get_linesize(tile->width ,frame->color_spec);
                                                data_pos += line_decoder->src_linesize;
                                        }
                                }
                        }
                } else { /* PT_VIDEO */
                        for(int i = 0; i < (int) decoder->max_substreams; ++i) {
                                data->nofec_frame->tiles[i].data_len = data->recv_frame->tiles[i].data_len;
                                data->nofec_frame->tiles[i].data = data->recv_frame->tiles[i].data;

                                if (data->recv_frame->tiles[i].data_len != (unsigned int) sum_map(data->pckt_list[i])) {
                                        debug_msg("Frame incomplete - substream %d, buffer %d: expected %u bytes, got %u.%s\n", i,
                                                        (unsigned int) data->buffer_num[i],
                                                        data->recv_frame->tiles[i].data_len,
                                                        (unsigned int) sum_map(data->pckt_list[i]),
                                                        decoder->decoder_type == EXTERNAL_DECODER && !decoder->accepts_corrupted_frame ? " dropped.\n" : "");
                                        data->is_corrupted = true;
                                        if(decoder->decoder_type == EXTERNAL_DECODER && !decoder->accepts_corrupted_frame) {
                                                goto cleanup;
                                        }
                                }
                        }
                }

                decoder->decompress_queue.push(std::move(data));
cleanup:
                ;
        }

        delete fec_state;

        return NULL;
}

static bool blacklist_current_out_codec(struct state_video_decoder *decoder){
        if(decoder->out_codec == VIDEO_CODEC_NONE)
                return false;

        auto to_erase = find(decoder->native_codecs.begin(), decoder->native_codecs.end(), decoder->out_codec);
        if (to_erase == decoder->native_codecs.end()) {
                return true; // should it really return true?
        }
        log_msg(LOG_LEVEL_DEBUG, "Blacklisting codec %s\n", get_codec_name(decoder->out_codec));
        decoder->native_codecs.erase(to_erase);

        return true;
}

struct decompress_data {
        struct state_video_decoder *decoder;
        int pos;
        struct video_frame *compressed;
        int buffer_num;
        decompress_status ret = DECODER_NO_FRAME;
        unsigned char *out;
        struct pixfmt_desc internal_prop; // set only if probing (ret == DECODER_GOT_CODEC)
};
static void *decompress_worker(void *data)
{
        auto d = (struct decompress_data *) data;
        struct state_video_decoder *decoder = d->decoder;

        if (!d->compressed->tiles[d->pos].data)
                return NULL;
        d->ret = decompress_frame(decoder->decompress_state.at(d->pos),
                        (unsigned char *) d->out,
                        (unsigned char *) d->compressed->tiles[d->pos].data,
                        d->compressed->tiles[d->pos].data_len,
                        d->buffer_num,
                        &decoder->frame->callbacks,
                        &d->internal_prop);
        return d;
}

ADD_TO_PARAM("decoder-drop-policy",
                "* decoder-drop-policy=blocking|nonblock|<sec>\n"
                "  Force specified blocking policy (default nonblock).\n"
                "  <sec> - specifies frame timeout in seconds (can have suffixes, eg. \"20ms\")\n");
static void *decompress_thread(void *args) {
        set_thread_name(__func__);
        struct state_video_decoder *decoder =
                (struct state_video_decoder *) args;
        int tile_width = decoder->received_vid_desc.width; // get_video_mode_tiles_x(decoder->video_mode);
        int tile_height = decoder->received_vid_desc.height; // get_video_mode_tiles_y(decoder->video_mode);

        long long force_putf_timeout = []() {
                auto drop_policy = commandline_params.find("decoder-drop-policy"s);
                if (drop_policy == commandline_params.end()) {
                        return -1LL;
                }
                if (drop_policy->second == "nonblock") {
                        return PUTF_NONBLOCK;
                }
                if (drop_policy->second == "blocking") {
                        return PUTF_BLOCKING;
                }
                return static_cast<long long>(
                    unit_evaluate_dbl(drop_policy->second.c_str(), true,
                                      nullptr) *
                    NS_IN_SEC);
        }();

        while(1) {
                unique_ptr<frame_msg> msg = decoder->decompress_queue.pop();

                if(!msg->recv_frame) { // poisoned
                        break;
                }

                auto t0 = std::chrono::high_resolution_clock::now();
                unique_ptr<char[]> tmp;

                if (decoder->out_codec == VIDEO_CODEC_END) {
                        tmp = unique_ptr<char[]>(new char[tile_height * (tile_width * MAX_BPS + MAX_PADDING)]);
                }

                if(decoder->decoder_type == EXTERNAL_DECODER) {
                        int tile_count = get_video_mode_tiles_x(decoder->video_mode) *
                                        get_video_mode_tiles_y(decoder->video_mode);
                        vector<task_result_handle_t> handle(tile_count);
                        vector<decompress_data> data(tile_count);
                        for (int pos = 0; pos < tile_count; ++pos) {
                                data[pos].decoder = decoder;
                                data[pos].pos = pos;
                                data[pos].compressed = msg->nofec_frame;
                                data[pos].buffer_num = msg->buffer_num[pos];
                                if (tmp.get()) {
                                        data[pos].out = (unsigned char *) tmp.get();
                                } else if (decoder->merged_fb) {
                                        // TODO: OK when rendering directly to display FB, otherwise, do not reflect pitch (we use PP)
                                        int x = pos % get_video_mode_tiles_x(decoder->video_mode),
                                            y = pos / get_video_mode_tiles_x(decoder->video_mode);
                                        data[pos].out = (unsigned char *) vf_get_tile(decoder->frame, 0)->data + y * decoder->pitch * tile_height +
                                                vc_get_linesize(tile_width, decoder->out_codec) * x;
                                } else {
                                        data[pos].out = (unsigned char *) vf_get_tile(decoder->frame, pos)->data;
                                }
                                if (tile_count > 1) {
                                        handle[pos] = task_run_async(decompress_worker, &data[pos]);
                                } else {
                                        decompress_worker(&data[pos]);
                                }
                        }
                        if (tile_count > 1) {
                                for (int pos = 0; pos < tile_count; ++pos) {
                                        wait_task(handle[pos]);
                                }
                        }
                        for (int pos = 0; pos < tile_count; ++pos) {
                                if (data[pos].ret == DECODER_GOT_CODEC) {
                                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Detected compression properties: " << get_pixdesc_desc(data[pos].internal_prop) << "\n";
                                        decoder->msg_queue.push(new main_msg_reconfigure(decoder->received_vid_desc, nullptr, false, data[pos].internal_prop));
                                        goto skip_frame;
                                }
                                if (data[pos].ret != DECODER_GOT_FRAME){
                                        if (data[pos].ret == DECODER_UNSUPP_PIXFMT) {
                                                if(blacklist_current_out_codec(decoder))
                                                        decoder->msg_queue.push(new main_msg_reconfigure(decoder->received_vid_desc, nullptr, true));
                                        }
                                        goto skip_frame;
                                }
                        }
                } else {
                        if (decoder->frame->decoder_overrides_data_len == TRUE) {
                                for (unsigned int i = 0; i < decoder->frame->tile_count; ++i) {
                                        decoder->frame->tiles[i].data_len = msg->nofec_frame->tiles[i].data_len;
                                }
                        }
                }

                LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "Decompress duration: " <<
                        duration_cast<nanoseconds>(high_resolution_clock::now() - t0).count() / 1000000.0 << " ms\n";

                if(decoder->change_il) {
                        for(unsigned int i = 0; i < decoder->frame->tile_count; ++i) {
                                struct tile *tile = vf_get_tile(decoder->frame, i);
                                decoder->change_il(tile->data, tile->data, vc_get_linesize(tile->width,
                                                        decoder->out_codec), tile->height, &decoder->change_il_state[i]);
                        }
                }

                {
                        long long putf_timeout = force_putf_timeout != -1 ? force_putf_timeout : PUTF_NONBLOCK; // originally was BLOCKING when !is_codec_interframe(decoder->received_vid_desc.color_spec)

                        decoder->frame->ssrc = msg->nofec_frame->ssrc;
                        decoder->frame->timestamp = msg->nofec_frame->timestamp;
                        const bool ret = display_put_frame(
                            decoder->display, decoder->frame, putf_timeout);
                        msg->is_displayed = ret;
                        decoder->frame = display_get_frame(decoder->display);
                        assert(decoder->frame != nullptr);
                }

skip_frame:
                {
                        unique_lock<mutex> lk(decoder->lock);
                        // we have put the video frame and requested another one which is
                        // writable so on
                        decoder->buffer_swapped = true;
                        lk.unlock();
                        decoder->buffer_swapped_cv.notify_one();
                }
        }

        return NULL;
}

static void decoder_set_video_mode(struct state_video_decoder *decoder, enum video_mode video_mode)
{
        decoder->video_mode = video_mode;
        decoder->max_substreams = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);
}

/**
 * @brief Initializes video decompress state.
 * @param video_mode  video_mode expected to be received from network
 * @param display     Video display that is supposed to be controlled from decoder.
 *                    display_get_frame(), display_put_frame(), display_get_propert()
 *                    and display_reconfigure() functions may be used. If set to NULL,
 *                    no decoding will take place.
 * @param encryption  Encryption config string. Currently, this is a passphrase to be
 *                    used. This may change eventually.
 * @return Newly created decoder state. If an error occured, returns NULL.
 */
struct state_video_decoder *video_decoder_init(struct module *parent,
                enum video_mode video_mode,
                struct display *display, const char *encryption)
{
        struct state_video_decoder *s;

        s = new state_video_decoder(parent);

        if (encryption) {
                s->dec_funcs = static_cast<const struct openssl_decrypt_info *>(load_library("openssl_decrypt",
                                        LIBRARY_CLASS_UNDEFINED, OPENSSL_DECRYPT_ABI_VERSION));
                if (!s->dec_funcs) {
                                log_msg(LOG_LEVEL_FATAL, "UltraGrid was build without OpenSSL support!\n");
                                delete s;
                                return NULL;
                }
                if (s->dec_funcs->init(&s->decrypt, encryption) != 0) {
                        log_msg(LOG_LEVEL_FATAL, "Unable to create decompress!\n");
                        delete s;
                        return NULL;
                }
        }

        decoder_set_video_mode(s, video_mode);

        if(!video_decoder_register_display(s, display)) {
                delete s;
                return NULL;
        }

        return s;
}

/**
 * @brief starts decompress and ldmg threads
 *
 * Called from video_decoder_register_display(). It is also called after
 * video_decoder_stop_threads() in reconfiguration.
 *
 * @invariant
 * decoder->display != NULL
 */
static void video_decoder_start_threads(struct state_video_decoder *decoder)
{
        assert(decoder->display); // we want to run threads only if decoder is active

        decoder->decompress_thread_id = thread(decompress_thread, decoder);
        decoder->fec_thread_id = thread(fec_thread, decoder);
}

/**
 * @brief This function stops running threads.
 *
 * @invariant
 * decoder->display != NULL
 */
static void video_decoder_stop_threads(struct state_video_decoder *decoder)
{
        assert(decoder->display);

        unique_ptr<frame_msg> msg(new frame_msg(decoder->control, decoder->stats));
        decoder->fec_queue.push(std::move(msg));

        decoder->fec_thread_id.join();
        decoder->decompress_thread_id.join();
}

static auto codec_list_to_str(vector<codec_t> const &codecs) {
        if (codecs.empty()) {
                return "(none)"s;
        }
        ostringstream oss;
        auto it = codecs.begin();
        oss << *it++;
        for ( ; it != codecs.end(); ++it) {
                oss << (it + 1 == codecs.end() ? " and " : ", ") << get_codec_name(*it);
        }
        return oss.str();
}

ADD_TO_PARAM("decoder-use-codec",
                "* decoder-use-codec=<codec>\n"
                "  Use specified pixel format for decoding (eg. v210). This overrides automatic\n"
                "  choice. The pixel format must be supported by the video display. Use 'help' to see\n"
                "  available options for a display (eg.: 'uv -d gl --param decoder-use-codec=help').\n");
/**
 * @brief Registers video display to be used for displaying decoded video frames.
 *
 * No display should be managed by this decoder when this function is called.
 *
 * @see decoder_remove_display
 *
 * @param decoder decoder state
 * @param display new display to be registered
 * @return        whether registration was successful
 */
bool video_decoder_register_display(struct state_video_decoder *decoder, struct display *display)
{
        assert(display != NULL);
        assert(decoder->display == NULL);

        int ret;

        decoder->display = display;

        codec_t native_codecs[VIDEO_CODEC_COUNT];
        size_t len = sizeof native_codecs;
        ret = display_ctl_property(decoder->display, DISPLAY_PROPERTY_CODECS, native_codecs, &len);
        decoder->native_codecs.clear();
        if (!ret) {
                log_msg(LOG_LEVEL_ERROR, "Failed to query codecs from video display.\n");
        } else {
                for (size_t i = 0; i < len / sizeof(codec_t); ++i) {
                        decoder->native_codecs.push_back(native_codecs[i]);
                }
        }
        if (get_commandline_param("decoder-use-codec")) {
                const char *codec_str = get_commandline_param("decoder-use-codec");
                codec_t req_codec = get_codec_from_name(codec_str);
                if ("help"s == codec_str) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Supported codecs for current display are: " << codec_list_to_str(decoder->native_codecs) << "\n";
                        return false;
                }
                if (req_codec == VIDEO_CODEC_NONE) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong decoder codec spec: %s.\n", codec_str);
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Supported codecs for current display are: " << codec_list_to_str(decoder->native_codecs) << "\n";
                        return false;
                }
                if (find(decoder->native_codecs.begin(), decoder->native_codecs.end(), req_codec) != end(decoder->native_codecs)) {
                        decoder->native_codecs.clear();
                        decoder->native_codecs.push_back(req_codec);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Display doesn't support requested codec: %s.\n", codec_str);
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Supported codecs for current display are: " << codec_list_to_str(decoder->native_codecs) << "\n";
                        return false;
                }
        }

        free(decoder->disp_supported_il);
        decoder->disp_supported_il_cnt = 20 * sizeof(enum interlacing_t);
        decoder->disp_supported_il = (enum interlacing_t*) calloc(decoder->disp_supported_il_cnt,
                        sizeof(enum interlacing_t));
        ret = display_ctl_property(decoder->display, DISPLAY_PROPERTY_SUPPORTED_IL_MODES, decoder->disp_supported_il, &decoder->disp_supported_il_cnt);
        if(ret) {
                decoder->disp_supported_il_cnt /= sizeof(enum interlacing_t);
        } else {
                enum interlacing_t tmp[] = { PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME}; /* default if not said othervise */
                memcpy(decoder->disp_supported_il, tmp, sizeof(tmp));
                decoder->disp_supported_il_cnt = sizeof(tmp) / sizeof(enum interlacing_t);
        }

        video_decoder_start_threads(decoder);

        return true;
}

/**
 * @brief This removes display from current decoder.
 *
 * From now on, no video frames will be decoded with current decoder.
 * @see decoder_register_display - the counterpart of this functon
 *
 * @param decoder decoder from which will the decoder be removed
 */
void video_decoder_remove_display(struct state_video_decoder *decoder)
{
        if (decoder->display) {
                video_decoder_stop_threads(decoder);
                if (decoder->frame) {
                        display_put_frame(decoder->display, decoder->frame, PUTF_DISCARD);
                        decoder->frame = NULL;
                }
                decoder->display = NULL;
                memset(&decoder->display_desc, 0, sizeof(decoder->display_desc));
        }
}

static void cleanup(struct state_video_decoder *decoder)
{
        decoder->decoder_type = UNSET;
        for (auto &d : decoder->decompress_state) {
                decompress_done(d);
        }
        decoder->decompress_state.clear();
        if(decoder->line_decoder) {
                free(decoder->line_decoder);
                decoder->line_decoder = NULL;
        }

        for (auto && item : decoder->change_il_state) {
                free(item);
        }
        decoder->change_il_state.resize(0);
}

/**
 * @brief Destroys decoder created with decoder_init()
 * @param decoder decoder to be destroyed. If NULL, no action is performed.
 */
void video_decoder_destroy(struct state_video_decoder *decoder)
{
        if(!decoder)
                return;

        if (decoder->dec_funcs) {
                decoder->dec_funcs->destroy(decoder->decrypt);
        }

        video_decoder_remove_display(decoder);

        cleanup(decoder);

        free(decoder->disp_supported_il);

        delete decoder;
}

/**
 * Reorders display codecs to match compression internal format.
 *
 * First try to add HW-accelerated codecs, then exactly comp_int_fmt (if
 * available) and then sort the rest - matching color-space first, higher bit
 * depths first.
 *
 * Always first try "native" than generic decoder (native is the one matching
 * comp_int_fmt, generic should be catch-all allowing decompression of
 * arbitrary compressed stream of received codec).
 */
static vector<pair<struct pixfmt_desc, codec_t>> video_decoder_order_output_codecs(struct pixfmt_desc comp_int_prop, vector<codec_t> const &display_codecs)
{
        vector<pair<struct pixfmt_desc, codec_t>> ret;
        set<codec_t> used;
        // first add hw-accelerated codecs
        for (auto codec : display_codecs) {
                if (codec_is_hw_accelerated(codec)) {
                        ret.emplace_back(comp_int_prop, codec);
                        if (comp_int_prop.depth != 0) {
                                ret.emplace_back(pixfmt_desc{}, codec);
                        }
                        used.insert(codec);
                }
        }
        // then codecs matching exactly internal codec
        for (auto codec : display_codecs) {
                if (used.find(codec) != used.end()) {
                        continue;
                };
                if (pixdesc_equals(get_pixfmt_desc(codec), comp_int_prop)) {
                        ret.emplace_back(comp_int_prop, codec);
                        if (comp_int_prop.depth != 0) {
                                ret.emplace_back(pixfmt_desc{}, codec);
                        }
                        used.insert(codec);
                }
        }
        // then add also all other codecs
        vector<codec_t> remaining;
        for (auto codec : display_codecs) {
                if (used.find(codec) != used.end()) {
                        continue;
                };
                remaining.emplace_back(codec);
        }
        if (comp_int_prop.depth != 0) {
                sort(remaining.begin(), remaining.end(), [comp_int_prop](const codec_t &a, const codec_t &b) {
                                struct pixfmt_desc desc_a = get_pixfmt_desc(a);
                                struct pixfmt_desc desc_b = get_pixfmt_desc(b);
                                return compare_pixdesc(&desc_a, &desc_b, &comp_int_prop) < 0;
                        });
        }
        for (auto & c : remaining) {
                ret.emplace_back(comp_int_prop, c);
                if (comp_int_prop.depth != 0) {
                        ret.emplace_back(pixfmt_desc{}, c);
                }
        }

        if (log_level >= LOG_LEVEL_VERBOSE) {
                LOG(LOG_LEVEL_VERBOSE) << "Trying codecs in this order:\n";
                for (auto it = ret.begin(); it != ret.end(); ++it) {
                        LOG(LOG_LEVEL_VERBOSE) << "\t" << get_codec_name((*it).second) << ", internal: " << get_pixdesc_desc((*it).first) << "\n";
                }
        }

        return ret;
}

/**
 * This function selects, according to given video description, appropriate
 *
 * @param[in]  decoder     decoder to be taken parameters from (video mode, native codecs etc.)
 * @param[in]  desc        incoming video description
 * @param[out] decode_line If chosen decoder is a linedecoder, this variable contains the
 *                         decoding function.
 * @return                 Output codec, if no decoding function found, -1 is returned.
 */
static codec_t choose_codec_and_decoder(struct state_video_decoder *decoder, struct video_desc desc,
                                decoder_t *decode_line, struct pixfmt_desc comp_int_prop)
{
        codec_t out_codec = VIDEO_CODEC_NONE;

        /* first check if the codec is natively supported */
        for (auto &codec : decoder->native_codecs) {
                if (desc.color_spec == codec) {
                        if ((desc.color_spec == DXT1 || desc.color_spec == DXT1_YUV ||
                                        desc.color_spec == DXT5)
                                        && decoder->video_mode != VIDEO_NORMAL)
                                continue; /// DXT1 it is a exception, see @ref vdec_note1

                        *decode_line = vc_memcpy;
                        decoder->decoder_type = LINE_DECODER;

                        if(desc.color_spec == RGBA || /* another exception - we may change shifts */
                                        desc.color_spec == RGB) { // should RGB be also handled
                                *decode_line = get_decoder_from_to(desc.color_spec, desc.color_spec);
                        }

                        out_codec = codec;
                        goto after_linedecoder_lookup;
                }
        }
        /* otherwise if we have line decoder (incl. slow codecs) */
        {
                vector<codec_t> native_codecs_copy = decoder->native_codecs;
                native_codecs_copy.push_back(VIDEO_CODEC_NONE); // this needs to be NULL-terminated
                *decode_line = get_best_decoder_from(desc.color_spec, native_codecs_copy.data(), &out_codec);
                if (*decode_line) {
                        decoder->decoder_type = LINE_DECODER;
                        goto after_linedecoder_lookup;
                }
        }

after_linedecoder_lookup:

        /* we didn't find line decoder. So try now regular (aka DXT) decoder */
        if(*decode_line == NULL) {
                decoder->decompress_state.resize(decoder->max_substreams);

                // try to probe video format
                if (comp_int_prop.depth == 0 && decoder->out_codec != VIDEO_CODEC_END) {
                        bool supports_autodetection = decompress_init_multi(desc.color_spec,
                                        pixfmt_desc{}, VIDEO_CODEC_NONE, decoder->decompress_state.data(),
                                        decoder->decompress_state.size());
                        if (supports_autodetection) {
                                decoder->decoder_type = EXTERNAL_DECODER;
                                return VIDEO_CODEC_END;
                        }
                }

                vector<pair<struct pixfmt_desc, codec_t>> formats_to_try; // comp_int_prop (may be empty), display_fmt
                formats_to_try = video_decoder_order_output_codecs(comp_int_prop, decoder->native_codecs);

                for (auto it = formats_to_try.begin(); it != formats_to_try.end(); ++it) {
                        out_codec = (*it).second;
                        if (decompress_init_multi(desc.color_spec, (*it).first,
                                                (*it).second,
                                                decoder->decompress_state.data(),
                                                decoder->decompress_state.size())) {
                                decoder->decoder_type = EXTERNAL_DECODER;
                                goto after_decoder_lookup;
                        }
                }
                decoder->decompress_state.clear();
        }
after_decoder_lookup:

        if(decoder->decoder_type == UNSET) {
                log_msg(LOG_LEVEL_ERROR, "Unable to find decoder for input codec \"%s\"!!!\n", get_codec_name(desc.color_spec));
                LOG(LOG_LEVEL_INFO) << "Compression internal codec is \"" << get_pixdesc_desc(comp_int_prop) << "\". Native codecs are: " << codec_list_to_str(decoder->native_codecs) << "\n";
                return VIDEO_CODEC_NONE;
        }

        return out_codec;
}

/**
 * This function finds interlacing mode changing function.
 *
 * @param[in]  in_il       input_interlacing
 * @param[in]  supported   list of supported output interlacing modes
 * @param[in]  il_out_cnt  count of supported items
 * @param[out] out_il      selected output interlacing
 * @return                 selected interlacing changing function, NULL if not needed or not found
 */
static change_il_t select_il_func(enum interlacing_t in_il, enum interlacing_t *supported,
                int il_out_cnt, /*out*/ enum interlacing_t *out_il)
{
        struct transcode_t { enum interlacing_t in; enum interlacing_t out; change_il_t func; };

        struct transcode_t transcode[] = {
                {LOWER_FIELD_FIRST, INTERLACED_MERGED, il_lower_to_merged},
                {UPPER_FIELD_FIRST, INTERLACED_MERGED, il_upper_to_merged},
                {INTERLACED_MERGED, UPPER_FIELD_FIRST, il_merged_to_upper}
        };

        int i;
        /* first try to check if it can be nativelly displayed */
        for (i = 0; i < il_out_cnt; ++i) {
                if(in_il == supported[i]) {
                        *out_il = in_il;
                        return NULL;
                }
        }

        for (i = 0; i < il_out_cnt; ++i) {
                size_t j;
                for (j = 0; j < sizeof(transcode) / sizeof(struct transcode_t); ++j) {
                        if(in_il == transcode[j].in && supported[i] == transcode[j].out) {
                                *out_il = transcode[j].out;
                                return transcode[j].func;
                        }
                }
        }

        log_msg(LOG_LEVEL_WARNING, "[Warning] Cannot find transition between incoming and display "
                        "interlacing modes!\n");
        return NULL;
}

/**
 * Reconfigures decoder if network received video data format has changed.
 *
 * @param decoder       the video decoder state
 * @param desc          new video description to be reconfigured to
 * @return              boolean value if reconfiguration was successful
 *
 * @invariant
 * decoder->display != NULL
 */
static bool reconfigure_decoder(struct state_video_decoder *decoder,
                struct video_desc desc, struct pixfmt_desc comp_int_prop)
{
        codec_t out_codec;
        decoder_t decode_line;
        enum interlacing_t display_il = PROGRESSIVE;
        //struct video_frame *frame;
        int display_requested_pitch = PITCH_DEFAULT;
        int display_requested_rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;

        // this code forces flushing the pipelined data
        video_decoder_stop_threads(decoder);
        if (decoder->frame)
                display_put_frame(decoder->display, decoder->frame, PUTF_DISCARD);
        decoder->frame = NULL;
        video_decoder_start_threads(decoder);

        cleanup(decoder);

        desc.tile_count = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);

        out_codec = choose_codec_and_decoder(decoder, desc, &decode_line, comp_int_prop);
        if (out_codec == VIDEO_CODEC_NONE) {
                LOG(LOG_LEVEL_ERROR) << "Could not find neither line conversion nor decompress from " <<
                        get_codec_name(desc.color_spec) << " to display supported formats (" << codec_list_to_str(decoder->native_codecs) << ").\n";
                return false;
        }
        decoder->out_codec = out_codec;
        struct video_desc display_desc = desc;

        int display_mode = DISPLAY_PROPERTY_VIDEO_MERGED; // default
        size_t len = sizeof(int);

        if (!display_ctl_property(decoder->display, DISPLAY_PROPERTY_VIDEO_MODE,
                        &display_mode, &len)) {
                debug_msg("Failed to get video display mode.\n");
        }

        if (display_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_3D) {
                display_mode = display_desc.tile_count == 2 ? DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES :
                        DISPLAY_PROPERTY_VIDEO_MERGED;
        }

        if (display_mode == DISPLAY_PROPERTY_VIDEO_MERGED) {
                display_desc.width *= get_video_mode_tiles_x(decoder->video_mode);
                display_desc.height *= get_video_mode_tiles_y(decoder->video_mode);
                display_desc.tile_count = 1;
        }

        decoder->change_il = select_il_func(desc.interlacing, decoder->disp_supported_il,
                        decoder->disp_supported_il_cnt, &display_il);
        decoder->change_il_state.resize(decoder->max_substreams);
        display_desc.interlacing = display_il;
        display_desc.color_spec  = out_codec;
        if (out_codec != VIDEO_CODEC_END && !video_desc_eq(decoder->display_desc, display_desc)) {
                /* reconfigure VO and give it opportunity to pass us pitch */
                bool ret = display_reconfigure(decoder->display, display_desc, decoder->video_mode);
                if(!ret) {
                        return false;
                }
                decoder->display_desc = display_desc;

                len = sizeof(display_requested_rgb_shift);
                ret = display_ctl_property(decoder->display, DISPLAY_PROPERTY_RGB_SHIFT,
                                &display_requested_rgb_shift, &len);
                if(!ret) {
                        debug_msg("Failed to get r,g,b shift property from video driver.\n");
                        int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;
                        memcpy(&display_requested_rgb_shift, rgb_shift, sizeof(rgb_shift));
                }

                ret = display_ctl_property(decoder->display, DISPLAY_PROPERTY_BUF_PITCH,
                                &display_requested_pitch, &len);
                if(!ret) {
                        debug_msg("Failed to get pitch from video driver.\n");
                        display_requested_pitch = PITCH_DEFAULT;
                }
        }

        int linewidth;
        if (display_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                linewidth = desc.width;
        } else {
                linewidth = desc.width * get_video_mode_tiles_x(decoder->video_mode);
        }

        if(display_requested_pitch == PITCH_DEFAULT)
                decoder->pitch = vc_get_linesize(linewidth, out_codec);
        else
                decoder->pitch = display_requested_pitch;


        int src_x_tiles = get_video_mode_tiles_x(decoder->video_mode);
        int src_y_tiles = get_video_mode_tiles_y(decoder->video_mode);

        if(decoder->decoder_type == LINE_DECODER) {
                decoder->line_decoder = (struct line_decoder *) malloc(src_x_tiles * src_y_tiles *
                                        sizeof(struct line_decoder));
                if(display_mode == DISPLAY_PROPERTY_VIDEO_MERGED && decoder->video_mode == VIDEO_NORMAL) {
                        struct line_decoder *out = &decoder->line_decoder[0];
                        out->base_offset = 0;
                        out->conv_num = get_pf_block_pixels(desc.color_spec) * get_pf_block_bytes(out_codec);
                        out->conv_den = get_pf_block_bytes(desc.color_spec) * get_pf_block_pixels(out_codec);
                        memcpy(out->shifts, display_requested_rgb_shift, 3 * sizeof(int));

                        out->decode_line = decode_line;
                        out->dst_pitch = decoder->pitch;
                        out->src_linesize = vc_get_linesize(desc.width, desc.color_spec);
                        out->dst_linesize = vc_get_linesize(desc.width, out_codec);
                        decoder->merged_fb = true;
                } else if(display_mode == DISPLAY_PROPERTY_VIDEO_MERGED
                                && decoder->video_mode != VIDEO_NORMAL) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x +
                                                        src_x_tiles * y];
                                        out->base_offset = y * (desc.height)
                                                        * decoder->pitch +
                                                        vc_get_linesize(x * desc.width, out_codec);

                                        out->conv_num = get_pf_block_pixels(desc.color_spec) * get_pf_block_bytes(out_codec);
                                        out->conv_den = get_pf_block_bytes(desc.color_spec) * get_pf_block_pixels(out_codec);
                                        memcpy(out->shifts, display_requested_rgb_shift,
                                                        3 * sizeof(int));

                                        out->decode_line = decode_line;

                                        out->dst_pitch = decoder->pitch;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, desc.color_spec);
                                        out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = true;
                } else if (display_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x +
                                                        src_x_tiles * y];
                                        out->base_offset = 0;
                                        out->conv_num = get_pf_block_pixels(desc.color_spec) * get_pf_block_bytes(out_codec);
                                        out->conv_den = get_pf_block_bytes(desc.color_spec) * get_pf_block_pixels(out_codec);
                                        memcpy(out->shifts, display_requested_rgb_shift,
                                                        3 * sizeof(int));

                                        out->decode_line = decode_line;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, desc.color_spec);
                                        out->dst_pitch =
                                                out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = false;
                }
        } else if (decoder->decoder_type == EXTERNAL_DECODER) {
                int buf_size;

                for(unsigned int i = 0; i < decoder->decompress_state.size(); ++i) {
                        buf_size = decompress_reconfigure(decoder->decompress_state.at(i), desc,
                                        display_requested_rgb_shift[0],
                                        display_requested_rgb_shift[1],
                                        display_requested_rgb_shift[2],
                                        decoder->pitch,
                                        out_codec == VIDEO_CODEC_END ? VIDEO_CODEC_NONE : out_codec);
                        if(!buf_size) {
                                return false;
                        }
                }
                decoder->merged_fb = display_mode != DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                int res = 0, ret;
                size_t size = sizeof(res);
                ret = decompress_get_property(decoder->decompress_state.at(0),
                                DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME,
                                &res, &size);
                decoder->accepts_corrupted_frame = ret && res;
        }

        // Pass metadata to receiver thread (it can tweak parameters)
        struct msg_receiver *msg = (struct msg_receiver *)
                new_message(sizeof(struct msg_receiver));
        msg->type = RECEIVER_MSG_VIDEO_PROP_CHANGED;
        msg->new_desc = decoder->received_vid_desc;
        struct response *resp =
                send_message_to_receiver(decoder->mod.parent, (struct message *) msg);
        free_response(resp);

        if (out_codec != VIDEO_CODEC_END) {
                decoder->frame = display_get_frame(decoder->display);
                assert(decoder->frame != nullptr);
        }

        return true;
}

bool parse_video_hdr(const uint32_t *hdr, struct video_desc *desc)
{
        uint32_t tmp;
        int fps_pt, fpsd, fd, fi;

        tmp = ntohl(hdr[0]);
        desc->tile_count = (tmp >> 22) + 1; // a bit hacky - assuming this packet is from last substream

        desc->width = ntohl(hdr[3]) >> 16;
        desc->height = ntohl(hdr[3]) & 0xffff;
        desc->color_spec = get_codec_from_fcc(hdr[4]);
        if(desc->color_spec == VIDEO_CODEC_NONE) {
                log_msg(LOG_LEVEL_ERROR, "Unknown FourCC \"%.4s\"!\n",
                        (const char *)&hdr[4]);
                return false;
        }

        tmp = ntohl(hdr[5]);
        desc->interlacing = (enum interlacing_t) (tmp >> 29);
        fps_pt = (tmp >> 19) & 0x3ff;
        fpsd = (tmp >> 15) & 0xf;
        fd = (tmp >> 14) & 0x1;
        fi = (tmp >> 13) & 0x1;

        desc->fps = compute_fps(fps_pt, fpsd, fd, fi);

        if (desc->fps == -1) {
                log_msg_once(LOG_LEVEL_WARNING, to_fourcc('U', 'F', 'P', 'S'), MOD_NAME "Unsupported FPS received (newer UG?), setting to 30.\n");
                desc->fps = 30.0;
        }

        return true;
}

static void reconfigure_helper(struct state_video_decoder *decoder,
                struct video_desc network_desc,
                struct pixfmt_desc comp_int_desc)
{
        decoder->received_vid_desc = network_desc;
        decoder->received_int_desc = comp_int_desc;

        bool ret = reconfigure_decoder(decoder, decoder->received_vid_desc, comp_int_desc);
        if (!ret) {
                log_msg(LOG_LEVEL_ERROR, "[video dec.] Reconfiguration failed!!!\n");
                decoder->frame = NULL;
                decoder->out_codec = VIDEO_CODEC_NONE;
        }
}

static void reconfigure_if_needed(struct state_video_decoder *decoder,
                struct video_desc network_desc,
                struct pixfmt_desc comp_int_desc)
{
        const bool desc_eq = video_desc_eq_excl_param(
            decoder->received_vid_desc, network_desc, PARAM_TILE_COUNT);
        const bool comp_int_eq =
            pixdesc_equals(decoder->received_int_desc, comp_int_desc);
        if (desc_eq && comp_int_eq) {
                return;
        }
        reconfigure_helper(decoder, network_desc, comp_int_desc);
}

/**
 * Checks if network format has changed.
 *
 * @param decoder decoder state
 * @param hdr     raw RTP payload header
 */
static void check_for_mode_change(struct state_video_decoder *decoder,
                const uint32_t *hdr)
{
        struct video_desc network_desc;

        parse_video_hdr(hdr, &network_desc);
        if (video_desc_eq_excl_param(decoder->received_vid_desc, network_desc,
                                      PARAM_TILE_COUNT)) {
                return;
        }
        LOG(LOG_LEVEL_NOTICE)
            << "[video dec.] New incoming video format detected: "
            << network_desc << endl;
        reconfigure_helper(decoder, network_desc, {});
}

#define ERROR_GOTO_CLEANUP ret = FALSE; goto cleanup;
#define max(a, b)       (((a) > (b))? (a): (b))

/**
 * @brief Decodes a participant buffer representing one video frame.
 * @param cdata        PBUF buffer
 * @param decoder_data @ref vcodec_state containing decoder state and some additional data
 * @retval TRUE        if decoding was successful.
 *                     It stil doesn't mean that the frame will be correctly displayed,
 *                     decoding may fail in some subsequent (asynchronous) steps.
 * @retval FALSE       if decoding failed
 */
int decode_video_frame(struct coded_data *cdata, void *decoder_data, struct pbuf_stats *stats)
{
        struct vcodec_state *pbuf_data = (struct vcodec_state *) decoder_data;
        struct state_video_decoder *decoder = pbuf_data->decoder;

        int ret = TRUE;
        int prints=0;
        int max_substreams = decoder->max_substreams;

        vector<uint32_t> buffer_num(max_substreams);
        // the following is just FEC related optimalization - normally we fill up
        // allocated buffers when we have compressed data. But in case of FEC, there
        // is just the FEC buffer present, so we point to it instead to copying
        struct video_frame *frame = vf_alloc(max_substreams);
        frame->callbacks.data_deleter = vf_data_deleter;
        unique_ptr<map<int, int>[]> pckt_list(new map<int, int>[max_substreams]);

        int buffer_number = 0;
        bool buffer_swapped = false;

        // We have no framebuffer assigned, exitting
        if(!decoder->display) {
                vf_free(frame);
                return FALSE;
        }

        main_msg_reconfigure *msg_reconf;
        while ((msg_reconf = decoder->msg_queue.pop(true /* nonblock */))) {
                if (msg_reconf->force) {
                        reconfigure_helper(decoder, msg_reconf->desc,
                                           msg_reconf->compress_internal_prop);
                        log_msg(LOG_LEVEL_VERBOSE, "forced reconf\n");
                } else {
                        reconfigure_if_needed(
                            decoder, msg_reconf->desc,
                            msg_reconf->compress_internal_prop);
                }
                if (msg_reconf->last_frame) {
                        decoder->fec_queue.push(std::move(msg_reconf->last_frame));
                }
                delete msg_reconf;
        }

        frame->ssrc = cdata->data->ssrc;
        frame->timestamp = cdata->data->ts;
        int pt = cdata->data->pt;
        if (PT_VIDEO_HAS_FEC(pt)) {
                const uint32_t *hdr = (uint32_t *)(void *)cdata->data->data;
                const uint32_t tmp = ntohl(hdr[3]);
                const int k = tmp >> 19;
                const int m = 0x1fff & (tmp >> 6);
                const int c = 0x3f & tmp;
                const int seed = ntohl(hdr[4]);
                frame->fec_params =
                    fec_desc(fec::fec_type_from_pt(pt), k, m, c, seed);
        }

        while (cdata != NULL) {
                int len;
                const char *data;
                rtp_packet *pckt = cdata->data;
                enum openssl_mode crypto_mode = MODE_AES128_NONE;

                pt = pckt->pt;
                const uint32_t *hdr = (uint32_t *)(void *) pckt->data;
                const uint32_t data_pos = ntohl(hdr[1]);
                uint32_t tmp = ntohl(hdr[0]);
                const uint32_t substream = tmp >> 22;
                buffer_number = tmp & 0x3fffff;
                const int buffer_length = ntohl(hdr[2]);

                if (PT_VIDEO_IS_ENCRYPTED(pt)) {
                        if(!decoder->decrypt) {
                                log_msg(LOG_LEVEL_ERROR, ENCRYPTED_ERR);
                                ERROR_GOTO_CLEANUP
                        }
                } else {
                        if(decoder->decrypt) {
                                log_msg(LOG_LEVEL_ERROR, NOT_ENCRYPTED_ERR);
                                ERROR_GOTO_CLEANUP
                        }
                }

                switch (pt) {
                case PT_VIDEO:
                        len = pckt->data_len - sizeof(video_payload_hdr_t);
                        data = (const char *) hdr + sizeof(video_payload_hdr_t);
                        break;
                case PT_VIDEO_RS:
                case PT_VIDEO_LDGM:
                        len = pckt->data_len - sizeof(fec_payload_hdr_t);
                        data = (const char *) hdr + sizeof(fec_payload_hdr_t);
                        break;
                case PT_ENCRYPT_VIDEO:
                case PT_ENCRYPT_VIDEO_LDGM:
                case PT_ENCRYPT_VIDEO_RS:
                        {
				size_t media_hdr_len = pt == PT_ENCRYPT_VIDEO ? sizeof(video_payload_hdr_t) : sizeof(fec_payload_hdr_t);
                                len = pckt->data_len - sizeof(crypto_payload_hdr_t) - media_hdr_len;
                                data = (const char *)hdr + sizeof(crypto_payload_hdr_t) + media_hdr_len;
                                uint32_t crypto_hdr = ntohl(*(const uint32_t *)(const void *)((const char *)hdr + media_hdr_len));
                                crypto_mode = (enum openssl_mode) (crypto_hdr >> 24);
				if (crypto_mode == MODE_AES128_NONE || crypto_mode > MODE_AES128_MAX) {
					log_msg(LOG_LEVEL_WARNING, "Unknown cipher mode: %d\n", (int) crypto_mode);
					ret = FALSE;
					goto cleanup;
				}
                        }
                        break;
                default:
                        if (pt == PT_Unassign_Type95) {
                                log_msg_once(LOG_LEVEL_WARNING, to_fourcc('U', 'V', 'P', 'T'), MOD_NAME "Unassigned PT 95 received, ignoring.\n");
                        } else {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unknown packet type: " << pckt->pt << ".\n";
                        }
                        ret = FALSE;
                        goto cleanup;
                }

                if ((int) substream >= max_substreams) {
                        log_msg(LOG_LEVEL_WARNING, "[decoder] received substream ID %d. Expecting at most %d substreams. Did you set -M option?\n",
                                        substream, max_substreams);
                        // the guess is valid - we start with highest substream number (anytime - since it holds a m-bit)
                        // in next iterations, index is valid
                        enum video_mode video_mode =
                                guess_video_mode(substream + 1);
                        if (video_mode != VIDEO_UNKNOWN) {
                                log_msg(LOG_LEVEL_NOTICE, "[decoder] Guessing mode: ");
                                decoder_set_video_mode(decoder, video_mode);
                                decoder->received_vid_desc.width = 0; // just for sure, that we reconfigure in next iteration
                                log_msg(LOG_LEVEL_NOTICE, "%s. Check if it is correct.\n", get_video_mode_description(decoder->video_mode));
                        } else {
                                log_msg(LOG_LEVEL_FATAL, "[decoder] Unknown video mode!\n");
                                handle_error(1);
                        }
                        // we need skip this frame (variables are illegal in this iteration
                        // and in case that we got unrecognized number of substreams - exit
                        ret = FALSE;
                        goto cleanup;
                }

                char plaintext[RTP_MAX_PACKET_LEN]; // will be actually shorter
                if (PT_VIDEO_IS_ENCRYPTED(pt)) {
                        int data_len;

                        if((data_len = decoder->dec_funcs->decrypt(decoder->decrypt,
                                        data, len,
                                        (const char *) hdr, pt == PT_ENCRYPT_VIDEO ?
                                        sizeof(video_payload_hdr_t) : sizeof(fec_payload_hdr_t),
                                        plaintext, crypto_mode)) == 0) {
                                goto next_packet;
                        }
                        data = (char *) plaintext;
                        len = data_len;
                }

                if (!PT_VIDEO_HAS_FEC(pt))
                {
                        /* Critical section
                         * each thread *MUST* wait here if this condition is true
                         */
                        check_for_mode_change(decoder, hdr);

                        // hereafter, display framebuffer can be used, so we
                        // check if we got it
                        if (FRAMEBUFFER_NOT_READY(decoder)) {
                                vf_free(frame);
                                return FALSE;
                        }
                }

                buffer_num[substream] = buffer_number;
                frame->tiles[substream].data_len = buffer_length;
                pckt_list[substream][data_pos] = len;

                if ((pt == PT_VIDEO || pt == PT_ENCRYPT_VIDEO) && decoder->decoder_type == LINE_DECODER) {
                        struct tile *tile = NULL;
                        if(!buffer_swapped) {
                                wait_for_framebuffer_swap(decoder);
                                buffer_swapped = true;
                                unique_lock<mutex> lk(decoder->lock);
                                decoder->buffer_swapped = false;
                        }

                        if (!decoder->merged_fb) {
                                tile = vf_get_tile(decoder->frame, substream);
                        } else {
                                tile = vf_get_tile(decoder->frame, 0);
                        }

                        struct line_decoder *line_decoder =
                                &decoder->line_decoder[substream];

                        /* End of critical section */

                        /* MAGIC, don't touch it, you definitely break it
                         *  *source* is data from network, *destination* is frame buffer
                         */

                        /* compute Y pos in source frame and convert it to
                         * byte offset in the destination frame
                         */
                        int y = (data_pos / line_decoder->src_linesize) * line_decoder->dst_pitch;

                        /* compute X pos in source frame */
                        int s_x = data_pos % line_decoder->src_linesize;

                        /* convert X pos from source frame into the destination frame.
                         * it is byte offset from the beginning of a line.
                         */
                        int d_x = s_x * line_decoder->conv_num / line_decoder->conv_den;

                        /* pointer to data payload in packet */
                        auto *source = (const unsigned char *)(data);

                        /* copy whole packet that can span several lines.
                         * we need to clip data (v210 case) or center data (RGBA, R10k cases)
                         */
                        while (len > 0) {
                                /* len id payload length in source BPP
                                 * decoder needs len in destination BPP, so convert it
                                 */
                                int l = len * line_decoder->conv_num / line_decoder->conv_den;

                                /* do not copy multiple lines, we need to
                                 * copy (& clip, center) line by line
                                 */
                                if (l + d_x > (int) line_decoder->dst_linesize) {
                                        l = line_decoder->dst_linesize - d_x;
                                }

                                /* compute byte offset in destination frame */
                                const uint32_t offset = y + d_x;

                                /* watch the SEGV */
                                if (l + line_decoder->base_offset + offset <= tile->data_len) {
                                        /*decode frame:
                                         * we have offset for destination
                                         * we update source contiguously
                                         * we pass {r,g,b}shifts */
                                        line_decoder->decode_line((unsigned char*)tile->data + line_decoder->base_offset + offset, source, l,
                                                        line_decoder->shifts[0], line_decoder->shifts[1],
                                                        line_decoder->shifts[2]);
                                        /* we decoded one line (or a part of one line) to the end of the line
                                         * so decrease *source* len by 1 line (or that part of the line */
                                        len -= line_decoder->src_linesize - s_x;
                                        /* jump in source by the same amount */
                                        source += line_decoder->src_linesize - s_x;
                                } else {
                                        /* this should not ever happen as we call reconfigure before each packet
                                         * iff reconfigure is needed. But if it still happens, something is terribly wrong
                                         * say it loudly
                                         */
                                        if((prints % 100) == 0) {
                                                log_msg(LOG_LEVEL_ERROR, "WARNING!! Discarding input data as frame buffer is too small.\n"
                                                                "Well this should not happened. Expect troubles pretty soon.\n");
                                        }
                                        prints++;
                                        len = 0;
                                }
                                /* each new line continues from the beginning */
                                d_x = 0;        /* next line from beginning */
                                s_x = 0;
                                y += line_decoder->dst_pitch;  /* next line */
                        }
                } else { /* PT_VIDEO_LDGM or external decoder */
                        if(!frame->tiles[substream].data) {
                                frame->tiles[substream].data = (char *) malloc(buffer_length + PADDING);
                        }

                        if (data_pos + len > (unsigned) buffer_length) {
                                if((prints % 100) == 0) {
                                        log_msg(LOG_LEVEL_ERROR, "WARNING!! Discarding input data as frame buffer is too small.\n"
                                                        "Well this should not happened. Expect troubles pretty soon.\n");
                                }
                                prints++;
                                len = max<int>(0, buffer_length - data_pos);
                        }
                        memcpy(frame->tiles[substream].data + data_pos,
                               (const unsigned char *)data, len);
                }

next_packet:
                cdata = cdata->nxt;
        }

        if (FRAMEBUFFER_NOT_READY(decoder) && (pt == PT_VIDEO || pt == PT_ENCRYPT_VIDEO)) {
                ret = FALSE;
                goto cleanup;
        }

        assert(ret == TRUE);

        /// Zero missing parts of framebuffer - this may be useful for compressed video
        /// (which may be also with FEC - but we use systematic codes therefore it may
        /// benefit from that as well).
        if (decoder->decoder_type != LINE_DECODER) {
                for(int i = 0; i < max_substreams; ++i) {
                        unsigned int last_end = 0;
                        for (auto const & packets : pckt_list[i]) {
                                unsigned int start = packets.first;
                                unsigned int len = packets.second;
                                if (last_end < start) {
                                        memset(frame->tiles[i].data + last_end, 0, start - last_end);
                                }
                                last_end = start + len;
                        }
                        if (last_end < frame->tiles[i].data_len) {
                                memset(frame->tiles[i].data + last_end, 0, frame->tiles[i].data_len - last_end);
                        }
                }
        }

        {
                unsigned int frame_size = 0;
                for (int i = 0; i < max_substreams; ++i) {
                        frame_size += frame->tiles[i].data_len;
                }
                pbuf_data->max_frame_size =
                    max(pbuf_data->max_frame_size, frame_size);
                // format message
                unique_ptr <frame_msg> fec_msg (new frame_msg(decoder->control, decoder->stats));
                fec_msg->buffer_num = std::move(buffer_num);
                fec_msg->recv_frame = frame;
                frame = NULL;
                fec_msg->pckt_list = std::move(pckt_list);
                fec_msg->received_pkts_cum = stats->received_pkts_cum;
                fec_msg->expected_pkts_cum = stats->expected_pkts_cum;

                auto t0 = std::chrono::high_resolution_clock::now();
                decoder->fec_queue.push(std::move(fec_msg));
                auto t1 = std::chrono::high_resolution_clock::now();
                double tpf = 1.0 / decoder->display_desc.fps;
                if (std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() > tpf && decoder->stats.displayed > 20) {
                        decoder->slow_msg.print("Your computer may be too SLOW to play this !!!\n");
                }

        }
cleanup:
        ;
        if(ret != TRUE) {
                vf_free(frame);
        }
        pbuf_data->decoded++;

        decoder->stats.update(buffer_number);

        return ret;
}

static void decoder_process_message(struct module *m)
{
        struct state_video_decoder *s = (struct state_video_decoder *) m->priv_data;

        struct message *msg;
        while ((msg = check_message(m))) {
                struct response *r;
                struct msg_universal *m_univ = (struct msg_universal *) msg;
                if (strcmp(m_univ->text, "get_format") == 0) {
                        s->lock.lock();
                        string video_desc = s->received_vid_desc;
                        s->lock.unlock();
                        r = new_response(RESPONSE_OK, video_desc.c_str());
                } else {
                        r = new_response(RESPONSE_NOT_FOUND, NULL);
                }
                free_message(msg, r);

        }
}

