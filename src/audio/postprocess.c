// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2012--2026 CESNET, zájmové sdružení právických osob/
/**
 * @file
 * @todo
 * unnecessarily complex - move mixing/scaling to audio receiver thread (if not
 * even remove)
 */

#include "postprocess.h"

#include <assert.h>   // for assert
#include <inttypes.h> // for memcpy, memset, strcasecmp...
#include <math.h>     // for log
#include <stdio.h>    // for snprintf
#include <stdlib.h>   // for strtoll, atof, free, realloc
#include <string.h>   // for memcpy, memset, strcasecmp...
#include <strings.h>  // for strcasecmp

#include "audio/resampler.hpp"       // for audio_frame2_resampler
#include "audio/utils.h"             // for channel_map, calculate_rms, mux_...
#include "compat/c23.h"              // IWYU pragma: keep
#include "control_socket.h"
#include "debug.h"
#include "host.h"
#include "messaging.h"               // for new_response, msg_universal, che...
#include "module.h"
#include "tv.h"
#include "types.h"                   // for fec_desc, fec_type
#include "utils/color_out.h"
#include "utils/debug.h"             // for DEBUG_TIMER_*
#include "utils/macros.h"
#include "utils/misc.h"              // for get_stat_color
#include "utils/worker.h"

#define MOD_NAME "[audio postpr.] "

#define OUR_CLASS MODULE_CLASS_POSTPROCESS
const enum module_class path_audio_postprocess[] = {
        MODULE_CLASS_AUDIO, MODULE_CLASS_RECEIVER, OUR_CLASS, MODULE_CLASS_NONE
};

struct scale_data {
        double vol_avg;
        int samples;
        double scale;
};

struct state_audio_postprocess {
        struct module mod;

        time_ns_t t0;
        struct control_state *control;

        unsigned int channel_remapping:1;
        struct channel_map channel_map;

        /// contains scaling metadata if we want to perform audio scaling
        struct scale_data *scale;
        bool fixed_scale;

        bool muted;

        struct audio_frame2 *decoded; ///< buffer that keeps audio samples from
                                      ///< last 5 seconds (for statistics)
        struct audio_frame2_resampler *resampler;
        struct audio_frame2           *resample_remainder;
        _Atomic uint64_t
            req_resample_to; // hi 32 - numerator; lo 32 - denominator
};

static const double VOL_UP   = 1.1;
static const double VOL_DOWN = 1.0 / 1.1;

static void compute_scale(struct scale_data *scale_data, double vol_avg, int samples, int sample_rate)
{
        scale_data->vol_avg = scale_data->vol_avg * (scale_data->samples / ((double) scale_data->samples + samples)) +
                vol_avg * (samples / ((double) scale_data->samples + samples));
        scale_data->samples += samples;

        if (scale_data->samples > sample_rate * 6) {
                double ratio = 1.0;

                if (scale_data->vol_avg > 0.25 || (scale_data->vol_avg > 0.05 && scale_data->scale > 1.0)) {
                        ratio = VOL_DOWN;
                } else if (scale_data->vol_avg < 0.20 && scale_data->scale < 1.0) {
                        ratio = VOL_UP;
                }

                scale_data->scale *= ratio;
                scale_data->vol_avg *= ratio;

                MSG(VERBOSE,
                    "Audio scale adjusted to: %f (average volume was %f)\n",
                    scale_data->scale, scale_data->vol_avg);

                scale_data->samples = 4 * sample_rate;
        }
}

static void audio_decoder_process_message(struct module *m)
{
        struct state_audio_postprocess *s = m->priv_data;

        struct message *msg = nullptr;
        while ((msg = check_message(m)) != nullptr) {
                struct msg_universal *muniv = (struct msg_universal *) msg;
                struct response *r = nullptr;
                if (strstr(muniv->text, MSG_UNIVERSAL_TAG_AUDIO_DECODER) == muniv->text) {
                        s->req_resample_to = strtoull(muniv->text + strlen(MSG_UNIVERSAL_TAG_AUDIO_DECODER), nullptr, 0);
                        MSG(VERBOSE,
                            "Resampling sound to: %" PRIu64 "/%llu\n",
                            (s->req_resample_to >> ADEC_CH_RATE_SHIFT),
                            (s->req_resample_to &
                             ((1LLU << ADEC_CH_RATE_SHIFT) - 1)));
                        r = new_response(RESPONSE_OK, nullptr);
                } else {
                        r = new_response(RESPONSE_NOT_FOUND, nullptr);
                }
                free_message(msg, r);
        }
}

static void
audio_channel_map_usage()
{
        color_printf("Usage:\n");
        color_printf("\t" TBOLD(TRED("--audio-channel-map <mapping>"))"   mapping of input audio channels\n");
        color_printf("\t                                to output audio channels comma-separated\n");
        color_printf("\t                                list of channel mapping (receiver only)\n");

        color_printf("\nExamples:\n");
        color_printf("\t- " TBOLD("0:0,1:0") " - mixes first 2 channels\n");
        color_printf("\t- " TBOLD("0:0") "     - play only first channel\n");
        color_printf("\t- " TBOLD("0:0,:1") "  - sets second channel to a silence, first "
                     "one is left as is\n");
        color_printf("\t- " TBOLD("0:0,0:1") " - splits mono into 2 channels\n");
}

static void
audio_scale_usage()
{
        color_printf("Usage:\n");
        color_printf(TBOLD(TRED("\t--audio-scale [<factor>|<method>]\n")));
        color_printf("\t        Floating point number that tells a static scaling factor for all\n");
        color_printf("\t        output channels. Scaling method can be one from these:\n");
        color_printf(TBOLD("\t          0.0-1.0") " - factor to scale to (usually 0-1 but can be more)\n");
        color_printf(TBOLD("\t          mixauto") " - automatically adjust volume if using channel\n");
        color_printf("\t                    mixing/remapping (default)\n");
        color_printf(TBOLD("\t          auto") " - automatically adjust volume\n");
        color_printf(TBOLD("\t          none") " - no scaling will be performed\n");
}

ADD_TO_PARAM("soft-resample", "* soft-resample=<num>/<den>\n"
                "  Resample to specified sampling rate, eg. 12288128/256 for 48000.5 Hz\n");
int
audio_postprocess_init(const char *channel_map, const char *scale,
                       struct module                   *parent,
                       struct state_audio_postprocess **out)
{
        assert(scale != nullptr);

        if (channel_map != nullptr && strcmp(channel_map, "help") == 0) {
                audio_channel_map_usage();
                return 1;
        }

        if (strcmp(scale, "help") == 0) {
                audio_scale_usage();
                return 1;
        }

        bool scale_auto = false;
        double scale_factor = 1.0;
        struct state_audio_postprocess *s = calloc(1, sizeof *s);
        s->resampler = audio_frame2_resampler_init();
        s->channel_map.max_output = -1;

        module_init_default(&s->mod);
        s->mod.cls = OUR_CLASS;
        s->mod.priv_data = s;
        s->mod.new_message = audio_decoder_process_message;
        module_register(&s->mod, parent);

        const char *val = get_commandline_param("soft-resample");
        if (val != nullptr) {
                assert(strchr(val, '/') != nullptr);
                s->req_resample_to = strtoll(val, NULL, 0) << 32LLU | strtoll(strchr(val, '/') + 1, NULL, 0);
        }

        s->t0 = get_time_in_ns();

        s->control = get_control_state(parent);

        if (channel_map != nullptr) {
                if (!parse_channel_map_cfg(&s->channel_map, channel_map)) {
                        audio_postprocess_done(s);
                        return -1;
                }
                s->channel_remapping = true;
        } else {
                s->channel_remapping = false;
                s->channel_map.map = NULL;
                s->channel_map.sizes = NULL;
                s->channel_map.size = 0;
        }

        if (strcasecmp(scale, "mixauto") == 0) {
                if(s->channel_remapping) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Channel remapping detected - automatically scaling audio levels. Use \"--audio-scale none\" to disable.\n");
                        scale_auto = true;
                } else {
                        scale_auto = false;
                }
        } else if (strcasecmp(scale, "auto") == 0) {
                scale_auto = true;
        } else if (strcasecmp(scale, "none") == 0) {
                scale_auto = false;
                scale_factor = 1.0;
        } else {
                scale_auto = false;
                scale_factor = atof(scale);
                if(scale_factor <= 0.0) {
                        log_msg(LOG_LEVEL_ERROR, "Invalid audio scaling factor!\n");
                        audio_postprocess_done(s);
                        return -1;
                }
        }

        s->fixed_scale = scale_auto ? false : true;
        s->scale = calloc(1, sizeof *s->scale);
        s->scale[0].scale   = scale_factor;
        s->scale[0].vol_avg = 1.0;

        *out = s;
        return 0;
}

void
audio_postprocess_done(struct state_audio_postprocess *postprocess)
{
        if (!postprocess) {
                return;
        }
        channel_map_free_data(&postprocess->channel_map);
        module_done(&postprocess->mod);
        audio_frame2_delete(postprocess->decoded);
        audio_frame2_delete(postprocess->resample_remainder);
        delete_resampler(postprocess->resampler);
        free(postprocess);
}

struct adec_stats_processing_data {
        struct audio_frame2 *frame;
        double seconds;
        long bytes_received;
        long bytes_expected;
        bool muted_receiver;
};

static void *adec_compute_and_print_stats(void *arg) {
        struct adec_stats_processing_data* d = arg;
        char loss[STR_LEN];
        if (d->bytes_received < d->bytes_expected) {
                snprintf_ch(loss, " (%ld lost)",
                            d->bytes_expected - d->bytes_received);
        } else {
                loss[0] = '\0';
        }

        const double exp_samples = audio_frame2_get_sample_rate(d->frame) * d->seconds;
        const char *dec_cnt_warn_col = get_stat_color(
            audio_frame2_get_sample_count(d->frame) / exp_samples);

        MSG(INFO,
            "Received %ld/%ld B%s, "
            "decoded %s%d samples" TERM_RESET " in %.2f sec.\n",
            d->bytes_received, d->bytes_expected, loss,
            dec_cnt_warn_col, audio_frame2_get_sample_count(d->frame), d->seconds);

        char volume[STR_LEN];
        char       *vol_start = volume;
        for (int i = 0; i < audio_frame2_get_channel_count(d->frame); ++i) {
                double rms = 0.0;
                double peak = 0.0;
                rms = calculate_rms2(d->frame, i, &peak);
                format_audio_channel_volume(i, rms, peak, TERM_FG_GREEN,
                                            &vol_start, volume + sizeof volume);
        }

        log_msg(LOG_LEVEL_INFO, TBOLD(TGREEN(MOD_NAME)) "Volume: %s dBFS RMS/peak%s\n", volume,
            d->muted_receiver ? TBOLD(TRED(" (muted receiver)")) : "");

        audio_frame2_delete(d->frame);
        free(d);

        return NULL;
}

int
audio_postprocess_reconfigure(struct state_audio_postprocess *postprocess,
                              int                             input_channels)
{
        if (postprocess->channel_remapping &&
            postprocess->channel_map.size > input_channels) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "Audio channel map references channels with "
                                 "idx higher than ch. count!\n");
        }
        int output_channels = postprocess->channel_remapping ?
                        postprocess->channel_map.max_output + 1: input_channels;

        if(!postprocess->fixed_scale) {
                free(postprocess->scale);
                postprocess->scale =
                    calloc(output_channels, sizeof *postprocess->scale);
                for (int i = 0; i < output_channels; i++) {
                        postprocess->scale[i].scale   = 1.0;
                        postprocess->scale[i].vol_avg = 1.0;
                }
        }
        audio_frame2_delete(postprocess->resample_remainder);
        postprocess->resample_remainder = nullptr;
        audio_frame2_delete(postprocess->decoded);
        postprocess->decoded = nullptr;
        return output_channels;
}

bool
decode_audio_frame_postprocess(struct state_audio_postprocess *postprocess,
                               struct audio_frame2            *decompressed,
                               struct audio_frame             *out,
                               long long int *expected_bytes_cum,
                               long long int *received_bytes_cum)
{
        // Perform a variable rate resample if any output device has requested it
        if (postprocess->req_resample_to != 0 ||
            out->sample_rate != audio_frame2_get_sample_rate(decompressed)) {
                int resampler_bps =
                    resampler_align_bps(postprocess->resampler, audio_frame2_get_bps(decompressed));
                if (resampler_bps <= 0) {
                        return false;
                }
                if (resampler_bps != audio_frame2_get_bps(decompressed)) {
                        audio_frame2_change_bps(decompressed,resampler_bps);
                }
                if (postprocess->req_resample_to != 0) {
                        if (postprocess->resample_remainder) {
                                audio_frame2_append(
                                    postprocess->resample_remainder,
                                    decompressed);
                                audio_frame2_replace(
                                    decompressed,
                                    &postprocess->resample_remainder);
                        }
                        struct audio_frame2 *remainder = nullptr;
                        bool                 ret = audio_frame2_resample_fake(
                            postprocess->resampler, decompressed,
                            postprocess->req_resample_to >> ADEC_CH_RATE_SHIFT,
                            postprocess->req_resample_to &
                                ((1LLU << ADEC_CH_RATE_SHIFT) - 1),
                            &remainder);
                        if (!ret) {
                                MSG(INFO, "You may try to set different "
                                          "sampling on sender.\n");
                                return false;
                        }
                        audio_frame2_delete(postprocess->resample_remainder);
                        postprocess->resample_remainder = remainder;
                } else {
                        if (!audio_frame2_resample(postprocess->resampler,
                                                   decompressed,
                                                   out->sample_rate)) {
                                MSG(INFO, "You may try to set different "
                                          "sampling on sender.\n");
                                return false;
                        }
                }
        }

        if (audio_frame2_get_bps(decompressed) != out->bps) {
                audio_frame2_change_bps(decompressed, out->bps);
        }

        out->data_len =
            audio_frame2_get_data_len(decompressed, 0) * out->ch_count;
        if (out->max_size < out->data_len) {
                out->max_size = out->data_len;
                out->data = (char *) realloc(out->data, out->data_len);
        }

        memset(out->data, 0, out->data_len);

        if (!postprocess->muted) {
                // there is a mapping for channel
                for(int channel = 0; channel < audio_frame2_get_channel_count(decompressed); ++channel) {
                        if (postprocess->channel_remapping) {
                                if(channel < postprocess->channel_map.size) {
                                        for(int i = 0; i < postprocess->channel_map.sizes[channel]; ++i) {
                                                int new_position = postprocess->channel_map.map[channel][i];
                                                if (new_position >= out->ch_count) {
                                                        continue;
                                                }
                                                mux_and_mix_channel(out->data,
                                                                audio_frame2_get_data(decompressed, channel),
                                                                audio_frame2_get_bps(decompressed), audio_frame2_get_data_len(decompressed, channel),
                                                                out->ch_count, new_position,
                                                                postprocess->scale[postprocess->fixed_scale ? 0 : new_position].scale);
                                        }
                                }
                        } else {
                                if (channel >= out->ch_count) {
                                        continue;
                                }
                                mux_and_mix_channel(out->data, audio_frame2_get_data(decompressed, channel),
                                                audio_frame2_get_bps(decompressed),
                                                audio_frame2_get_data_len(decompressed, channel), out->ch_count, channel,
                                                postprocess->scale[postprocess->fixed_scale ? 0 : audio_frame2_get_channel_count(decompressed)].scale);
                        }
                }
        }
        out->timestamp = audio_frame2_get_timestamp(decompressed);

        if (postprocess->decoded == nullptr) {
                postprocess->decoded = audio_frame2_alloc(
                    audio_frame2_get_channel_count(decompressed), AC_PCM,
                    audio_frame2_get_bps(decompressed),
                    audio_frame2_get_sample_rate(decompressed));
                audio_frame2_reserve(postprocess->decoded, 6);
        }
        audio_frame2_append(postprocess->decoded, decompressed);

        if (control_stats_enabled(postprocess->control)) {
                char report[STR_LEN];
                char *start = report;
                char *end = report + sizeof report;
                start += snprintf(start, end - start, "ARECV");
                int num_ch = MIN(audio_frame2_get_channel_count(decompressed), control_audio_ch_report_count(postprocess->control));
                for(int i = 0; i < num_ch; i++){
                        double rms, peak;
                        rms = calculate_rms2(decompressed, i, &peak);
                        double rms_dbfs0 = 20 * log(rms) / log(10);
                        double peak_dbfs0 = 20 * log(peak) / log(10);
                        start += snprintf(start, end - start,
                                          " volrms%d %f volpeak%d %f", i,
                                          rms_dbfs0, i, peak_dbfs0);
                }

                control_report_stats(postprocess->control, report);
        }

        time_ns_t t = get_time_in_ns();
        if ((t - postprocess->t0) > SEC_TO_NS(5)) {
                struct adec_stats_processing_data *d = calloc(1, sizeof *d);
                d->frame                             = audio_frame2_alloc(
                    audio_frame2_get_channel_count(decompressed), AC_PCM,
                    audio_frame2_get_bps(decompressed),
                    audio_frame2_get_sample_rate(decompressed));
                audio_frame2_reserve(d->frame, 6);
                SWAP_PTR(d->frame, postprocess->decoded);
                d->seconds = NS_TO_SEC_DBL(t - postprocess->t0);
                d->bytes_received = *received_bytes_cum;
                d->bytes_expected = *expected_bytes_cum;
                d->muted_receiver = postprocess->muted;

                task_run_async_detached(adec_compute_and_print_stats, d);

                postprocess->t0 = t;
                *received_bytes_cum = 0;
                *expected_bytes_cum = 0;
        }

        DEBUG_TIMER_START(audio_decode_compute_autoscale);
        if(!postprocess->fixed_scale) {
                int output_channels = postprocess->channel_remapping ?
                        postprocess->channel_map.max_output + 1: audio_frame2_get_channel_count(decompressed);
                for(int i = 0; i <= postprocess->channel_map.max_output; ++i) {
                        if (postprocess->channel_map.contributors[i] <= 1) {
                                continue;
                        }
                        double avg = get_avg_volume(out->data, out->bps,
                                        out->data_len / output_channels / out->bps, output_channels, i);
                        compute_scale(&postprocess->scale[i], avg,
                                        out->data_len / output_channels / out->bps, out->sample_rate);
                }
        }
        DEBUG_TIMER_STOP(audio_decode_compute_autoscale);
        return true;
}
 
