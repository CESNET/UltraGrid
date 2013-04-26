#include "audio/resampler.h"
#include "audio/utils.h"

#include <speex/speex_resampler.h>

struct resampler
{
        audio_frame2 *resampled;

        char *muxed;
        char *resample_buffer;
        SpeexResamplerState *resampler;
        int resample_from, resample_ch_count;
        int resample_to;
};

struct resampler *resampler_init(int dst_sample_rate)
{
        struct resampler *s = calloc(1, sizeof(struct resampler));

        s->resample_buffer = malloc(1024 * 1024);
        s->muxed = malloc(1024 * 1024);
        s->resampled = audio_frame2_init();
        s->resample_to = dst_sample_rate;

        return s;
}

void resampler_done(struct resampler *s)
{
        free(s);
}

audio_frame2 *resampler_resample(struct resampler *s, audio_frame2 *frame)
{
        if(s->resample_to == frame->sample_rate) {
                return frame;
        } else {
                audio_frame2_allocate(s->resampled, frame->ch_count,
                                (long long) frame->data_len[0] * s->resample_to / frame->sample_rate * 2);
                uint32_t write_frames = 1024 * 1024;

                if(s->resample_from != frame->sample_rate || s->resample_ch_count != frame->ch_count) {
                        s->resample_from = frame->sample_rate;
                        s->resample_ch_count = frame->ch_count;
                        if(s->resampler) {
                                speex_resampler_destroy(s->resampler);
                        }
                        int err;
                        s->resampler = speex_resampler_init(frame->ch_count, s->resample_from,
                                        s->resample_to, 10, &err);
                        if(err) {
                                abort();
                        }
                }

                for(int i = 0; i < frame->ch_count; ++i) {
                        mux_channel(s->muxed, frame->data[i], frame->bps, frame->data_len[i],
                                        frame->ch_count, i, 1.0);
                }
                char *in_buf;
                int data_len = frame->data_len[0] * frame->ch_count; // uncompressed, so all channels
                                                                     // have the same size
                if(frame->bps != 2) {
                        change_bps(s->resample_buffer, 2, s->muxed, frame->bps,
                                        data_len);
                        in_buf = s->resample_buffer;
                        data_len = data_len / frame->bps * 2;
                } else {
                        in_buf = s->muxed;
                }

                uint32_t in_frames = data_len / frame->ch_count / 2;
                speex_resampler_process_interleaved_int(s->resampler, (spx_int16_t *)(void *) in_buf, &in_frames,
                                (spx_int16_t *)(void *) s->muxed, &write_frames);

                for(int i = 0; i < frame->ch_count; ++i) {
                        s->resampled->data_len[i] = write_frames * 2 /* bps */;
                        demux_channel(s->resampled->data[i], s->muxed, 2,
                                        s->resampled->data_len[i] * s->resampled->ch_count,
                                        s->resampled->ch_count, i);

                }
                s->resampled->sample_rate = s->resample_to;
                s->resampled->bps = 2;
        }

        return s->resampled;
}

