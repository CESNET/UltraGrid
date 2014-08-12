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
        struct resampler *s = (struct resampler *) calloc(1, sizeof(struct resampler));

        s->resample_buffer = (char *) malloc(1024 * 1024);
        s->muxed = (char *) malloc(1024 * 1024);
        s->resampled = new audio_frame2;
        s->resample_to = dst_sample_rate;

        return s;
}

void resampler_done(struct resampler *s)
{
        free(s->resample_buffer);
        free(s->muxed);
        delete s->resampled;
        free(s);
}

audio_frame2 *resampler_resample(struct resampler *s, audio_frame2 *frame)
{
        if(s->resample_to == frame->get_sample_rate()) {
                return frame;
        } else {
                s->resampled->init(frame->get_channel_count(), AC_PCM, 2, s->resample_to);
                uint32_t write_frames = 1024 * 1024;

                if(s->resample_from != frame->get_sample_rate() || s->resample_ch_count != frame->get_channel_count()) {
                        s->resample_from = frame->get_sample_rate();
                        s->resample_ch_count = frame->get_channel_count();
                        if(s->resampler) {
                                speex_resampler_destroy(s->resampler);
                        }
                        int err;
                        s->resampler = speex_resampler_init(frame->get_channel_count(), s->resample_from,
                                        s->resample_to, 10, &err);
                        if(err) {
                                abort();
                        }
                }

                for (int i = 0; i < frame->get_channel_count(); ++i) {
                        mux_channel(s->muxed, frame->get_data(i), frame->get_bps(), frame->get_data_len(i),
                                        frame->get_channel_count(), i, 1.0);
                }
                int data_len = frame->get_data_len(0) * frame->get_channel_count(); // uncompressed, so all channels
                                                                     // have the same size
                if(frame->get_bps() != 2) {
                        change_bps(s->resample_buffer, 2, s->muxed, frame->get_bps(),
                                        data_len);
                        data_len = data_len / frame->get_bps() * 2;
                } else {
                        memcpy(s->resample_buffer, s->muxed, data_len);
                }

                uint32_t in_frames = data_len / frame->get_channel_count() / 2;
                uint32_t in_frames_orig = in_frames;
                speex_resampler_process_interleaved_int(s->resampler,
                                (spx_int16_t *)(void *) s->resample_buffer, &in_frames,
                                (spx_int16_t *)(void *) s->muxed, &write_frames);
                assert(in_frames_orig == in_frames);

                for(int i = 0; i < frame->get_channel_count(); ++i) {
                        size_t data_len = write_frames * 2 /* bps */;
                        //s->resampled->data_len[i] = 
                        char *data = (char *) malloc(data_len);
                        demux_channel(data, s->muxed, 2, data_len * s->resampled->get_channel_count(),
                                        s->resampled->get_channel_count(), i);
                        s->resampled->append(i, data, data_len);
                        free(data);
                }
        }

        return s->resampled;
}

