#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_win32.h"
#include "config_unix.h"
#endif

#include "audio/audio.h"

struct resampler;

struct resampler *resampler_init(int dst_sample_rate);
void              resampler_done(struct resampler *);
audio_frame2     *resampler_resample(struct resampler *, audio_frame2 *);

