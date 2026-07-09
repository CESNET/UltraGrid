/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2012-2026 CESNET, zájmové sdružení právických osob */

#ifndef AUDIO_POSTPROCESS_H_043B95E6_9C7C_485D_96BF_769D17474460
#define AUDIO_POSTPROCESS_H_043B95E6_9C7C_485D_96BF_769D17474460

#include "compat/c23.h" // IWYU pragma: keep for bool

struct audio_frame;
struct audio_frame2;
struct module;
struct state_audio_postprocess;

#define MSG_UNIVERSAL_TAG_AUDIO_DECODER "ADEC "
#define ADEC_CH_RATE_SHIFT (32LLU)

#ifdef __cplusplus
extern "C" {
#endif

int  audio_postprocess_init(const char *channel_map, const char *scale,
                            struct module                   *parent,
                            struct state_audio_postprocess **out);
int  audio_postprocess_reconfigure(struct state_audio_postprocess *postprocess,
                                   int input_channels);
void audio_postprocess_done(struct state_audio_postprocess *postprocess);
bool decode_audio_frame_postprocess(struct state_audio_postprocess *postprocess,
                                    struct audio_frame2 *decompressed,
                                    struct audio_frame  *out,
                                    long long int       *received_bytes_cum);

#ifdef __cplusplus
}
#endif

#endif // defined AUDIO_POSTPROCESS_H_043B95E6_9C7C_485D_96BF_769D17474460
