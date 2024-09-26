#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/libavcodec/to_lavc_vid_conv.c"
#include "debug.h"
#include "video_codec.h"

#define W 3840
#define H 2160

bool cuda_devices_explicit = false;

int
main()
{
        log_level = LOG_LEVEL_ERROR; // silence warnings

        printf("UltraGrid to FFmpeg conversions duration:\n");
        const struct uv_to_av_conversion *conv = get_uv_to_av_conversions();
        char                             *in   = malloc(8 * W * H);

        while (conv->func != NULL) {
                struct timespec t0, t1;
                timespec_get(&t0, TIME_UTC);
                struct to_lavc_vid_conv *s =
                    to_lavc_vid_conv_init(conv->src, W, H, conv->dst, 1);
                struct AVFrame *out_frame = to_lavc_vid_conv(s, in);
                timespec_get(&t1, TIME_UTC);
                printf("%s->%s:\t%.2f ms\n", get_codec_name(conv->src),
                       av_get_pix_fmt_name(conv->dst),
                       (t1.tv_sec - t0.tv_sec) * 1E3 +
                           ((t1.tv_nsec - t0.tv_nsec) / 1E6));

                av_frame_free(&out_frame);
                conv++;
        }
        free(in);
}
