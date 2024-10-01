#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/libavcodec/from_lavc_vid_conv.c"
#undef MOD_NAME
#include "../src/libavcodec/to_lavc_vid_conv.c"
#include "debug.h"
#include "video_codec.h"

#define W 3840
#define H 2160

bool cuda_devices_explicit = false;

static void
benchmark_to_lavc_convs()
{
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

static void
benchmark_from_lavc_convs()
{
        printf("UltraGrid from FFmpeg conversions duration:\n");
        char *out = malloc(8 * W * H);
        for (unsigned i = 0; i < AV_TO_UV_CONVERSION_COUNT; i++) {
                const struct av_to_uv_conversion *conv =
                    &av_to_uv_conversions[i];
                if (conv->uv_codec == DRM_PRIME) { // crashes
                        continue;
                }
                AVFrame *in = av_frame_alloc();
                in->format  = conv->av_codec;
                in->width   = W;
                in->height  = H;
                av_frame_get_buffer(in, 0);
                struct timespec t0, t1;
                timespec_get(&t0, TIME_UTC);
                conv->convert((struct conv_data){
                    out, in, vc_get_linesize(W, conv->uv_codec),
                    DEFAULT_RGB_SHIFT_INIT });
                timespec_get(&t1, TIME_UTC);
                printf("%s->%s:\t%.2f ms\n",
                       av_get_pix_fmt_name(conv->av_codec),
                       get_codec_name(conv->uv_codec),
                       (t1.tv_sec - t0.tv_sec) * 1E3 +
                           ((t1.tv_nsec - t0.tv_nsec) / 1E6));
                av_frame_free(&in);
        }
        free(out);
}

int
main(int argc, char *argv[])
{
        // silence warnings about reducing bit depth - not relevant here
        log_level = LOG_LEVEL_ERROR;
        bool bench_to = true;
        bool bench_from = true;

        if (argc > 1) {
                if (argc == 2 && strcmp(argv[1], "to") == 0) {
                        bench_from = false;
                } else if (argc == 2 && strcmp(argv[1], "from") == 0) {
                        bench_to = false;
                } else {
                        printf("Usage:\n%s [to|from]\n", argv[0]);
                        return strcmp(argv[1], "help") != 0;
                }
        }

        if (bench_to) {
                benchmark_to_lavc_convs();
        }
        if (bench_from) {
                benchmark_from_lavc_convs();
        }
}
