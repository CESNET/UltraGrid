#include <stdio.h>
#include "v_codec.h"

const struct codec_info_t codec_info[] = {
        {RGBA, "RGBA", 0, 0, 4.0, 1},
        {UYVY, "UYVY", 846624121, 0, 2, 0},
        {DVS8, "DVS8", 0, 0, 2, 0},
        {R10k, "R10k", 1378955371, 0, 4, 1},
        {v210, "v210", 1983000880, 48, 8.0/3.0, 0},
        {DVS10, "DVS10", 0, 48, 8.0/3.0, 0},
        {0, NULL, 0, 0, 0.0, 0}};


void
show_codec_help(void)
{
        printf("\tSupported codecs:\n");
        printf("\t\t8bits\n");
        printf("\t\t\t'RGBA' - Red Green Blue Alpha 32bit\n");
        printf("\t\t\t'UYVY' - YUV 4:2:2\n");
        printf("\t\t\t'DVS8' - Centaurus 8bit YUV 4:2:2\n");
        printf("\t\t10bits\n");
        printf("\t\t\t'R10k' - RGB 4:4:4\n");
        printf("\t\t\t'v210' - YUV 4:2:2\n");
        printf("\t\t\t'DVS10' - Centaurus 10bit YUV 4:2:2\n");
}
