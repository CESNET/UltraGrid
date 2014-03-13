#include "capture_filter/resize_utils.h"

int resize(char *indata, char *outdata, unsigned int *data_len, struct opencv_tile_struct *opencv){
    int res = 0;

    if (indata == NULL || outdata == NULL || data_lent == NULL || opencv == NULL) {
        return 1;
    }

    opencv->in.data = indata;
    opencv->out.data = outdata;

    cvtColor(opencv->in, opencv->out, CV_YUV2RGB_UYVY);
    resize(out, out, Size(0,0), opencv->scale_factor, opencv->scale_factor, INTER_LINEAR);

    *data_len = out.step * out.rows * sizeof(char);

    return res;
}

int reconfigure_opencv_tile_struct(struct opencv_tile_struct *opencv,unsigned int width, unsigned int height, float sc_fact){
    int res = 0;

    if (opencv == NULL || width == 0 || height == 0 || scale_factor == 0) {
        return 1;
    }

    opencv->in = Mat.create(height, width, CV_8UC2);
    opencv->width = width;
    opencv->height = height;

    return 0;
}
