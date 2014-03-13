#include "capture_filter/resize_utils.h"

int resize_frame(char *indata, char *outdata, unsigned int *data_len, unsigned int width, unsigned int height, double scale_factor){
    int res = 0;
    Mat out, in;

    if (indata == NULL || outdata == NULL || data_lent == NULL || opencv == NULL) {
        return 1;
    }

    in = Mat.create(height, width, CV_8UC2);
    in.data = indata;
    out.data = outdata;

    cvtColor(in, out, CV_YUV2RGB_UYVY);
    resize(out, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);

    *data_len = out.step * out.rows * sizeof(char);

    return res;
}
