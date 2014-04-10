#include "capture_filter/resize_utils.h"

using namespace cv;

int resize_frame(char *indata, codec_t in_color, char *outdata, unsigned int *data_len, unsigned int width, unsigned int height, double scale_factor){
    int res = 0;
    Mat out, in, rgb;
    assert(in_color == UYVY || in_color == RGB);
    if (indata == NULL || outdata == NULL || data_len == NULL) {
        return 1;
    }

    switch(in_color){
		case UYVY:
			in.create(height, width, CV_8UC2);
		    in.data = (uchar*)indata;
			cvtColor(in, rgb, CV_YUV2RGB_UYVY);
			resize(rgb, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);
			break;
		case RGB:
			in.create(height, width, CV_8UC3);
		    in.data = (uchar*)indata;
			resize(in, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);
		    break;
		case RGBA:
			in.create(height, width, CV_8UC4);
		    in.data = (uchar*)indata;
			resize(in, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);
			break;
		case BGR:
			in.create(height, width, CV_8UC3);
		    in.data = (uchar*)indata;
			resize(in, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);
			break;
    }


    *data_len = out.step * out.rows * sizeof(char);
    memcpy(outdata,out.data,*data_len);

    return res;
}
