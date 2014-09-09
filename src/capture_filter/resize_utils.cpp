/*
 * FILE:    capture_filter/resize_utils.cpp
 * AUTHORS: Gerard Castillo     <gerard.castillo@i2cat.net>
 *          Marc Palau          <marc.palau@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the Fundació i2CAT,
 *      Internet I Innovació Digital a Catalunya. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "capture_filter/resize_utils.h"

using namespace cv;

int resize_frame(char *indata, codec_t in_color, char *outdata, unsigned int *data_len, unsigned int width, unsigned int height, double scale_factor){
    assert(in_color == UYVY || in_color == YUYV || in_color == RGB);

    int res = 0;
    Mat out, in, rgb;

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
		case YUYV:
			in.create(height, width, CV_8UC2);
		    in.data = (uchar*)indata;
			cvtColor(in, rgb, CV_YUV2RGB_YUYV);
			resize(rgb, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);
			break;
		case RGB:
			in.create(height, width, CV_8UC3);
		    in.data = (uchar*)indata;
			resize(in, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);
			break;
    }

    *data_len = out.step * out.rows * sizeof(char);
    memcpy(outdata,out.data,*data_len);

    return res;
}
