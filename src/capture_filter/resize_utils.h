#ifndef RESIZE_UTILS_H_
#define RESIZE_UTILS_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif

    struct opencv_tile_struct {
        int width;
        int height;
        float scale_factor;
        Mat in, out;
    };

    int resize(char *indata, char *outdata, unsigned int *data_len, struct opencv_tile_struct *opencv);
    int reconfigure_opencv_tile_struct(struct opencv_tile_struct *opencv,unsigned int width, unsigned int height, float sc_fact);

#ifdef __cplusplus
}
#endif

#endif// RESIZE_UTILS_H_
