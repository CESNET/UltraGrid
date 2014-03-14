#ifndef RESIZE_UTILS_H_
#define RESIZE_UTILS_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <opencv/cv.hpp>
#include <opencv/cv.h>

int resize_frame(char *indata, char *outdata, unsigned int *data_len, unsigned int width, unsigned int height, double scale_factor);

#endif// RESIZE_UTILS_H_
