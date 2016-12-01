/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   pixel.h
 * Author: xminarik
 *
 * Created on October 11, 2016, 1:31 PM
 */

#ifndef PIXEL_H
#define PIXEL_H

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"

struct state_pixelization {
public:
    struct video_frame *filter(struct video_frame *in);
    
    struct module mod;

    int x, y, width, height;
    double x_relative, y_relative, width_relative, height_relative;
    int pixelSize;

    struct video_desc saved_desc;
    bool in_relative_units;
private:
};

#endif /* PIXEL_H */

