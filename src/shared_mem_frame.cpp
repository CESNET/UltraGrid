/**
 * @file   shared_mem_frame.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 */

#include "shared_mem_frame.hpp"

#include <cmath>

#ifndef GUI_BUILD
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_codec.h"
#include "video_frame.h"
#include "debug.h"

#else

#include <stdio.h>
#define error_msg(...) fprintf(stderr, __VA_ARGS__)
#endif // GUI_BUILD

Shared_mem::Shared_mem(const char *key) :
        mem_size(4096),
        locked(false),
        owner(false)
{
        setKey(key);
}

Shared_mem::Shared_mem() : Shared_mem("ultragrid_preview") {  }

Shared_mem::~Shared_mem(){
        destroy();
}

void Shared_mem::setKey(const char *key){
        shared_mem.setKey(key);
}

bool Shared_mem::create(){
        if(shared_mem.create(mem_size) == false){
                /* Creating shared memory could fail because of shared memory
                 * left over after crash. Here we try to release such memory.*/
                shared_mem.attach();
                shared_mem.detach();
                if(shared_mem.create(mem_size) == false){
                        error_msg("[Shared mem] Can't create shared memory!\n");
                        return false;
                }
        }

        Shared_mem_frame *frame = get_frame_and_lock();
        if(!frame)
                return false;
        frame->width = 20;
        frame->height = 20;
        frame->should_detach = false;
        unlock();
        owner = true;
        return true;
}

bool Shared_mem::attach(){
        return shared_mem.attach();
}

bool Shared_mem::detach(){
        if(locked)
                unlock();

        owner = false;
        return shared_mem.detach();
}

void Shared_mem::destroy(){
        if(!owner)
                return;

        Shared_mem_frame *sframe = get_frame_and_lock();
        if(sframe){
                sframe->should_detach = true;
        }
        detach();
}

bool Shared_mem::is_attached(){
        return shared_mem.isAttached();
}

bool Shared_mem::lock(){
        /* Locking should fail when the shared memory is not attached,
         * but for some reason QT sometimes succeeds in locking unattached
         * memory (bug in QT maybe?)
         */
        if(!is_attached())
                return false;

        if(shared_mem.lock()){
                locked = true;
                return true;
        }
        return false;
}

bool Shared_mem::unlock(){
        if(shared_mem.unlock()){
                locked = false;
                return true;
        }
        return false;
}

Shared_mem_frame *Shared_mem::get_frame_and_lock(){
#ifndef GUI_BUILD
        if(reconfiguring){
                if(shared_mem.attach()){
                        // Shared mem is still not detached by the GUI
                        shared_mem.detach();
                        return nullptr;
                } else {
                        if(shared_mem.create(mem_size) == false){
                                error_msg("[Shared mem] Can't create shared memory (get_frame)!\n");
                                return nullptr;
                        } else {
                                owner = true;
                        }
                        lock();
                        struct Shared_mem_frame *sframe = (Shared_mem_frame*) shared_mem.data();
                        sframe->width = scaledW_pad;
                        sframe->height = scaledH;
                        sframe->should_detach = false;
                        reconfiguring = false;
                }
        } else {
                if(!is_attached()) attach();
                lock();
        }
#else
        if(!is_attached()) attach();
        lock();
#endif // GUI_BUILD

        struct Shared_mem_frame *sframe = (Shared_mem_frame*) shared_mem.data();
        if(sframe && sframe->should_detach){
                detach();
                return nullptr;
        }

        return sframe;
}

#ifndef GUI_BUILD

void Shared_mem::check_reconf(struct video_desc in_desc){
        if (video_desc_eq(desc, in_desc))
                return;

        desc = in_desc;

        /* We need to destroy the shared memory segment
         * and recreate it with a new size. To destroy it all processes
         * must detach it. We detach here and then wait until the GUI detaches */
        destroy();

        const float target_width = 960;
        const float target_height = 540;

        float scale = ((in_desc.width / target_width) + (in_desc.height / target_height)) / 2.f;
        if(scale < 1)
                scale = 1;
        scale = std::round(scale);
        scaleF = (int) scale;

        scaledW = in_desc.width / scaleF;
        //OpenGL wants the width to be divisable by 4
        scaledW_pad = ((scaledW + 4 - 1) / 4) * 4;
        scaledH = in_desc.height / scaleF;
        scaled_frame.resize(get_bpp(in_desc.color_spec) * scaledW * scaledH);
        mem_size = get_bpp(preview_codec) * scaledW_pad * scaledH + sizeof(Shared_mem_frame);

        reconfiguring = true;
}

void Shared_mem::put_frame(struct video_frame *frame){
        // Only he instance that created the shared frame can write to it
        if(!owner){
                return;
        }

        decoder_t dec = get_decoder_from_to(frame->color_spec, preview_codec);
        if (!dec) {
                LOG(LOG_LEVEL_WARNING) << "[Shared mem] Cannot find decoder from " <<
                        get_codec_name(frame->color_spec) << " to " <<
                        get_codec_name(preview_codec) << ".\n";
                return;
        }

        check_reconf(video_desc_from_frame(frame));

        int src_line_len = vc_get_linesize(desc.width, frame->color_spec);
        int block_size = get_pf_block_bytes(frame->color_spec);
        assert(block_size > 0);
        int dst = 0;
        for(unsigned y = 0; y < desc.height; y += scaleF){
                for(int x = 0; x + scaleF * block_size <= src_line_len; x += scaleF * block_size){
                        memcpy(scaled_frame.data() + dst, frame->tiles[0].data + y*src_line_len + x, block_size);
                        dst += block_size;
                }
        }

        struct Shared_mem_frame *sframe = get_frame_and_lock();

        if(!sframe)
                return;

        int dst_line_len_pad = vc_get_linesize(scaledW_pad, preview_codec);
        int dst_line_len = vc_get_linesize(scaledW, preview_codec);
        src_line_len = vc_get_linesize(scaledW, frame->color_spec);
        for(int i = 0; i < scaledH; i++){
                dec(sframe->pixels + dst_line_len_pad * i,
                                scaled_frame.data() + src_line_len * i,
                                dst_line_len,
                                0, 8, 16);
        }

        unlock();
}

#endif // GUI_BUILD
