#include "shared_mem_frame.hpp"

#ifndef GUI_BUILD
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_codec.h"
#include "video_frame.h"
#endif // GUI_BUILD

#include <stdio.h>

Shared_mem::Shared_mem(const char *key) :
        key(key),
        mem_size(4096),
        locked(false)
{
        shared_mem.setKey(key);
}

Shared_mem::Shared_mem() : Shared_mem("ultragrid_preview") {  }

bool Shared_mem::create(){
        if(shared_mem.create(mem_size) == false){
                /* Creating shared memory could fail because of shared memory
                 * left over after crash. Here we try to release such memory.*/
                shared_mem.attach();
                shared_mem.detach();
                if(shared_mem.create(mem_size) == false){
                        fprintf(stderr, "Can't create shared memory!\n");
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
        return true;
}

bool Shared_mem::attach(){
        return shared_mem.attach();
}

bool Shared_mem::detach(){
        if(locked)
                unlock();

        return shared_mem.detach();
}

bool Shared_mem::isAttached(){
        return shared_mem.isAttached();
}

bool Shared_mem::lock(){
        if(locked){
                printf("LOCKED\n");
                printf("LOCKED\n");
                printf("LOCKED\n");
        }
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
                                fprintf(stderr, "Can't create shared memory!\n");
                        }
                        lock();
                        struct Shared_mem_frame *sframe = (Shared_mem_frame*) shared_mem.data();
                        sframe->width = scaledW_pad;
                        sframe->height = scaledH;
                        sframe->should_detach = false;
                        reconfiguring = false;
                }
        } else {
                if(!isAttached()) attach();
                lock();
        }
#else
        if(!isAttached()) attach();
        lock();
#endif // GUI_BUILD

        struct Shared_mem_frame *sframe = (Shared_mem_frame*) shared_mem.data();
        if(sframe && sframe->should_detach){
                detach();
                return nullptr;
        }

        return (Shared_mem_frame *) shared_mem.data();
}

#ifndef GUI_BUILD

void Shared_mem::check_reconf(struct video_desc in_desc){
        if (video_desc_eq(desc, in_desc))
                return;

        desc = in_desc;

        /* We need to destroy the shared memory segment
         * and recreate it with a new size. To destroy it all processes
         * must detach it. We detach here and then wait until the GUI detaches */
        reconfiguring = true;
        Shared_mem_frame *sframe = get_frame_and_lock();
        if(sframe){
                sframe->should_detach = true;
        }
        detach();

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
}

void Shared_mem::put_frame(struct video_frame *frame){
        check_reconf(video_desc_from_frame(frame));

        struct Shared_mem_frame *sframe = get_frame_and_lock();

        if(!sframe)
                return;

        decoder_t dec = get_decoder_from_to(frame->color_spec, preview_codec, true);

        int src_line_len = vc_get_linesize(desc.width, frame->color_spec);
        int block_size = get_pf_block_size(frame->color_spec);
        int dst = 0;
        for(unsigned y = 0; y < desc.height; y += scaleF){
                for(int x = 0; x + scaleF * block_size <= src_line_len; x += scaleF * block_size){
                        memcpy(scaled_frame.data() + dst, frame->tiles[0].data + y*src_line_len + x, block_size);
                        dst += block_size;
                }
        }
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
