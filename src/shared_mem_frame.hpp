/**
 * @file   shared_mem_frame.hpp
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

#ifndef SHARED_MEM_FRAME_HPP
#define SHARED_MEM_FRAME_HPP

#include <QSharedMemory>

#ifndef GUI_BUILD
#include "types.h"
#endif // GUI_BUILD

struct Shared_mem_frame{
        int width, height;
        bool should_detach;
        unsigned char pixels[];
};

class Shared_mem{
public:
        Shared_mem(const char *key);
        Shared_mem();
        ~Shared_mem();
        bool create();
        bool attach();
        bool detach();
        bool is_attached();

        bool lock();
        bool unlock();

        void destroy();

        void setKey(const char *key);

#ifndef GUI_BUILD
        void put_frame(struct video_frame *frame);
#endif // GUI_BUILD

        Shared_mem_frame *get_frame_and_lock();

private:
        size_t mem_size;
        bool locked;
        bool owner;
        QSharedMemory shared_mem;

#ifndef GUI_BUILD
        void check_reconf(struct video_desc in_desc);
        static const codec_t preview_codec = RGB;
        int scaledW = 0;
        int scaledH = 0;
        int scaleF = 0;
        int scaledW_pad = 0;
        struct video_desc desc = video_desc();
        std::vector<unsigned char> scaled_frame;

        bool reconfiguring = false;
#endif // GUI_BUILD
};


#endif
