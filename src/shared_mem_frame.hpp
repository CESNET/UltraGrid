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
        bool create();
        bool attach();
        bool detach();
        bool isAttached();

        bool lock();
        bool unlock();

#ifndef GUI_BUILD
        void put_frame(struct video_frame *frame);
#endif // GUI_BUILD

        Shared_mem_frame *get_frame_and_lock();

private:
        const char *key;
        size_t mem_size;
        bool locked;
        QSharedMemory shared_mem;

#ifndef GUI_BUILD
        void check_reconf(struct video_desc in_desc);
        static const codec_t preview_codec = RGB;
        int scaledW, scaledH;
        int scaleF;
        int scaledW_pad;
        struct video_desc desc = {};
        std::vector<unsigned char> scaled_frame;

        bool reconfiguring = false;
#endif // GUI_BUILD
};


#endif
