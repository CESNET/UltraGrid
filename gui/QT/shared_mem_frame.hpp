#ifndef SHARED_MEM_FRAME_HPP
#define SHARED_MEM_FRAME_HPP

struct Shared_mem_frame{
	int width, height;
	unsigned char pixels[];
};


#endif
