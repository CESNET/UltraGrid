#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "ipc_frame.h"

namespace{
int32_t read_int(const char *buf){
	int32_t result = 0;
	auto char_ptr = reinterpret_cast<char *>(&result);

	char_ptr[0] = buf[0];
	char_ptr[1] = buf[1];
	char_ptr[2] = buf[2];
	char_ptr[3] = buf[3];

	return result;
}

void write_int(char *dst, uint32_t val){
        auto src = reinterpret_cast<char *>(&val);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
}

} //anon namespace


bool ipc_frame_parse_header(Ipc_frame_header *hdr, const char *buf){
	hdr->width = read_int(buf + 0);
	hdr->height = read_int(buf + 4);
	hdr->data_len = read_int(buf + 8);
	hdr->color_spec = static_cast<Ipc_frame_color_spec>(read_int(buf + 12));

	return true;
}

void ipc_frame_write_header(const struct Ipc_frame_header *hdr, char *dst){
	memset(dst, 0, IPC_FRAME_HEADER_LEN);
	write_int(dst + 0, hdr->width);
	write_int(dst + 4, hdr->height);
	write_int(dst + 8, hdr->data_len);
	write_int(dst + 12, hdr->color_spec);
}

Ipc_frame *ipc_frame_new(){
	auto frame = static_cast<Ipc_frame *>(malloc(sizeof(Ipc_frame)));
	frame->header.width = 0;
	frame->header.height = 0;
	frame->header.data_len = 0;
	frame->header.color_spec = IPC_FRAME_COLOR_NONE;

	frame->data = nullptr;
	frame->alloc_size = 0;

	return frame;
}

void ipc_frame_free(Ipc_frame *frame){
	free(frame->data);
	free(frame);
}

bool ipc_frame_reserve(Ipc_frame *frame, size_t size){
	if(size <= frame->alloc_size)
		return true;

	auto newbuf = static_cast<char *>(realloc(frame->data, size));
	if(!newbuf)
		return false;
	frame->data = newbuf;
	frame->alloc_size = size;

	return true;
}
