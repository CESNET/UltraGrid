#ifndef IPC_FRAME_84d5ffd7aad2
#define IPC_FRAME_84d5ffd7aad2

#include <stdbool.h>
#include <stddef.h>

#define IPC_FRAME_HEADER_LEN 128

enum Ipc_frame_color_spec{
	IPC_FRAME_COLOR_NONE = 0,
	IPC_FRAME_COLOR_RGBA = 1,
	IPC_FRAME_COLOR_UYVY = 2,
	IPC_FRAME_COLOR_RGB = 11
};

struct Ipc_frame_header{
	int width;
	int height;
	int data_len;
	enum Ipc_frame_color_spec color_spec;
};

struct Ipc_frame{
	Ipc_frame_header header;
	char *data;

	size_t alloc_size;
};

bool ipc_frame_parse_header(struct Ipc_frame_header *hdr, const char *buf);
void ipc_frame_write_header(const struct Ipc_frame_header *hdr, char *dst);

Ipc_frame *ipc_frame_new();
bool ipc_frame_reserve(struct Ipc_frame *frame, size_t size);
void ipc_frame_free(struct Ipc_frame *frame);

#ifdef __cplusplus
#include <memory>
struct Ipc_frame_deleter{ void operator()(Ipc_frame *f){ ipc_frame_free(f); } };
using Ipc_frame_uniq = std::unique_ptr<Ipc_frame, Ipc_frame_deleter>;
#endif //__cplusplus

#endif
