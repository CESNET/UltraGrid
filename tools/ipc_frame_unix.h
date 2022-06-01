#ifndef IPC_FRAME_UNIX_e3ab60622626
#define IPC_FRAME_UNIX_e3ab60622626

#include "ipc_frame.h"

struct Ipc_frame_reader;

Ipc_frame_reader *ipc_frame_reader_new(const char *path);
void ipc_frame_reader_free(struct Ipc_frame_reader *reader);

bool ipc_frame_reader_has_frame(struct Ipc_frame_reader *reader);
bool ipc_frame_reader_is_connected(struct Ipc_frame_reader *reader);
bool ipc_frame_reader_read(struct Ipc_frame_reader *reader, struct Ipc_frame *dst);

#ifdef __cplusplus
#include <memory>
struct Ipc_frame_reader_deleter{
	void operator()(Ipc_frame_reader *r){ ipc_frame_reader_free(r); }
};
using Ipc_frame_reader_uniq = std::unique_ptr<Ipc_frame_reader, Ipc_frame_reader_deleter>;
#endif //__cplusplus

#endif
