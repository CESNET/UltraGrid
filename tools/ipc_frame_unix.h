#ifndef IPC_FRAME_UNIX_e3ab60622626
#define IPC_FRAME_UNIX_e3ab60622626

#include "ipc_frame.h"

#ifdef _WIN32
#define PLATFORM_TMP_DIR "C:/temp/"
#else
#define PLATFORM_TMP_DIR "/tmp/"
#endif

struct Ipc_frame_reader;

Ipc_frame_reader *ipc_frame_reader_new(const char *path);
void ipc_frame_reader_free(struct Ipc_frame_reader *reader);

bool ipc_frame_reader_has_frame(struct Ipc_frame_reader *reader);
bool ipc_frame_reader_is_connected(struct Ipc_frame_reader *reader);
bool ipc_frame_reader_read(struct Ipc_frame_reader *reader, struct Ipc_frame *dst);

struct Ipc_frame_writer;

Ipc_frame_writer *ipc_frame_writer_new(const char *path);
void ipc_frame_writer_free(struct Ipc_frame_writer *writer);

bool ipc_frame_writer_write(struct Ipc_frame_writer *writer, const struct Ipc_frame *f);

#ifdef __cplusplus
#include <memory>
struct Ipc_frame_reader_deleter{
	void operator()(Ipc_frame_reader *r){ ipc_frame_reader_free(r); }
};
using Ipc_frame_reader_uniq = std::unique_ptr<Ipc_frame_reader, Ipc_frame_reader_deleter>;

struct Ipc_frame_writer_deleter{
	void operator()(Ipc_frame_writer *w){ ipc_frame_writer_free(w); }
};
using Ipc_frame_writer_uniq = std::unique_ptr<Ipc_frame_writer, Ipc_frame_writer_deleter>;
#endif //__cplusplus


#endif
