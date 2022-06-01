#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include "ipc_frame_unix.h"

struct Ipc_frame_reader{
	int listen_fd;
	int data_fd;
};

Ipc_frame_reader *ipc_frame_reader_new(const char *path){
	auto reader = static_cast<Ipc_frame_reader *>(malloc(sizeof(Ipc_frame_reader)));
	reader->data_fd = -1;

	reader->listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);

	sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path), path);
	unlink(path);

	bind(reader->listen_fd, (const sockaddr *) &addr, sizeof(addr.sun_path)); //TODO check return

	listen(reader->listen_fd, 5);

	return reader;
}

void ipc_frame_reader_free(struct Ipc_frame_reader *reader){
	if(reader->data_fd >= 0)
		close(reader->data_fd);
	if(reader->listen_fd >= 0)
		close(reader->listen_fd);

	free(reader);
}

static size_t blocking_read(int fd, char *dst, size_t size){
	size_t bytes_read = 0;

	while(bytes_read < size){
		int read_now = read(fd, dst + bytes_read, size - bytes_read);
		if(read_now <= 0)
			break;

		bytes_read += read_now;
	}

	return bytes_read;
}

static bool socket_read_avail(int fd){
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(fd, &fds);

	struct timeval tv;
	tv.tv_sec = 0;
	tv.tv_usec = 0;

	int ret = select(fd + 1, &fds, nullptr, nullptr, &tv);

	return ret > 0;
}

static bool try_accept(struct Ipc_frame_reader *reader){
	if(!socket_read_avail(reader->listen_fd))
		return false;

	reader->data_fd = accept(reader->listen_fd, nullptr, 0);
	return true;
}

bool ipc_frame_reader_has_frame(struct Ipc_frame_reader *reader){
	if(reader->data_fd < 0 && !try_accept(reader))
		return false;

	return socket_read_avail(reader->data_fd);
}

bool ipc_frame_reader_is_connected(struct Ipc_frame_reader *reader){
	return reader->data_fd >= 0 || try_accept(reader);
}

static bool do_frame_read(Ipc_frame_reader *reader, Ipc_frame *dst){
	char header_buf[IPC_FRAME_HEADER_LEN];

	if(blocking_read(reader->data_fd, header_buf, IPC_FRAME_HEADER_LEN) != IPC_FRAME_HEADER_LEN)
		return false;

	if(!ipc_frame_parse_header(&dst->header, header_buf))
		return false;

	if(!ipc_frame_reserve(dst, dst->header.data_len))
		return false;

	int read_data = blocking_read(reader->data_fd, dst->data, dst->header.data_len);

	return read_data == dst->header.data_len;
}

bool ipc_frame_reader_read(Ipc_frame_reader *reader, Ipc_frame *dst){
	bool ret = do_frame_read(reader, dst);
	if(!ret){
		close(reader->data_fd);
		reader->data_fd = -1;
	}

	return ret;
}
