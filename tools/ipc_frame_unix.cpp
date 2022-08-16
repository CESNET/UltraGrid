#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <array>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#define CLOSESOCKET close
#define INVALID_SOCKET -1
typedef int fd_t;
#else
#include <winsock2.h>
#include <afunix.h>
#define CLOSESOCKET closesocket
typedef SOCKET fd_t;
#endif

#include <cerrno>
#include "ipc_frame_unix.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

struct Ipc_frame_reader{
        fd_t listen_fd;
        fd_t data_fd;
};

Ipc_frame_reader *ipc_frame_reader_new(const char *path){
        auto reader = new Ipc_frame_reader();
        reader->data_fd = INVALID_SOCKET;

        reader->listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if(reader->listen_fd == INVALID_SOCKET){
                delete reader;
                return nullptr;
        }

        sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path, sizeof(addr.sun_path));
        addr.sun_path[sizeof(addr.sun_path) - 1] = '\0';
        unlink(path);

        int ret = 0;
        ret = bind(reader->listen_fd, (const sockaddr *) &addr, sizeof(addr.sun_path));
        if(ret == -1){
                CLOSESOCKET(reader->listen_fd);
                delete reader;
                return nullptr;
        }

        ret = listen(reader->listen_fd, 5);
        if(ret == -1){
                CLOSESOCKET(reader->listen_fd);
                delete reader;
                return nullptr;
        }

        return reader;
}

void ipc_frame_reader_free(struct Ipc_frame_reader *reader){
        if(reader->data_fd != INVALID_SOCKET)
                CLOSESOCKET(reader->data_fd);
        if(reader->listen_fd != INVALID_SOCKET)
                CLOSESOCKET(reader->listen_fd);

        delete reader;
}

static size_t blocking_read(fd_t fd, char *dst, size_t size){
        size_t bytes_read = 0;

        while(bytes_read < size){
                int read_now = recv(fd, dst + bytes_read, size - bytes_read, 0);
                if(read_now <= 0)
                        break;

                bytes_read += read_now;
        }

        return bytes_read;
}

static bool socket_read_avail(fd_t fd){
        if(fd == INVALID_SOCKET)
                return false;

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
        if(reader->data_fd == INVALID_SOCKET && !try_accept(reader))
                return false;

        return socket_read_avail(reader->data_fd);
}

bool ipc_frame_reader_is_connected(struct Ipc_frame_reader *reader){
        return reader->data_fd != INVALID_SOCKET || try_accept(reader);
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
                CLOSESOCKET(reader->data_fd);
                reader->data_fd = INVALID_SOCKET;
        }

        return ret;
}

struct Ipc_frame_writer{
        fd_t data_fd;
};

Ipc_frame_writer *ipc_frame_writer_new(const char *path){
        auto writer = new Ipc_frame_writer;
        writer->data_fd = INVALID_SOCKET;

        sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path, sizeof(addr.sun_path));
        addr.sun_path[sizeof(addr.sun_path) - 1] = '\0';

        writer->data_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if(writer->data_fd == INVALID_SOCKET){
                delete writer;
                return nullptr;
        }

        int ret = connect(writer->data_fd, (const struct sockaddr *) &addr, sizeof(addr));
        if(ret == -1){
                CLOSESOCKET(writer->data_fd);
                delete writer;
                return nullptr;
        }

        return writer;
}

void ipc_frame_writer_free(struct Ipc_frame_writer *writer){
        if(writer->data_fd != INVALID_SOCKET)
                CLOSESOCKET(writer->data_fd);

        delete writer;
}

namespace{

void block_write(fd_t fd, void *buf, size_t size){
        size_t written = 0;
        char *src = static_cast<char *>(buf);

        while(written < size){
                int ret = send(fd, src + written, size - written, MSG_NOSIGNAL);
                if(ret == -1)
                        return;
                written += ret;
        }
}

} //anon namespace

bool ipc_frame_writer_write(struct Ipc_frame_writer *writer, const struct Ipc_frame *f){
        std::array<char, IPC_FRAME_HEADER_LEN> header;

        ipc_frame_write_header(&f->header, header.data());

        errno = 0;
        block_write(writer->data_fd, header.data(), header.size());
        block_write(writer->data_fd, f->data, f->header.data_len);

        return errno == 0;
}

