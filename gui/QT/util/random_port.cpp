#include <QtGlobal>
#include <cerrno>

#ifndef Q_OS_WIN
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#define CLOSESOCKET close
#define WIN_WSA_CLEAN()
#define INVALID_SOCKET -1
typedef int fd_t;
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#define CLOSESOCKET closesocket
#define WIN_WSA_CLEAN WSACleanup
typedef SOCKET fd_t;
#endif

#include "random_port.hpp"

int getRandomFreePort(){
#ifdef Q_OS_WIN
	WSAData wsadata;
	WSAStartup(MAKEWORD(2,2), &wsadata);
#endif

	bool ipv4fallback = false;

	fd_t socket_fd = socket(AF_INET6, SOCK_STREAM, 0);
	if(socket_fd == INVALID_SOCKET && errno == EAFNOSUPPORT){
		ipv4fallback = true;
		socket_fd = socket(AF_INET, SOCK_STREAM, 0);
	}

	if(socket_fd == INVALID_SOCKET){
		WIN_WSA_CLEAN();
		return -1;
	}

	struct sockaddr_storage ss{};
	socklen_t s_len = 0;
	if(!ipv4fallback){
		auto *s_in6 = reinterpret_cast<struct sockaddr_in6 *>(&ss);
		s_in6->sin6_family = AF_INET6;
		s_in6->sin6_addr = in6addr_any;
		s_in6->sin6_port = 0;
		s_len = sizeof(*s_in6);
	} else {
		auto *s_in = reinterpret_cast<struct sockaddr_in *>(&ss);
		s_in->sin_family = AF_INET;
		s_in->sin_addr.s_addr = htonl(INADDR_ANY);
		s_in->sin_port = 0;
		s_len = sizeof(*s_in);
	}

	int rc = ::bind(socket_fd, reinterpret_cast<const struct sockaddr *>(&ss), s_len);

	if(rc != 0){
		CLOSESOCKET(socket_fd);
		WIN_WSA_CLEAN();
		return -1;
	}

	rc = getsockname(socket_fd, reinterpret_cast<struct sockaddr *>(&ss), &s_len);
	CLOSESOCKET(socket_fd);
	WIN_WSA_CLEAN();

	if(rc != 0)
		return -1;

	if(ipv4fallback){
		auto *s_in = reinterpret_cast<struct sockaddr_in *>(&ss);
		return ntohs(s_in->sin_port);
	}

	auto *s_in6 = reinterpret_cast<struct sockaddr_in6 *>(&ss);
	return ntohs(s_in6->sin6_port);
}
