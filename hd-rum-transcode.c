#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <arpa/inet.h>
#include <netdb.h>
#define __USE_GNU 1
#include <pthread.h>
#include <fcntl.h>


#define SIZE    10000

struct replica {
    const char *host;
    unsigned short port;
    int sock;
};


struct item {
    struct item *next;
    long size;
    char buf[SIZE];
};

static struct item *queue;
static struct item *qhead;
static struct item *qtail;
static int qempty = 1;
static int qfull = 0;
static pthread_mutex_t qempty_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t qfull_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t qempty_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t qfull_cond = PTHREAD_COND_INITIALIZER;

struct replica *replicas;
int count;

void qinit(int qsize)
{
    int i;

    printf("initializing packet queue for %d items\n", qsize);

    queue = (struct item *) calloc(qsize, sizeof(struct item));
    if (queue == NULL) {
        fprintf(stderr, "not enough memory\n");
        exit(2);
    }

    for (i = 0; i < qsize; i++)
        queue[i].next = queue + i + 1;
    queue[qsize - 1].next = queue;

    qhead = qtail = queue;
}


int buffer_size(int sock, int optname, int size)
{
    socklen_t len = sizeof(int);

    if (setsockopt(sock, SOL_SOCKET, optname, (void *) &size, len)
        || getsockopt(sock, SOL_SOCKET, optname, (void *) &size, &len)) {
        perror("[sg]etsockopt()");
        return -1;
    }

    printf("UDP %s buffer size set to %d bytes\n",
           (optname == SO_SNDBUF) ? "send" : "receiver", size);

    return 0;
}


int output_socket(unsigned short port, const char *host, int bufsize)
{
    int s;
    struct addrinfo hints;
    struct addrinfo *res;
    char saddr[INET_ADDRSTRLEN];
    char p[6];

    memset(&hints, '\0', sizeof(hints));
    hints.ai_flags = AI_CANONNAME;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    hints.ai_family = PF_INET;

    snprintf(p, 6, "%d", port);
    p[5] = '\0';

    if (getaddrinfo(host, p, &hints, &res)) {
        int err = errno;
        if (err == 0)
            fprintf(stderr, "Address not found: %s\n", host);
        else
            fprintf(stderr, "%s: %s\n", gai_strerror(err), host);
        exit(2);
    }

    inet_ntop(AF_INET, &((struct sockaddr_in *) res->ai_addr)->sin_addr,
              saddr, sizeof(saddr));
    printf("connecting to %s (%s) port %d\n",
           res->ai_canonname, saddr, port);

    if ((s = socket(res->ai_family, res->ai_socktype, 0)) == -1) {
        perror("socket");
        exit(2);
    }

    if (buffer_size(s, SO_SNDBUF, bufsize))
        exit(2);

    if (connect(s, res->ai_addr, res->ai_addrlen)) {
        perror("connect");
        exit(2);
    }

    return s;
}


void *writer(void *arg)
{
    int i;

    while (1) {
        while (qhead != qtail) {
            for (i = 0; i < count; i++)
                write(replicas[i].sock, qhead->buf, qhead->size);

            qhead = qhead->next;

            pthread_mutex_lock(&qfull_mtx);
            qfull = 0;
            pthread_cond_signal(&qfull_cond);
            pthread_mutex_unlock(&qfull_mtx);
        }

        pthread_mutex_lock(&qempty_mtx);
        if (qempty)
            pthread_cond_wait(&qempty_cond, &qempty_mtx);
        qempty = 1;
        pthread_mutex_unlock(&qempty_mtx);
    }

    return NULL;
}


int main(int argc, char **argv)
{
    unsigned short port;
    int qsize;
    int bufsize;
    struct sockaddr_in addr;
    int sock_in;
    pthread_t thread;
    int err = 0;
    int i;

    if (argc < 4) {
        fprintf(stderr, "%s buffer_size port host...\n", argv[0]);
        return 1;
    }

    if ((bufsize = atoi(argv[1])) <= 0) {
        fprintf(stderr, "invalid buffer size: %d\n", bufsize);
        return 1;
    }
    switch (argv[1][strlen(argv[1]) - 1]) {
    case 'K':
    case 'k':
        bufsize *= 1024;
        break;

    case 'M':
    case 'm':
        bufsize *= 1024*1024;
        break;
    }

    qsize = bufsize / 8000;

    printf("using UDP send and receive buffer size of %d bytes\n", bufsize);

    if ((port = atoi(argv[2])) <= 0) {
        fprintf(stderr, "invalid port: %d\n", port);
        return 1;
    }

    qinit(qsize);

    /* input socket */
    if ((sock_in = socket(PF_INET, SOCK_DGRAM, 0)) == -1) {
        perror("input socket");
        return 2;
    }

    if (buffer_size(sock_in, SO_RCVBUF, bufsize))
        exit(2);

    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sock_in, (struct sockaddr *) &addr,
             sizeof(struct sockaddr_in))) {
        perror("bind");
        return 2;
    }

    printf("listening on *:%d\n", port);

    /* output socket(s) */
    count = argc - 3;
    replicas = (struct replica *) calloc(count, sizeof(struct replica));
    if (replicas == NULL) {
        fprintf(stderr, "not enough memory for replica array");
        return 2;
    }

    for (i = 0; i < count; i++) {
        replicas[i].host = argv[3 + i];

        if (i > 0 && strcmp(replicas[i - 1].host, replicas[i].host) == 0)
            replicas[i].port = replicas[i - 1].port + 1;
        else
            replicas[i].port = port;

        replicas[i].sock = output_socket(replicas[i].port, replicas[i].host,
                                         bufsize);
    }

    if (pthread_create(&thread, NULL, writer, NULL)) {
        fprintf(stderr, "cannot create writer thread\n");
        return 2;
    }

    /* main loop */
    while (1) {
        fcntl(sock_in, F_SETFL, O_NONBLOCK);

        while (qtail->next != qhead
               && ((qtail->size = read(sock_in, qtail->buf, SIZE)) > 0
                   || (err = errno) == EAGAIN)) {

            if (qtail->size <= 0) {
                pthread_yield();
            }
            else {
                qtail = qtail->next;

                pthread_mutex_lock(&qempty_mtx);
                qempty = 0;
                pthread_cond_signal(&qempty_cond);
                pthread_mutex_unlock(&qempty_mtx);
            }
        }

        if (qtail->size <= 0)
            break;

        pthread_mutex_lock(&qfull_mtx);
        if (qfull)
            pthread_cond_wait(&qfull_cond, &qfull_mtx);
        qfull = 1;
        pthread_mutex_unlock(&qfull_mtx);
    }

    if (qtail->size < 0) {
        printf("read: %s\n", strerror(err));
        return 2;
    }

    return 0;
}

