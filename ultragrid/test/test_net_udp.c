/*
 * FILE:    test_net_udp.c
 * AUTHORS: Colin Perkins
 * 
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/net_udp.h"
#include "test_net_udp.h"

#define BUFSIZE 1024

static void randomize(char buf[], int buflen)
{
        int i;

        for (i = 0; i < buflen; i++) {
                buf[i] = (lrand48() && 0xff00) >> 8;
        }
}

int test_net_udp(void)
{
        struct timeval timeout;
        socket_udp *s1, *s2;
        char buf1[BUFSIZE], buf2[BUFSIZE];
        const char *hname;
        int rc, i;

#ifndef WIN32

        /* "BSD" bug test that appears in this function uses fork and
         * exec, that are not present on WIN32.  Since the original
         * bug has not been seen on Win32 it's probably not worth
         * converting to use CreateProcess() */

        int status_parent, status_child;

#endif                          /* WIN32 */

        srand48(time(NULL));

        /**********************************************************************/
        /* The first test is to loopback a packet to ourselves...             */
        printf
            ("Testing UDP/IP networking (IPv4 loopback) ................................ ");
        fflush(stdout);
        s1 = udp_init("127.0.0.1", 5004, 5004, 1);
        if (s1 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }
        randomize(buf1, BUFSIZE);
        randomize(buf2, BUFSIZE);
        if (udp_send(s1, buf1, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot send packet");
                goto abort_loopback;
        }
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        udp_fd_zero();
        udp_fd_set(s1);
        rc = udp_select(&timeout);
        if (rc < 0) {
                printf("FAIL\n");
                perror("  Select failed");
                goto abort_loopback;
        }
        if (rc == 0) {
                printf("FAIL\n");
                printf("  No data waiting\n");
                goto abort_loopback;
        }
        if (!udp_fd_isset(s1)) {
                printf("FAIL\n");
                printf("  No data on file descriptor\n");
                goto abort_loopback;
        }
        if (udp_recv(s1, buf2, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Receive failed");
                goto abort_loopback;
        }
        if (memcmp(buf1, buf2, BUFSIZE) != 0) {
                printf("FAIL\n");
                printf("  Buffer corrupt\n");
                goto abort_loopback;
        }
        printf("Ok\n");
 abort_loopback:
        hname = udp_host_addr(s1);      /* we need this for the unicast test... */
        udp_exit(s1);

        /**********************************************************************/
        /* Now we send a packet to ourselves via our real network address...  */
        printf
            ("Testing UDP/IP networking (IPv4 unicast) ................................. ");
        fflush(stdout);
        s1 = udp_init(hname, 5004, 5005, 1);
        if (s1 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }

        s2 = udp_init(hname, 5005, 5004, 1);
        if (s2 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }

        randomize(buf1, BUFSIZE);
        randomize(buf2, BUFSIZE);
        if (udp_send(s1, buf1, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot send");
                goto abort_unicast;
        }
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        udp_fd_zero();
        udp_fd_set(s1);
        udp_fd_set(s2);
        rc = udp_select(&timeout);
        if (rc < 0) {
                printf("FAIL\n");
                perror("  Select failed");
                goto abort_unicast;
        }
        if (rc == 0) {
                printf("FAIL\n");
                printf("  No data waiting (no route to %s?)\n", hname);
                goto abort_unicast;
        }
        if (!udp_fd_isset(s2)) {
                printf("FAIL\n");
                printf("  No data on file descriptor\n");
                goto abort_unicast;
        }
        if (udp_recv(s2, buf2, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot receive");
                goto abort_unicast;
        }
        if (memcmp(buf1, buf2, BUFSIZE) != 0) {
                printf("FAIL\n");
                printf("  Buffer corrupt\n");
                goto abort_unicast;
        }
        printf("Ok\n");
 abort_unicast:
        udp_exit(s1);
        udp_exit(s2);

        /**********************************************************************/
        /* Loopback a packet to ourselves via multicast...                    */
        printf
            ("Testing UDP/IP networking (IPv4 multicast) ............................... ");
        fflush(stdout);
        s1 = udp_init("224.2.0.1", 5004, 5004, 1);
        if (s1 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }
        randomize(buf1, BUFSIZE);
        randomize(buf2, BUFSIZE);
        if (udp_send(s1, buf1, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot send");
                goto abort_multicast;
        }
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        udp_fd_zero();
        udp_fd_set(s1);
        rc = udp_select(&timeout);
        if (rc < 0) {
                printf("FAIL\n");
                perror("  Select failed");
                goto abort_multicast;
        }
        if (rc == 0) {
                printf("FAIL\n");
                printf("  No data waiting (no multicast loopback route?)\n");
                goto abort_multicast;
        }
        if (!udp_fd_isset(s1)) {
                printf("FAIL\n");
                printf("  No data on file descriptor\n");
                goto abort_multicast;
        }
        if (udp_recv(s1, buf2, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot receive");
                goto abort_multicast;
        }
        if (memcmp(buf1, buf2, BUFSIZE) != 0) {
                printf("FAIL\n");
                printf("  Buffer corrupt\n");
                goto abort_multicast;
        }
        printf("Ok\n");
 abort_multicast:
        udp_exit(s1);

        /**********************************************************************/
        /* Loopback a packet to ourselves via multicast, checking lengths...  */
        printf
            ("Testing UDP/IP networking (IPv4 length check) ............................ ");
        fflush(stdout);
        s1 = udp_init("224.2.0.1", 5004, 5004, 1);
        if (s1 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }
        for (i = 1; i < BUFSIZE; i++) {
                randomize(buf1, i);
                randomize(buf2, i);
                if (udp_send(s1, buf1, i) < 0) {
                        printf("FAIL\n");
                        perror("  Cannot send");
                        goto abort_length;
                }
                timeout.tv_sec = 1;
                timeout.tv_usec = 0;
                udp_fd_zero();
                udp_fd_set(s1);
                rc = udp_select(&timeout);
                if (rc < 0) {
                        printf("FAIL\n");
                        perror("  Select failed");
                        goto abort_length;
                }
                if (rc == 0) {
                        printf("FAIL\n");
                        printf
                            ("  No data waiting (no multicast loopback route?)\n");
                        goto abort_length;
                }
                if (!udp_fd_isset(s1)) {
                        printf("FAIL\n");
                        printf("  No data on file descriptor\n");
                        goto abort_length;
                }
                if (udp_recv(s1, buf2, BUFSIZE) != i) {
                        printf("FAIL\n");
                        perror("  Cannot receive");
                        goto abort_length;
                }
                if (memcmp(buf1, buf2, i) != 0) {
                        printf("FAIL\n");
                        printf("  Buffer corrupt\n");
                        goto abort_length;
                }
        }
        printf("Ok\n");
 abort_length:
        udp_exit(s1);

#ifdef HAVE_IPv6
        /**********************************************************************/
        /* The first test is to loopback a packet to ourselves...             */
        printf
            ("Testing UDP/IP networking (IPv6 loopback) ................................ ");
        fflush(stdout);
        s1 = udp_init("::1", 5004, 5004, 1);
        if (s1 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }
        randomize(buf1, BUFSIZE);
        randomize(buf2, BUFSIZE);
        if (udp_send(s1, buf1, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot send");
                goto abort_loopback_ipv6;
        }
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        udp_fd_zero();
        udp_fd_set(s1);
        rc = udp_select(&timeout);
        if (rc < 0) {
                printf("FAIL\n");
                perror("  Select failed");
                goto abort_loopback_ipv6;
        }
        if (rc == 0) {
                printf("FAIL\n");
                printf("  No data waiting\n");
                goto abort_loopback_ipv6;
        }
        if (!udp_fd_isset(s1)) {
                printf("FAIL\n");
                printf("  No data on file descriptor\n");
                goto abort_loopback_ipv6;
        }
        if (udp_recv(s1, buf2, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot receive");
                goto abort_loopback_ipv6;
        }
        if (memcmp(buf1, buf2, BUFSIZE) != 0) {
                printf("FAIL\n");
                printf("  Buffer corrupt\n");
                goto abort_loopback_ipv6;
        }
        printf("Ok\n");
 abort_loopback_ipv6:
        udp_exit(s1);

        /**********************************************************************/
        /* Loopback a packet to ourselves via multicast. The address is the   */
        /* SAP address, but we use a different port.                          */
        printf
            ("Testing UDP/IP networking (IPv6 multicast) ............................... ");
        fflush(stdout);
        s1 = udp_init("ff01::2:7ffe", 5004, 5004, 1);
        if (s1 == NULL) {
                printf("FAIL\n");
                printf("  Cannot initialize socket\n");
                return 1;
        }
        randomize(buf1, BUFSIZE);
        randomize(buf2, BUFSIZE);
        if (udp_send(s1, buf1, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot send");
                goto abort_multicast_ipv6;
        }
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        udp_fd_zero();
        udp_fd_set(s1);
        rc = udp_select(&timeout);
        if (rc < 0) {
                printf("FAIL\n");
                perror("  Select failed");
                goto abort_multicast_ipv6;
        }
        if (rc == 0) {
                printf("FAIL\n");
                printf("  No data waiting (no multicast loopback route?)\n");
                goto abort_multicast_ipv6;
        }
        if (!udp_fd_isset(s1)) {
                printf("FAIL\n");
                printf("  No data on file descriptor\n");
                goto abort_multicast_ipv6;
        }
        if (udp_recv(s1, buf2, BUFSIZE) < 0) {
                printf("FAIL\n");
                perror("  Cannot receive");
                goto abort_multicast_ipv6;
        }
        if (memcmp(buf1, buf2, BUFSIZE) != 0) {
                printf("FAIL\n");
                printf("  Buffer corrupt\n");
                goto abort_multicast_ipv6;
        }
        hname = udp_host_addr(s1);      /* we need this for the unicast test... */
        printf("Ok\n");
 abort_multicast_ipv6:
        udp_exit(s1);
#else
        printf
            ("Testing UDP/IP networking (IPv6 loopback) ................................ --\n");
        printf
            ("Testing UDP/IP networking (IPv6 unicast) ................................. --\n");
        printf
            ("Testing UDP/IP networking (IPv6 multicast) ............................... --\n");
#endif

        /**********************************************************************/

#ifdef WIN32
        printf
            ("Testing UDP/IP networking (FreeBSD bug) .................................. --\n");
#else
        printf
            ("Testing UDP/IP networking (FreeBSD bug) .................................. ");
        fflush(stdout);
        status_parent = 0;
        randomize(buf1, 64);
        s1 = udp_init("224.2.0.1", 5004, 5004, 1);
        if (s1 == NULL) {
                printf("fail (parent): cannot initialize socket\n");
                return 1;
        }
        rc = fork();
        if (rc == -1) {
                printf("fail: cannot fork\n");
                goto abort_bsd;
        } else if (rc == 0) {
                /* child */
                s2 = udp_init("224.2.0.1", 5004, 5004, 1);
                if (s2 == NULL) {
                        printf("FAIL\n");
                        printf("  Child cannot initialize socket\n");
                        exit(0);
                }
                if (udp_send(s2, buf1, 64) < 0) {
                        printf("FAIL\n");
                        perror("  Child cannot send");
                        exit(0);
                }
                timeout.tv_sec = 10;
                timeout.tv_usec = 0;
                udp_fd_zero();
                udp_fd_set(s2);
                rc = udp_select(&timeout);
                if (rc < 0) {
                        printf("FAIL\n");
                        perror("  Child select");
                        exit(0);
                }
                if (rc == 0) {
                        printf("FAIL\n");
                        printf
                            ("  Child: no data waiting (no multicast loopback route?)\n");
                        exit(0);
                }
                if (!udp_fd_isset(s2)) {
                        printf("FAIL\n");
                        printf("  Child: no data on file descriptor\n");
                        exit(0);
                }
                rc = udp_recv(s2, buf2, BUFSIZE);
                if (rc < 0) {
                        printf("FAIL\n");
                        perror("  Child cannot receive");
                        exit(0);
                }
                if (rc != 64) {
                        printf("FAIL\n");
                        printf("  Child: read size incorrect (%d != %d)\n", rc,
                               64);
                        exit(0);
                }
                if (memcmp(buf1, buf2, 64) != 0) {
                        printf("FAIL\n");
                        printf("  Child: buffer corrupt\n");
                        exit(0);
                }
                udp_exit(s2);
                exit(1);
        } else {
                /* parent */
                timeout.tv_sec = 10;
                timeout.tv_usec = 0;
                udp_fd_zero();
                udp_fd_set(s1);
                rc = udp_select(&timeout);
                if (rc < 0) {
                        printf("FAIL\n");
                        perror("  Parent select");
                        goto abort_bsd;
                }
                if (rc == 0) {
                        printf("FAIL\n");
                        printf
                            ("  Parent: no data waiting (no multicast loopback route?)\n");
                        goto abort_bsd;
                }
                if (!udp_fd_isset(s1)) {
                        printf("FAIL\n");
                        printf("  Parent: no data on file descriptor\n");
                        goto abort_bsd;
                }
                rc = udp_recv(s1, buf2, BUFSIZE);
                if (rc < 0) {
                        printf("FAIL\n");
                        perror("  Parent cannot receive");
                        goto abort_bsd;
                }
                if (rc != 64) {
                        printf("FAIL\n");
                        printf("  Parent: read size incorrect (%d != %d)\n", rc,
                               64);
                        goto abort_bsd;
                }
                if (memcmp(buf1, buf2, 64) != 0) {
                        printf("FAIL\n");
                        printf("  Parent: buffer corrupt\n");
                        goto abort_bsd;
                }
                status_parent = 1;
        }
 abort_bsd:
        wait(&status_child);
        if (status_parent && status_child) {
                printf("Ok\n");
        }
        udp_exit(s1);
#endif                          /* WIN32 */

        return 0;
}
