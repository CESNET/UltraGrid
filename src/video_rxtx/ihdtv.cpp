/*
 * FILE:    ihdtv.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <string>

#include "debug.h"
#include "host.h"
#include "ihdtv/ihdtv.h"
#include "video_display.h"
#include "video_capture.h"
#include "video_rxtx/ihdtv.h"
#include "video.h"

using namespace std;

void ihdtv_video_rxtx::send_frame(struct video_frame *tx_frame)
{
#ifdef HAVE_IHDTV
        ihdtv_send(&m_tx_connection, tx_frame, 9000000);      // FIXME: fix the use of frame size!!
        VIDEO_FRAME_DISPOSE(tx_frame);
#else // IHDTV
        UNUSED(tx_frame);
#endif // IHDTV
}

void *ihdtv_video_rxtx::receiver_loop()
{
#ifdef HAVE_IHDTV
        ihdtv_connection *connection = &m_rx_connection;

        struct video_frame *frame_buffer = NULL;

        while (!should_exit_receiver) {
                frame_buffer = display_get_frame(m_display_device);
                if (ihdtv_receive
                    (connection, frame_buffer->tiles[0].data, frame_buffer->tiles[0].data_len))
                        return 0;       // we've got some error. probably empty buffer
                display_put_frame(m_display_device, frame_buffer, PUTF_BLOCKING);
        }
#endif // IHDTV
        return NULL;
}

#if 0
static void *ihdtv_sender_thread(void *arg)
{
        ihdtv_connection *connection = (ihdtv_connection *) ((void **)arg)[0];
        struct vidcap *capture_device = (struct vidcap *)((void **)arg)[1];
        struct video_frame *tx_frame;
        struct audio_frame *audio;

        while (1) {
                if ((tx_frame = vidcap_grab(capture_device, &audio)) != NULL) {
                        ihdtv_send(connection, tx_frame, 9000000);      // FIXME: fix the use of frame size!!
                        free(tx_frame);
                } else {
                        fprintf(stderr,
                                "Error receiving frame from capture device\n");
                        return 0;
                }
        }

        return 0;
}
#endif

ihdtv_video_rxtx::ihdtv_video_rxtx(struct module *parent, struct video_export *video_exporter,
                                        const char *requested_compression,
                                        struct vidcap *capture_device, struct display *display_device,
                                        int requested_mtu, int argc, char **argv) :
        video_rxtx(parent, video_exporter, requested_compression)
{
#ifdef HAVE_IHDTV
        if ((argc != 0) && (argc != 1) && (argc != 2)) {
                throw string("Wrong number of parameters");
        }

        printf("Initializing ihdtv protocol\n");

        // we cannot act as both together, because parameter parsing would have to be revamped
        if (capture_device && display_device) {
               throw string 
                        ("Error: cannot act as both sender and receiver together in ihdtv mode");
        }

        m_display_device = display_device;

        if (requested_mtu == 0)     // mtu not specified on command line
        {
                requested_mtu = 8112;       // not really a mtu, but a video-data payload per packet
        }

        if (display_device) {
                if (ihdtv_init_rx_session
                                (&m_rx_connection, (argc == 0) ? NULL : argv[0],
                                 (argc ==
                                  0) ? NULL : ((argc == 1) ? argv[0] : argv[1]),
                                 3000, 3001, requested_mtu) != 0) {
                        throw string("Error initializing receiver session");
                }
        }

        if (capture_device != 0) {
                if (argc == 0) {
                        throw string("Error: specify the destination address");
                }

                if (ihdtv_init_tx_session
                                (&m_tx_connection, argv[0],
                                 (argc == 2) ? argv[1] : argv[0],
                                 requested_mtu) != 0) {
                        throw string("Error initializing sender session");
                }
        }
#else
        UNUSED(parent);
        UNUSED(video_exporter);
        UNUSED(requested_compression);
        UNUSED(capture_device);
        UNUSED(display_device);
        UNUSED(requested_mtu);
        UNUSED(argc);
        UNUSED(argv);
        throw string("Support for iHDTV not compiled in");
#endif // HAVE_IHDTV
}

ihdtv_video_rxtx::~ihdtv_video_rxtx()
{
}

