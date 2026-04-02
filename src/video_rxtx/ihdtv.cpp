/*
 * FILE:    ihdtv.cpp
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
 * Copyright (c) 2005-2026 CESNET, zájmové sdružení právnických osob
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

#include <cassert>          // for assert
#include <cstdio>           // for NULL, printf
#include <memory>           // for shared_ptr
#include <string>           // for basic_string, string
#include <utility>          // for move

#include "debug.h"          // for LOG_LEVEL_WARNING, bug_msg
#include "host.h"           // for common_opts, register_should_exit_callback
#include "ihdtv/ihdtv.h"    // for ihdtv_connection, ihdtv_init_rx_session
#include "lib_common.h"     // for REGISTER_MODULE, library_class
#include "types.h"          // for tile, video_frame
#include "video_display.h"  // for PUTF_BLOCKING, display_get_frame, display...
#include "video_rxtx.h"     // for vrxtx_params, VIDEO_RXTX_ABI_VERSION, vid...

class ihdtv_video_rxtx {
public:
        ihdtv_video_rxtx(const struct vrxtx_params *params,
                         const struct common_opts  *common);
        ~ihdtv_video_rxtx();
        void send_frame(std::shared_ptr<video_frame>) noexcept;
        static void *receiver_thread(void *arg) {
                ihdtv_video_rxtx *s = static_cast<ihdtv_video_rxtx *>(arg);
                return s->receiver_loop();
        }
private:
        struct module *m_parent;
        void *receiver_loop();
        ihdtv_connection m_tx_connection, m_rx_connection;
        struct display *m_display_device;
};

using namespace std;

void
ihdtv_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        ihdtv_send(&m_tx_connection, tx_frame.get(), 9000000);      // FIXME: fix the use of frame size!!
}

static void
should_exit_callback(void *should_exit)
{
        *(bool *) should_exit = true;
}

void *ihdtv_video_rxtx::receiver_loop()
{
        ihdtv_connection *connection = &m_rx_connection;

        struct video_frame *frame_buffer = NULL;

        bool should_exit = false;
        register_should_exit_callback(m_parent, should_exit_callback, &should_exit);
        while (!should_exit) {
                frame_buffer = display_get_frame(m_display_device);
                if (ihdtv_receive
                    (connection, frame_buffer->tiles[0].data, frame_buffer->tiles[0].data_len))
                        return 0;       // we've got some error. probably empty buffer
                display_put_frame(m_display_device, frame_buffer, PUTF_BLOCKING);
        }
        unregister_should_exit_callback(m_parent, should_exit_callback,
                                        &should_exit);
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

ihdtv_video_rxtx::ihdtv_video_rxtx(const struct vrxtx_params *params,
                                   const struct common_opts  *common)
    : m_parent(common->parent), m_tx_connection(), m_rx_connection()
{
        const struct rxtx_medium_params *params_video =
            &params->medium[TX_MEDIA_VIDEO];
        int    argc = uv_argc;
        char **argv = uv_argv;
        if ((argc != 0) && (argc != 1) && (argc != 2)) {
                throw string("Wrong number of parameters");
        }

        printf("Initializing ihdtv protocol\n");

        // we cannot act as both together, because parameter parsing would have to be revamped
        if (params_video->rxtx_mode == (MODE_SENDER | MODE_RECEIVER)) {
               throw string 
                        ("Error: cannot act as both sender and receiver together in ihdtv mode");
        }

        m_display_device = params->display_device;

        if ((params_video->rxtx_mode & MODE_RECEIVER) != 0U) {
                assert(params->display_device != nullptr);
                if (ihdtv_init_rx_session
                                (&m_rx_connection, (argc == 0) ? NULL : argv[0],
                                 (argc ==
                                  0) ? NULL : ((argc == 1) ? argv[0] : argv[1]),
                                 3000, 3001, common->mtu) != 0) {
                        throw string("Error initializing receiver session");
                }
        }

        if ((params_video->rxtx_mode & MODE_SENDER) != 0U) {
                assert(params->capture_device != nullptr);
                if (argc == 0) {
                        throw string("Error: specify the destination address");
                }

                if (ihdtv_init_tx_session
                                (&m_tx_connection, argv[0],
                                 (argc == 2) ? argv[1] : argv[0],
                                 common->mtu) != 0) {
                        throw string("Error initializing sender session");
                }
        }
}

ihdtv_video_rxtx::~ihdtv_video_rxtx()
{
}

static void *
create_video_rxtx_ihdtv(const struct vrxtx_params *params,
                        const struct common_opts  *common)
{
        bug_msg(LOG_LEVEL_WARNING,
                "Warning: iHDTV support may be currently broken. ");
        return new ihdtv_video_rxtx(params, common);
}

static void done(void *state) {
        auto *s = static_cast<ihdtv_video_rxtx *>(state);
        delete s;
}

static void
send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = static_cast<ihdtv_video_rxtx *>(state);
        s->send_frame(std::move(f));
}

static const struct video_rxtx_info ihdtv_video_rxtx_info = {
        .long_name              = "iHDTV",
        .create                 = create_video_rxtx_ihdtv,
        .done                   = done,
        .send_frame             = send_frame,
        .join_sender            = nullptr,
        .set_sender_audio_spec  = nullptr,
        .receiver_routine       = ihdtv_video_rxtx::receiver_thread,
};

REGISTER_MODULE(ihdtv, &ihdtv_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

