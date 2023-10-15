/*
 * FILE:    rtsp/ultragrid_rtsp.h
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec     <pulec@cesnet.cz>
 *          Jakub Kováč      <xkovac5@mail.muni.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2010-2023 CESNET, z. s. p. o.
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
 *      This product includes software developed by the Fundació i2CAT,
 *      Internet I Innovació Digital a Catalunya. This product also includes
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

#ifndef ULTRAGRID_RTSP_HH
#define ULTRAGRID_RTSP_HH

#include <pthread.h>
#include "rtsp/rtsp_utils.h"
#include "audio/types.h"
#include "module.h"

class UltragridRTSPServer;

class ultragrid_rtsp {
public:
    /** 
     * @param rtsp_port The caller is responsible for ensuring the port is available.
     *                  If set as 0, the server will use the default value.
    */
    ultragrid_rtsp(unsigned int rtsp_port, struct module* mod, rtsp_media_type_t media_type, audio_codec_t audio_codec,
            int audio_sample_rate, int audio_channels, int audio_bps, int rtp_video_port, int rtp_audio_port);
    /**
     * Stops server and frees any allocated memory
    */
    ~ultragrid_rtsp();

    /**
     * Copy constructor and copy assignment operator do not make sense in this context
    */
    ultragrid_rtsp(const ultragrid_rtsp&) = delete;
    ultragrid_rtsp& operator=(const ultragrid_rtsp&) = delete;

    /**
     * Start server in new thread
     * 
     * @retval 0    New thread created and server started successfully
    */
    int start_server();
    /**
     * @note can be started again by calling start_server
    */
    void stop_server();

private:
    static void* server_runner(void* args);

    pthread_t server_thread;
    bool thread_running;
    char server_stop_flag; // 0 server can run, 1 server should stop
    std::unique_ptr<UltragridRTSPServer> rtsp_server; // pointer to avoid name clashes with live555
};

#endif // ULTRAGRID_RTSP_HH
