/*
 * AUTHOR:   David Cassany   <david.cassany@i2cat.net>,
 *           Ignacio Contreras <ignacio.contreras@i2cat.net>,
 *           Gerard Castillo <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2015-2024 CESNET
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
 *      California Information Sciences Institute.
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

#ifndef RTP_RTPDEC_STATE_H_1712C4BC_F5CA_44DE_AF8B_EB2996D0E8A5
#define RTP_RTPDEC_STATE_H_1712C4BC_F5CA_44DE_AF8B_EB2996D0E8A5

struct coded_data;
struct video_frame;

struct decode_data_rtsp {
        int (*decode)(struct coded_data *cdata, void *decode_data);
        struct video_frame *frame;
        int offset_len;
        int video_pt;

        // codec specific
        union {
                struct {
                        /// SPS/PPS headers extracted from RTSP
                        unsigned char offset_buffer[2048];
                } h264;
                struct {
                        char *dqt_start;
                        int   not_first_run; ///< decrease verbosity next time
                } jpeg;
        };
};

#endif // defined RTP_RTPDEC_STATE_H_1712C4BC_F5CA_44DE_AF8B_EB2996D0E8A5
