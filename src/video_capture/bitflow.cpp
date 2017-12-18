/**
 * @file   video_capture/bitflow.cpp
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017 CESNET z.s.p.o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "bitflow/BFciLib.h"
#include <inttypes.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"


struct vidcap_state_bitflow {
        tCIp        sCIp;                    /* device open token */
        int     sNdx;                        /* device index */
        tCIU32                      nPtrs;
        tCIU8                       **uPtrs;
        struct video_desc desc;
        tCIU32 nFrames;
};

static void vidcap_bitflow_done(void *state);

static bool map_frame_buffers(struct vidcap_state_bitflow *s) {
        tCIRC circ;
        if (kCIEnoErr != (circ = CiMapFrameBuffers(s->sCIp,
                                        0, // 0-read only, 1- read/write
                                        &s->nPtrs,
                                        &s->uPtrs)))
        {
                log_msg(LOG_LEVEL_ERROR, "CiMapFrameBuffers gave '%s'\n",CiErrStr(circ));
                if ((NULL != s->uPtrs) && (kCIEnoErr != (circ = CiUnmapFrameBuffers(s->sCIp)))){
                        log_msg(LOG_LEVEL_ERROR, "CiUnmapFrameBuffers gave '%s'\n",CiErrStr(circ));
                }
                return false;
        }
        if (s->nPtrs != s->nFrames) {
                log_msg(LOG_LEVEL_ERROR, "Error requested - %" PRIu32 ", mapped :%p\n", s->nFrames, s->nPtrs);
        }

        return true;
}

static bool reset_acquisition(struct vidcap_state_bitflow *s) {
        tCIRC circ;
        if (kCIEnoErr != (circ = CiAqSWreset(s->sCIp))) {
                log_msg(LOG_LEVEL_ERROR, "CiAqSWreset gave '%s'\n", CiErrStr(circ));
                if ((NULL != s->uPtrs) && (kCIEnoErr != (circ = CiUnmapFrameBuffers(s->sCIp)))){
                        log_msg(LOG_LEVEL_ERROR, "CiUnmapFrameBuffers gave '%s'\n",CiErrStr(circ));
                }
                return false;
        }

        return true;
}

static bool start_acquisition(struct vidcap_state_bitflow *s) {
        tCIRC circ;
        int numberOfFrames = 0;
        if (kCIEnoErr != (circ = CiAqStart(s->sCIp, numberOfFrames))) // 0-continous, n-frame number
        {
                log_msg(LOG_LEVEL_ERROR, "CiAqStart gave '%s'\n",CiErrStr(circ));
                if ((NULL != s->uPtrs) && (kCIEnoErr != (circ = CiUnmapFrameBuffers(s->sCIp)))){
                        log_msg(LOG_LEVEL_ERROR, "CiUnmapFrameBuffers gave '%s'\n",CiErrStr(circ));
                }
                return false;
        }
        return true;
}

static float get_fps(struct vidcap_state_bitflow *s) {
        tCIRC circ;
        tCITDP cfgFN, cfg, camFN, cam, firmFN, firm, serial, cxp, spare;

        if (kCIEnoErr != (circ = CiVFGinquire2(s->sCIp, &cfgFN, &cfg, &camFN,
                                        &cam, &firmFN, &firm, &serial, &cxp, &spare))) {
                log_msg(LOG_LEVEL_ERROR, "CiVFGinquire gave '%s'\n", CiErrStr(circ));
                return 0.0f;
        }

        int x = 0;
        while (cam[x].descriptor != NULL) {
                if (strcmp(cam[x].descriptor, "CAMREGAFTER") == 0) {
                        tCIU32  *ptr = (tCIU32 *)(cam[x].datap);
                        int y = *ptr++;
                        while (y-- > 0) {
                                if (ptr[0] == 0x000200C4) {
                                        return *(float *) (ptr + 1);
                                }
                                ptr += 2;
                        }
                }
                x += 1;
        }

        return 0;
}

static int vidcap_bitflow_init(const struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }
        struct vidcap_state_bitflow *s = (struct vidcap_state_bitflow *) calloc(1, sizeof(struct vidcap_state_bitflow));

        s->sNdx = 0;

        tCIRC circ;
        //Open the ndx'th frame grabber with write permission.
        if (kCIEnoErr != (circ = CiVFGopen(s->sNdx,                //!< from CiSysBrdInfo()
                                        kCIBO_writeAccess,   //!< access mode flags
                                        &s->sCIp)))             //!< access token
        {
                log_msg(LOG_LEVEL_ERROR, "CiVFGopen gave '%s'\n",CiErrStr(circ));
                goto error;
        }

        if (kCIEnoErr != (circ = CiVFGinitialize(s->sCIp,NULL)))
        {
                log_msg(LOG_LEVEL_ERROR, "CiVFGinitialize gave '%s'\n",CiErrStr(circ));
                goto error;
        }

        if (kCIEnoErr != (circ = CiDrvrBuffConfigure(s->sCIp, 20, 0, 0, 0, 0))) {
                log_msg(LOG_LEVEL_ERROR, "CiDrvrBuffConfigure gave '%s'\n",CiErrStr(circ));
                goto error;
        }

        tCIU32 bitsPerPix, hROIsize, vROIsize, hROIoffset, vROIoffset, stride;
        if (kCIEnoErr != (circ = CiBufferInterrogate(s->sCIp, &s->nFrames, &bitsPerPix, &hROIoffset, &hROIsize, &vROIoffset, &vROIsize, &stride)))
        {
                log_msg(LOG_LEVEL_ERROR, "CiBufferInterrogate gave '%s'\n",CiErrStr(circ));
                goto error;
        }

        s->desc = {hROIsize, vROIsize, RGB, get_fps(s), PROGRESSIVE, 1};

        if (s->desc.fps == 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to get FPS!");
                goto error;
        }

        if (!map_frame_buffers(s)) {
                goto error;
        }

        if (!reset_acquisition(s)) {
                goto error;
        }

        if (!start_acquisition(s)) {
                goto error;
        }

        *state = s;
        return VIDCAP_INIT_OK;

error:
        vidcap_bitflow_done(s);
        return -1;
}

static void vidcap_bitflow_done(void *state)
{
        struct vidcap_state_bitflow *s = (struct vidcap_state_bitflow *) state;
        tCIRC circ;

        /*
         **  Unmap the frame buffers.
         */
        if ((NULL != s->uPtrs) && (kCIEnoErr != (circ = CiUnmapFrameBuffers(s->sCIp)))){
                log_msg(LOG_LEVEL_ERROR, "CiUnmapFrameBuffers gave '%s'\n",CiErrStr(circ));
        }
        /*
         **  Close the access.
         */
        if ((NULL != s->sCIp) && (kCIEnoErr != (circ = CiVFGclose(s->sCIp)))) {
                log_msg(LOG_LEVEL_ERROR, "CiVFGclose gave '%s'\n",CiErrStr(circ));
        }

        free(s);
}

static void debayer(char *out, char *in, int width, int height)
{
        for (int y = 0; y < height; y += 2) {
                for (int x = 0; x < width; ++x) {
                        *out++ = in[x + 1];
                        *out++ = in[x];
                        *out++ = in[x + width];
                }
                for (int x = 0; x < width; ++x) {
                        *out++ = in[x + 1];
                        *out++ = in[x + width + 1];
                        *out++ = in[x + width];
                }

                in += 2 * width;
        }
}

static void restartAcquisition(struct vidcap_state_bitflow *s)
{
        if (!map_frame_buffers(s)) {
                goto fail;
        }

        if (!reset_acquisition(s)) {
                goto fail;
        }

        if (!start_acquisition(s)) {
                goto fail;
        }

        return;
fail:
        log_msg(LOG_LEVEL_FATAL, "[bitflow] Unable to restart acquisition!\n");
        exit_uv(1);
}

static struct video_frame *vidcap_bitflow_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_state_bitflow *s = (struct vidcap_state_bitflow *) state;

        tCIU32                      frameID;
        tCIU8                       *frameP;
        tCIRC                       circ;

        //Check to see if a frame is already available before waiting.
        circ = CiGetOldestNotDeliveredFrame(s->sCIp, &frameID, &frameP);

        switch (circ) {
                case kCIEnoErr:
                        // we have the frame
                        break;
                case kCIEnoNewData:
                        if (kCIEnoErr != (circ = CiWaitNextUndeliveredFrame(s->sCIp,-1))) // negative - infinite wait
                        {
                                switch (circ) {
                                        case kCIEaqAbortedErr:
                                                log_msg(LOG_LEVEL_ERROR, "CiWaitNextUndeliveredFrame gave '%s'\n",CiErrStr(circ));
                                                break;
                                        case kCIEdataHWerr:
                                                log_msg(LOG_LEVEL_ERROR, "CiWaitNextUndeliveredFrame gave '%s'\n",CiErrStr(circ));
                                                restartAcquisition(s);

                                                break;
                                        default:
                                                log_msg(LOG_LEVEL_ERROR, "CiWaitNextUndeliveredFrame gave '%s'\n",CiErrStr(circ));
                                }
                                return NULL;
                        }
                        circ = CiGetOldestNotDeliveredFrame(s->sCIp, &frameID, &frameP);
                        if (circ == kCIEnoErr) {
                                break;
                        } else {
                                return NULL;
                        }
                case kCIEaqAbortedErr:
                        log_msg(LOG_LEVEL_ERROR, "CiGetOldestNotDeliveredFrame: acqistion aborted\n");
                        return NULL;
                case kCIEdataHWerr:
                        log_msg(LOG_LEVEL_ERROR, "CiWaitNextUndeliveredFrame gave '%s'\n",CiErrStr(circ));
                        restartAcquisition(s);
                        return NULL;
                default:
                        log_msg(LOG_LEVEL_ERROR, "CiGetOldestNotDeliveredFrame gave '%s'\n",CiErrStr(circ));
                        return NULL;
        }

        tCIU32 count;
        switch (circ = CiGetUndeliveredCount(s->sCIp,&count)) {
                case kCIEnoErr:
                        /*
                         **    We have the frame.
                         */
                        break;
                case kCIEnoNewData:
                        /*
                         **    We need to wait for another frame.
                         */
                        log_msg(LOG_LEVEL_ERROR, "CiGetUndeliveredCount error\n");
                        return NULL;
                case kCIEaqAbortedErr:
                        log_msg(LOG_LEVEL_ERROR, "CiGetOldestNotDeliveredFrame: acqistion aborted\n");
                        return NULL;
                default:
                        log_msg(LOG_LEVEL_ERROR, "CiGetOldestNotDeliveredFrame gave '%s'\n",CiErrStr(circ));
                        return NULL;
        }

        if (count > 0) {
                log_msg(LOG_LEVEL_WARNING, "Undelivered count: %" PRIu32 "\n");
        }

        video_frame *out = vf_alloc_desc_data(s->desc);
        debayer((char *) out->tiles[0].data, (char *) frameP, s->desc.width, s->desc.height);
        out->dispose = vf_free;

        *audio = NULL;
        return out;
}

static struct vidcap_type *vidcap_bitflow_probe(bool verbose)
{
        UNUSED(verbose);
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "bitflow";
                vt->description = "No video capture device";
        }
        return vt;
}

static const struct video_capture_info vidcap_bitflow_info = {
        vidcap_bitflow_probe,
        vidcap_bitflow_init,
        vidcap_bitflow_done,
        vidcap_bitflow_grab,
};

REGISTER_MODULE(bitflow, &vidcap_bitflow_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

