/**
 * @file   video_display/gl_vdpau.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2026 CESNET, zájmové sdružení právnických osob
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
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include <cassert>
#include "debug.h"

#include "gl_vdpau.hpp"

/**
 * @brief Checks VdpMixer parameters and reinitializes it if they don't match the video parameters
 */
static void check_mixer(struct state_vdpau *vdp, hw_vdpau_frame *frame){
        uint32_t frame_w;
        uint32_t frame_h;
        VdpChromaType ct;

        VdpStatus st = vdp->funcs.videoSurfaceGetParameters(frame->surface,
                        &ct,
                        &frame_w,
                        &frame_h);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to get surface parameters: %s\n", vdp->funcs.getErrorString(st));
        }

        if(vdp->surf_width != frame_w ||
                        vdp->surf_height != frame_h ||
                        vdp->surf_ct != ct ||
                        !vdp->mixerInitialized)
        {
                vdp->initMixer(frame_w, frame_h, ct);
        }
}

void state_vdpau::loadFrame(hw_vdpau_frame *frame){
        assert(initialized);

        glBindTexture(GL_TEXTURE_2D, 0);

        int state = 0;
        int len = 0;
        if(vdpgl_surf){
                glVDPAUGetSurfaceivNV(vdpgl_surf,
                                GL_SURFACE_STATE_NV,
                                1,
                                &len,
                                &state
                                );
        }

        if(state == GL_SURFACE_MAPPED_NV)
                glVDPAUUnmapSurfacesNV(1, &vdpgl_surf);

        checkInterop(frame->hwctx.device, frame->hwctx.get_proc_address);
        check_mixer(this, frame);

        mixerRender(frame->surface);

        glVDPAUMapSurfacesNV(1, &vdpgl_surf);

        glBindTexture(GL_TEXTURE_2D, textures[0]);

        hw_vdpau_frame_unref(&lastFrame);
        lastFrame = hw_vdpau_frame_copy(frame);
}

/**
 * @brief Uninitializes VdpMixer
 */
void state_vdpau::uninitMixer(){
        VdpStatus st;

        if (mixer != VDP_INVALID_HANDLE){
                st = funcs.videoMixerDestroy(mixer);
                mixer = VDP_INVALID_HANDLE;
                if(st != VDP_STATUS_OK){
                        log_msg(LOG_LEVEL_ERROR, "Failed to destroy VdpVideoMixer: %s\n", funcs.getErrorString(st));
                }
        } 

        if (out_surf != VDP_INVALID_HANDLE){
                st = funcs.outputSurfaceDestroy(out_surf);
                if(st != VDP_STATUS_OK){
                        log_msg(LOG_LEVEL_ERROR, "Failed to destroy VdpOutputSurface: %s\n", funcs.getErrorString(st));
                }
                out_surf = VDP_INVALID_HANDLE;
                surf_width = 0;
                surf_height = 0;
                surf_ct = 0;
        }

        glVDPAUUnregisterSurfaceNV(vdpgl_surf);
        mixerInitialized = false;
}

/**
 * @brief Initializes VdpMixer
 */
void state_vdpau::initMixer(uint32_t w, uint32_t h, VdpChromaType ct){
        uninitMixer();

        surf_ct = ct;
        surf_width = w;
        surf_height = h;

        VdpStatus st;

        VdpRGBAFormat rgbaFormat = VDP_RGBA_FORMAT_B8G8R8A8;

        st = funcs.outputSurfaceCreate(device,
                        rgbaFormat,
                        surf_width,
                        surf_height,
                        &out_surf);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to create VdpOutputSurface: %s\n", funcs.getErrorString(st));
        }

        VdpVideoMixerParameter params[] = {
                VDP_VIDEO_MIXER_PARAMETER_CHROMA_TYPE,
                VDP_VIDEO_MIXER_PARAMETER_VIDEO_SURFACE_WIDTH,
                VDP_VIDEO_MIXER_PARAMETER_VIDEO_SURFACE_HEIGHT
        };

        void *param_vals[] = {
                &surf_ct,
                &surf_width,
                &surf_height
        };

        st = funcs.videoMixerCreate(device,
                        0,
                        nullptr,
                        3,
                        params,
                        param_vals,
                        &mixer);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to create VdpVideoMixer: %s\n", funcs.getErrorString(st));
        }

        vdpgl_surf = glVDPAURegisterOutputSurfaceNV(NV_CAST(out_surf),
                        GL_TEXTURE_2D,
                        1,
                        textures);

        glVDPAUSurfaceAccessNV(vdpgl_surf, GL_WRITE_DISCARD_NV);

        mixerInitialized = true;
}

/**
 * @brief Renders VdpVideoSurface into a VdpOutputSurface
 */
void state_vdpau::mixerRender(VdpVideoSurface f){
        VdpStatus st = funcs.videoMixerRender(mixer,
                        VDP_INVALID_HANDLE,
                        nullptr,
                        VDP_VIDEO_MIXER_PICTURE_STRUCTURE_FRAME,
                        0,
                        nullptr,
                        f,
                        0,
                        nullptr,
                        nullptr,
                        out_surf,
                        nullptr,
                        nullptr,
                        0,
                        nullptr);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to render: %s\n", funcs.getErrorString(st));
        }
}

/**
 * @brief Checks if the vdpau-GL interoperability is initialized with the same hw contexts
 * and reinitializes it if needed
 */
void state_vdpau::checkInterop(VdpDevice device, VdpGetProcAddress *get_proc_address){
        if(this->device != device || this->get_proc_address != get_proc_address){
                uninitInterop();
                initInterop(device, get_proc_address);
        }
}

/**
 * @brief Initializes vdpau-GL interoperability
 */
void state_vdpau::initInterop(VdpDevice device, VdpGetProcAddress *get_proc_address){
        if(interopInitialized)
                uninitInterop();

        glVDPAUInitNV(NV_CAST(device), (void *) get_proc_address);
        this->device = device;
        this->get_proc_address = get_proc_address;

        vdp_funcs_load(&funcs, device, get_proc_address);
        interopInitialized = true;
}

/**
 * @brief Uninitializes vdpau-GL interoperability
 */
void state_vdpau::uninitInterop(){
        if(!interopInitialized)
                return;

        glVDPAUFiniNV();
        device = 0;
        get_proc_address = nullptr;
        interopInitialized = false;
        mixerInitialized = false;

        //VDPAUFiniNV() unmaps and unregisters all surfaces automatically
        vdpgl_surf = 0;
}

/**
 * @brief Initializes state_vdpau
 */
bool state_vdpau::init(){
        initialized = true;
        glGenTextures(4, textures);
        hw_vdpau_frame_init(&lastFrame);

        return true;
}

/**
 * @brief Uninitializes state_vdpau
 */
void state_vdpau::uninit(){
        uninitMixer();
        uninitInterop();
        hw_vdpau_frame_unref(&lastFrame);
        glDeleteTextures(4, textures);
        for(int i = 0; i < 4; i++){
                textures[i] = 0;
        }
}

struct state_vdpau *
vdp_init()
{
        struct state_vdpau *vdp = new state_vdpau();
        vdp->init();
        return vdp;
}

void
vdp_load_frame(struct state_vdpau *vdp, hw_vdpau_frame *frame)
{
        vdp->loadFrame(frame);
}

void
vdp_destroy(struct state_vdpau *vdp)
{
        delete vdp;
}
