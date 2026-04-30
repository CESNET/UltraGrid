/**
 * @file   video_display/gl_vdpau.hpp
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

#ifndef GL_VDPAU_HPP_667244de5757
#define GL_VDPAU_HPP_667244de5757

#include "config.h"  // for HWACC_VDPAU

#ifdef HWACC_VDPAU

#include "hwaccel_vdpau.h"

#ifdef __cplusplus
extern "C" {
#endif // defined __cplusplus

struct state_vdpau *vdp_init();
void vdp_load_frame(struct state_vdpau *vdp, hw_vdpau_frame *frame);
void vdp_destroy(struct state_vdpau *vdp);

#ifdef __cplusplus
}
#endif // defined __cplusplus

#else

#define vdp_init() 0
#define vdp_destroy(...)
#define vdp_load_frame(...)

#endif //HWACC_VDPAU
#endif //GL_VDPAU_HPP_667244de5757
