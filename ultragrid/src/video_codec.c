/*
 * FILE:   video_codec.c
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2004 University of Glasgow
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
 *    must display the following acknowledgement: "This product includes 
 *    software developed by the University of Glasgow".
 * 
 * 4. The name of the University may not be used to endorse or promote 
 *    products derived from this software without specific prior written 
 *    permission.
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
 *
 * $Revision: 1.2 $
 * $Date: 2007/12/11 19:16:45 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/rtp.h"		/* Needed by rtp/pbuf.h  */
#include "rtp/pbuf.h"		/* For struct coded_data */
#include "video_types.h"
#include "video_codec.h"
#include "video_codec/dv.h"
#include "video_codec/uv_yuv.h"

struct vcodec_api {
	int                  (*vcx_init)           (void);
	void                 (*vcx_exit)           (void);
	const char *         (*vcx_get_name)       (void);
	const char *         (*vcx_get_desc)       (void);
	int                  (*vcx_can_encode)     (void);
	int                  (*vcx_can_decode)     (void);
	struct vcodec_state *(*vcx_encoder_create) (void);
	void                 (*vcx_encoder_destroy)(struct vcodec_state *state);
	int                  (*vcx_encode)         (struct vcodec_state *state, 
                                                    struct video_frame *in, 
					            struct coded_data *out);
	struct vcodec_state *(*vcx_decoder_create) (void);
	void                 (*vcx_decoder_destroy)(struct vcodec_state *state);
	int                  (*vcx_decode)         (struct vcodec_state *state, 
                                                    struct coded_data *in, 
						    struct video_frame *out);
};

/* Table of pointers to codec functions, one for each codec we */
/* support. These are not necessarily all available for use as */
/* some codecs may fail their initialization (for example, if  */
/* they depend on hardware devices that are not present). The  */
/* available_vcodecs[] array contains pointers to those codecs */
/* that are usable.                                            */
static struct vcodec_api vcodec_table[] = {
#ifdef HAVE_DV_CODEC
	{
		dv_init,
		dv_exit,
		dv_get_name,
		dv_get_desc,
		dv_can_encode,
		dv_can_decode,
		dv_encoder_create,
		dv_encoder_destroy,
		dv_encode,
		dv_decoder_create,
		dv_decoder_destroy,
		dv_decode
	},
#endif /* HAVE_DV_CODEC */
	{
		uv_yuv_init,
		uv_yuv_exit,
		uv_yuv_get_name,
		uv_yuv_get_desc,
		uv_yuv_can_encode,
		uv_yuv_can_decode,
		uv_yuv_encoder_create,
		uv_yuv_encoder_destroy,
		uv_yuv_encode,
		uv_yuv_decoder_create,
		uv_yuv_decoder_destroy,
		uv_yuv_decode
	}
};

#define NUM_VCODEC_INTERFACES (sizeof(vcodec_table)/sizeof(struct vcodec_api))

/* Pointers to entries in the vcodec_table[], for codecs that  */
/* are usable...                                               */
static struct vcodec_api *available_vcodecs[NUM_VCODEC_INTERFACES];
static unsigned	          num_vcodecs;

/*
 * Public interface functions follow...
 */

void
vcodec_init(void)
{
	unsigned i;

	num_vcodecs = 0;
	for (i = 0; i < NUM_VCODEC_INTERFACES; i++) {
		if (vcodec_table[i].vcx_init()) {
			available_vcodecs[num_vcodecs++] = &vcodec_table[i];
		}
	}
}

void
vcodec_done(void)
{
	unsigned i;

	for (i = 0; i < num_vcodecs; i++) {
		available_vcodecs[i]->vcx_exit();
	}
	num_vcodecs = 0;
}

unsigned
vcodec_get_num_codecs(void)
{
	return num_vcodecs;
}

const char *
vcodec_get_name(unsigned id)
{
	assert(id < num_vcodecs);
	return available_vcodecs[id]->vcx_get_name();
}

const char *
vcodec_get_description(unsigned id)
{
	assert(id < num_vcodecs);
	return available_vcodecs[id]->vcx_get_desc();
}

int
vcodec_can_encode(unsigned id)
{
	assert(id < num_vcodecs);
	return available_vcodecs[id]->vcx_can_encode();
}

int
vcodec_can_decode(unsigned id)
{
	assert(id < num_vcodecs);
	return available_vcodecs[id]->vcx_can_decode();
}

int
vcodec_map_payload(uint8_t pt, unsigned id)
{
	UNUSED(id);
	UNUSED(pt);
	return 0;
}

int
vcodec_unmap_payload(uint8_t pt)
{
	UNUSED(pt);
	return 0;
}

unsigned
vcodec_get_by_payload (uint8_t pt)
{
	UNUSED(pt);
	return 0; /* FIXME */
}

uint8_t
vcodec_get_payload(unsigned id)
{
	UNUSED(id);
	return 0;
}

/*****************************************************************************/
struct vcodec_state {
	uint32_t	magic;
};

struct vcodec_state *
vcodec_encoder_create (unsigned id)
{
	UNUSED(id);
	return NULL;
}

void
vcodec_encoder_destroy(struct vcodec_state *state)
{
	UNUSED(state);
}

int
vcodec_encode(struct vcodec_state *state, struct video_frame  *in, struct coded_data *out)
{
	UNUSED(state);
	UNUSED(in);
	UNUSED(out);
	return 0;
}

struct vcodec_state *
vcodec_decoder_create(unsigned id)
{
	UNUSED(id);
	return NULL;
}

void
vcodec_decoder_destroy(struct vcodec_state *state)
{
	UNUSED(state);
}

int
vcodec_decode(struct vcodec_state *state, struct coded_data   *in, struct video_frame  *out)
{
	UNUSED(state);
	UNUSED(in);
	UNUSED(out);
	return 0;
}

