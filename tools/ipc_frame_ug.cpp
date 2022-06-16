#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#endif

#include <array>

#include "ipc_frame_ug.h"
#include "ipc_frame.h"
#include "types.h"
#include "video_codec.h"

namespace {
void scale_frame(char *dst, char *src,
	   	int src_w, int src_h,
		int f, codec_t codec)
{
	int src_line_len = vc_get_linesize(src_w, codec);
	int block_size = get_pf_block_bytes(codec);
	assert(block_size > 0);
	int written = 0;
	for(int y = 0; y + f <= src_h; y += f){
		for(int x = 0; x + f * block_size <= src_line_len; x += f * block_size){
			memcpy(dst + written, src + y*src_line_len + x, block_size);
			written += block_size;
		}
	}
}

}//anon namespace

bool ipc_frame_from_ug_frame(struct Ipc_frame *dst,
		const struct video_frame *src,
		codec_t codec,
		unsigned scale_factor)
{
	if(!src)
		return false;

	decoder_t dec = nullptr;
	if(codec != VIDEO_CODEC_NONE){
        dec = get_decoder_from_to(src->color_spec, codec);
		if(!dec){
			return false;
		}
	} else {
		codec = src->color_spec;
		dec = vc_memcpy;
	}

	dst->header.width = src->tiles[0].width;
	dst->header.height = src->tiles[0].height;
	dst->header.color_spec = static_cast<Ipc_frame_color_spec>(codec);

	int dst_frame_to_allocate = 0;

	if(scale_factor != 0){
		int block_size_px = get_pf_block_pixels(src->color_spec);
		int block_count = (dst->header.width + block_size_px - 1) / block_size_px;
		dst->header.width = (block_count / scale_factor) * block_size_px;
		dst->header.height /= scale_factor;

		if(dec != vc_memcpy){
			//When both scaling and converting we need a tmp space - allocate extra
			dst_frame_to_allocate += get_bpp(src->color_spec) * dst->header.width * dst->header.height;
		}
	}

	int dst_frame_size = get_bpp(codec) * dst->header.width * dst->header.height;
	dst_frame_to_allocate += dst_frame_size;

	if(!ipc_frame_reserve(dst, dst_frame_to_allocate))
		return false;

	dst->header.data_len = dst_frame_size;

	char *scale_dst = dst->data;
	unsigned char *dec_dst = (unsigned char *) dst->data;
	unsigned char *dec_src = (unsigned char *) src->tiles[0].data;

	if(scale_factor != 0){
		if(dec != vc_memcpy){
			scale_dst = dst->data + dst_frame_size;
		}
		dec_src = (unsigned char *) scale_dst;

		scale_frame(scale_dst, src->tiles[0].data,
				src->tiles[0].width, src->tiles[0].height,
				scale_factor, src->color_spec);

	}


	if(dec_src == dec_dst){
		assert(dec == vc_memcpy);
		return true;
	}

	int dst_line_len = vc_get_linesize(dst->header.width, codec);
	int src_line_len = vc_get_linesize(dst->header.width, src->color_spec);
	for(int i = 0; i < dst->header.height; i++){
		dec(dec_dst + dst_line_len * i,
				dec_src + src_line_len * i,
				dst_line_len,
				0, 8, 16);
	}

	return true;
}
