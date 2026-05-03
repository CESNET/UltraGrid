#include <oapv/oapv.h>

#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "from_planar.h"
#include "openapv/openapv_conversions.h"

namespace {

#define MOD_NAME "[OpenAPV dec.] "

struct from_openapv_conversion {
        codec_t         dst_codec;
        int             required_src_cs;
        decode_planar_func_t *convert;
        int             log2_chroma_h;
};

static const from_openapv_conversion from_openapv_conversions[] = {
        { UYVY, OAPV_CS_YCBCR422_10LE, yuv422pXX_to_uyvy,   0 },
        { YUYV, OAPV_CS_YCBCR422_10LE, yuv422p_to_yuyv,     0 },
        { v210, OAPV_CS_YCBCR422_10LE, yuv422p10le_to_v210, 0 },
        { VIDEO_CODEC_NONE, OAPV_CS_UNKNOWN, nullptr, 0 },
};

struct state_video_decompress_openapv {
        
        oapvd_t decoder_handle{};
        oapvm_t metadata_handle{};
        oapvd_cdesc_t cdesc{};

        oapv_frms_t decoded_frames{};
        oapv_bitb_t input_buffer{};

        struct video_desc desc{};
        int pitch = 0;
        codec_t out_codec = VIDEO_CODEC_NONE;
};

static const from_openapv_conversion *get_from_openapv_conversion(codec_t dst_codec) {
        for (const auto *c = from_openapv_conversions; c->dst_codec != VIDEO_CODEC_NONE; ++c) {
                if (c->dst_codec == dst_codec) {
                        return c;
                }
        }
        return nullptr;
}

static bool configure_with(struct state_video_decompress_openapv *s, unsigned char *bitstream_buffer, size_t codestream_size) {
        int ret = -1;
        oapv_au_info_t aui{};

        ret = oapvd_info(bitstream_buffer, codestream_size, &aui);
        if (OAPV_FAILED(ret) || aui.num_frms < 1) {
                return false;
        }

        if (s->decoder_handle == nullptr) {
                s->cdesc.threads = OAPV_CDESC_THREADS_AUTO;
                s->decoder_handle = oapvd_create(&s->cdesc, &ret);
                if (OAPV_FAILED(ret)) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "oapvd_create failed (ret=%d).\n", ret);
                        return false;
                }
        }

        if (s->metadata_handle == nullptr) {
                s->metadata_handle = oapvm_create(&ret);
                if (OAPV_FAILED(ret)) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "oapvm_create failed (ret=%d).\n", ret);
                        oapvd_delete(s->decoder_handle);
                        s->decoder_handle = nullptr;
                        return false;
                }
        }

        if (s->decoded_frames.frm[0].imgb != nullptr) {
                for (int i = 0; i < OAPV_MAX_CC; ++i) {
                        free(s->decoded_frames.frm[0].imgb->baddr[i]);
                }
                delete s->decoded_frames.frm[0].imgb;
                s->decoded_frames.frm[0].imgb = nullptr;
        }

        s->decoded_frames.num_frms = aui.num_frms;
        for (int fi = 0; fi < aui.num_frms; ++fi) {
                const oapv_frm_info_t &frm_info = aui.frm_info[fi];
                int fmt = OAPV_CS_GET_FORMAT(frm_info.cs);
                if (fmt != OAPV_CF_YCBCR422 && fmt != OAPV_CF_YCBCR444 && fmt != OAPV_CF_YCBCR4444) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported APV color format %d for frame %d.\n", fmt, fi);
                        return false;
                }

                s->decoded_frames.frm[fi].imgb = new oapv_imgb_t();
                memset(s->decoded_frames.frm[fi].imgb, 0, sizeof(oapv_imgb_t));

                oapv_imgb_t *imgb = s->decoded_frames.frm[fi].imgb;
                imgb->cs = frm_info.cs;
                imgb->w[0] = frm_info.w;
                imgb->h[0] = frm_info.h;
                int cf = OAPV_CS_GET_FORMAT(frm_info.cs);
                switch(cf) {
                        case OAPV_CF_YCBCR422:
                                imgb->w[1] = imgb->w[2] = (frm_info.w + 1) / 2;
                                imgb->h[1] = imgb->h[2] = frm_info.h;
                                imgb->np = 3;
                                break;
                        case OAPV_CF_YCBCR444:
                                imgb->w[1] = imgb->w[2] = frm_info.w;
                                imgb->h[1] = imgb->h[2] = frm_info.h;
                                imgb->np = 3;
                                break;
                        case OAPV_CF_YCBCR4444:
                                imgb->w[1] = imgb->w[2] = imgb->w[3] = frm_info.w;
                                imgb->h[1] = imgb->h[2] = imgb->h[3] = frm_info.h;
                                imgb->np = 4;
                                break;
                        default:
                                printf("Unsupported color format for input buffer: %d\n", cf);
                                return false;
                }

                for (int i = 0; i < imgb->np; ++i) {
                        imgb->aw[i] = ((imgb->w[i] + OAPV_MB_W - 1) / OAPV_MB_W) * OAPV_MB_W;
                        imgb->ah[i] = ((imgb->h[i] + OAPV_MB_H - 1) / OAPV_MB_H) * OAPV_MB_H;
                        imgb->s[i] = imgb->aw[i] * OAPV_CS_GET_BYTE_DEPTH(imgb->cs);
                        imgb->e[i] = imgb->ah[i];
                        imgb->bsize[i] = imgb->s[i] * imgb->e[i];
                        imgb->baddr[i] = malloc(imgb->bsize[i]);
                        if (imgb->baddr[i] == nullptr) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate plane %d for frame %d (%zu bytes).\n",
                                        i, fi, (size_t) imgb->bsize[i]);
                                for (int j = 0; j < i; ++j) {
                                        free(imgb->baddr[j]);
                                }
                                delete imgb;
                                s->decoded_frames.frm[fi].imgb = nullptr;
                                return false;
                        }
                        memset(imgb->baddr[i], 0, imgb->bsize[i]);
                        imgb->a[i] = imgb->baddr[i];
                }

                s->decoded_frames.frm[fi].pbu_type = frm_info.pbu_type;
                s->decoded_frames.frm[fi].group_id = frm_info.group_id;
        }

        s->input_buffer.ssize = codestream_size;
        s->input_buffer.bsize = codestream_size;

        return true;
}

static void *openapv_decompress_init(void) {
        state_video_decompress_openapv *s = new state_video_decompress_openapv();
        return s;
}

static int openapv_decompress_reconfigure(void *state, struct video_desc desc,
        int rshift, int gshift, int bshift, int pitch, codec_t out_codec) {
        state_video_decompress_openapv *s = (state_video_decompress_openapv *) state;

        s->desc = desc;
        s->pitch = pitch;
        s->out_codec = out_codec;

        if (out_codec != VIDEO_CODEC_NONE && get_from_openapv_conversion(out_codec) == nullptr) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported output codec %s.\n", get_codec_name(out_codec));
                return false;
        }

        return true;
}

static decompress_status openapv_decompress(void *state, unsigned char *dst, unsigned char *buffer, 
        unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_prop) {
        state_video_decompress_openapv *s = (state_video_decompress_openapv *) state;

        if (s->out_codec == VIDEO_CODEC_NONE) {
                oapv_au_info_t aui{};
                if (OAPV_FAILED(oapvd_info(buffer, src_len, &aui)) || aui.num_frms < 1) {
                        return DECODER_NO_FRAME;
                }
                int frm_idx = 0;
                for (int i = 0; i < aui.num_frms; ++i) {
                        if (aui.frm_info[i].pbu_type == OAPV_PBU_TYPE_PRIMARY_FRAME) {
                                frm_idx = i;
                                break;
                        }
                }
                *internal_prop = {};
                internal_prop->depth = aui.frm_info[frm_idx].bit_depth;
                internal_prop->subsampling = SUBS_422;
                internal_prop->rgb = false;
                return DECODER_GOT_CODEC;
        }

        const from_openapv_conversion *conv = get_from_openapv_conversion(s->out_codec);
        if (conv == nullptr) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unsupported out_codec=%s.\n", get_codec_name(s->out_codec));
                return DECODER_UNSUPP_PIXFMT;
        }

        if (!configure_with(s, buffer, src_len)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "configure_with failed for src_len=%u.\n", src_len);
                return DECODER_NO_FRAME;
        }
        s->input_buffer.addr = buffer;
        oapvd_stat_t stat{};
        oapvm_rem_all(s->metadata_handle);
        int decode_ret = oapvd_decode(s->decoder_handle, &s->input_buffer, &s->decoded_frames, s->metadata_handle, &stat);
        if (OAPV_FAILED(decode_ret)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "oapvd_decode failed ret=%d src_len=%u.\n", decode_ret, src_len);
                return DECODER_NO_FRAME;
        }

        int frm_idx = -1;
        for (int i = 0; i < s->decoded_frames.num_frms; ++i) {
                if (s->decoded_frames.frm[i].pbu_type == OAPV_PBU_TYPE_PRIMARY_FRAME) {
                        frm_idx = i;
                        break;
                }
        }
        if (frm_idx < 0) {
                return DECODER_NO_FRAME;
        }

        oapv_imgb_t *imgb = s->decoded_frames.frm[frm_idx].imgb; 
        if (imgb == nullptr) {
                return DECODER_NO_FRAME;
        }
        
        if (imgb->cs != conv->required_src_cs) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Decoded colorspace 0x%x does not match required 0x%x for output codec %s.\n",
                        imgb->cs, conv->required_src_cs, get_codec_name(s->out_codec));
                return DECODER_NO_FRAME;
        }

        struct from_planar_data d = {};
        d.width = imgb->w[0];
        d.height = imgb->h[0];
        d.out_data = dst;
        d.out_pitch = s->pitch;
        d.in_data[0] = (const unsigned char *) imgb->a[0];
        d.in_data[1] = (const unsigned char *) imgb->a[1];
        d.in_data[2] = (const unsigned char *) imgb->a[2];
        d.in_linesize[0] = imgb->s[0];
        d.in_linesize[1] = imgb->s[1];
        d.in_linesize[2] = imgb->s[2];
        d.in_depth = OAPV_CS_GET_BIT_DEPTH(imgb->cs);
        d.log2_chroma_h = conv->log2_chroma_h;
        decode_planar_parallel(conv->convert, d, FROM_PLANAR_THREADS_AUTO);

        return DECODER_GOT_FRAME;
}

static int openapv_decompress_get_property(void *state, int property, void *val, size_t *len) {
        return true;
}

static void openapv_decompress_done(void *state) {
       state_video_decompress_openapv *s = (state_video_decompress_openapv *) state;
       delete s;
}

static int openapv_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        if (compression != APV) {
                return VDEC_PRIO_NA;
        }
        if (ugc == VIDEO_CODEC_NONE) {
                return VDEC_PRIO_PROBE_HI;
        }
        return get_from_openapv_conversion(ugc) != nullptr ? VDEC_PRIO_PREFERRED : VDEC_PRIO_NA;
}

static const struct video_decompress_info openapv_info = {
        openapv_decompress_init,
        openapv_decompress_reconfigure,
        openapv_decompress,
        openapv_decompress_get_property,
        openapv_decompress_done,
        openapv_decompress_get_priority,
};

REGISTER_MODULE(openapv, &openapv_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

}
