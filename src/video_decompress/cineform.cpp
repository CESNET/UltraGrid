#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/resource_manager.h"
#include "video.h"
#include "video_decompress.h"

#include "CFHDTypes.h"
#include "CFHDDecoder.h"

struct state_cineform_decompress {
        int              width, height;
        int              pitch;
        int              rshift, gshift, bshift;
        int              max_compressed_len;
        codec_t          in_codec;
        codec_t          out_codec;

        unsigned         last_frame_seq:22; // This gives last sucessfully decoded frame seq number. It is the buffer number from the packet format header, uses 22 bits.
        bool             last_frame_seq_initialized;
        bool             prepared_to_decode;

        CFHD_DecoderRef decoderRef;

        struct video_desc saved_desc;
};

static void * cineform_decompress_init(void)
{
        struct state_cineform_decompress *s;

        s = new state_cineform_decompress();

        s->width = s->height = s->pitch = 0;
        s->prepared_to_decode = false;

        CFHD_Error status;
        status = CFHD_OpenDecoder(&s->decoderRef, nullptr);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to open decoder\n");
        }

        return s;
}

static void cineform_decompress_done(void *state)
{
        struct state_cineform_decompress *s =
                (struct state_cineform_decompress *) state;

        CFHD_CloseDecoder(s->decoderRef);
        delete s;
}

static bool configure_with(struct state_cineform_decompress *s,
                struct video_desc desc)
{
        s->last_frame_seq_initialized = false;
        s->prepared_to_decode = false;
        s->saved_desc = desc;

        return true;
}

static int cineform_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_cineform_decompress *s =
                (struct state_cineform_decompress *) state;

        s->pitch = pitch;
        assert(out_codec == UYVY ||
                        out_codec == RGB ||
                        out_codec == v210);

        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->in_codec = desc.color_spec;
        s->out_codec = out_codec;
        s->width = desc.width;
        s->height = desc.height;

        return configure_with(s, desc);
}

static bool prepare(struct state_cineform_decompress *s,
                unsigned char *src,
                unsigned int src_len)
{
        if(s->prepared_to_decode){
                return true;
        }

        CFHD_Error status;

        int actualWidth;
        int actualHeight;
        CFHD_PixelFormat actualFormat;
        status = CFHD_PrepareToDecode(s->decoderRef,
                        s->saved_desc.width,
                        s->saved_desc.height,
                        CFHD_PIXEL_FORMAT_2VUY,
                        CFHD_DECODED_RESOLUTION_FULL,
                        CFHD_DECODING_FLAGS_NONE,
                        src,
                        src_len,
                        &actualWidth,
                        &actualHeight,
                        &actualFormat
                        );
        assert(actualWidth == s->width);
        assert(actualHeight == s->height);
        int actualPitch;
        CFHD_GetImagePitch(actualWidth, actualFormat, &actualPitch);
        assert(actualPitch == s->pitch);
        assert(actualFormat == CFHD_PIXEL_FORMAT_2VUY);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to prepare for decoding\n");
                return false;
        } 

        s->prepared_to_decode = true;
        return true;
}

static decompress_status cineform_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks)
{
        struct state_cineform_decompress *s = (struct state_cineform_decompress *) state;
        decompress_status res = DECODER_NO_FRAME;

        CFHD_Error status;

        if(!prepare(s, src, src_len)){
                return res;
        }

        status = CFHD_DecodeSample(s->decoderRef,
                        src,
                        src_len,
                        dst,
                        s->pitch);

        if(status == CFHD_ERROR_OKAY){
                res = DECODER_GOT_FRAME;
        } else {
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to decode %i\n", status);
        }

        return res;
}

static int cineform_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_cineform_decompress *s =
                (struct state_cineform_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
#ifdef CINEFORM_ACCEPT_CORRUPTED
                                *(int *) val = TRUE;
#else
                                *(int *) val = FALSE;
#endif
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

static const struct decode_from_to *cineform_decompress_get_decoders() {
        static const struct decode_from_to dec_static[] = {
                { CFHD, UYVY, 500 },
        };

        return dec_static;
}

static const struct video_decompress_info cineform_info = {
        cineform_decompress_init,
        cineform_decompress_reconfigure,
        cineform_decompress,
        cineform_decompress_get_property,
        cineform_decompress_done,
        cineform_decompress_get_decoders,
};

REGISTER_MODULE(cineform, &cineform_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
