/*
 * FILE:    audio_fmt.c
 * PROGRAM: RAT
 * AUTHOR:  Orion Hodson
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 */
 
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "memory.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_codec/g711.h"

int
audio_format_get_common(audio_format* ifmt, 
                        audio_format *ofmt, 
                        audio_format *comfmt)
{
        int isamples, osamples;

        /* Pre-conditions for finding a common format */
        if (ifmt->sample_rate != ofmt->sample_rate) {
                return FALSE;
        }

        if ((ifmt->channels != 1 && ifmt->channels != 2) ||
            (ofmt->channels != 1 && ofmt->channels != 2)) {
                return FALSE;
        }
        
        if ((ifmt->encoding != DEV_PCMU && ifmt->encoding != DEV_S16) ||
            (ofmt->encoding != DEV_PCMU && ofmt->encoding != DEV_S16)) {
                return FALSE;
        }

        if (ifmt->encoding == DEV_PCMU && ofmt->encoding == DEV_PCMU) {
                comfmt->encoding = DEV_PCMU;
        } else {
                comfmt->encoding = DEV_S16;
        }

        comfmt->sample_rate = ifmt->sample_rate;

        switch(comfmt->encoding) {
        case DEV_PCMU: comfmt->bits_per_sample = 8;  break;
        case DEV_S16:  comfmt->bits_per_sample = 16; break;
        case DEV_PCMA:
        case DEV_U8:
        case DEV_S8:   return FALSE; /* not supported currently */
        }
        comfmt->channels = max(ifmt->channels, ofmt->channels);

        isamples = ifmt->bytes_per_block * 8 / (ifmt->channels * ifmt->bits_per_sample);
        osamples = ofmt->bytes_per_block * 8 / (ofmt->channels * ofmt->bits_per_sample);
        
        comfmt->bytes_per_block = min(isamples, osamples) * comfmt->channels * comfmt->bits_per_sample / 8;
        
        return TRUE;
}

int
audio_format_match(audio_format *fmt1, audio_format *fmt2)
{
        return !memcmp(fmt1, fmt2, sizeof(audio_format));
}

audio_format*
audio_format_dup(const audio_format *src)
{
        audio_format *dst = (audio_format*)malloc(sizeof(audio_format));
        memcpy(dst, src, sizeof(audio_format));
        return dst;
}

void
audio_format_free(audio_format **bye)
{
        free(*bye);
        *bye = NULL;
}

static int
convert_buffer_channels(audio_format *src, 
                        u_char       *src_buf, 
                        int           src_bytes, 
                        audio_format *dst, 
                        u_char       *dst_buf, 
                        int           dst_bytes)
{
        u_char *se;
        int     out_bytes;

        assert(src->channels != dst->channels);
        assert(src->channels == 1 || src->channels == 2);
        assert(dst->channels == 1 || dst->channels == 2);
        assert(src_buf != dst_buf);

        out_bytes = src_bytes * dst->channels / src->channels;

        assert(out_bytes <= dst_bytes); /* Memory overwrite ahoy */

        se = src_buf + src_bytes; /* End of source buffer */

        if (src->channels == 1) {
                /* mono -> stereo */
                if (src->bits_per_sample == 16) {
                        sample *s = (sample*)src_buf;
                        sample *d = (sample*)dst_buf;
                        assert(src->encoding == DEV_S16);
                        while((u_char*)s < se) {
                                *d++ = *s;
                                *d++ = *s++;
                        }
                } else {
                        u_char *s = src_buf;
                        u_char *d = dst_buf;
                        while((u_char*)s < se) {
                                *d++ = *s;
                                *d++ = *s++;
                        }
                }
        } else {
                /* stereo -> mono, nothing smart just average channels */
                if (src->encoding == DEV_S16) {
                        sample *s = (sample*)src_buf;
                        sample *d = (sample*)dst_buf;
                        int tmp;
                        while((u_char*)s < se) {
                                tmp  = *s++;
                                tmp += *s++;
                                *d++ = (sample)(tmp / 2);
                        }
                } else if (src->encoding == DEV_PCMU) {
                        u_char *s = src_buf;
                        u_char *d = dst_buf;
                        int tmp;
                        while(s < se) {
                                tmp  = u2s(*s); s++;
                                tmp += u2s(*s); s++;
                                tmp /= 2;
                                *d++ = s2u(tmp);
                        }
                } else if (src->encoding == DEV_PCMA) {
                        u_char *s = src_buf;
                        u_char *d = dst_buf;
                        int tmp;
                        while(s < se) {
                                tmp  = a2s(*s); s++;
                                tmp += a2s(*s); s++;
                                tmp /= 2;
                                *d++ = s2a(tmp);
                        }
                } else if (src->encoding == DEV_S8) {
                        u_char *s = src_buf;
                        u_char *d = dst_buf;
                        u_char tmp;
                        while(s < se) {
                                tmp  = *s/2; s++;
                                tmp += *s/2; s++;
                                *d++ = tmp;
                        }
                } else if (src->encoding == DEV_U8) {
                        u_char *s = src_buf;
                        u_char *d = dst_buf;
                        while (s != se) {
                                *d  = *s >> 1; s++;
                                *d += *s >> 1; s++;
                                d++;
                        }
                } else {
                        debug_msg("Sample type not recognized\n");
                        assert(0);
                }
        }

        return out_bytes;
}

static int
convert_buffer_sample_type(audio_format *src, u_char *src_buf, int src_bytes,
                           audio_format *dst, u_char *dst_buf, int dst_bytes)
{
        u_char *se;
        u_char *src8, *dst8;          /* 8 bit source and destination buffers */
        sample *src16, *dst16, tmp16; /* 16 bit source and destintion buffers, plus temp */

        int out_bytes;

        out_bytes = src_bytes * dst->bits_per_sample / src->bits_per_sample;

        src8 = dst8 = se = NULL;
        src16 = dst16 = NULL;

        if (out_bytes > dst_bytes) {
                debug_msg("Conversion failed: output buffer too small (%d < %d)\n", dst_bytes, out_bytes);
                return 0;
        }

        if (src_buf == dst_buf) {
                debug_msg("Conversion failed: source buffer and destination buffer are the same\n");
                return 0;
        }

        se = src_buf + src_bytes;

        switch (src->encoding) {
        case DEV_S16:
                src16 = (sample*)src_buf;
                dst8  = dst_buf;
                switch (dst->encoding) {
                case DEV_PCMU:
                        while((u_char*)src16 < se) {
                                *dst8 = s2u(*src16);
                                src16++; dst8++;
                        }
                        break;
                case DEV_PCMA:
                        while((u_char*)src16 < se) {
                                *dst8 = s2a(*src16);
                                src16++; dst8++;
                        }
                        break;
                case DEV_S8:
                        while((u_char*)src16 < se) {
                                *dst8 = *src16 >> 8;
                                src16++; dst8++;
                        }
                        break;
                case DEV_U8:
                        while((u_char*)src16 < se) {
                                *dst8 = (*src16 >> 8) + 128;
                                src16++; dst8++;
                        }
                        break;
                case DEV_S16:
                        break; /* Nothing to do! */
                }
                break;
        case DEV_PCMU:
                src8 = src_buf;
                switch (dst->encoding) {
                case DEV_S16:
                        dst16 = (sample*)dst_buf;
                        while(src8 < se) {
                                *dst16 = u2s(*src8);
                                src8++; dst16++;
                        }
                        break;
                case DEV_S8:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = u2s(*src8);
                                *dst8 = (u_char)(tmp16 >> 8);
                                src8++; dst8++;
                        }
                        assert((dst8 - dst_buf) == (src8 - src_buf));
                        break;
                case DEV_U8:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = u2s(*src8);
                                *dst8 = (u_char)((tmp16 >> 8) + 128);
                                src8++; dst8++;
                        }
                        break;
                case DEV_PCMA:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = u2s(*src8);
                                *dst8 = s2a(tmp16);
                                src8++; dst8++;
                        }
                break;
                case DEV_PCMU:
                        break; /* Nothing to do! */
                }
                break;
        case DEV_PCMA:
                src8 = src_buf;
                switch (dst->encoding) {
                case DEV_S16:
                        dst16 = (sample*)dst_buf;
                        while(src8 < se) {
                                *dst16 = a2s(*src8);
                                src8++; dst16++;
                        }
                        break;
                case DEV_S8:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = a2s(*src8);
                                *dst8 = (u_char)(tmp16 >> 8);
                                src8++; dst8++;
                        }
                break;
                case DEV_U8:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = a2s(*src8);
                                *dst8 = (u_char)((tmp16 >> 8) + 128);
                                src8++; dst8++;
                        }
                        break;
                case DEV_PCMU:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = a2s(*src8);
                                *dst8 = s2u(tmp16);
                                src8++; dst8++;
                        }
                break;
                case DEV_PCMA:
                        break; /* Nothing to do! */
                }
                break;
        case DEV_S8:
                src8 = src_buf;
                switch (dst->encoding) {
                case DEV_S16:
                        dst16 = (sample*)dst_buf;
                        while (src8 < se) {
                                *dst16 = *src8 << 8;
                                src8++; dst16++;
                        }
                        break;
                case DEV_PCMU:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = *src8 << 8;
                                *dst8 = s2u(tmp16);
                                src8++; dst8++;
                        }
                        break;
                case DEV_PCMA:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = *src8 << 8;
                                *dst8 = s2a(tmp16);
                                src8++; dst8++;
                        }
                        break;
                case DEV_U8:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                *dst8 = *src8 + 128;
                                src8++; dst8++;
                        }
                        break;
                case DEV_S8:
                        break; /* Nothing to do! */
                }
                break;
        case DEV_U8:
                src8 = src_buf;
                switch(dst->encoding) {
                case DEV_S8:
                        dst8 = dst_buf;
                        while (src8 < se) {
                                *dst8 = *src8 - 128;
                                src8++; dst8++;
                        }
                        break;
                case DEV_S16:
                        dst16 = (sample*)dst_buf;
                        while(src8 < se) {
                                *dst16 = (sample)((*src8 - 128) << 8);
                                src8++; dst16++;
                        }
                        break;
                case DEV_PCMU:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = (sample)((*src8 - 128) << 8);
                                *dst8 = s2u(tmp16);
                                src8++; dst8++;
                        }
                case DEV_PCMA:
                        dst8 = dst_buf;
                        while(src8 < se) {
                                tmp16 = (sample)((*src8 - 128) << 8);
                                *dst8 = s2a(tmp16);
                                src8++; dst8++;
                        }
                case DEV_U8:
                        break; /* Nothing to do! */
                }
                break;
        }
        return out_bytes;
}

int /* Returns number of bytes put in converted buffer */
audio_format_buffer_convert(audio_format *src, 
                            u_char *src_buf, 
                            int src_bytes, 
                            audio_format *dst, 
                            u_char *dst_buf,
                            int dst_bytes)
{
        int out_bytes;
        u_char *se;

        out_bytes = src_bytes * dst->bits_per_sample * dst->channels / (src->bits_per_sample * src->channels);
        se        = src_buf + src_bytes;

        assert(out_bytes <= dst_bytes);
        if (src->channels == dst->channels || src->encoding == dst->encoding) {
                /* Only 1 buffer needed */
                if (src->channels != dst->channels) {
                        convert_buffer_channels    (src, src_buf, src_bytes, dst, dst_buf, dst_bytes); 
                } else {
                        convert_buffer_sample_type (src, src_buf, src_bytes, dst, dst_buf, dst_bytes); 
                }
        } else {
                /* Additional buffer needed since we have to change
                 * channels and sample type- do everything in steps 
                 */
#define AF_TMP_BUF_SZ   160
#define AF_TMP_BUF_SMPLS 80                
                u_char ibuf[AF_TMP_BUF_SZ];
                int    done = 0, src_adv, ret;
                while (src_bytes > 0) {
                        src_adv = min(AF_TMP_BUF_SMPLS, src_bytes);
                        ret = convert_buffer_channels(src, src_buf, src_adv, dst, ibuf, AF_TMP_BUF_SZ);
                        src_bytes -= src_adv;
                        src_buf   += src_adv;
                        ret = convert_buffer_sample_type(src, ibuf, ret, dst, dst_buf, dst_bytes);
                        dst_bytes -= ret;
                        dst_buf   += ret;
                        done      += ret;
                }
                assert(done == out_bytes);
                assert(dst_bytes >= 0);
        }

        return out_bytes;
}

int
audio_format_change_encoding(audio_format *cur, deve_e new_enc)
{
        int bits_per_sample = 0;

        switch (new_enc) {
        case DEV_S16:  
                bits_per_sample = 16; 
                break;
        case DEV_S8:
        case DEV_U8:
        case DEV_PCMU:
        case DEV_PCMA:
                bits_per_sample = 8;  
                break;
        }
        
        cur->bytes_per_block = cur->bytes_per_block * bits_per_sample / cur->bits_per_sample;
        cur->bits_per_sample = bits_per_sample;
        cur->encoding        = new_enc;
/* If this is zero something has gone wrong! */
        return cur->bytes_per_block; 
}

static const char *
deve_get_name(deve_e encoding)
{
        switch(encoding) {
        case DEV_PCMU: return "8-bit mu-law";
        case DEV_PCMA: return "8-bit a-law";
        case DEV_S16:  return "16-bit signed linear";
        case DEV_S8:   return "8-bit signed linear";
        case DEV_U8:   return "8-bit unsigned linear";
        }
        return NULL;
}

static const char *
audio_channels_name(int channels)
{
        switch(channels) {
        case 1: return "Mono";
        case 2: return "Stereo";
        default: assert(0); 
        }                
        return NULL;
}

int
audio_format_name(const audio_format *cur, char *buf, int buf_len)
{
        assert(cur != NULL);
        assert(buf != NULL);
        assert(buf_len > 0);

        buf[0] = '\0';
        if (buf_len >= 38) {
                /* 38 = strlen("16-bit signed linear,48-kHz,stereo") + 1; */
                sprintf(buf, "%s,%d-kHz,%s",
                        deve_get_name(cur->encoding),
                        cur->sample_rate/1000,
                        audio_channels_name(cur->channels));
                return TRUE;
        }

        return FALSE;
}

