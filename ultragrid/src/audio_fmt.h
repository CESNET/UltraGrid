/*
 * FILE:    audio_fmt.h
 * PROGRAM: RAT
 * AUTHOR:  Orion Hodson
 *
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Id: audio_fmt.h,v 1.1 2007/11/08 09:48:58 hopet Exp $
 */

#ifndef _AUDIO_FMT_H_
#define _AUDIO_FMT_H_

int           audio_format_get_common (audio_format  *fmt1, 
                                       audio_format  *fmt2, 
                                       audio_format  *common_fmt);
int           audio_format_match      (audio_format  *fmt1, 
                                       audio_format  *fmt2);
audio_format* audio_format_dup        (const audio_format  *src);
void          audio_format_free       (audio_format **bye);
int           audio_format_buffer_convert (audio_format *src, 
                                           u_char       *src_buf, 
                                           int           src_bytes, 
                                           audio_format *dst, 
                                           u_char       *dst_buf,
                                           int           dst_bytes);
int           audio_format_change_encoding (audio_format *cur, 
                                            deve_e        new_enc);
int           audio_format_name(const audio_format *cur, char *buf, int buf_len);

#endif /* _AUDIO_FMT_H_ */

