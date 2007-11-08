/*
 * FILE:    codec_state.h
 * PROGRAM: RAT
 * AUTHOR:  Orion Hodson
 *
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Id: audio_codec_state.h,v 1.1 2007/11/08 09:48:58 hopet Exp $
 */

#ifndef __CODEC_STATE_H__
#define __CODEC_STATE_H__

typedef enum {
        ENCODER,
        DECODER
} codec_mode;

struct s_codec_state_store;

int    codec_state_store_create  (struct s_codec_state_store **css, codec_mode m);
void   codec_state_store_destroy (struct s_codec_state_store **css);
codec_state *    
       codec_state_store_get     (struct s_codec_state_store *, 
                                  codec_id_t id);

void   codec_state_store_remove  (struct s_codec_state_store *cs, 
                                  codec_id_t id);

#endif /* __CODEC_STATE_H__ */
