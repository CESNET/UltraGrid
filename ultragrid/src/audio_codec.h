/*
 * FILE:    audio_codec.h
 * AUTHORS: Orion Hodson
 *          Colin Perkins
 *
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: audio_codec.h,v 1.1 2007/11/08 09:48:58 hopet Exp $
 */

#ifndef _ACODEC_H_
#define _ACODEC_H_

/* Codec module startup and end */

void acodec_init (void);
void acodec_exit (void);

/* Use these two functions to finder number of available codecs 
 * and to get an codec id of the num'th codec.
 */
uint32_t    acodec_get_number_of_codecs (void);
acodec_id_t acodec_get_codec_number     (uint32_t num);

/* Use this function to check if codec id is valid / corrupted */

int acodec_id_is_valid(acodec_id_t id);

/* Use these functions to see what formats a codec supports
 * and whether they are encode from or decode to.
 */

const acodec_format_t* acodec_get_format        (acodec_id_t id);
int                    acodec_can_encode_from   (acodec_id_t id, 
                                                const audio_format *qfmt);
int                    acodec_can_encode        (acodec_id_t id);
int                    acodec_can_decode_to     (acodec_id_t id, 
                                                const audio_format *qfmt);
int                    acodec_can_decode        (acodec_id_t id);
int                    acodec_audio_formats_compatible(acodec_id_t id1,
                                                       acodec_id_t id2);

/* This is easily calculable but crops up everywhere */
uint32_t               acodec_get_samples_per_frame (acodec_id_t id);

/* Codec encoder functions */
int  acodec_encoder_create  (acodec_id_t id, acodec_state **cs);
void acodec_encoder_destroy (acodec_state **cs);
int  acodec_encode          (acodec_state* cs, 
                            coded_unit*  in_native,
                            coded_unit*  out);

/* Codec decoder functions */
int  acodec_decoder_create  (acodec_id_t id, acodec_state **cs);
void acodec_decoder_destroy (acodec_state **cs);
int  acodec_decode          (acodec_state* cs, 
                            coded_unit*  in,
                            coded_unit*  out_native);

/* Repair related */

int  acodec_decoder_can_repair (acodec_id_t id);
int  acodec_decoder_repair     (acodec_id_t id, 
                               acodec_state *cs,
                               uint16_t consec_missing,
                               coded_unit *prev, 
                               coded_unit *miss, 
                               coded_unit *next);

/* Peek function for variable frame size codecs */
uint32_t acodec_peek_frame_size(acodec_id_t id, u_char *data, uint16_t blk_len);

int     acodec_clear_coded_unit(coded_unit *u);

/* RTP payload mapping interface */
int         payload_is_valid      (u_char pt);
int         acodec_map_payload    (acodec_id_t id, u_char pt);
int         acodec_unmap_payload  (acodec_id_t id, u_char pt);
u_char      acodec_get_payload    (acodec_id_t id);
acodec_id_t acodec_get_by_payload (u_char pt);

/* For compatibility only */
acodec_id_t acodec_get_first_mapped_with(uint16_t sample_rate, uint16_t channels);

/* Name to codec mappings */
acodec_id_t acodec_get_by_name      (const char *name);
acodec_id_t acodec_get_matching     (const char *short_name, uint16_t sample_rate, uint16_t channels);

acodec_id_t acodec_get_native_coding (uint16_t sample_rate, uint16_t channels);

int        acodec_is_native_coding  (acodec_id_t id);

int        acodec_get_native_info   (acodec_id_t cid, 
                                    uint16_t *sample_rate, 
                                    uint16_t *channels);
/* For layered codecs */
uint8_t     acodec_can_layer         (acodec_id_t id);
int        acodec_get_layer         (acodec_id_t id,
                                    coded_unit *cu_whole,
                                    uint8_t layer,
                                    uint16_t *markers,
                                    coded_unit *cu_layer);
int        acodec_combine_layer     (acodec_id_t id,
                                    coded_unit *cu_layer,
                                    coded_unit *whole,
                                    uint8_t nelem,
                                    uint16_t *markers);


#endif /* _ACODEC_H_ */
