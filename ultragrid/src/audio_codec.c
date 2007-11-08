/*
 * FILE:    audio_codec.c
 * AUTHORS: Orion Hodson
 *          Colin Perkins
 *
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:58 $
 */
 
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_codec_types.h"
#include "audio_codec.h"
#include "audio_codec/l8.h"
#include "audio_codec/l16.h"
#include "audio_codec/g711.h"
#include "audio_codec/g726.h"
#include "audio_codec/g728.h"
#include "audio_codec/dvi.h"
#include "audio_codec/gsm.h"
#include "audio_codec/lpc.h"
#include "audio_codec/vdvi.h"

/* Codec class initialization - hey didn't c++ happen somewhere along the 
 * timeline ;-) 
 */

/* One time codec {con,de}struction */
typedef void    (*cx_init_f) (void);
typedef void    (*cx_exit_f) (void);

/* Codec Probing functions */
typedef uint16_t               (*cx_get_formats_count_f) (void);
typedef const acodec_format_t* (*cx_get_format_f)        (uint16_t);

/* Codec Encoding functions */
typedef int     (*cx_encoder_create_f)    (uint16_t idx, u_char **state);
typedef void    (*cx_encoder_destroy_f)   (uint16_t idx, u_char **state);
typedef int     (*cx_encode_f)            (uint16_t idx, u_char *state, 
                                           sample *in, coded_unit *out);
typedef int     (*cx_can_encode_f)        (uint16_t idx);

/* Codec Decoding functions */
typedef int     (*cx_decoder_create_f)    (uint16_t idx, u_char **state);
typedef void    (*cx_decoder_destroy_f)   (uint16_t idx, u_char **state);
typedef int     (*cx_decode_f)            (uint16_t idx, u_char *state, 
                                           coded_unit *in, sample *out);
typedef int     (*cx_can_decode_f)        (uint16_t idx);

/* For determining frame sizes of variable bit rate codecs */
typedef int     (*cx_peek_size_f)         (uint16_t idx, u_char *data, int data_len);

/* For codec domain repair schemes */
typedef int     (*cx_repair_f)            (uint16_t idx, 
                                           u_char *state,
                                           uint16_t consec_missing,
                                           coded_unit *prev,
                                           coded_unit *missing,
                                           coded_unit *next);

/* For layered codecs */
typedef uint8_t		(*cx_can_layer_f)	  (void);
typedef int             (*cx_get_layer_f)         (uint16_t idx, coded_unit *cu_whole, uint8_t layer,
                                                   uint16_t *markers, coded_unit *cu_layer);
typedef int             (*cx_combine_layer_f)     (uint16_t idx, coded_unit *cu_layer, coded_unit *cu_whole, uint8_t nelem, uint16_t *markers);

typedef struct s_acodec_fns {
        cx_init_f               cx_init;
        cx_exit_f               cx_exit;
        cx_get_formats_count_f  cx_get_formats_count;
        cx_get_format_f         cx_get_format;
        cx_encoder_create_f     cx_encoder_create;
        cx_encoder_destroy_f    cx_encoder_destroy;
        cx_encode_f             cx_encode;
        cx_can_encode_f         cx_can_encode;
        cx_decoder_create_f     cx_decoder_create;
        cx_decoder_destroy_f    cx_decoder_destroy;
        cx_decode_f             cx_decode;
        cx_can_decode_f         cx_can_decode;
        cx_peek_size_f          cx_peek_size;
        cx_repair_f             cx_repair;
	cx_can_layer_f	        cx_can_layer;
        cx_get_layer_f          cx_get_layer;
        cx_combine_layer_f      cx_combine_layer;
} acodec_fns_t;

static acodec_fns_t acodec_table[] = {
	{
                NULL, 
                NULL,
                l16_get_formats_count,
                l16_get_format,
                NULL, /* No encoder setup / tear down */
                NULL, 
                l16_encode,
                NULL,
                NULL,
                NULL, /* No decoder setup / tear down */
                l16_decode,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
        },
        {
                NULL, 
                NULL,
                l8_get_formats_count,
                l8_get_format,
                NULL, /* No encoder setup / tear down */
                NULL, 
                l8_encode,
                NULL,
                NULL,
                NULL, /* No decoder setup / tear down */
                l8_decode,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
        },
        {       g711_init,
                NULL,
                g711_get_formats_count,
                g711_get_format,
                NULL,
                NULL,
                g711_encode,
                NULL,
                NULL,
                NULL,
                g711_decode,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
        },
#ifdef HAVE_G728
        {
                NULL,
                NULL,
                g728_get_formats_count,
                g728_get_format,
                g728_encoder_create,
                g728_encoder_destroy,
                g728_encoder_do,
                NULL,
                g728_decoder_create,
                g728_decoder_destroy,
                g728_decoder_do,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
        },
#endif /* HAVE_G728 */
        {
                NULL,
                NULL,
                g726_get_formats_count,
                g726_get_format,
                g726_state_create,
                g726_state_destroy,
                g726_encode,
                NULL,
                g726_state_create,
                g726_state_destroy,
                g726_decode,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
        },
        {
                NULL,
                NULL,
                dvi_get_formats_count,
                dvi_get_format,
                dvi_state_create,
                dvi_state_destroy,
                dvi_encode,
                NULL,
                dvi_state_create,
                dvi_state_destroy,
                dvi_decode,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
        },
        {
                NULL,
                NULL,
                vdvi_get_formats_count,
                vdvi_get_format,
                vdvi_state_create,
                vdvi_state_destroy,
                vdvi_encoder,
                NULL,
                vdvi_state_create,
                vdvi_state_destroy,
                vdvi_decoder,
                NULL,
                vdvi_peek_frame_size,
                NULL,
                NULL,
                NULL,
                NULL
        },
        {
                NULL,
                NULL,
                gsm_get_formats_count,
                gsm_get_format,
                gsm_state_create,
                gsm_state_destroy,
                gsm_encoder,
                NULL,
                gsm_state_create,
                gsm_state_destroy,
                gsm_decoder,
                NULL,
                NULL,
                gsm_repair,
                NULL,
                NULL,
                NULL
        },
        {
                lpc_setup,
                NULL,
                lpc_get_formats_count,
                lpc_get_format,
                lpc_encoder_state_create,
                lpc_encoder_state_destroy,
                lpc_encoder,
                NULL,
                lpc_decoder_state_create,
                lpc_decoder_state_destroy,
                lpc_decoder,
                NULL,
                NULL,
                lpc_repair,
                NULL,
                NULL,
                NULL
        }
};

/* NUM_CODEC_INTERFACES = number of codec interfaces */
/* Applications never know this...                   */
#define NUM_ACODEC_INTERFACES (sizeof(acodec_table)/sizeof(acodec_fns_t))

/* These are used to save multiple function calls in
 * codec_get_codec_number function. */
static uint16_t num_fmts_supported[NUM_ACODEC_INTERFACES];
static uint16_t total_fmts_supported;

/* Codec identifier is 32 bits int. It's like an MS handle.
 * First byte is always zero.
 * Second byte is index in codec_table above plus 1.
 * Third and fourth bytes hold index for encoding format
 * used by codec.
 */ 

#define ACODEC_GET_IFS_INDEX(id)     (uint16_t)(((id  & 0x00ff0000) >> 16) - 1)
#define ACODEC_GET_FMT_INDEX(id)     (uint16_t)((id & 0x0000ffff) - 1)
#define ACODEC_MAKE_ID(ifs,fmt)      (((ifs) + 1) << 16)|((fmt+1)&0xffff)

#define ACODEC_VALID_PAD(id)          (!(id & 0xff000000))
#define ACODEC_VALID_IFS(id)           (id & 0x00ff0000)
#define ACODEC_VALID_FMT(id)           (id & 0x0000ffff)

int
acodec_id_is_valid(acodec_id_t id)
{
        uint32_t ifs, fmt;

        if (acodec_is_native_coding(id)) {
                /* Native codings should be tested with
                 * codec_is_native_coding */
                debug_msg("Coding is invalid because it is a native coding\n");
                return FALSE;
        }
        
        if (!ACODEC_VALID_PAD(id) || 
            !ACODEC_VALID_IFS(id) ||
            !ACODEC_VALID_FMT(id)) {
                debug_msg("Codec id (0x%08x) invalid (pad %x, ifs %x, fmt %x)",
                          id, ACODEC_VALID_PAD(id), ACODEC_VALID_IFS(id),
                          ACODEC_VALID_FMT(id));
                return FALSE;
        }

        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);

        if (ifs >= NUM_ACODEC_INTERFACES) {
                /* Table index too large */
                debug_msg("Codec index outside table\n");
                return FALSE; 
        }

        if (fmt >= num_fmts_supported[ifs]) {
                /* Format index too large */
                debug_msg("Format index outside table %d / %d\n", fmt, num_fmts_supported[ifs]);
                return FALSE;
        }
        return TRUE;
}

static void acodec_map_init(void);
static void acodec_map_exit(void);

void
acodec_init()
{
        const acodec_format_t *cf;
        uint32_t i;
        uint16_t j;

        if (total_fmts_supported == 0) {
                for(i = 0; i < NUM_ACODEC_INTERFACES; i++) {
                        if (acodec_table[i].cx_init) acodec_table[i].cx_init();
                        num_fmts_supported[i] = acodec_table[i].cx_get_formats_count();
                        total_fmts_supported += num_fmts_supported[i];
                        for(j = 0; j < num_fmts_supported[i]; j++) {
                                cf = acodec_table[i].cx_get_format(j);
                                /* Most compilers should spot this check anyway */
                                assert(strlen(cf->short_name) < ACODEC_SHORT_NAME_LEN);
                                assert(strlen(cf->long_name) < ACODEC_LONG_NAME_LEN);
                                assert(strlen(cf->description) < ACODEC_DESCRIPTION_LEN);
                        }
                }
                acodec_map_init();
        } else {
                debug_msg("acodec_init already called - ignoring.\n");
        }
}        

void
acodec_exit()
{
        uint32_t i;
        if (total_fmts_supported != 0) {
                for(i = 0; i < NUM_ACODEC_INTERFACES; i++) {
                        if (acodec_table[i].cx_exit) acodec_table[i].cx_exit();
                }
                acodec_map_exit();
                total_fmts_supported = 0;
        } else {
                debug_msg("acodec_exit not inited - ignoring\n");
        }
}

uint32_t
acodec_get_number_of_codecs()
{
        return total_fmts_supported;
}

acodec_id_t
acodec_get_codec_number(uint32_t n)
{
        acodec_id_t id;
        uint32_t    ifs;
        assert(n < total_fmts_supported);
        
        for(ifs = 0; n >= num_fmts_supported[ifs]; ifs++) {
                n = n - num_fmts_supported[ifs];
        }

        id = ACODEC_MAKE_ID(ifs, n);

        assert(acodec_id_is_valid(id));        

        return id;
}

const acodec_format_t*
acodec_get_format(acodec_id_t id)
{
        uint16_t ifs, fmt;

        assert(acodec_id_is_valid(id));

        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);
        return acodec_table[ifs].cx_get_format(fmt);
}

int
acodec_can_encode(acodec_id_t id)
{
        uint16_t ifs;

        ifs = ACODEC_GET_IFS_INDEX(id);

        assert(acodec_id_is_valid(id));                
        assert(ifs < NUM_ACODEC_INTERFACES);

        if (acodec_table[ifs].cx_can_encode) {
                /* cx_can_encode only needs to exist if encoder and decoder are asymmetric */
                return acodec_table[ifs].cx_can_encode(ACODEC_GET_FMT_INDEX(id));
        } else {
                const acodec_format_t *cf;
                cf = acodec_get_format(id);
                if (cf->format.sample_rate % 8000) {
                        return FALSE; /* only m * 8k at the moment */
                }
        }
        
        return TRUE;
}

int
acodec_can_decode(acodec_id_t id)
{
        uint32_t ifs;

        ifs = ACODEC_GET_IFS_INDEX(id);

        assert(acodec_id_is_valid(id));        
        assert(ifs < NUM_ACODEC_INTERFACES);

        if (acodec_table[ifs].cx_can_decode) {
                /* cx_can_encode only needs to exist if encoder and decoder are asymmetric */
                return acodec_table[ifs].cx_can_decode(ACODEC_GET_FMT_INDEX(id));
        }  else {
                const acodec_format_t *cf;
                cf = acodec_get_format(id);
                if (cf->format.sample_rate % 8000) {
                        return FALSE; /* Only m * 8k at moment */
                }
        }
        
        return TRUE;
}

int
acodec_audio_formats_compatible(acodec_id_t id1, acodec_id_t id2)
{
        const acodec_format_t *cf1, *cf2;
        int match;
        
        assert(acodec_id_is_valid(id1));
        assert(acodec_id_is_valid(id2));
        
        cf1 = acodec_get_format(id1);
        cf2 = acodec_get_format(id2);

        match = !memcmp(&cf1->format, &cf2->format, sizeof(audio_format)); 

        return match;
}

uint32_t
acodec_get_samples_per_frame(acodec_id_t id)
{
        const acodec_format_t *cf;
        uint32_t spf;

        assert(acodec_id_is_valid(id));
        cf = acodec_get_format(id);
        spf = cf->format.bytes_per_block * 8 / 
                (cf->format.channels * cf->format.bits_per_sample);

        return spf;
}

/* Encoder related ***********************************************************/
int
acodec_encoder_create(acodec_id_t id, acodec_state **cs)
{
        if (acodec_id_is_valid(id)) {
                uint16_t ifs, fmt;
                *cs = (acodec_state*) malloc(sizeof(acodec_state));
                if (!cs) {
                        *cs = NULL;
                        return 0;
                }
                (*cs)->state = NULL;
                (*cs)->id = id;
                ifs = ACODEC_GET_IFS_INDEX(id);
                fmt = ACODEC_GET_FMT_INDEX(id);
                if (acodec_table[ifs].cx_encoder_create) {
                        /* Must also have a destructor */
                        assert(acodec_table[ifs].cx_encoder_destroy != NULL);
                        acodec_table[ifs].cx_encoder_create(fmt, 
                                                           &(*cs)->state);
                }
                return TRUE;
        } else {
                debug_msg("Attempting to initiate invalid acodec\n");
                abort();
        }
        return 0;
}

void
acodec_encoder_destroy(acodec_state **cs)
{
        acodec_id_t id;
        assert(*cs != NULL);
        id = (*cs)->id;
        if (acodec_id_is_valid(id)) {
                uint16_t ifs, fmt;
                ifs = ACODEC_GET_IFS_INDEX(id);
                fmt = ACODEC_GET_FMT_INDEX(id);
                if (acodec_table[ifs].cx_encoder_destroy) {
                        /* Must also have a destructor */
                        acodec_table[ifs].cx_encoder_destroy(fmt, 
                                                               &(*cs)->state);
                }
                free(*cs);
                *cs = NULL;
        } else {
                debug_msg("Destroying corrupted acodec\n");
                abort();
        }
}

int
acodec_encode(acodec_state *cs,
             coded_unit  *in_native,
             coded_unit  *cu)
{
        uint16_t    ifs, fmt;
        int        success;

        assert(cs        != NULL);
        assert(in_native != NULL);
        assert(cu        != NULL);

        assert(acodec_is_native_coding(in_native->id));
        assert (in_native->state == NULL);
#ifdef DEBUG
        {
                const acodec_format_t *cf = acodec_get_format(cs->id);
                assert (cf->format.bytes_per_block == in_native->data_len);
        }
#endif
        cu->id = cs->id;
        ifs = ACODEC_GET_IFS_INDEX(cu->id);
        fmt = ACODEC_GET_FMT_INDEX(cu->id);

        success = acodec_table[ifs].cx_encode(fmt, cs->state, (sample*)in_native->data, cu);
        return success;
}

/* Decoder related ***********************************************************/
int
acodec_decoder_create(acodec_id_t id, acodec_state **cs)
{
        if (acodec_id_is_valid(id)) {
                uint16_t ifs, fmt;
                *cs = (acodec_state*) malloc(sizeof(acodec_state));
                if (!cs) {
                        *cs = NULL;
                        return 0;
                }
                (*cs)->state = NULL;
                (*cs)->id = id;
                ifs = ACODEC_GET_IFS_INDEX(id);
                fmt = ACODEC_GET_FMT_INDEX(id);
                if (acodec_table[ifs].cx_decoder_create) {
                        /* Must also have a destructor */
                        assert(acodec_table[ifs].cx_decoder_destroy != NULL);
                        acodec_table[ifs].cx_decoder_create(fmt, 
                                                               &(*cs)->state);
                }
                return TRUE;
        } else {
                debug_msg("Attempting to initiate invalid acodec\n");
                abort();
        }
        return 0;
}

void
acodec_decoder_destroy(acodec_state **cs)
{
        acodec_id_t id;
        assert(*cs != NULL);
        id = (*cs)->id;
        if (acodec_id_is_valid(id)) {
                uint16_t ifs, fmt;
                ifs = ACODEC_GET_IFS_INDEX(id);
                fmt = ACODEC_GET_FMT_INDEX(id);
                if (acodec_table[ifs].cx_decoder_destroy) {
                        /* Must also have a destructor */
                        acodec_table[ifs].cx_decoder_destroy(fmt, 
                                                               &(*cs)->state);
                }
                free(*cs);
                *cs = NULL;
        } else {
                debug_msg("Destroying corrupted acodec\n");
                abort();
        }
}

int
acodec_decode(acodec_state *cs,
             coded_unit  *in,
             coded_unit  *out)
{
        const acodec_format_t *cf;
        acodec_id_t           id;
        uint16_t              ifs, fmt, rate, channels;
        int                  success;

        assert(cs  != NULL);
        assert(out != NULL);
        assert(in  != NULL);
        
        id = cs->id;
        assert(in->id == cs->id);
        assert(acodec_is_native_coding(in->id) == FALSE);

        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);

        /* Setup outgoing data block */
        cf = acodec_get_format(id);
        assert(out->state == NULL);
        assert(out->data  == NULL);
        rate     = (uint16_t)cf->format.sample_rate;
        channels = (uint16_t)cf->format.channels;
        out->id       = acodec_get_native_coding(rate, channels);
        out->data_len = cf->format.bytes_per_block;
        out->data     = (u_char*) malloc(out->data_len);

        /* Decode */
        success = acodec_table[ifs].cx_decode(fmt, cs->state, in, (sample*)out->data);
        return success;
}

int
acodec_decoder_can_repair (acodec_id_t id) 
{
        uint16_t ifs;
        assert(acodec_id_is_valid(id));
        ifs = ACODEC_GET_IFS_INDEX(id);
        if (acodec_table[ifs].cx_repair) {
                return TRUE;
        } else {
                return FALSE;
        }
}

int
acodec_decoder_repair(acodec_id_t id, acodec_state *cs, 
                     uint16_t consec_missing, 
                     coded_unit *prev, 
                     coded_unit *miss, 
                     coded_unit *next)
{
        uint16_t    ifs, fmt;

        assert(acodec_id_is_valid(id));
        assert(id == cs->id);

        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);

        if (miss->id != id) {
                debug_msg("Wrong previous unit supplied for repair.  Probably a transition in stream.\n");
                return FALSE;
        }
        miss->id = prev->id;

        assert(acodec_table[ifs].cx_repair != NULL);
        return acodec_table[ifs].cx_repair(fmt, 
                                          cs->state, 
                                          consec_missing,
                                          prev, miss, next);
}

uint32_t 
acodec_peek_frame_size(acodec_id_t id, u_char *data, uint16_t blk_len)
{
        uint16_t    ifs, fmt;

        assert(acodec_id_is_valid(id));

        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);        

        if (acodec_table[ifs].cx_peek_size) {
                return acodec_table[ifs].cx_peek_size(fmt, data, (int)blk_len);
        } else {
                const acodec_format_t *cf = acodec_get_format(id);
                return cf->mean_coded_frame_size;
        }
}

/* Codec clear coded unit - not sure where this should go - away? :-) */
int
acodec_clear_coded_unit(coded_unit *u)
{
        if (u->state_len) {
                assert(u->state != NULL);
                free(u->state);
        }
        if (u->data_len) {
                assert(u->data != NULL);
                free(u->data);
        }
        memset(u, 0, sizeof(coded_unit));
        return TRUE;
}

/* RTP related things like mapping and payload sanity checking ***************/

int
payload_is_valid(u_char pt)
{
        /* Per rfc1890.txt (actually 72-95 is unassigned, but we use it anyway */
        if (pt < 29 || (pt>=72 && pt <=127)) return TRUE;
        return FALSE;
}

/* RTP Mapping interface - 
 * 2 maps one from payload to codec id and the other from codec id to 
 * to payload.
 */

#define NUM_PAYLOADS 128
static acodec_id_t payload_map[NUM_PAYLOADS];
static u_char    *acodec_map[NUM_ACODEC_INTERFACES];

static void
acodec_map_init()
{
        uint16_t i,j;
        const acodec_format_t *fmt;

        memset(payload_map, 0, NUM_PAYLOADS * sizeof(acodec_id_t));
        for(i = 0; i < NUM_ACODEC_INTERFACES; i++) {
                acodec_map[i] = (u_char*) malloc(num_fmts_supported[i]);
                memset(acodec_map[i], ACODEC_PAYLOAD_DYNAMIC, num_fmts_supported[i]);
                for(j = 0; j < num_fmts_supported[i]; j++) {
                        fmt = acodec_table[i].cx_get_format(j);
                        if (fmt->default_pt == ACODEC_PAYLOAD_DYNAMIC) {
                                continue;
                        }
                        acodec_map_payload(ACODEC_MAKE_ID(i,j), fmt->default_pt);
                }
        }
}

static void
acodec_map_exit()
{
        uint32_t i;
        for(i = 0; i < NUM_ACODEC_INTERFACES; i++) {
                free(acodec_map[i]);
        }
}

int
acodec_map_payload(acodec_id_t id, u_char pt)
{
        if (payload_is_valid(pt) && acodec_id_is_valid(id)) {
                if (payload_map[pt] != 0) {
                        acodec_unmap_payload(id, pt);
                }
                payload_map[pt] = id;
                acodec_map[ACODEC_GET_IFS_INDEX(id)][ACODEC_GET_FMT_INDEX(id)] = pt;
                return TRUE;
        }
#ifdef DEBUG
        {
                const acodec_format_t *cf;
                cf = acodec_get_format(id);
                debug_msg("Failed to map payload for %s\n", cf->long_name);
        }
#endif /* DEBUG */
        return FALSE;
}

u_char
acodec_get_payload(acodec_id_t id)
{
        u_char pt;

        assert(acodec_id_is_valid(id));
        pt = acodec_map[ACODEC_GET_IFS_INDEX(id)][ACODEC_GET_FMT_INDEX(id)];
        if (payload_is_valid(pt)) {
                assert(acodec_get_by_payload(pt) == id);
                return pt;
        }
        return ACODEC_PAYLOAD_DYNAMIC;
}

int 
acodec_unmap_payload(acodec_id_t id, u_char pt)
{
        if (payload_is_valid(pt) && 
            acodec_id_is_valid(id) &&
            payload_map[pt] == id) {
                payload_map[pt] = 0;
                acodec_map[ACODEC_GET_IFS_INDEX(id)][ACODEC_GET_FMT_INDEX(id)]
		= ACODEC_PAYLOAD_DYNAMIC;
                return TRUE;
        }
        debug_msg("Failed to unmap payload\n");
        return FALSE;
}

acodec_id_t
acodec_get_by_payload (u_char pt)
{
        if (payload_is_valid(pt)) {
#ifdef DEBUG
                if (payload_map[pt] == 0) {
                        debug_msg("No acodec for payload %d\n", pt);
                }
#endif       
                return payload_map[pt];
        } else {
                debug_msg("acodec_get_by_payload - invalid payload (%d)\n", pt);
                return 0;
        }
}

/* For compatibility only */
acodec_id_t 
acodec_get_first_mapped_with(uint16_t sample_rate, uint16_t channels)
{
        const acodec_format_t *cf;
        int pt;
        
        for(pt = 0; pt < NUM_PAYLOADS; pt++) {
                if (payload_map[pt]) {
                        cf = acodec_get_format(payload_map[pt]);
                        if (cf->format.sample_rate == sample_rate &&
                            cf->format.channels    == channels) {
                                return payload_map[pt];
                        }
                }
        }
        debug_msg("No mapped acodecs compatible (%d, %d)\n",
                  sample_rate, channels);
        return 0;
}


acodec_id_t 
acodec_get_by_name(const char *name)
{
        const acodec_format_t *cf;
        uint16_t ifs, fmt;

        for(ifs = 0; ifs < NUM_ACODEC_INTERFACES; ifs++) {
                for(fmt = 0; fmt < num_fmts_supported[ifs]; fmt++) {
                        cf = acodec_table[ifs].cx_get_format(fmt);
                        if (!strcasecmp(cf->long_name, name)) {
                                return ACODEC_MAKE_ID(ifs,fmt);
                        }
                }
        }

        return 0;
}


acodec_id_t
acodec_get_matching(const char *short_name, uint16_t freq, uint16_t channels)
{
        /* This has been changed to try really hard to find a matching acodec.
         * The reason is that it's now called as part of the command-line      
         * parsing, and so has to cope with user entered acodec names. Also, it 
         * should recognise the names sdr gives the codecs, for compatibility 
         * with rat-v3.0.                                                [csp] 
         */

        /* This is not quite as inefficient as it looks, since stage 1 will
         * almost always find a match.                                     
         */

        const acodec_format_t  *cf = NULL;
        acodec_id_t             cid = 0; 
        uint32_t                i, codecs;
        char                  *long_name;

        /* Stage 1: Try the designated short names... */
        codecs = acodec_get_number_of_codecs();
        for(i = 0; i < codecs; i++) {
                cid = acodec_get_codec_number(i);
                cf  = acodec_get_format(cid);
                if (cf->format.sample_rate == freq  && 
                    cf->format.channels == channels && 
                    !strcasecmp(short_name, cf->short_name)) {
                        return cid;
                }
        }

        /* Stage 2: Try to generate a matching name... */
        long_name = (char *) malloc(strlen(short_name) + 12);
        sprintf(long_name, "%s-%dK-%s", short_name, freq/1000, channels==1?"MONO":"STEREO");
        for(i = 0; i < codecs; i++) {
                cid = acodec_get_codec_number(i);
                cf  = acodec_get_format(cid);
                if (cf->format.sample_rate == freq  && 
                    cf->format.channels == channels && 
                    !strcasecmp(long_name, cf->long_name)) {
                        free(long_name);
                        return cid;
                }
        }

        /* Stage 3: Nasty hack... PCM->PCMU for compatibility with sdr 
         * and old rat versions 
         */
        if (strncasecmp(short_name, "pcm", 3) == 0) {
                sprintf(long_name, "PCMU-%dK-%s", freq/1000, channels==1?"MONO":"STEREO");
                for(i = 0; i < codecs; i++) {
                        cid = acodec_get_codec_number(i);
                        cf  = acodec_get_format(cid);
                        if (cf->format.sample_rate == freq  && 
                            cf->format.channels == channels && 
                            !strcasecmp(long_name, cf->long_name)) {
                                free(long_name);
                                return cid;
                        }
                }
        }

        free(long_name);

        debug_msg("Unable to find codec \"%s\" at rate %d channels %d\n", short_name, freq, channels);
        return 0;
}


/* These constants are what are supported as native codings */

static uint16_t sampling_rates[] = {8000, 11025, 16000, 22050, 32000, 44100, 48000};
static uint16_t num_sampling_rates = sizeof(sampling_rates)/sizeof(sampling_rates[0]);
static uint16_t max_channels = 2;

/* The following three functions are a hack so we can have encode and
 * decode functions take coded_units as input and output.  This makes
 * paths cleaner since we don't have two data types for coded and raw
 * units.  */

acodec_id_t 
acodec_get_native_coding(uint16_t sample_rate, uint16_t channels)
{
        acodec_id_t cid;
        uint32_t i, index;

        for (i = 0; i < num_sampling_rates; i++) {
                if (sampling_rates[i] == sample_rate) {
                        break;
                }
        }
        assert(i != num_sampling_rates);
        assert(channels <= max_channels);
        index = i * max_channels + (channels - 1);
        /* There is no codec corresponding to this but make it
         * have right form we set it interfaces to the number
         * of interfaces, i.e. one more than is legal.
         */
        cid = ACODEC_MAKE_ID(NUM_ACODEC_INTERFACES, index);
        return cid;
}

int
acodec_is_native_coding(acodec_id_t cid)
{
        return (ACODEC_GET_IFS_INDEX(cid) == NUM_ACODEC_INTERFACES &&
                ACODEC_GET_FMT_INDEX(cid) < num_sampling_rates * max_channels);
}

int 
acodec_get_native_info(acodec_id_t cid, 
                      uint16_t   *p_rate, 
                      uint16_t   *p_channels)
{
        uint32_t i, c, index;

        if (acodec_is_native_coding(cid)) {
                index = ACODEC_GET_FMT_INDEX(cid);
                /* Calculate and verify index in table of acceptable rates */
                i = index / max_channels;
                if (p_rate != NULL) {
                        *p_rate = sampling_rates[i];
                }
                /* Calculate and verify number of channels */
                c = (index % max_channels) + 1;
                if (p_channels != NULL) {
                        *p_channels = (uint16_t)c;
                }
                return TRUE;
        }
        return FALSE;
}

/* Layered codecs ***********************************************************/
uint8_t
acodec_can_layer(acodec_id_t id)
{
        uint16_t ifs;

        ifs = ACODEC_GET_IFS_INDEX(id);

        assert(acodec_id_is_valid(id));                
        assert(ifs < NUM_ACODEC_INTERFACES);

        if (acodec_table[ifs].cx_can_layer) {
                return acodec_table[ifs].cx_can_layer();
        }
        return 1;

}

int
acodec_get_layer(acodec_id_t id, coded_unit *cu_whole, uint8_t layer, uint16_t *markers, coded_unit *cu_layer)
{
        uint16_t ifs, fmt;
		
        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);        

        if (acodec_table[ifs].cx_get_layer) {
                return acodec_table[ifs].cx_get_layer(fmt, cu_whole, layer, markers, cu_layer);
        }
        return FALSE;
}

int
acodec_combine_layer (acodec_id_t id, coded_unit *cu_layer, coded_unit *cu_whole, uint8_t nelem, uint16_t *markers)
{
        uint16_t ifs, fmt;
		
        ifs = ACODEC_GET_IFS_INDEX(id);
        fmt = ACODEC_GET_FMT_INDEX(id);        
		
        assert(acodec_table[ifs].cx_combine_layer);
        return acodec_table[ifs].cx_combine_layer(fmt, cu_layer, cu_whole, nelem, markers);
}
