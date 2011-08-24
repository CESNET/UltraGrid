#ifndef _DVS_RENDER_H
#define _DVS_RENDER_H

#ifdef __cplusplus
extern "C" {
#endif

#define SV_RENDER_LUTFLAGS_NOTLINEAR 0x01

#if !defined(DOCUMENTATION)

typedef union {
  struct {
    void * plut;              ///< Pointer to the buffer containing the LUT data.
    int    size;              ///< Size of the LUT data, e.g. RGBA * 4 bytes * entries
                              ///< -> 4 * 4 * 1024 or 4 * 4 * 1025 for a 10-bit LUT.
                              ///< The components are non-interleaved, i.e.
                              ///< "R0R1R2...G0G1G2...B0B1B2...A0A1A2...".
    int    lutid;             ///< LUT ID in case there are multiple LUTs available on
                              ///< the data path (0..n).
    int    flags;             ///< Flags for LUT processing (see
                              ///< SV_RENDER_LUTFLAGS_<xxx> in function
                              ///< sv_render_push_1dlut()).
    int    spare[4];          ///< Reserved for future use.
  } v1;                       ///< Version 1 of the structure.
} sv_render_1dlut;


typedef union {
  struct {
    void * plut;              ///< Pointer to the buffer containing the LUT data.
    int    size;              ///< Size of the LUT data. See function
                              ///< sv_render_push_3dlut() for details.
    int    lutid;             ///< LUT ID in case there are multiple 3D LUTs available
                              ///< on the data path (0..n).
    int    flags;             ///< Reserved for future use.
    int    spare[4];          ///< Reserved for future use.
  } v1;                       ///< Version 1 of the structure.
} sv_render_3dlut;


typedef union {
  struct {
    void *  addr;             ///< Address of the JPEG2000 source data (incl.
                              ///< plaintext).
    int     size;             ///< Size of the JPEG2000 source data.
    int     encryption;       ///< Encryption mode (SV_ENCRYPTION_<xxx>).
    int     keyid;            ///< Decryption key ID.
    int     payload;          ///< Amount of data (incl. plaintext and padding).
    int     plaintext;        ///< Plaintext offset.
    int     sourcelength;     ///< Original size of the non-encrypted data.
  } v1;                       ///< Version 1 of the structure.
} sv_render_jpeg2000_decode;


typedef union {
  struct {
    int xsize;                ///< X-size of uncompressed data.
    int ysize;                ///< Y-size of uncompressed data.
    int storagemode;          ///< Storage mode (SV_MODE_<xxx>).
    int lineoffset;           ///< Offset from line to line (default is zero (0)).
    int matrixtype;           ///< Color space (SV_MATRIXTYPE_<xxx>).
    int dataoffset;           ///< Offset to the start of the data.
  } v1;                       ///< Version 1 of the structure used for uncompressed
                              ///< buffers.
  struct {
    int xsize;                ///< X-size of compressed data.
    int ysize;                ///< Y-size of compressed data.
    int buffersize;           ///< Size of the compressed data.
    int dataoffset;           ///< Offset to the start of the data.
    int storagemode;          ///< Storage mode (SV_MODE_<xxx>).
    int matrixtype;           ///< Color space (SV_MATRIXTYPE_<xxx>).
  } v2;                       ///< Version 2 of the structure used for compressed
                              ///< buffers.
} sv_render_bufferlayout;


typedef union {
  struct {
    double matrix[10];        ///< Matrix coefficients (3 * 3 plus 1 for alpha).
    double offset[4];         ///< Matrix offsets (one for each component).
  } v1;                       ///< Version 1 of the structure.
  struct {
    double matrix[10];        ///< Matrix coefficients (3 * 3 plus 1 for alpha).
    double inoffset[4];       ///< Matrix in-offsets (one for each component).
    double outoffset[4];      ///< Matrix out-offsets (one for each component).
    int matrixid;             ///< Matrix position within processing pipeline (0..n).
  } v2;                       ///< Version 2 of the structure.
  struct {
    int matrixtype_source;    ///< Source color space (SV_MATRIXTYPE_<xxx>).
    int matrixtype_dest;      ///< Destination color space (SV_MATRIXTYPE_<xxx>).
    int matrixid;             ///< Matrix position within processing pipeline (0..n).
  } v3;                       ///< Version 3 of the structure.
} sv_render_matrix;


typedef union {
  struct {
    int xsize;                ///< Destination x-size.
    int ysize;                ///< Destination y-size.
    int xoffset;              ///< Currently not used, set to zero (0).
    int yoffset;              ///< Currently not used, set to zero (0).
  } v1;                       ///< Version 1 of the structure.
  struct {
    int xsize;                ///< Destination x-size.
    int ysize;                ///< Destination y-size.
    int xoffset;              ///< Currently not used, set to zero (0).
    int yoffset;              ///< Currently not used, set to zero (0). 
    int sharpness;            ///< Sharpness, valid from -0xffff to +0xffff, default is zero (0).
  } v2;                       ///< Version 2 of the structure.
} sv_render_scaler;


#endif /* !DOCUMENTATION */


#ifndef CBLIBINT_H
typedef void sv_render_handle;
typedef void sv_render_context;

typedef struct _sv_render_image {
  int bufferid;
  int bufferoffset;
  int buffersize;
  int xsize;
  int ysize;
  int lineoffset;
  int storagemode;
  int matrixtype;
  int dataoffset;
} sv_render_image;
#endif


/*
//  Official renderoption parameters
*/
#define SV_RENDER_OPTION_SAFERENDER         1
//...

#ifndef _DVS_RENDER_H_SV_DEFINESONLY_
export int sv_render_open(sv_handle * sv, sv_render_handle ** pprender);
export int sv_render_close(sv_handle * sv, sv_render_handle * prender);

export int sv_render_option_set(sv_handle * sv, sv_render_handle * prender, int option, int value);
export int sv_render_option_get(sv_handle * sv, sv_render_handle * prender, int option, int * value);

export int sv_render_begin(sv_handle * sv, sv_render_handle * prender, sv_render_context ** ppcontext);
export int sv_render_reuse(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext);
export int sv_render_issue(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, sv_overlapped * poverlapped);
export int sv_render_ready(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, int timeout, sv_overlapped * poverlapped);
export int sv_render_end(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext);
export int sv_render_malloc(sv_handle * sv, sv_render_handle * prender, sv_render_image ** ppimage, int version, int size, sv_render_bufferlayout * pstorage);
export int sv_render_free(sv_handle * sv, sv_render_handle * prender, sv_render_image * pimage);
export int sv_render_dma(sv_handle * sv, sv_render_handle * prender, int btocard, sv_render_image * pimage, void * buffer, int bufferoffset, int transfersize, sv_overlapped * poverlapped);
export int sv_render_dmaex(sv_handle * sv, sv_render_handle * prender, int btocard, sv_render_image * pimage, char * memoryaddr, int memorysize, int memoryoffset, int memorylineoffset, int cardoffset, int cardlineoffset, int linesize, int linecount, int spare, sv_overlapped * poverlapped);
export int sv_render_memory_info(sv_handle * sv, sv_render_handle * prender, int * ptotal, int * pfree, int * pfreeblock);

export int sv_render_push_matrix(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, int version, int size, sv_render_matrix * pvalue);
export int sv_render_push_1dlut(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, int version, int size, sv_render_1dlut * pvalue);
export int sv_render_push_3dlut(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, int version, int size, sv_render_3dlut * pvalue);
export int sv_render_push_scaler(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, sv_render_image * pdest, int version, int size, sv_render_scaler * pvalue);
export int sv_render_push_jpeg2000_decode(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, sv_render_image * pimage, int version, int size, sv_render_jpeg2000_decode * pvalue);
export int sv_render_push_image(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, sv_render_image * pimage);
export int sv_render_push_render(sv_handle * sv, sv_render_handle * prender, sv_render_context * pcontext, sv_render_image * pimage);
#endif

#ifdef __cplusplus
}
#endif

#include "dvs_render_optional.h"

#endif /* _DVS_RENDER_H */
