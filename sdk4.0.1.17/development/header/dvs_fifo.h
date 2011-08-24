/*
//      DVS C-Library FifoAPI Header File, Copyright (c) 1998-2010 DVS Digital Video Systems AG
*/

#ifndef _DVS_FIFO_H
#define _DVS_FIFO_H


#ifdef __cplusplus 
  extern "C" { 
#endif 

#if !defined(DOCUMENTATION)

typedef struct {
  int version;            ///< Used internally only, i.e. do not change.
  int size;               ///< Size of this structure in bytes. Do not change.
  int fifoid;             ///< FIFO/frame ID of the buffer. Do not change.
  int flags;              ///< Used internally only, i.e. do not change.
  struct {
    char * addr;          ///< Pointer for the DMA.
    int    size;          ///< Size of the DMA.
  } dma;                  ///< DMA substructure (not used when setting the define
                          ///< SV_FIFO_FLAG_NODMAADDR).
  struct {
    char * addr;          ///< Pointer/offset to the video data.
    int    size;          ///< Size of the video data.
  } video[2];             ///< Array containing two video fields.
  struct {
    char * addr[4];       ///< Pointer/offset to the audio data of the channels 1 to 4.
    int    size;          ///< Size of the audio buffer.
  } audio[2];             ///< Array containing two audio fields.
  struct {
    int    ltc_tc;        ///< Analog LTC timecode without bit masking.
    int    ltc_ub;        ///< Analog LTC user bytes.
    int    vitc_tc;       ///< Analog VITC timecode.
    int    vitc_ub;       ///< Analog VITC user bytes.
    int    vtr_tick;      ///< Capture tick for VTR timecode and user bytes.
    int    vtr_tc;        ///< VTR timecode.
    int    vtr_ub;        ///< VTR user bytes.
    int    vtr_info3;     ///< VTR info bytes 8 to 11.
    int    vtr_info;      ///< VTR info bytes 0 to 3.
    int    vtr_info2;     ///< VTR info bytes 4 to 7.
    int    vitc_tc2;      ///< Analog VITC timecode of field 2.
    int    vitc_ub2;      ///< Analog VITC user bytes of field 2.
    int    pad[4];        ///< Reserved for future use.
  } timecode;             ///< Timecode substructure.
  struct {
    int tick;             ///< Tick when the frame was captured. Valid for an input
                          ///< only.
    int clock_high;       ///< Clock of the MSBs (most significant bytes) when a frame
                          ///< was captured. Valid for an input only.
    int clock_low;        ///< Clock of the LSBs (least significant bytes) when a frame
                          ///< was captured. Valid for an input only.
    int gpi;              ///< GPI information of the FIFO buffer.
    int aclock_high;      ///< Clock of the MSBs (most significant bytes) when audio was
                          ///< captured. Valid for an input only.
    int aclock_low;       ///< Clock of the LSBs (least significant bytes) when audio
                          ///< was captured. Valid for an input only.
    int pad[2];           ///< Reserved for future use.
  } control;              ///< Various buffer related values.
  struct {
    int cmd;              ///< The received VTR command. Valid for an input only.
    int length;           ///< Number of data bytes. Valid for an input only.
    unsigned char data[16];///< Data bytes. Valid for an input only.
  } vtrcmd;               ///< Substructure containing the incoming VTR commands.
  struct {
    char * addr[4];       ///< Pointer/offset to the audio data of the channels 5 to 8.
  } audio2[2];            ///< Array containing two audio fields.
  struct {
    int storagemode;      ///< Image data storage mode.
    int xsize;            ///< Image data x-size.
    int ysize;            ///< Image data y-size.
    int xoffset;          ///< Image data x-offset from center.
    int yoffset;          ///< Image data y-offset from center.
    int dataoffset;       ///< Offset to the first pixel in the buffer.
    int lineoffset;       ///< Offset from line to line or zero (0) for default.
    int compression;      ///< Compression code of the video data.
    int encryption;       ///< Obsolete. Instead use sv_fifo_buffer.encryption.code.
    int matrixtype;       ///< Matrix type.
    int bufferid;         ///< Buffer ID from the Render API. For a normal FIFO
                          ///< operation it should be set to zero (0).
    int pad[4];           ///< Reserved for future use.
  } storage;              ///< Dynamic storage mode. Valid for an output only.
  struct {
    int dvitc_tc[2];      ///< Digital/ANC VITC timecode.
    int dvitc_ub[2];      ///< Digital/ANC VITC user bytes.
    int film_tc[2];       ///< Digital/ANC film timecode (RP201).
    int film_ub[2];       ///< Digital/ANC film user bytes (RP201).
    int prod_tc[2];       ///< Digital/ANC production timecode (RP201).
    int prod_ub[2];       ///< Digital/ANC production user bytes (RP201).
    int dltc_tc;          ///< Digital/ANC LTC timecode.
    int dltc_ub;          ///< Digital/ANC LTC user bytes.
    int closedcaption[2]; ///< Analog closed caption. Valid for an input only.
    int afilm_tc[2];      ///< Analog film timecode.
    int afilm_ub[2];      ///< Analog film user bytes.
    int aprod_tc[2];      ///< Analog production timecode.
    int aprod_ub[2];      ///< Analog production user bytes.
  } anctimecode;          ///< ANC timecode substructure.
  struct {
    char * addr;          ///< Pointer/offset to the video data channel B (second video
                          ///< image, see the define SV_FIFO_FLAG_VIDEO_B).
    int    size;          ///< Size of the video data.
  } video_b[2];           ///< Array containing two video fields (second video image,
                          ///< see the define SV_FIFO_FLAG_VIDEO_B).
  struct {
    char * addr;          ///< Pointer/offset to the ANC data.
    int    size;          ///< Size of the ANC data.
  } anc[2];               ///< Array containing two ANC fields.
  struct {
    int keyid;            ///< Decryption key ID.
    int payload;          ///< Amount of data (incl. plaintext and padding).
    int plaintext;        ///< Plaintext offset.
    int sourcelength;     ///< Original size of the non-encrypted data.
    int code;             ///< Encryption code of the video data.
    int pad[3];           ///< Reserved for future use.
  } encryption;           ///< Substructure for a decryption of video. It can be used
                          ///< with HydraCinema only.
  struct {
    int keyid;            ///< Decryption key ID.
    int payload;          ///< Amount of data (incl. plaintext and padding).
    int plaintext;        ///< Plaintext offset.
    int sourcelength;     ///< Original size of the non-encrypted data.
    int code;             ///< Encryption code of the audio data.
    int bits;             ///< Bit width of the decrypted data.
    int channels;         ///< Channel count of the decrypted data.
    int frequency;        ///< Frequency of the decrypted data.
    int littleendian;     ///< Endianness of the decrypted data. It is a boolean value
                          ///< and TRUE if the data is in little endian byte order.
    int bsigned;          ///< Signedness of the decrypted data. It is a boolean value
                          ///< and TRUE if the data must be interpreted as a signed
                          ///< integer.
  } encryption_audio;     ///< Substructure for a decryption of audio. It can be used
                          ///< with HydraCinema only.
  int pad[6];             ///< Reserved for future use.
} sv_fifo_buffer;


typedef struct {
  int nbuffers;               ///< Absolute FIFO depth. The real FIFO depth is:
                              ///< <nReal> = sv_fifo_info.nbuffers - 1.
  int availbuffers;           ///< For an output this element returns the number of
                              ///< free/empty buffers, i.e. the remaining
                              ///< sv_fifo_getbuffer() calls at your disposal. You can
                              ///< calculate the number of filled buffers with:
                              ///< <nFilled> = sv_fifo_info.nbuffers - sv_fifo_info.availbuffers.
                              ///< For an input this element returns the number of filled buffers,
                              ///< i.e. the remaining sv_fifo_getbuffer() calls at your disposal.
                              ///< You can calculate the number of empty buffers with:
                              ///< <nEmpty> = sv_fifo_info.nbuffers - sv_fifo_info.availbuffers.
  int tick;                   ///< Current tick.
  int clock_high;             ///< Clock time of the last vertical sync (upper 32 bits).
  int clock_low;              ///< Clock time of the last vertical sync (lower 32 bits).
  int dropped;                ///< Number of frames that were dropped since calling the
                              ///< function sv_fifo_start().
  int clocknow_high;          ///< Current clock time (upper 32 bits).
  int clocknow_low;           ///< Current clock time (lower 32 bits).
  int waitdropped;            ///< For DVS internal use. Number of dropped waits for the
                              ///< vertical sync of the function sv_fifo_getbuffer().
  int waitnotintime;          ///< For DVS internal use. Number of times with waits for
                              ///< the vertical sync that occurred not in real time.
  int audioinerror;           ///< Audio input error code.
  int videoinerror;           ///< Video input error code.
  int displaytick;            ///< Current display tick.
  int recordtick;             ///< Current record tick.
  int openprogram;            ///< Program which tried to open the device.
  int opentick;               ///< Tick of the time when the program tried to open the
                              ///< device.
  int pad26[8];               ///< Reserved for future use.
} sv_fifo_info;


typedef struct {
  int version;                ///< Version of this structure.
  int size;                   ///< Size of the structure.
  int when;                   ///< Buffer tick.
  int clock_high;             ///< Buffer clock (upper 32 bits).
  int clock_low;              ///< Buffer clock (lower 32 bits).
  int clock_tolerance;        ///< Buffer clock operation tolerance.
  int padding[32-6];          ///< Reserved for future use.
} sv_fifo_bufferinfo;
#endif /* !DOCUMENTATION */

#if defined(DOCUMENTATION_SDK)
/**
//  \ingroup fifoapi
//
//  \def SV_FIFO_BUFFERINFO_VERSION_1
//
//  This define is implemented to distinguish between different versions of the structure <i>\ref sv_fifo_bufferinfo</i>.
//  Currently the SDK supports this struct version only. It has to be set at all times.
*/
#define SV_FIFO_BUFFERINFO_VERSION_1    1
#endif /* DOKUMENTATION_SDK */

#if !defined(DOCUMENTATION)
typedef struct {
  int     entries;            ///< Number of valid entries in this structure.
  int     dmaalignment;       ///< Needed alignment of a DMA buffer.
  int     nbuffers;           ///< Maximum number of buffers in this video mode.
  int     vbuffersize;        ///< Size of one video frame in the board memory plus an
                              ///< internal alignment. For the exact size use the
                              ///< function sv_storage_status().
  int     abuffersize;        ///< Size of the audio data in the board memory that
                              ///< corresponds to one frame of video.
  void *  unused1;            ///< No longer used.
  void *  unused2;            ///< No longer used.
  int     dmarect_xoffset;    ///< X-offset for the current DMA rectangle.
  int     dmarect_yoffset;    ///< Y-offset for the current DMA rectangle.
  int     dmarect_xsize;      ///< X-size for the current DMA rectangle.
  int     dmarect_ysize;      ///< Y-size for the current DMA rectangle.
  int     dmarect_lineoffset; ///< Line to line offset for the current DMA rectangle.
  int     field1offset;       ///< Offset to the start of field 1.
  int     field2offset;       ///< Offset to the start of field 2.
  int     ancbuffersize;      ///< Size of the ANC data in the board memory that
                              ///< corresponds to one frame of video.
  int     pad[64-15];         ///< Reserved for future use.
} sv_fifo_configinfo;


typedef struct {
  int mode;       ///< Memory mode (SV_FIFO_MEMORYMODE_<xxx> flags). See the function
                  ///< sv_fifo_memorymode().
  int size;       ///< Memory size, zero (0) is default.
  int pad[16];    ///< Reserved for future use. Set to zero (0).
} sv_fifo_memory;


typedef struct {
  int            linenr;       ///< Line number of this ANC packet.
  int            did;          ///< Data ID of this ANC packet.
  int            sdid;         ///< Secondary data ID of this ANC packet.
  int            datasize;     ///< Data payload, i.e. the number of bytes in the data
                               ///< element.
  int            vanc;         ///< Position of this packet. Either VANC (1) or
                               ///< HANC (0).
  int            field;        ///< Field index (0 or 1) of this ANC packet.
  int            pad[6];       ///< Reserved for future use. Set to zero (0).
  unsigned char  data[256];    ///< Buffer for the data payload (element 'datasize').
} sv_fifo_ancbuffer;


typedef struct {
  struct {
    int activate;              ///< Activate encryption.

    struct {
      unsigned char key[16];   ///< Encryption AES key (most significant byte first).
      unsigned char attr[8];   ///< Encryption AES attribute (most significant byte
                               ///< first).
    } aes;                     ///< Substructure that contains all parameters used for a
                               ///< link encryption.

    struct {
      int linenr;              ///< Line number of the metadata ANC packet.
      int current_key;         ///< ID of the current key.
      int next_key;            ///< ID of the next key.
      int pad[5];              ///< Reserved for future use. Set to zero (0).
    } metadata;                ///< Substructure that contains all parameters to control
                               ///< the ANC packet.
  } link[2];                   ///< Array for the SDI links A and B.
} sv_fifo_linkencrypt_info;

#endif /* !DOCUMENTATION */

#ifndef CBLIBINT_H
/**
//  \ingroup fifoapi
//
//  \typedef sv_fifo
//
//  Void handle to the internal structure describing the FIFO.
*/
typedef void * sv_fifo;
#else
#if defined(WIN32) && defined(DLL) && !defined(DVS_CLIB_NOEXPORTS)
#define export __declspec(dllexport)
#else
#define export
#endif
#endif

#ifndef _DVS_CLIB_NO_FUNCTIONS_
export int sv_fifo_init(sv_handle * sv, sv_fifo ** ppfifo, int jack, int bShared, int bDMA, int flagbase, int nframes);
export int sv_fifo_free(sv_handle * sv, sv_fifo * pfifo);
export int sv_fifo_reset(sv_handle * sv, sv_fifo * pfifo);
export int sv_fifo_start(sv_handle * sv, sv_fifo * pfifo);
export int sv_fifo_startex(sv_handle * sv, sv_fifo * pfifo, int * pwhen, int * pclockhigh, int * pclocklow, int * pspare);
export int sv_fifo_stop(sv_handle * sv, sv_fifo * pfifo, int flags);
export int sv_fifo_stopex(sv_handle * sv, sv_fifo * pfifo, int flags, int * pwhen, int * pclockhigh, int * pclocklow, int * pspare);
export int sv_fifo_wait(sv_handle * sv, sv_fifo * pfifo);
export int sv_fifo_status(sv_handle * sv, sv_fifo * pfifo, sv_fifo_info * pinfo);
export int sv_fifo_getbuffer(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer ** pbuffer, sv_fifo_bufferinfo * bufferinfo, int flags);
export int sv_fifo_putbuffer(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer * pbuffer, sv_fifo_bufferinfo * bufferinfo);
export int sv_fifo_vsyncwait(sv_handle * sv, sv_fifo * pfifo);
export int sv_fifo_configstatus(sv_handle * sv, sv_fifo * pfifo, sv_fifo_configinfo * pconfig);
export int sv_fifo_dmarectangle(sv_handle * sv, sv_fifo * pfifo, int xoffset, int yoffset, int xsize, int ysize, int lineoffset);
export int sv_fifo_anc(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer * pbuffer, sv_fifo_ancbuffer * panc);
export int sv_fifo_ancdata(sv_handle * sv, sv_fifo * pfifo, unsigned char * buffer, int buffersize, int * pcount);
export int sv_fifo_lut(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer * pbuffer, unsigned char * buffer, int buffersize, int cookie, int flags);
export int sv_fifo_bypass(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer * pbuffer, int video, int audio);
export int sv_fifo_linkencrypt(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer * pbuffer, sv_fifo_linkencrypt_info * pinfo);
export int sv_fifo_sanitycheck(sv_handle * sv, sv_fifo * pfifo);
export int sv_fifo_sanitylevel(sv_handle * sv, sv_fifo * pfifo, int level, int version);
export int sv_fifo_memorymode(sv_handle * sv, sv_fifo_memory * memory);
export int sv_fifo_anclayout(sv_handle * sv, sv_fifo * pfifo, char * description, int size, int * required);
export int sv_fifo_matrix(sv_handle * sv, sv_fifo * pfifo, sv_fifo_buffer * pbuffer, unsigned int * pmatrix);
#endif

#define SV_FIFO_FLAG_DONTBLOCK            0x000001
#define SV_FIFO_FLAG_VIDEOONLY            0x000002
#define SV_FIFO_FLAG_AUDIOONLY            0x000004
#define SV_FIFO_FLAG_NODMAADDR            0x000008
#define SV_FIFO_FLAG_FLUSH                0x000010
#define SV_FIFO_FLAG_NODMA                0x000020
#define SV_FIFO_FLAG_DMARECTANGLE         0x000040
#define SV_FIFO_FLAG_TIMEDOPERATION       0x000080
#define SV_FIFO_FLAG_PULLDOWN             0x000100
#define SV_FIFO_FLAG_VSYNCWAIT            0x000200
#define SV_FIFO_FLAG_VSYNCWAIT_RT         0x000400 ///< Obsolete
#define SV_FIFO_FLAG_PHYSADDR             0x000800 ///< Obsolete
#define SV_FIFO_FLAG_SETAUDIOSIZE         0x001000
#define SV_FIFO_FLAG_CLOCKEDOPERATION     0x002000
#define SV_FIFO_FLAG_STORAGEMODE          0x004000
#define SV_FIFO_FLAG_STORAGENOAUTOCENTER  0x008000
#define SV_FIFO_FLAG_AUDIOINTERLEAVED     0x010000
#define SV_FIFO_FLAG_TIMECODEVALID        0x020000
#define SV_FIFO_FLAG_FIELD                0x040000
#define SV_FIFO_FLAG_VIDEO_B              0x080000

#define SV_FIFO_FLAG_REPEAT_ONCE          0x000000
#define SV_FIFO_FLAG_REPEAT_2TIMES        0x100000
#define SV_FIFO_FLAG_REPEAT_3TIMES        0x200000
#define SV_FIFO_FLAG_REPEAT_4TIMES        0x300000
#define SV_FIFO_FLAG_REPEAT_MASK          0x300000

#define SV_FIFO_FLAG_ANC                  0x400000
#define SV_FIFO_FLAG_NO_LIVE              0x800000
#define SV_FIFO_FLAG_N0_LIVE              0x800000 ///< Obsolete
#define SV_FIFO_FLAG_RESERVED1            0x01000000 ///< For internal use
#define SV_FIFO_FLAG_STEREOMODE_A         0x02000000 ///< Currently not used.
#define SV_FIFO_FLAG_STEREOMODE_B         0x04000000 ///< Currently not used.

#define SV_FIFO_MEMORYMODE_DEFAULT        0x000000
#define SV_FIFO_MEMORYMODE_FIFO_ALL       0x000000
#define SV_FIFO_MEMORYMODE_SHARE_RENDER   0x000001
#define SV_FIFO_MEMORYMODE_FIFO_NONE      0x000002


#define SV_FIFO_SANITY_VERSION_DEFAULT    0
#define SV_FIFO_SANITY_VERSION_1          1

#define SV_FIFO_SANITY_LEVEL_OFF          0
#define SV_FIFO_SANITY_LEVEL_FATAL        1
#define SV_FIFO_SANITY_LEVEL_ERROR        2
#define SV_FIFO_SANITY_LEVEL_WARN         3

#define SV_FIFO_LUT_ID_DEFAULT             0x00000000
#define SV_FIFO_LUT_ID_MASK                0x000000ff
#define SV_FIFO_LUT_TYPE_1D_RGBA           0x00000000
#define SV_FIFO_LUT_TYPE_3D                0x00000100
#define SV_FIFO_LUT_TYPE_1D_RGB            0x00000200
#define SV_FIFO_LUT_TYPE_1D_RGBA_NONLINEAR 0x00000300
#define SV_FIFO_LUT_TYPE_1D_RGBA_4K        0x00000400
#define SV_FIFO_LUT_TYPE_MASK              0x00000f00
#define SV_FIFO_LUT_FLAG_MASK              0x0000f000
#define SV_FIFO_LUT_FLAG_LASTNODE          0x00001000

#define SV_FIFO_DMA_MEMMAP                 0x00 // obsolete
#define SV_FIFO_DMA_ON                     0x01
#define SV_FIFO_DMA_OFF                    0x02
#define SV_FIFO_DMA_VIDEO                  0x04
#define SV_FIFO_DMA_AUDIO                  0x08

#ifdef __cplusplus 
  } 
#endif 

#endif



