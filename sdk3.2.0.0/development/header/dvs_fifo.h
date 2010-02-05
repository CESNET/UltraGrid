/*
//      DVS C-Library FifoAPI Header File, Copyright (c) 1998-2008 DVS Digital Video Systems AG
*/

#ifndef _DVS_FIFO_H
#define _DVS_FIFO_H

#ifdef __cplusplus 
  extern "C" { 
#endif 

/**
//  \ingroup fifoapi
//  
//  \struct sv_fifo_buffer
//
//  The following details the FIFO buffer structure used by the functions \e sv_fifo_getbuffer() and
//  \e sv_fifo_putbuffer().
*/
typedef struct {
  int version;            ///< Used internally only, i.e. do not change.
  int size;               ///< Size of this structure in bytes (do not change).
  int fifoid;             ///< FIFO/frame ID of the buffer, do not change.
  int flags;              ///< Used internally only, i.e. do not change.
  struct {
    char * addr;          ///< Pointer for the DMA.
    int    size;          ///< Size for DMA.
  } dma;                  ///< DMA substructure (not used when setting \c SV_FIFO_FLAG_NODMADDR).
  struct {
    char * addr;          ///< Pointer/offset to the video data.
    int    size;          ///< Size of video data.
  } video[2];             ///< Array containing two video fields.
  struct {
    char * addr[4];       ///< Pointer/offset to the audio data of channels 1 to 4.
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
    int tick;             ///< Tick when the frame was captured. Valid for an input only.
    int clock_high;       ///< Clock of the MSBs (most significant bytes) when a frame was captured. Valid for an input only.
    int clock_low;        ///< Clock of the LSBs (least significant bytes) when a frame was captured. Valid for an input only.
    int gpi;              ///< GPI information of the FIFO buffer.
    int aclock_high;      ///< Clock of the MSBs (most significant bytes) when audio was captured. Valid for an input only.
    int aclock_low;       ///< Clock of the LSBs (least significant bytes) when audio was captured. Valid for an input only.
    int pad[2];           ///< Reserved for future use.
  } control;              ///< Various buffer related values.
  struct {
    int cmd;              ///< The received VTR command. Valid for an input only.
    int length;           ///< Number of data bytes. Valid for an input only.
    unsigned char data[16];///< Data bytes. Valid for an input only.
  } vtrcmd;               ///< Substructure containing the incoming VTR commands.
  struct {
    char * addr[4];       ///< Pointer/offset to the audio data of channels 5 to  8.
  } audio2[2];            ///< Array containing two audio fields.
  struct {
    int storagemode;      ///< Image data storage mode.
    int xsize;            ///< Image data x-size.
    int ysize;            ///< Image data y-size.
    int xoffset;          ///< Image data x-offset from center.
    int yoffset;          ///< Image data y-offset from center.
    int dataoffset;       ///< Offset to the first pixel in the buffer.
    int lineoffset;       ///< Offset from line to line, or zero (\c 0) for default.
    int pad[8];           ///< Reserved for future use.
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
    char * addr;          ///< Pointer/offset to the video data channel B (second video image, see define \c #SV_FIFO_FLAG_VIDEO_B.
    int    size;          ///< Size of video data.
  } video_b[2];           ///< Array containing two video fields (second video image, see define \c #SV_FIFO_FLAG_VIDEO_B).
  struct {
    char * addr;          ///< Pointer/offset to the ANC data.
    int    size;          ///< Size of ANC data.
  } anc[2];               ///< Array containing two ANC fields.
  int pad[24];            ///< Reserved for future use.
} sv_fifo_buffer;


/**
//  \ingroup fifoapi
//
//  \struct sv_fifo_info
//
//  The following describes the structure \e sv_fifo_info which is used by the function \e sv_fifo_status().
*/
typedef struct {
  int nbuffers;               ///< Total number of buffers.
  int availbuffers;           ///< Number of free buffers.
  int tick;                   ///< Current tick.
  int clock_high;             ///< Clock time of the last vertical sync (upper 32 bits).
  int clock_low;              ///< Clock time of the last vertical sync (lower 32 bits).
  int dropped;                ///< Number of dropped frames.
  int clocknow_high;          ///< Current clock time (upper 32 bits).
  int clocknow_low;           ///< Current clock time (lower 32 bits).
  int waitdropped;            ///< Number of dropped waits for the vertical sync of the function \e sv_fifo_getbuffer().
  int waitnotintime;          ///< Number of times with waits for the vertical sync that occurred not in real time.
  int audioinerror;           ///< Audio input error code.
  int videoinerror;           ///< Video input error code.
  int displaytick;            ///< Current display tick.
  int recordtick;             ///< Current record tick.
  int openprogram;            ///< Program which tried to open the device.
  int opentick;               ///< Tick of the time when the program tried to open the device.
  int pad26[8];               ///< Reserved for future use.
} sv_fifo_info;


/**
//  \ingroup fifoapi
//
//  \struct sv_fifo_bufferinfo
//
//  The following details the structure of the buffer related timing information used by the functions
//  \e sv_fifo_getbuffer() and \e sv_fifo_putbuffer().
*/
typedef struct {
  int version;                ///< Version of this structure.
  int size;                   ///< Size of the structure.
  int when;                   ///< Buffer tick.
  int clock_high;             ///< Buffer clock (upper 32 bits).
  int clock_low;              ///< Buffer clock (lower 32 bits).
  int clock_tolerance;        ///< Buffer clock operation tolerance.
  int padding[32-6];          ///< Reserved for future use.
} sv_fifo_bufferinfo;

/**
//  \ingroup fifoapi
//
//  \def SV_FIFO_BUFFERINFO_VERSION_1
//
//  This define is implemented to distinguish between different versions of the structure \e sv_fifo_bufferinfo.
//  Currently the SDK supports this struct version only. It has to be set at all times.
*/
#define SV_FIFO_BUFFERINFO_VERSION_1    1


/**
//  \ingroup fifoapi
//
//  \struct sv_fifo_configinfo
//
//  The following provides details about the structure \e sv_fifo_configinfo used by
//  the function \e sv_fifo_configstatus() to return system parameters.
*/
typedef struct {
  int     entries;            ///< Number of valid entries in this structure.
  int     dmaalignment;       ///< Needed alignment of a DMA buffer.
  int     nbuffers;           ///< Maximum number of buffers in this video mode.
  int     vbuffersize;        ///< Size of one video frame in the board memory.
  int     abuffersize;        ///< Size of one audio frame in the board memory.
  void *  unused1;            ///< No longer used.
  void *  unused2;            ///< No longer used.
  int     dmarect_xoffset;    ///< X-offset for the current DMA rectangle.
  int     dmarect_yoffset;    ///< Y-offset for the current DMA rectangle.
  int     dmarect_xsize;      ///< X-size for the current DMA rectangle.
  int     dmarect_ysize;      ///< Y-size for the current DMA rectangle.
  int     dmarect_lineoffset; ///< Line to line offset for the current DMA rectangle.
  int     field1offset;       ///< Offset to the start of field 1.
  int     field2offset;       ///< Offset to the start of field 2.
  int     ancbuffersize;      ///< Size of one ANC frame in the board memory.
  int     pad[64-15];         ///< Reserved for future use.
} sv_fifo_configinfo;

/**
//  \ingroup fifoapi
//  
//  \struct sv_fifo_ancbuffer
//
//  The following details the ANC data structure used by the function \e sv_fifo_anc().
*/
typedef struct {
  int            linenr;       ///< Line number of this ANC packet.
  int            did;          ///< Data ID of this ANC packet.
  int            sdid;         ///< Secondary data ID of this ANC packet.
  int            datasize;     ///< Data payload, i.e. the number of bytes in the data element.
  int            vanc;         ///< Position of this packet. Either VANC&nbsp;(\c 1 ) or HANC&nbsp;(\c 0 ).
  int            field;        ///< Field index (\c 0 or \c 1) of this ANC packet.
  int            pad[6];       ///< Reserved for future use. Set to zero&nbsp;(\c 0 ).
  unsigned char  data[256];    ///< Buffer for the data payload (element \e datasize ).
} sv_fifo_ancbuffer;

/**
//  \ingroup fifoapi
//  
//  \struct sv_fifo_linkencrypt_info
//
//  The following details the link encryption data structure used by the function \e sv_fifo_linkencrypt().
*/
typedef struct {
  struct {
    int activate;              ///< Activate encryption.

    struct {
      unsigned char key[16];   ///< Encryption AES key (most significant byte first).
      unsigned char attr[8];   ///< Encryption AES attribute (most significant byte first).
    } aes;

    struct {
      int linenr;              ///< Line number of the metadata ANC packet.
      int current_key;         ///< ID of the current key.
      int next_key;            ///< ID of the next key.
      int pad[5];              ///< Reserved for future use. Set to zero&nbsp;(\c 0 ).
    } metadata;
  } link[2];                   ///< Array for the SDI links&nbsp;A and&nbsp;B.
} sv_fifo_linkencrypt_info;

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
export int sv_fifo_anclayout(sv_handle * sv, sv_fifo * pfifo, char * description, int size, int * required);

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
#define SV_FIFO_FLAG_VSYNCWAIT_RT         0x000400
#define SV_FIFO_FLAG_PHYSADDR             0x000800
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

#define SV_FIFO_SANITY_VERSION_DEFAULT    0
#define SV_FIFO_SANITY_VERSION_1          1

#define SV_FIFO_SANITY_LEVEL_OFF          0
#define SV_FIFO_SANITY_LEVEL_FATAL        1
#define SV_FIFO_SANITY_LEVEL_ERROR        2
#define SV_FIFO_SANITY_LEVEL_WARN         3

#define SV_FIFO_LUT_ID_DEFAULT            0x00000000
#define SV_FIFO_LUT_ID_MASK               0x000000ff
#define SV_FIFO_LUT_TYPE_1D_RGBA          0x00000000
#define SV_FIFO_LUT_TYPE_3D               0x00000100
#define SV_FIFO_LUT_TYPE_1D_RGB           0x00000200
#define SV_FIFO_LUT_TYPE_MASK             0x00000f00

#ifdef __cplusplus 
  } 
#endif 

#endif

