/*
//      DVS C-Library Header File, Copyright (c) 1994-2010 DVS Digital Video Systems AG
//
//      Avoid using constant that have OBSOLETE defined after them
//
//
//      For definition of error codes look in the file dvs_errors.h
//
//
//      _DVS_CLIB_H_SV_DEFINESONLY_
*/

#ifndef _DVS_CLIB_H
#define _DVS_CLIB_H

#ifdef __cplusplus 
  extern "C" { 
#endif 

#ifndef _DVS_VERSION_H_
#include "dvs_version.h"
#endif

#if !defined(_DVS_ERRORS_H_) && !defined(_SV_ERRORS_H_)
#include "dvs_errors.h"
#endif

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef O_BINARY
#define O_BINARY        0
#endif


#if !defined(uint32) 
#define uint32 unsigned int
#endif

#if defined(__STDC__) || defined(WIN32)
#if !defined __PROTOTYPES__
#define __PROTOTYPES__
#endif
#endif



#if defined(WIN32) && defined(DLL) && !defined(DVS_CLIB_NOEXPORTS)
#define export __declspec(dllexport)
#else
#define export
#endif

#define STRMATCH(a,b)           (!strcmp((a),(b)))   /*  mv style  */
#define SV_STRMATCH(a,b)        (!strcmp((a),(b)))   /*  sv style  */
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */


 /*
 *              Device Type Definitions
 */

#define SV_DEVTYPE_UNKNOWN             0
#define SV_DEVTYPE_SCSIVIDEO_1         1 
#define SV_DEVTYPE_SCSIVIDEO_2         2 
#define SV_DEVTYPE_PRONTOVIDEO         3 
#define SV_DEVTYPE_PRONTOVIDEO_RGB     4 
#define SV_DEVTYPE_PRONTOVIDEO_PICO    5
#define SV_DEVTYPE_PCISTUDIO           6
#define SV_DEVTYPE_CLIPSTATION         7
#define SV_DEVTYPE_MOVIEVIDEO          8
#define SV_DEVTYPE_PRONTOVISION        9
#define SV_DEVTYPE_PRONTOSERVER        10
#define SV_DEVTYPE_CLIPBOARD           11
#define SV_DEVTYPE_HDSTATIONPRO        12
#define SV_DEVTYPE_HDBOARD             13
#define SV_DEVTYPE_SDSTATIONPRO        14
#define SV_DEVTYPE_SDBOARD             15
#define SV_DEVTYPE_HDXWAY              16
#define SV_DEVTYPE_SDXWAY              17
#define SV_DEVTYPE_CLIPSTER            18
#define SV_DEVTYPE_CENTAURUS           19
#define SV_DEVTYPE_HYDRA               20
#define SV_DEVTYPE_DIGILAB             21
#define SV_DEVTYPE_ATOMIX              22
#define SV_DEVTYPE_HYDRAX              23
//#define SV_DEVTYPE_RESERVED            24   // internal use
#define SV_DEVTYPE_ATOMIXLT            25
    
#define SV_DEVMODE_UNKNOWN             0
#define SV_DEVMODE_BLACK               1
#define SV_DEVMODE_LIVE                2
#define SV_DEVMODE_DISPLAY             3
#define SV_DEVMODE_RECORD              4
#define SV_DEVMODE_VTRLOAD             5
#define SV_DEVMODE_VTRSAVE             6
#define SV_DEVMODE_VTREDIT             7
#define SV_DEVMODE_TIMELINE            8
#define SV_DEVMODE_COLORBAR            9

#define SV_DEVSTATE_UNKNOWN            0
#define SV_DEVSTATE_START              1
#define SV_DEVSTATE_RUNNING            2
#define SV_DEVSTATE_READY              3
#define SV_DEVSTATE_ERROR              4

/*
//              Licence Management 
*/
#define SV_LICENCETYPE_ORIGINAL         0
#define SV_LICENCETYPE_CSPLICENCE       1
#define SV_LICENCETYPE_LICENCED         2
#define SV_LICENCETYPE_LICENCEBITS      3

#define SV_LICENCED_DISKRECORDER        0x00000001  // Clipster-DDR
#define SV_LICENCED_CLIPSTER            0x00000002  // Clipster
#define SV_LICENCED_OEM                 0x00000004  // OEM Allowed, if not not fifoapi / openml disabled
#define SV_LICENCED_MULTIDEVICE         0x00000008  // Multidevice link for Clipster
#define SV_LICENCED_AUDIO               0x00000010  // Enables all audio channels
#define SV_LICENCED_HIRES               0x00000020  // Enables Analog >= 165MHz and DVI Duallink
#define SV_LICENCED_SDTV                0x00000040  // Enables PAL,PALHR,NTSC,NTSCHR,...
#define SV_LICENCED_HDTV                0x00000080  // Enables all HD SDI 74.25 and 74.25/1.001 rasters
#define SV_LICENCED_FILM2K              0x00000100  // Enables > 2000 x 1300 
#define SV_LICENCED_FILM2KPLUS          0x00000200  // Enables > 2000 x 1300 10bit RGB
#define SV_LICENCED_FILM4K              0x00000400  // Enables > 3000 x 2000 
#define SV_LICENCED_CUSTOMRASTER        0x00000800  // Enables custom rasters
#define SV_LICENCED_HSDL                0x00001000  // Enables Dual link film2k transfer <= 20 Hz
#define SV_LICENCED_HSDL4K              0x00002000  // Enables Dual link film4k transfer
#define SV_LICENCED_12BIT               0x00004000  // Enables support of 12 Bit iomodes
#define SV_LICENCED_PROCESSING          0x00008000  // Enables subpixel / interpolating zoomandpan
#define SV_LICENCED_MIXER               0x00010000  // Enables mixer
#define SV_LICENCED_EVALUATION          0x00020000  // Evaluation unit
#define SV_LICENCED_ZOOMANDPAN          0x00040000  // Enables pixel / pixel duplication zoom and pan
#define SV_LICENCED_COLORCORRECTOR      0x00080000  // Enables colorcorrector in Clipster
#define SV_LICENCED_HDTVKEYCHANNEL      0x00100000  // Enables the keychannel in HDTV 
#define SV_LICENCED_HDTVDUALLINK        0x00200000  // Enables duallink mode in HDTV 
#define SV_LICENCED_SDTVKEYCHANNEL      0x00400000  // Enables the keychannel in SDTV 
#define SV_LICENCED_SDTVDUALLINK        0x00800000  // Enables duallink mode in SDTV 
#define SV_LICENCED_AUTOCONFORMING      0x01000000  // Enables Autoconforming
#define SV_LICENCED_COLORMANAGEMENT     0x02000000  // Enables ColorManagement
#define SV_LICENCED_SGI                 0x04000000  // Custom licence bit
#define SV_LICENCED_2K_1080PLAY         0x08000000  // Enables 2048x1080 playout
#define SV_LICENCED_DVI16               0x10000000  // Enables DVI up to 16 bit

#define SV_OPSYS_WINDOWS                0x01
#define SV_OPSYS_LINUX                  0x02
#define SV_OPSYS_IRIX                   0x04
#define SV_OPSYS_MACOS                  0x08
#define SV_OPSYS_SOLARIS                0x10


#define SV_LICENCEBIT_OPSYS_WINDOWS             0
#define SV_LICENCEBIT_OPSYS_LINUX               1
#define SV_LICENCEBIT_OPSYS_IRIX                2
#define SV_LICENCEBIT_OPSYS_MACOS               3
#define SV_LICENCEBIT_OPSYS_SOLARIS             4
#define SV_LICENCEBIT_OEM                       5           // OEM Allowed, if not not fifoapi / openml disabled
#define SV_LICENCEBIT_QUICKTIME                 6           // 

#define SV_LICENCEBIT_HW_AUDIO                  16          // Enables 8 (mono) audio channels
#define SV_LICENCEBIT_HW_AUDIO16                17          // Enables 16 (mono) audio channels
#define SV_LICENCEBIT_HW_COLORMANAGEMENT        18
#define SV_LICENCEBIT_HW_HIRES                  19          // Enables Analog >= 165MHz and DVI Duallink
#define SV_LICENCEBIT_HW_SDTV                   20          // Enables PAL,PALHR,NTSC,NTSCHR,...
#define SV_LICENCEBIT_HW_HDTV                   21          // Enables all HD SDI 74.25 and 74.25/1.001 rasters
#define SV_LICENCEBIT_HW_PHDTV                  22
#define SV_LICENCEBIT_HW_PHDTVARRI              23
#define SV_LICENCEBIT_HW_FILM2K                 24          // Enables > 2000 x 1300 
#define SV_LICENCEBIT_HW_FILM4K                 25          // Enables > 3000 x 2000 
#define SV_LICENCEBIT_HW_FILM6K                 26          // Enables > 5000 x 4000 
#define SV_LICENCEBIT_HW_CUSTOMRASTER           27          // Enables custom rasters
#define SV_LICENCEBIT_HW_HSDL                   28          // Enables Dual link film2k transfer <= 20 Hz
#define SV_LICENCEBIT_HW_HSDL4K                 29          // Enables Dual link film2k transfer <= 20 Hz
#define SV_LICENCEBIT_HW_12BIT                  30          // Enables support of 12 Bit iomodes
#define SV_LICENCEBIT_HW_PROCESSING             31          // Enables subpixel / interpolating zoomandpan
#define SV_LICENCEBIT_HW_MIXER                  32          // Enables mixer
#define SV_LICENCEBIT_HW_MIXEREE                33          // Enables mixer
#define SV_LICENCEBIT_HW_MIXERREMOTE            34          // Enables mixer
#define SV_LICENCEBIT_HW_ZOOMANDPAN             35          // Enables pixel / pixel duplication zoom and pan
#define SV_LICENCEBIT_HW_ZOOMANDPANREMOTE       36          // Enables pixel / pixel duplication zoom and pan
#define SV_LICENCEBIT_HW_HDTVKEYCHANNEL         37          // Enables the keychannel in HDTV 
#define SV_LICENCEBIT_HW_HDTVDUALLINK           38          // Enables duallink mode in HDTV 
#define SV_LICENCEBIT_HW_SDTVKEYCHANNEL         39          // Enables the keychannel in SDTV 
#define SV_LICENCEBIT_HW_SDTVDUALLINK           40          // Enables duallink mode in SDTV 
#define SV_LICENCEBIT_HW_EUREKA1980X1152        41
#define SV_LICENCEBIT_HW_SLOWPAL                42
#define SV_LICENCEBIT_HW_DVI16                  43
#define SV_LICENCEBIT_HW_SDTVFF                 44
#define SV_LICENCEBIT_HW_DVIINPUT               45
#define SV_LICENCEBIT_HW_CINE4KRASTER           46
#define SV_LICENCEBIT_HW_MULTICHANNEL           47
#define SV_LICENCEBIT_HW_LINKENCRYPT            48
#define SV_LICENCEBIT_HW_ANCCOMPLETE            49
#define SV_LICENCEBIT_HW_JPEG2000RAW            50
#define SV_LICENCEBIT_HW_RENDER                 51
#define SV_LICENCEBIT_HW_JPEG2000ENC4K          52
#define SV_LICENCEBIT_HW_JPEG2000CODEC4K        53
#define SV_LICENCEBIT_HW_UPSCALER               54



#define SV_LICENCEBIT_CLIPSTER_VENICE           61          // Venice Enable bit
#define SV_LICENCEBIT_CLIPSTER_DCISTATION       62          // DCI Station Enable bit
#define SV_LICENCEBIT_CLIPSTER_SPYCERBOX        63          // SpycerBox Enable bit
#define SV_LICENCEBIT_CLIPSTER_PRONTO           64          // Pronto Enable bit
#define SV_LICENCEBIT_CLIPSTER_CLIPSTER         65          // Clipster Enable bit
#define SV_LICENCEBIT_CLIPSTER_AUTOCONFORMING   66          // Enables Autoconforming
#define SV_LICENCEBIT_CLIPSTER_CCPRIMARY        67          // Enables colorcorrector in Clipster
#define SV_LICENCEBIT_CLIPSTER_CCSECONDARY      68
#define SV_LICENCEBIT_CLIPSTER_EVALUATION       69          // Evaluation Device
#define SV_LICENCEBIT_CLIPSTER_MULTIDEVICE      70          // Multidevice link for Clipster
#define SV_LICENCEBIT_CLIPSTER_PLUGINCOIPP      71
#define SV_LICENCEBIT_CLIPSTER_PLUGINOPENFX     72
#define SV_LICENCEBIT_CLIPSTER_VARIFRAME        73
#define SV_LICENCEBIT_CLIPSTER_VTRTIMELINE      74          // VTR Emulation Timeline
#define SV_LICENCEBIT_CLIPSTER_WORKFLOW4K       75
#define SV_LICENCEBIT_CLIPSTER_WORKFLOW6K       76
#define SV_LICENCEBIT_CLIPSTER_WORKFLOW8K       77
#define SV_LICENCEBIT_CLIPSTER_METADATA         78
#define SV_LICENCEBIT_CLIPSTER_DVSCOPY          79
#define SV_LICENCEBIT_CLIPSTER_DVSSAN           80
#define SV_LICENCEBIT_CLIPSTER_PANELJLCOOPER    81
#define SV_LICENCEBIT_CLIPSTER_PANELJLCOOPERCC  82
#define SV_LICENCEBIT_CLIPSTER_PANELTANGENT     83
#define SV_LICENCEBIT_CLIPSTER_PANELTANGENTCC   84
#define SV_LICENCEBIT_CLIPSTER_SOFTWAREWIPES    85
#define SV_LICENCEBIT_CLIPSTER_PRERELEASE       86
#define SV_LICENCEBIT_CLIPSTER_DCIMASTERING     87
#define SV_LICENCEBIT_CLIPSTER_CAPTURETOOL      88
#define SV_LICENCEBIT_CLIPSTER_CONFORMINGTOOL   89
#define SV_LICENCEBIT_CLIPSTER_OBSOLETE1        90
#define SV_LICENCEBIT_CLIPSTER_OBSOLETE2        91
#define SV_LICENCEBIT_CLIPSTER_CODEC_H264       92
#define SV_LICENCEBIT_CLIPSTER_CODEC_VC1        93
#define SV_LICENCEBIT_CLIPSTER_TIMELINEMARKERS  94
#define SV_LICENCEBIT_CLIPSTER_CODEC_DNXHD      95
#define SV_LICENCEBIT_CLIPSTER_EXTENDEDMXF      96
#define SV_LICENCEBIT_CLIPSTER_CODEC_STANDARD   97
#define SV_LICENCEBIT_CLIPSTER_CODEC_ENHANCED   98
#define SV_LICENCEBIT_CLIPSTER_CODEC_AVCINTRA   99
#define SV_LICENCEBIT_CLIPSTER_FFMPEG_DNXHD     100
#define SV_LICENCEBIT_CLIPSTER_FFMPEG_DVCPRO    101
#define SV_LICENCEBIT_CLIPSTER_FFMPEG_MPEG      102
#define SV_LICENCEBIT_CLIPSTER_WORKFLOWSTEREO   103
#define SV_LICENCEBIT_CLIPSTER_BURNIN           104
#define SV_LICENCEBIT_CLIPSTER_SOFTWARE3x0      105
#define SV_LICENCEBIT_CLIPSTER_SOAP             106
#define SV_LICENCEBIT_CLIPSTER_SOAP_LDAP        107
#define SV_LICENCEBIT_CLIPSTER_SOAP_PASSWORD    108
#define SV_LICENCEBIT_CLIPSTER_SOAP_NU          109
#define SV_LICENCEBIT_CLIPSTER_SUBTITLING       110
#define SV_LICENCEBIT_CLIPSTER_ARRIRAW          111
#define SV_LICENCEBIT_CLIPSTER_PHANTOMRAW       112
#define SV_LICENCEBIT_CLIPSTER_PANASONIC_AVC    113
#define SV_LICENCEBIT_CLIPSTER_CODEC_PRORES     114
#define SV_LICENCEBIT_CLIPSTER_CODEC_CINEFORM   115
#define SV_LICENCEBIT_CLIPSTER_SUBTITLING_3D    116

/* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// There are further private licence bits in dvs_private.h
*/


 
/*
//              Licence Management ClipStationPro/HDStationPro/SDStationPro 
*/
#define SV_CSPLICENCE_VALID             0x00000001
#define SV_CSPLICENCE_HD360             0x00000002
#define SV_CSPLICENCE_ODETICS           0x00000004
#define SV_CSPLICENCE_BETA              0x00000008
#define SV_CSPLICENCE_STREAMER          0x00000010
#define SV_CSPLICENCE_MULTICHANNEL      0x00000020
#define SV_CSPLICENCE_CLIPMANAGEMENT    0x00000040
#define SV_CSPLICENCE_PULLDOWN          0x00000080
#define SV_CSPLICENCE_LOUTHVDCP         0x00000100
#define SV_CSPLICENCE_MULTIDEVICE       0x00000200
#define SV_CSPLICENCE_RGBSUPPORT        0x00000400
#define SV_CSPLICENCE_DISKRECORDER      0x00000800
#define SV_CSPLICENCE_AUDIO1            0x00001000
#define SV_CSPLICENCE_AUDIO2            0x00002000
#define SV_CSPLICENCE_AUDIO_GET(x)      (((x>>12)&3)?(1<<((x>>12)&3)):0)
#define SV_CSPLICENCE_KEYCHANNEL        0x00004000
#define SV_CSPLICENCE_MIXER             0x00008000
#define SV_CSPLICENCE_DUALLINK          0x00010000
#define SV_CSPLICENCE_SDTV              0x00020000
#define SV_CSPLICENCE_FILM2K            0x00040000
#define SV_CSPLICENCE_OSFS              0x00080000
#define SV_CSPLICENCE_FILM2KPLUS        0x00100000
#define SV_CSPLICENCE_HSDL              0x00200000
#define SV_CSPLICENCE_TILEMODE          0x00400000
#define SV_CSPLICENCE_IRIXOEM           0x00800000
#define SV_CSPLICENCE_IRIXSGI           0x01000000
/*
//              Licence Bits for Cine Control Set Capture Software 
*/
#define SV_CSPLICENCE_CINECONTROL       0x02000000


#define SV_CSPLICENCE_RAMSIZE_SIZE(x)       ((32|((x>>1)&24))<<(x&15))
#define SV_CSPLICENCE_DISKSIZE_SIZE(x)      ((16|((x>>4)&15))<<((x&15)+9))
#define SV_CSPLICENCE_DISKSIZE_LIMITED(x)   (x != 255)

 
/*
//              Licence Management ProntoVideo
*/
#define SV_LICENCE_FLAG_MAIN            0x00000001
#define SV_LICENCE_FLAG_VTR_MASTER      0x00000002
#define SV_LICENCE_FLAG_VTR_SLAVE       0x00000004
#define SV_LICENCE_FLAG_D5              0x00000008
#define SV_LICENCE_FLAG_JPEG            0x00000010
#define SV_LICENCE_FLAG_AUDIO           0x00000020
#define SV_LICENCE_FLAG_LTC             0x00000040
#define SV_LICENCE_FLAG_KEY             0x00000080
#define SV_LICENCE_FLAG_ETHERNET        0x00000100
#define SV_LICENCE_FLAG_RASTER          0x00000200
#define SV_LICENCE_FLAG_10BIT           0x00000400
#define SV_LICENCE_FLAG_ULTRA           0x00000800
#define SV_LICENCE_FLAG_RGB             0x00001000
#define SV_LICENCE_FLAG_SERVER          0x00002000
#define SV_LICENCE_FLAG_HOTKEY          0x00004000
#define SV_LICENCE_FLAG_HOPPER          0x00008000
#define SV_LICENCE_FLAG_PULLDOWN        0x00010000
#define SV_LICENCE_FLAG_CLIP            0x00020000
#define SV_LICENCE_FLAG_MRES            0x40000000
#define SV_LICENCE_FLAG_SPECIAL         0x80000000


/*
//              Video modes
*/

#define SV_MODE_MASK              0x00001FFF    /* dropframe bit must be included       */
#define SV_MODE_FLAG_DROPFRAME    0x00001000    /* divide raster parameters by 1/1.001  */
#define SV_MODE_FLAG_RESERVED     0x00002000    /* reserved for future expansion        */
#define SV_MODE_FLAG_PACKED       0x00004000
#define SV_MODE_FLAG_RASTERINDEX  0x00008000    /* mode is number instead of sv index   */
#define SV_MODE_RASTERMASK        0x00009fff

#define SV_STORAGEMODE_BLACKLINE  0x00008000

#define SV_FORCEDETECT_ENABLE     0x00004000

#define SV_MODE_PAL               0x00  /* pal          720x576 25.00hz Interlaced        */
#define SV_MODE_NTSC              0x01  /* ntsc         720x486 29.97hz Interlaced        */
#define SV_MODE_PALHR             0x02  /* pal          960x576 25.00hz Interlaced        */
#define SV_MODE_NTSCHR            0x03  /* ntsc         960x486 29.97hz Interlaced        */
#define SV_MODE_PALFF             0x04  /* pal          720x592 25.00hz Interlaced        */
#define SV_MODE_NTSCFF            0x05  /* ntsc         720x502 29.97hz Interlaced        */
#define SV_MODE_PAL608            0x06  /* pal608       720x608 25.00hz Interlaced        */
//#define SV_MODE_UNUSED            0x07  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x08  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x09  /* OBSOLETE */
#define SV_MODE_HD360             0x0a  /* HD360        960x504 29.97hz Compressed HDTV   */
#define SV_MODE_SMPTE293_59P      0x0b  /* SMPTE293/59P 720x483 59.94hz Progressive       */
#define SV_MODE_PAL_24I           0x0c  /* SLOWPAL      720x576 24.00hz Interlaced        */
#define SV_MODE_TEST              0x0d  /*              Test Raster                       */
#define SV_MODE_VESASDI_1024x768_60P  0x0e
//#define SV_MODE_UNUSED            0x0f  /* OBSOLETE */

#define SV_MODE_PAL_25P           0x10  /* pal          25Hz (1:1)                        */
#define SV_MODE_PAL_50P           0x11  /* pal          50Hz (1:1)                        */
#define SV_MODE_PAL_100P          0x12  /* pal         100Hz (1:1)                        */
#define SV_MODE_NTSC_29P          (0x13 | SV_MODE_FLAG_DROPFRAME) /* ntsc         29.97Hz (1:1) */
#define SV_MODE_NTSC_59P          (0x14 | SV_MODE_FLAG_DROPFRAME) /* ntsc         59.94Hz (1:1) */
#define SV_MODE_NTSC_119P         (0x15 | SV_MODE_FLAG_DROPFRAME) /* ntsc        119.88Hz (1:1) */
//#define SV_MODE_UNUSED            0x16  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x17  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x18  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x19  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x1A  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x1B  /* OBSOLETE */
//#define SV_MODE_UNUSED            0x1C  /* OBSOLETE */

#define SV_MODE_SMPTE274_25sF     0x1d  /*              1920x1080 25.00hz Segmented Frame */
#define SV_MODE_SMPTE274_29sF     0x1e  /*              1920x1080 29.97hz Segmented Frame */
#define SV_MODE_SMPTE274_30sF     0x1f  /*              1920x1080 30.00hz Segmented Frame */

#define SV_MODE_EUREKA            0x20
#define SV_MODE_SMPTE240_30I      0x21  /*              1920x1035 30.00hz Interlaced      */
#define SV_MODE_SMPTE274_30I      0x22  /*              1920x1038 30.00hz Interlaced      */
#define SV_MODE_SMPTE296_60P      0x23  /*               1280x720 60.00hz Progressive     */
#define SV_MODE_SMPTE240_29I      0x24  /*              1920x1035 29.97hz Interlaced      */
#define SV_MODE_SMPTE274_29I      0x25  /*              1920x1080 29.97hz Interlaced      */
#define SV_MODE_SMPTE296_59P      0x26  /*               1280x720 59.94hz Progressive     */
#define SV_MODE_SMPTE295_25I      0x27  /*              1920x1080 1250/25Hz Interlaced    */
#define SV_MODE_SMPTE274_25I      0x28  /*              1920x1080 25.00hz Interlaced      */
#define SV_MODE_SMPTE274_24sF     0x29  /*              1920x1080 24.00hz Segmented Frame */
#define SV_MODE_SMPTE274_23sF     0x2A  /*              1920x1080 23.98hz Segmented Frame */
#define SV_MODE_SMPTE274_24P      0x2B  /*              1920x1080 24.00hz Progressive     */
#define SV_MODE_SMPTE274_23P      0x2C  /*              1920x1080 23.98hz Progressive     */

#define SV_MODE_SMPTE274_25P      0x2D  /*              1920x1080 25.00hz Progressive     */
#define SV_MODE_SMPTE274_29P      0x2E  /*              1920x1080 29.97hz Progressive     */
#define SV_MODE_SMPTE274_30P      0x2F  /*              1920x1080 30.00hz Progressive     */

//#define SV_MODE_UNUSED            0x30  /* OBSOLETE */
#define SV_MODE_SMPTE296_72P      0x31  /*              1280x720 72.00hz Progressive      */
#define SV_MODE_SMPTE296_71P      0x32  /*              1280x720 71.93hz Progressive      */
#define SV_MODE_SMPTE296_72P_89MHZ 0x33  /*             1280x720 72.00hz Progressive Analog 89 Mhz */
#define SV_MODE_SMPTE296_71P_89MHZ 0x34  /*             1280x720 71.93hz Progressive Analog 89 Mhz */

#define SV_MODE_SMPTE274_23I       0x35  /*             1920x1080 23.98hz Interlaced      */
#define SV_MODE_SMPTE274_24I       0x36  /*             1920x1080 24.00hz Interlaced      */

#define SV_MODE_SMPTE274_47P       0x37  /*             1920x1080 47.95hz Progressive     */
#define SV_MODE_SMPTE274_48P       0x38  /*             1920x1080 48.00hz Progressive     */
#define SV_MODE_SMPTE274_59P       0x39  /*             1920x1080 59.94hz Progressive     */
#define SV_MODE_SMPTE274_60P       0x3a  /*             1920x1080 60.00hz Progressive     */
#define SV_MODE_SMPTE274_71P       0x3b  /*             1920x1080 71.93hz Progressive     */
#define SV_MODE_SMPTE274_72P       0x3c  /*             1920x1080 72.00hz Progressive     */

#define SV_MODE_SMPTE274_2560_24P  0x3d
#define SV_MODE_SMPTE274_2560_23P  (0x3d | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_SMPTE296_24P_30MHZ 0x3e
#define SV_MODE_SMPTE296_24P       0x3f
#define SV_MODE_SMPTE296_23P       (0x3f | SV_MODE_FLAG_DROPFRAME)

#define SV_MODE_FILM2K_1556_12P         0x40
#define SV_MODE_FILM2K_1556_6P          0x41
#define SV_MODE_FILM2K_1556_3P          0x42

#define SV_MODE_FILM2K_2048x1536_24P    0x45  /* Telecine formats with 4:3 aspect ratio   */
#define SV_MODE_FILM2K_2048x1536_24sF   0x46
#define SV_MODE_FILM2K_2048x1536_48P    0x47

#define SV_MODE_FILM2K_1536_24P         SV_MODE_FILM2K_2048x1536_24P
#define SV_MODE_FILM2K_1536_24sF        SV_MODE_FILM2K_2048x1536_24sF
#define SV_MODE_FILM2K_1536_48P         SV_MODE_FILM2K_2048x1536_48P

#define SV_MODE_FILM2K_2048x1556_24P    0x48
#define SV_MODE_FILM2K_2048x1556_23P    (0x48 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1556_24sF   0x49
#define SV_MODE_FILM2K_2048x1556_23sF   (0x49 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1556_48P    0x4a

#define SV_MODE_FILM2K_1556_24P         SV_MODE_FILM2K_2048x1556_24P
#define SV_MODE_FILM2K_1556_24sF        SV_MODE_FILM2K_2048x1556_24sF
#define SV_MODE_FILM2K_1556_48P         SV_MODE_FILM2K_2048x1556_48P

#define SV_MODE_FILM2K_2048x1556_25sF   0x4b
//#define SV_MODE_UNUSED            0x4c  /* OBSOLETE was palfield/ntscfield modes */
//#define SV_MODE_UNUSED            0x4d  /* OBSOLETE was palfield/ntscfield modes */
//#define SV_MODE_UNUSED            0x4e  /* OBSOLETE was palfield/ntscfield modes */

#define SV_MODE_SGI_1280x1024_NTSC 0x4f
#define SV_MODE_SGI_1280x1024_59P  0x50
#define SV_MODE_SGI_1280x2048_29I  0x51

#define SV_MODE_FILM2K_2048x1556_14sF   0x52
#define SV_MODE_FILM2K_2048x1556_15sF   0x53

#define SV_MODE_FILM2K_1556_14sF        SV_MODE_FILM2K_2048x1556_14sF
#define SV_MODE_FILM2K_1556_15sF        SV_MODE_FILM2K_2048x1556_15sF

#define SV_MODE_VESA_1024x768_30I  0x54
#define SV_MODE_VESA_1024x768_29I  (0x54 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1280x1024_30I 0x55
#define SV_MODE_VESA_1280x1024_29I (0x55 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1600x1200_30I 0x56
#define SV_MODE_VESA_1600x1200_29I (0x56 | SV_MODE_FLAG_DROPFRAME)

#define SV_MODE_VESA_640x480_60P   0x57
#define SV_MODE_VESA_640x480_59P   (0x57 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_800x600_60P   0x58
#define SV_MODE_VESA_800x600_59P   (0x58 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1024x768_60P  0x59
#define SV_MODE_VESA_1024x768_59P  (0x59 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1280x1024_60P 0x5a
#define SV_MODE_VESA_1280x1024_59P (0x5a | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1600x1200_60P 0x5b
#define SV_MODE_VESA_1600x1200_59P (0x5b | SV_MODE_FLAG_DROPFRAME)

#define SV_MODE_VESA_640x480_72P   0x5c
#define SV_MODE_VESA_640x480_71P   (0x5c | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_800x600_72P   0x5d
#define SV_MODE_VESA_800x600_71P   (0x5d | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1024x768_72P  0x5e
#define SV_MODE_VESA_1024x768_71P  (0x5e | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1280x1024_72P 0x5f
#define SV_MODE_VESA_1280x1024_71P (0x5f | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_VESA_1600x1200_72P 0x60
#define SV_MODE_VESA_1600x1200_71P (0x60 | SV_MODE_FLAG_DROPFRAME)

#define SV_MODE_STREAMER           0x61  
#define SV_MODE_STREAMERDF         0x62
#define SV_MODE_STREAMERSD         0x63

#define SV_MODE_SMPTE296_25P       0x64
#define SV_MODE_SMPTE296_29P       (0x65 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_SMPTE296_30P       0x65
#define SV_MODE_SMPTE296_50P       0x66

#define SV_MODE_FILM2K_2048x1556_29sF     (0x67 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1556_30sF     (0x67)
#define SV_MODE_FILM2K_2048x1556_36sF     (0x68)

#define SV_MODE_FILM2K_1556_29sF          SV_MODE_FILM2K_2048x1556_29sF
#define SV_MODE_FILM2K_1556_30sF          SV_MODE_FILM2K_2048x1556_30sF
#define SV_MODE_FILM2K_1556_36sF          SV_MODE_FILM2K_2048x1556_36sF


#define SV_MODE_FILM2K_2048x1080_23sF     (0x69 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1080_24sF     (0x69)
#define SV_MODE_FILM2K_2048x1080_23P      (0x6a | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1080_24P      (0x6a)

#define SV_MODE_FILM2K_1080_23sF          SV_MODE_FILM2K_2048x1080_23sF
#define SV_MODE_FILM2K_1080_24sF          SV_MODE_FILM2K_2048x1080_24sF
#define SV_MODE_FILM2K_1080_23P           SV_MODE_FILM2K_2048x1080_23P
#define SV_MODE_FILM2K_1080_24P           SV_MODE_FILM2K_2048x1080_24P

#define SV_MODE_FILM2K_2048x1556_19sF     (0x6b | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1556_20sF     (0x6b)

#define SV_MODE_FILM2K_1556_19sF          SV_MODE_FILM2K_2048x1556_19sF
#define SV_MODE_FILM2K_1556_20sF          SV_MODE_FILM2K_2048x1556_20sF

#define SV_MODE_1980x1152_25I             (0x6c)

#define SV_MODE_SMPTE274_50P              (0x6d)

#define SV_MODE_FILM4K_4096x2160_24sF   0x6e
#define SV_MODE_FILM4K_4096x2160_24P   	0x6f

#define SV_MODE_FILM4K_2160_24sF        SV_MODE_FILM4K_4096x2160_24sF
#define SV_MODE_FILM4K_2160_24P         SV_MODE_FILM4K_4096x2160_24P

#define SV_MODE_3840x2400_24sF     0x70
#define SV_MODE_3840x2400_24P      0x71

#define SV_MODE_FILM4K_3112_24sF   0x72
#define SV_MODE_FILM4K_3112_24P    0x73

#define SV_MODE_FILM4K_3112_5sF    0x74

#define SV_MODE_3840x2400_12P      0x75

#define SV_MODE_ARRI_1920x1080_47P (0x76 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_ARRI_1920x1080_48P 0x76
#define SV_MODE_ARRI_1920x1080_50P 0x77
#define SV_MODE_ARRI_1920x1080_59P (0x78 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_ARRI_1920x1080_60P 0x78

#define SV_MODE_SMPTE296_100P      0x79
#define SV_MODE_SMPTE296_119P      (0x7a | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_SMPTE296_120P      0x7a

#define SV_MODE_1920x1200_24P      0x7b
#define SV_MODE_1920x1200_60P      0x7c

#define SV_MODE_WXGA_1366x768_50P  0x7d
#define SV_MODE_WXGA_1366x768_59P  (0x7e | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_WXGA_1366x768_60P  0x7e
#define SV_MODE_WXGA_1366x768_90P  0x7f
#define SV_MODE_WXGA_1366x768_120P 0x80

#define SV_MODE_1400x1050_60P      0x81

#define SV_MODE_FILM2K_2048x858_23sF    (0x82 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x858_24sF    0x82
#define SV_MODE_FILM2K_1998x1080_23sF   (0x83 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_1998x1080_24sF   0x83

#define SV_MODE_ANALOG_1920x1080_47P  (0x84 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_ANALOG_1920x1080_48P   0x84
#define SV_MODE_ANALOG_1920x1080_50P   0x85
#define SV_MODE_ANALOG_1920x1080_59P  (0x86 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_ANALOG_1920x1080_60P   0x86

#define SV_MODE_QUADDVI_3840x2160_48P   0x87
#define SV_MODE_QUADDVI_3840x2160_48Pf2 0x88
#define SV_MODE_QUADDVI_3840x2160_60P   0x89
#define SV_MODE_QUADDVI_3840x2160_60Pf2 0x8a

#define SV_MODE_QUADSDI_3840x2160_23P   (0x8b | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_QUADSDI_3840x2160_24P   (0x8b)
#define SV_MODE_QUADSDI_4096x2160_23P   (0x8c | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_QUADSDI_4096x2160_24P   (0x8c)

#define SV_MODE_FILM2K_2048x1080_25P      0x8d
#define SV_MODE_FILM2K_1080_25P           SV_MODE_FILM2K_2048x1080_25P


//#define SV_MODE_RESERVED              0x8e
//#define SV_MODE_RESERVED              0x8f
//#define SV_MODE_RESERVED              0x90
//#define SV_MODE_RESERVED              0x91
//#define SV_MODE_RESERVED              0x92
//#define SV_MODE_RESERVED              0x93
//#define SV_MODE_RESERVED              0x94
//#define SV_MODE_RESERVED              0x95

#define SV_MODE_FILM2K_2048x1744_24P	  (0x96)
#define SV_MODE_FILM2K_2048x1080_47P	  (0x97 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1080_48P	  (0x97)

//#define SV_MODE_RESERVED              0x98

#define SV_MODE_FILM2K_2048x1080_25sF     0x99
#define SV_MODE_FILM2K_1080_25sF          SV_MODE_FILM2K_2048x1080_25sF

//#define SV_MODE_RESERVED              0x9a
//#define SV_MODE_RESERVED              0x9b

#define SV_MODE_FILM2K_2048x1080_29P	  (0x9c | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1080_30P	  (0x9c) 
#define SV_MODE_FILM2K_2048x1080_50P	  (0x9d) 
#define SV_MODE_FILM2K_2048x1080_59P	  (0x9e | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x1080_60P	  (0x9e)

#define SV_MODE_FILM2K_2048x858_23P       (0x9f | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_2048x858_24P       (0x9f)
#define SV_MODE_FILM2K_1998x1080_23P      (0xa0 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_FILM2K_1998x1080_24P      (0xa0)

#define SV_MODE_ANALOG_2048x1080_50P	  (0xa1) 
#define SV_MODE_ANALOG_2048x1080_59P	  (0xa2 | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_ANALOG_2048x1080_60P	  (0xa2)

//#define SV_MODE_RESERVED              0xa3
//#define SV_MODE_RESERVED              0xa4
//#define SV_MODE_RESERVED              0xa5
//#define SV_MODE_RESERVED              0xa6
//#define SV_MODE_RESERVED              0xa7
//#define SV_MODE_RESERVED              0xa8
//#define SV_MODE_RESERVED              0xa9
//#define SV_MODE_RESERVED              0xaa
//#define SV_MODE_RESERVED              0xab
//#define SV_MODE_RESERVED              0xac

#define SV_MODE_QUADSDI_3840x2400_23P    (0xad | SV_MODE_FLAG_DROPFRAME)
#define SV_MODE_QUADSDI_3840x2400_24P     0xad
#define SV_MODE_QUADDVI_3840x2160_30P     0xae
#define SV_MODE_QUADDVI_3840x2400_24P     0xaf

#define SV_MODE_FILM2K_2048x1772_24P      0xb0

#define SV_MODE_MAX                       0xb1

#define SV_MODE_CUSTOM                    0xfff


#define SV_MODE_COLOR_MASK        0x001F0000
#define SV_MODE_COLOR_YUV422      0x00000000
#define SV_MODE_COLOR_YUV422_UYVY 0x00000000
#define SV_MODE_COLOR_RGBA        0x00010000
#define SV_MODE_COLOR_LUMA        0x00020000
#define SV_MODE_COLOR_CHROMA      0x00030000
//#define SV_MODE_COLOR_RGB         0x00040000    /* OBSOLETE */
#define SV_MODE_COLOR_RGB_BGR     0x00040000
#define SV_MODE_COLOR_YUV422A     0x00050000
#define SV_MODE_COLOR_YUV444      0x00060000
#define SV_MODE_COLOR_YUV444A     0x00070000
//#define SV_MODE_COLOR_RGBVIDEO    0x00080000  /* OBSOLETE */
//#define SV_MODE_COLOR_RGBAVIDEO   0x00090000  /* OBSOLETE */
#define SV_MODE_COLOR_YUV2QT      0x000A0000    /* OBSOLETE */
#define SV_MODE_COLOR_RGB_RGB     0x000B0000
#define SV_MODE_COLOR_BGRA        0x000C0000
#define SV_MODE_COLOR_YUV422_YUYV 0x000D0000
#define SV_MODE_COLOR_ARGB        0x000E0000
#define SV_MODE_COLOR_ABGR        0x000F0000
#define SV_MODE_COLOR_ALPHA       0x00100000
#define SV_MODE_COLOR_ALPHA_422A  0x00110000
#define SV_MODE_COLOR_ALPHA_444A  0x00120000
#define SV_MODE_COLOR_ALPHA_A444  0x00130000
#define SV_MODE_COLOR_BAYER_BGGR  0x00140000
#define SV_MODE_COLOR_XYZ         0x00150000
#define SV_MODE_COLOR_YCC         0x00160000
#define SV_MODE_COLOR_YCC422      0x00170000
#define SV_MODE_COLOR_YUV444_VYU  0x00180000
#define SV_MODE_COLOR_WHITE       0x00190000
#define SV_MODE_COLOR_BLACK       0x001a0000
#define SV_MODE_COLOR_BAYER_GBRG  0x001b0000
#define SV_MODE_COLOR_BAYER_GRBG  0x001c0000
#define SV_MODE_COLOR_BAYER_RGGB  0x001d0000

#define SV_MODE_AUDIO_MASK        0x00e00000
#define SV_MODE_AUDIO_NOAUDIO     0x00000000
#define SV_MODE_AUDIO_1CHANNEL    0x00400000 /* OBSOLETE */
#define SV_MODE_AUDIO_2CHANNEL    0x00800000 /* OBSOLETE */
#define SV_MODE_AUDIO_4CHANNEL    0x00c00000 /* OBSOLETE */
#define SV_MODE_AUDIO_6CHANNEL    0x00200000 /* OBSOLETE */
#define SV_MODE_AUDIO_8CHANNEL    0x00600000

#define SV_MODE_NBIT_MASK         0x2b000000
#define SV_MODE_NBIT_8B           0x00000000 
#define SV_MODE_NBIT_10BDVS       0x01000000
#define SV_MODE_NBIT_10B          0x02000000
#define SV_MODE_NBIT_10BRALE      0x02000000  /* Right Aligned - Little Endian (QT)*/
#define SV_MODE_NBIT_10BDPX       0x03000000
#define SV_MODE_NBIT_10BLABE      0x03000000  /* Left Aligned - Big Endian (Cineon / defacto dpx format, format 'B') */
#define SV_MODE_NBIT_12B          0x08000000  /* 12 bit 'simple' */
#define SV_MODE_NBIT_16BLE        0x09000000
#define SV_MODE_NBIT_16BBE        0x0a000000
#define SV_MODE_NBIT_10BLALE      0x0b000000  /* Left Aligned - Little Endian */
#define SV_MODE_NBIT_10BRABE      0x20000000  /* Right Aligned - BigEndian (DPX format 'A') */
#define SV_MODE_NBIT_12BDPX       0x21000000  /* 12 Bit DPX ordering */
#define SV_MODE_NBIT_8BSWAP       0x22000000  /* 32 bit swapped data dpxv2 */
#define SV_MODE_NBIT_10BRALEV2    0x23000000  /* component 1/3 swapped data dpxv2 */
#define SV_MODE_NBIT_10BLABEV2    0x28000000  /* component 1/3 swapped data dpxv2 */
#define SV_MODE_NBIT_10BLALEV2    0x29000000  /* component 1/3 swapped data dpxv2 */
#define SV_MODE_NBIT_10BRABEV2    0x2a000000  /* component 1/3 swapped data dpxv2 */
#define SV_MODE_NBIT_12BDPXLE     0x2b000000  /* 12 Bit DPX LE ordering */



#define SV_MODE_AUDIOBITS_MASK    0x04000000
#define SV_MODE_AUDIOBITS_32      0x00000000
#define SV_MODE_AUDIOBITS_16      0x04000000 /* OBSOLETE */

#define SV_MODE_ACTIVE_VIDEO      0x00000000
#define SV_MODE_RESERVEDBIT        0x10000000 /* reserved for future expansion        */
#define SV_MODE_STORAGE_FRAME      0x40000000
#define SV_MODE_STORAGE_BOTTOM2TOP 0x80000000


#define SV_NBITTYPE_8B              0x008
#define SV_NBITTYPE_8BSWAP          0x108
#define SV_NBITTYPE_10B             0x00a /* OBSOLETE */
#define SV_NBITTYPE_10BRALE         0x00a 
#define SV_NBITTYPE_10BDVS          0x10a
#define SV_NBITTYPE_10BDPX          0x20a /* OBSOLETE */
#define SV_NBITTYPE_10BLABE         0x20a
#define SV_NBITTYPE_10BRABE         0x30a
#define SV_NBITTYPE_10BLALE         0x40a
#define SV_NBITTYPE_10BLABEV2       0x50a
#define SV_NBITTYPE_10BLALEV2       0x60a
#define SV_NBITTYPE_10BRABEV2       0x70a
#define SV_NBITTYPE_10BRALEV2       0x80a 
#define SV_NBITTYPE_12B             0x00c
#define SV_NBITTYPE_12BDPX          0x10c
#define SV_NBITTYPE_12BDPXLE        0x20c
#define SV_NBITTYPE_16BLE           0x010
#define SV_NBITTYPE_16BBE           0x110
#define SV_NBITTYPE_NBITS(x)        ((x)&0xff)



/*
 *      Color modes - only used in sv_status() and sv_storage_status()
 */  
#define SV_COLORMODE_MONO               0x00
#define SV_COLORMODE_RGB                0x01    /* OBSOLETE */
#define SV_COLORMODE_RGB_BGR            0x01
#define SV_COLORMODE_YUV422             0x02
#define SV_COLORMODE_YUV411             0x03
#define SV_COLORMODE_YUV422A            0x04
#define SV_COLORMODE_RGBA               0x05
#define SV_COLORMODE_YUV422STEREO       0x06
#define SV_COLORMODE_YUV444             0x07
#define SV_COLORMODE_YUV444A            0x08
#define SV_COLORMODE_YUV420             0x09
#define SV_COLORMODE_RGBVIDEO           0x0a    /* OBSOLETE */
#define SV_COLORMODE_RGBAVIDEO          0x0b    /* OBSOLETE */
#define SV_COLORMODE_YUV2QT             0x0c
#define SV_COLORMODE_RGB_RGB            0x0d
#define SV_COLORMODE_BGRA               0x0e
#define SV_COLORMODE_YUV422_YUYV        0x0f
#define SV_COLORMODE_CHROMA             0x10
#define SV_COLORMODE_ARGB               0x11
#define SV_COLORMODE_ABGR               0x12
#define SV_COLORMODE_BAYER_RGGB         0x13
#define SV_COLORMODE_XYZ                0x14
#define SV_COLORMODE_YCC                0x15
#define SV_COLORMODE_YCC422             0x16
#define SV_COLORMODE_YUV444_VYU         0x17
#define SV_COLORMODE_BAYER_GRBG         0x18
#define SV_COLORMODE_BAYER_BGGR         0x19
#define SV_COLORMODE_BAYER_GBRG         0x1a

#define SV_YUVMATRIX_CCIR601            0
#define SV_YUVMATRIX_CCIR709            1
#define SV_YUVMATRIX_RGB                2
#define SV_YUVMATRIX_RGB_CGR            3

 
 /*
//  Used with sv_matrix().
*/
typedef struct {
  int matrixmode;           ///< Returns the matrix mode set, not used for setting.
  int matrix[10];           ///< Fixed point float matrix, i.e. set divisor to 0x10000.
  int dematrix[10];         ///< Fixed point float dematrix, i.e. set divisor to 0x10000.
  int divisor;              ///< Common divisor for all coefficient.
  int inputfilter;          ///< SV_INPUTFILTER_XXX
  int outputfilter;         ///< SV_OUTPUTFILTER_XXX
  int maxpreloaded;         ///< Max number of preloaded matrices.
  int spare[1];             ///< Reserved for future expansion.
} sv_matrixinfo;

typedef struct {
  int matrixtype;           ///< Returns the matrix mode set, not used for setting.
  int matrix[18];           ///< Fixed point float matrix, i.e. set divisor to 0x10000.
  int dematrix[18];         ///< Fixed point float dematrix, i.e. set divisor to 0x10000.
  int divisor;              ///< Common divisor for all coefficient.
  int spare[5];
} sv_matrixexinfo;


#define SV_MATRIXTYPE_DEFAULT           0
#define SV_MATRIXTYPE_RGBFULL           1
#define SV_MATRIXTYPE_RGBHEAD           2
#define SV_MATRIXTYPE_601FULL           3
#define SV_MATRIXTYPE_601HEAD           4
#define SV_MATRIXTYPE_274FULL           5
#define SV_MATRIXTYPE_274HEAD           6
#define SV_MATRIXTYPE_XYZ               7
#define SV_MATRIXTYPE_YCC               8
#define SV_MATRIXTYPE_P3RGBFULL         9
#define SV_MATRIXTYPE_P3YUVFULL        10
#define SV_MATRIXTYPE_P7RGBFULL        11
#define SV_MATRIXTYPE_P7YUVFULL        12

#define SV_MATRIX_DEFAULT               0x0000
#define SV_MATRIX_CUSTOM                0x0001
#define SV_MATRIX_QUERY                 0x0002
#define SV_MATRIX_CCIR601               0x0003
#define SV_MATRIX_CCIR601CGR            0x0004
#define SV_MATRIX_CCIR709               0x0005
#define SV_MATRIX_CCIR709CGR            0x0006
#define SV_MATRIX_SMPTE274              0x0007
#define SV_MATRIX_SMPTE274CGR           0x0008
#define SV_MATRIX_CCIR601INV            0x0009
#define SV_MATRIX_SMPTE274INV           0x000a
#define SV_MATRIX_274TO601              0x000b
#define SV_MATRIX_601TO274              0x000c
#define SV_MATRIX_IDENTITY              0x000d
#define SV_MATRIX_RGBHEAD2FULL          0x000e
#define SV_MATRIX_RGBFULL2HEAD          0x000f
#define SV_MATRIX_YUVHEAD2FULL          0x0010
#define SV_MATRIX_YUVFULL2HEAD          0x0011
#define SV_MATRIX_274FTO601H            0x0012
#define SV_MATRIX_274HTO601F            0x0013
#define SV_MATRIX_601FTO274H            0x0014
#define SV_MATRIX_601HTO274F            0x0015
#define SV_MATRIX_MASK                  0x00ff

#define SV_MATRIX_FLAG_CGRMATRIX        0x0100    /* Only evaluated for custom matrix, and  */
                                                  /* not evaluated by sv_matrixex(), as is in coefficients */
#define SV_MATRIX_FLAG_SETINPUTFILTER   0x0400    /* Not evaluated by sv_matrixex() */
#define SV_MATRIX_FLAG_SETOUTPUTFILTER  0x0800    /* Not evaluated by sv_matrixex() */
#define SV_MATRIX_FLAG_FORCEMATRIX      0x1000
#define SV_MATRIX_FLAG_MASK             0xff00

 
/*
//      Data modes, parameters to the sv_host2sv/sv_sv2host calls
*/
#define SV_TYPE_YUV422                  0x00000000
#define SV_TYPE_MONO                    0x00010000
#define SV_TYPE_RGB                     0x00020000 /* OBSOLETE */
#define SV_TYPE_RGB_BGR                 0x00020000
#define SV_TYPE_YUV422A                 0x00030000
#define SV_TYPE_RGBA                    0x00040000 /* OBSOLETE */
#define SV_TYPE_RGBA_RGBA               0x00040000
#define SV_TYPE_STREAMER                0x00050000
#define SV_TYPE_HEADER                  0x00060000 /* OBSOLETE */
#define SV_TYPE_YUV444                  0x00070000
#define SV_TYPE_YUV444A                 0x00080000
#define SV_TYPE_CHROMA                  0x00090000
#define SV_TYPE_YUV2QT                  0x000a0000
#define SV_TYPE_RGB_RGB                 0x000b0000
#define SV_TYPE_RGBA_BGRA               0x000c0000
#define SV_TYPE_YUV422_YUYV             0x000d0000
#define SV_TYPE_RGBA_ARGB               0x000e0000
#define SV_TYPE_RGBA_ABGR               0x000f0000
#define SV_TYPE_BAYER_BGGR              0x00100000
#define SV_TYPE_YUV444_VYU              0x00110000
#define SV_TYPE_BAYER_GBRG              0x00120000
#define SV_TYPE_BAYER_GRBG              0x00130000
#define SV_TYPE_BAYER_RGGB              0x00140000
#define SV_TYPE_AUDIO12                 0x01000000 
#define SV_TYPE_AUDIO34                 0x02000000
#define SV_TYPE_KEY                     0x03000000
#define SV_TYPE_AUDIO56                 0x04000000
#define SV_TYPE_AUDIO78                 0x05000000
#define SV_TYPE_AUDIO9a                 0x06000000
#define SV_TYPE_AUDIObc                 0x07000000
#define SV_TYPE_AUDIOde                 0x08000000
#define SV_TYPE_AUDIOfg                 0x09000000
#define SV_TYPE_AUDIOall                0x0a000000
#define SV_TYPE_MASK                    0xFFFF0000

#define SV_DATARANGE_SCALE              0x00001000
#define SV_DATARANGE_SCALE_KEY          0x00002000

#define SV_DATASIZE_8BIT                0x00000000
#define SV_DATASIZE_16BIT_BIG           0x00000001
#define SV_DATASIZE_16BIT_LITTLE        0x00000002
#define SV_DATASIZE_32BIT_BIG           0x00000003
#define SV_DATASIZE_32BIT_LITTLE        0x00000004
#define SV_DATASIZE_10BIT               0x00000005  /* OBSOLETE */
#define SV_DATASIZE_10BITRALE           0x00000005  
#define SV_DATASIZE_10BITDVS            0x00000006
#define SV_DATASIZE_10BITDPX            0x00000007  /* OBSOLETE */
#define SV_DATASIZE_10BITLABE           0x00000007
#define SV_DATASIZE_12BIT               0x00000008
#define SV_DATASIZE_10BITRABE           0x00000009
#define SV_DATASIZE_10BITLALE           0x0000000a
#define SV_DATASIZE_12BITDPX            0x0000000b
#define SV_DATASIZE_MASK                0x000000FF

#define SV_DATA_YUV_NORMAL              0x00000000  /* OBSOLETE */
#define SV_DATA_YUV_CLIP                0x00000100  /* OBSOLETE */
#define SV_DATA_YUV_SCALE               0x00000200  /* OBSOLETE */
#define SV_DATA_YUV_MASK                0x00000F00  /* OBSOLETE */

/*------------------------------------------------------------------------------------------------*/
#define SV_MASTER_NOP                   0
#define SV_MASTER_STOP                  1
#define SV_MASTER_EJECT                 2
#define SV_MASTER_RECORD                3
#define SV_MASTER_PLAY                  4
#define SV_MASTER_FORWARD               5
#define SV_MASTER_REWIND                6
#define SV_MASTER_STANDBY               7
#define SV_MASTER_LIVE                  8
#define SV_MASTER_PAUSE                 9
#define SV_MASTER_GOTO                  10
#define SV_MASTER_SYNC                  11    /* OBSOLETE */
#define SV_MASTER_ASYNC                 12    /* OBSOLETE */
#define SV_MASTER_PRESET                13
#define SV_MASTER_JOG                   14
#define SV_MASTER_SHUTTLE               15
#define SV_MASTER_STBOFF                16
#define SV_MASTER_RAW                   17
#define SV_MASTER_AUTOPARK              18
#define SV_MASTER_PARKTIME              19
#define SV_MASTER_PREROLL               20
#define SV_MASTER_POSTROLL              21
#define SV_MASTER_EDITLAG               22
#define SV_MASTER_TIMECODE              23
#define SV_MASTER_TOLERANCE             24
#define SV_MASTER_FORCEDROPFRAME        25
#define SV_MASTER_DEBUG                 26
#define SV_MASTER_RECOFFSET             27
#define SV_MASTER_DISOFFSET             28
#define SV_MASTER_MOVETO                29
#define SV_MASTER_STEP                  30
#define SV_MASTER_CODE                  31
#define SV_MASTER_FLAG                  32
#define SV_MASTER_VAR                   33
#define SV_MASTER_PREVIEW               34
#define SV_MASTER_AUTOEDIT              35
#define SV_MASTER_REVIEW                36
#define SV_MASTER_INPOINT               37
#define SV_MASTER_OUTPOINT              38
#define SV_MASTER_NFRAMES               39
#define SV_MASTER_EDITFIELD_START       40
#define SV_MASTER_EDITFIELD_END         41
#define SV_MASTER_AUTOEDITONOFF         42
#define SV_MASTER_DEVICETYPE            43
#define SV_MASTER_AUTOPLAY              44
#define SV_MASTER_AUTOEDITFLAGS         45
#define SV_MASTER_GOTOPREROLL           46
#define SV_MASTER_EDITTICK              47
#define SV_MASTER_RECORDING             48
#define SV_MASTER_STEP_MOVETO           49
#define SV_MASTER_STEP_GOTO             50

#define SV_MASTER_FORCEDROPFRAME_AUTO   0
#define SV_MASTER_FORCEDROPFRAME_ON     1
#define SV_MASTER_FORCEDROPFRAME_OFF    2

//#define SV_MASTER_TIMECODE_ASTC         0     /* OBSOLETE */
#define SV_MASTER_TIMECODE_VITC         0
#define SV_MASTER_TIMECODE_LTC          1
#define SV_MASTER_TIMECODE_AUTO         2     /* Ask for both VITC and LTC, ie let the vtr choose */
#define SV_MASTER_TIMECODE_TIMER1       3
#define SV_MASTER_TIMECODE_TIMER2       4


#define SV_MASTER_AUTOPARK_OFF          0
#define SV_MASTER_AUTOPARK_ON           1

#define SV_MASTER_FLAG_AUTOEDIT         0x0001
#define SV_MASTER_FLAG_FORCEDROPFRAME   0x0002
#define SV_MASTER_FLAG_EMULATESTEPCMD   0x0004


#define SV_MASTER_STANDBY_OFF           0
#define SV_MASTER_STANDBY_ON            1
                                        
#define SV_MASTER_TOLERANCE_NONE        0
#define SV_MASTER_TOLERANCE_NORMAL      1
#define SV_MASTER_TOLERANCE_LARGE       2
#define SV_MASTER_TOLERANCE_ROUGH       3

#define SV_MASTER_PRESET_VIDEO          0x00000001
#define SV_MASTER_PRESET_AUDIO1         0x00000002
#define SV_MASTER_PRESET_AUDIO2         0x00000004
#define SV_MASTER_PRESET_AUDIO3         0x00000008
#define SV_MASTER_PRESET_AUDIO4         0x00000010
#define SV_MASTER_PRESET_DIGAUDIO1      0x00000100
#define SV_MASTER_PRESET_DIGAUDIO2      0x00000200
#define SV_MASTER_PRESET_DIGAUDIO3      0x00000400
#define SV_MASTER_PRESET_DIGAUDIO4      0x00000800
#define SV_MASTER_PRESET_DIGAUDIO5      0x00001000
#define SV_MASTER_PRESET_DIGAUDIO6      0x00002000
#define SV_MASTER_PRESET_DIGAUDIO7      0x00004000
#define SV_MASTER_PRESET_DIGAUDIO8      0x00008000
#define SV_MASTER_PRESET_AUDIOMASK      0x0000001e
#define SV_MASTER_PRESET_ASSEMBLE       0x00010000
#define SV_MASTER_PRESET_DIGAUDIO9      0x00100000
#define SV_MASTER_PRESET_DIGAUDIO10     0x00200000
#define SV_MASTER_PRESET_DIGAUDIO11     0x00400000
#define SV_MASTER_PRESET_DIGAUDIO12     0x00800000
#define SV_MASTER_PRESET_DIGAUDIO13     0x01000000
#define SV_MASTER_PRESET_DIGAUDIO14     0x02000000
#define SV_MASTER_PRESET_DIGAUDIO15     0x04000000
#define SV_MASTER_PRESET_DIGAUDIO16     0x08000000
#define SV_MASTER_PRESET_DIGAUDIOMASK   0x0ff0ff00
#define SV_MASTER_PRESET_DIGAUDIOLSB    0x0000ff00
#define SV_MASTER_PRESET_DIGAUDIOMSB    0x0ff00000

/*
 *              Bits in master.info from sv_status
 */

#define SV_MASTER_INFO_CUEUPREADY       0x00000001
#define SV_MASTER_INFO_STILL            0x00000002
#define SV_MASTER_INFO_REVERSE          0x00000004
#define SV_MASTER_INFO_VAR              0x00000008
#define SV_MASTER_INFO_JOG              0x00000010
#define SV_MASTER_INFO_SHUTTLE          0x00000020
#define SV_MASTER_INFO_SERVOLOCK        0x00000080
#define SV_MASTER_INFO_PLAY             0x00000100
#define SV_MASTER_INFO_RECORD           0x00000200
#define SV_MASTER_INFO_FORWARD          0x00000400
#define SV_MASTER_INFO_REWIND           0x00000800
#define SV_MASTER_INFO_EJECT            0x00001000
#define SV_MASTER_INFO_STOP             0x00002000
#define SV_MASTER_INFO_STANDBY          0x00008000
#define SV_MASTER_INFO_LOCAL            0x00010000
#define SV_MASTER_INFO_HARDERROR        0x00040000
#define SV_MASTER_INFO_TAPETROUBLE      0x00080000
#define SV_MASTER_INFO_REFVIDEOMISSING  0x00100000
#define SV_MASTER_INFO_CASSETTEOUT      0x00200000
#define SV_MASTER_INFO_OFFLINE          0x01000000

#define SV_MASTER_INFO_ACTION_MASK      0xf0000000
#define SV_MASTER_INFO_ACTION_LOAD      0x10000000
#define SV_MASTER_INFO_ACTION_SAVE      0x20000000
#define SV_MASTER_INFO_ACTION_EDIT      0x40000000

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
/*
//  Used with sv_vtrmaster(SV_MASTER_RAW), obsolete, use the function sv_vtrmaster_raw() instead.
*/
typedef struct {
  int   command;
  int   length;
  unsigned char data[16];
} sv_vtr_cmd;
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */

#define SV_VTRCONTROL_MODE_DEFAULT   0x00000000
#define SV_VTRCONTROL_MODE_AUTOEDIT  0x00000001
#define SV_VTRCONTROL_MODE_REVIEW    0x00000002
#define SV_VTRCONTROL_MODE_PREVIEW   0x00000003
#define SV_VTRCONTROL_MODE_MASK      0x00000007




#if defined(DOCUMENTATION_SDK)
/*----------------------------------------------------------------------*/
/**
//
//  \weakgroup svoption
//  @{
//
*/
#endif /* DOCUMENTATION_SDK */

#define SV_OPTION_NOP                       0
#define SV_OPTION_DEBUG                     1   /* INTERNAL */
#define SV_OPTION_REPEAT                    2   /* OBSOLETE */
#define SV_OPTION_HDELAY                    3
#define SV_OPTION_DEBUGVALUE                4   /* INTERNAL */
#define SV_OPTION_DISKACCESS                5   /* OBSOLETE */
#define SV_OPTION_LOOPMODE                  6   /* OBSOLETE */
#define SV_OPTION_SPEED                     7   /* OBSOLETE */
#define SV_OPTION_SLOWMOTION                8   /* OBSOLETE */
#define SV_OPTION_DISKSETUP                 9   /* OBSOLETE */
#define SV_OPTION_SPEEDBASE                 10  /* OBSOLETE */
#define SV_OPTION_BUMPMODE                  11  /* OBSOLETE */
#define SV_OPTION_COMPRESSION               12  /* OBSOLETE */
#define SV_OPTION_COMPR_FORMAT              13  /* OBSOLETE */
#define SV_OPTION_FASTMODE                  14  /* OBSOLETE */
#define SV_OPTION_OVERLAY_MODE              15  /* OBSOLETE */
#define SV_OPTION_BLOCKSIZE                 16  /* OBSOLETE */
#define SV_OPTION_AUDIOMODE                 17  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_YUV422          18  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_YUV422A         19  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_RGB             20  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_RGBA            21  /* OBSOLETE */
#define SV_OPTION_CLUT                      22  /* OBSOLETE */
#define SV_OPTION_IOMODE                    23
#define SV_OPTION_CONNECT                   24  /* OBSOLETE */
#define SV_OPTION_SVHS                      25  /* OBSOLETE */
#define SV_OPTION_ANALOG                    25
#define SV_OPTION_RECORD_LOOPMODE           26  /* OBSOLETE */
#define SV_OPTION_RECORD_SPEED              27  /* OBSOLETE */
#define SV_OPTION_GPI                       28
#define SV_OPTION_DISKSETUP_MONO            29  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_YUV422STEREO    30  /* OBSOLETE */
#define SV_OPTION_INPUTPORT                 31
#define SV_OPTION_AUDIOFREQ                 32
#define SV_OPTION_AUDIOBITS                 33
#define SV_OPTION_GPI_WAIT                  34  /* OBSOLETE */
#define SV_OPTION_STRIPE_FORMAT             35  /* OBSOLETE */
#define SV_OPTION_CYCLES                    36  /* OBSOLETE */
#define SV_OPTION_SPEED_IMM                 37  /* OBSOLETE */
#define SV_OPTION_VDELAY                    38
#define SV_OPTION_OVERLAY_FRAMEOFFSET       39  /* OBSOLETE */
#define SV_OPTION_AUDIO_OFFSET              40  /* OBSOLETE */
#define SV_OPTION_AUDIOMUTE                 41
#define SV_OPTION_DEFRAG_START              42  /* OBSOLETE */
#define SV_OPTION_SUPERVISORMODE            43  /* OBSOLETE */
#define SV_OPTION_STEREOMODE                44  /* OBSOLETE */
#define SV_OPTION_AUDIOLOCK                 45  /* OBSOLETE */
#define SV_OPTION_DUALCHANNELMODE           46  /* OBSOLETE */
#define SV_OPTION_PREREAD_OFFSET            47  /* OBSOLETE */
#define SV_OPTION_AUDIOINPUT                48
#define SV_OPTION_DISKSCAN                  49  /* OBSOLETE */
#define SV_OPTION_DISKSCAN_DISK             50  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_YUV420          51  /* OBSOLETE */
#define SV_OPTION_AUDIOSEGMENT              52  /* OBSOLETE */
#define SV_OPTION_LTCSOURCE                 53
#define SV_OPTION_LTCOFFSET                 54
#define SV_OPTION_LTCFLAGS                  55  /* OBSOLETE */
#define SV_OPTION_OVERLAY_TYPE              56  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_YUV444          57  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_YUV444A         58  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_RGBVIDEO        59  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_RGBAVIDEO       60  /* OBSOLETE */
#define SV_OPTION_OUTDURINGREC              61  /* OBSOLETE */
#define SV_OPTION_LTCSOURCE_REC             62  /* OBSOLETE */
#define SV_OPTION_AUDIOCHANNELS             63
#define SV_OPTION_SYNCMODE                  64
#define SV_OPTION_WORDCLOCK                 65
#define SV_OPTION_SYNCOUT                   66
#define SV_OPTION_SYNCOUTDELAY              67
#define SV_OPTION_VIDEOMODE                 68
#define SV_OPTION_AUTOPULLDOWN              69
#define SV_OPTION_NOHSWTRANSFER             70  /* OBSOLETE */
#define SV_OPTION_GAMMA                     71  /* OBSOLETE */
#define SV_OPTION_GAMMA_RED                 72  /* OBSOLETE */
#define SV_OPTION_GAMMA_GREEN               73  /* OBSOLETE */
#define SV_OPTION_GAMMA_BLUE                74  /* OBSOLETE */
#define SV_OPTION_SYNCOUTVDELAY             75
#define SV_OPTION_ANALOG_GAIN_BASE          76  /* OBSOLETE */
#define SV_OPTION_ANALOG_GAIN_G_Y           77  /* OBSOLETE */
#define SV_OPTION_ANALOG_GAIN_B_U           78  /* OBSOLETE */
#define SV_OPTION_ANALOG_GAIN_R_V           79  /* OBSOLETE */
#define SV_OPTION_ANALOG_OFFSET_BASE        80  /* OBSOLETE */
#define SV_OPTION_ANALOG_OFFSET_G_Y         81  /* OBSOLETE */
#define SV_OPTION_ANALOG_OFFSET_B_U         82  /* OBSOLETE */
#define SV_OPTION_ANALOG_OFFSET_R_V         83  /* OBSOLETE */
#define SV_OPTION_ANALOG_BLACKLEVEL_BASE    84  /* OBSOLETE */
#define SV_OPTION_ANALOG_BLACKLEVEL_G_Y     85  /* OBSOLETE */
#define SV_OPTION_ANALOG_BLACKLEVEL_B_U     86  /* OBSOLETE */
#define SV_OPTION_ANALOG_BLACKLEVEL_R_V     87  /* OBSOLETE */
#define SV_OPTION_ANALOG_RW_DEFAULT         88  /* OBSOLETE */
#define SV_OPTION_ANALOG_RW_USERDEF         89  /* OBSOLETE */ 
#define SV_OPTION_TIMECODE_DROPFRAME        90  /* OBSOLETE */
#define SV_OPTION_TIMECODE_OFFSET           91  /* OBSOLETE */
#define SV_OPTION_VITCLINE                  92
#define SV_OPTION_GAMMA_RW_DEFAULT          93  /* OBSOLETE */
#define SV_OPTION_GPIIN                     94
#define SV_OPTION_GPIOUT                    95
#define SV_OPTION_OUTPUTPORT                96
#define SV_OPTION_VTRMASTER_EDITLAG         97
#define SV_OPTION_VTRMASTER_FLAGS           98
#define SV_OPTION_VTRMASTER_POSTROLL        99
#define SV_OPTION_VTRMASTER_PREROLL         100
#define SV_OPTION_VTRMASTER_TCTYPE          101
#define SV_OPTION_VTRMASTER_TOLERANCE       102
#define SV_OPTION_TILEMODE                  103  /* OBSOLETE */
#define SV_OPTION_VIDEOMODE_NOTRASTER       104  /* OBSOLETE */
#define SV_OPTION_MULTIDEVICE               105
#define SV_OPTION_GAMMA_LUT                 106  /* OBSOLETE */
#define SV_OPTION_AUDIO_SPEED_COMPENSATION  107  /* OBSOLETE */
#define SV_OPTION_INPUTFILTER               108
#define SV_OPTION_OUTPUTFILTER              109
#define SV_OPTION_VITCREADERLINE            110
#define SV_OPTION_REHASH                    111  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_ARGB            112  /* OBSOLETE */
#define SV_OPTION_DISKSETUP_ABGR            113  /* OBSOLETE */
#define SV_OPTION_ANCREADER                 114
#define SV_OPTION_ANCGENERATOR              115
#define SV_OPTION_ANCUSER_DID               116
#define SV_OPTION_ANCUSER_SDID              117
#define SV_OPTION_ANCUSER_LINENR            118
#define SV_OPTION_ANCUSER_FLAGS             119
#define SV_OPTION_DLTC_SOURCE               120  /* OBSOLETE */
#define SV_OPTION_DVITC_SOURCE              121  /* OBSOLETE */
#define SV_OPTION_FILMTC_SOURCE             122  /* OBSOLETE */
#define SV_OPTION_PRODTC_SOURCE             123  /* OBSOLETE */
#define SV_OPTION_RECORDMODE                124  /* OBSOLETE */
#define SV_OPTION_WATCHDOG_TIMEOUT          125
#define SV_OPTION_WATCHDOG_ACTION           126
#define SV_OPTION_PROXY_VIDEOMODE           127
#define SV_OPTION_PROXY_ASPECTRATIO         128
#define SV_OPTION_PROXY_SYNCMODE            129
#define SV_OPTION_AUDIOANALOGOUT            130
#define SV_OPTION_DROPMODE                  131
#define SV_OPTION_MULTIDEVICE_STATE         132  /* OBSOLETE */
#define SV_OPTION_MULTIDEVICE_OFFSET        133  /* OBSOLETE */
#define SV_OPTION_TRACE                     134
#define SV_OPTION_RESETBEFORERECORD         135  /* OBSOLETE */
#define SV_OPTION_PROXY_TIMECODE            136
#define SV_OPTION_OUTPUTDVI                 137  /* OBSOLETE */
#define SV_OPTION_PROXY_OPTIONS             138
#define SV_OPTION_AUDIOMAXAIV               139
#define SV_OPTION_LTCFILTER                 140
#define SV_OPTION_PROXY_OUTPUT              141
#define SV_OPTION_LTCDROPFRAME              142
#define SV_OPTION_DISABLESWITCHINGLINE      143
#define SV_OPTION_RS422A                    144
#define SV_OPTION_RS422A_PINOUT             144  /* OBSOLETE */
#define SV_OPTION_RS422B                    145
#define SV_OPTION_RS422B_PINOUT             145  /* OBSOLETE */
#define SV_OPTION_RS422C                    146
#define SV_OPTION_RS422C_PINOUT             146  /* OBSOLETE */
#define SV_OPTION_RS422D                    147
#define SV_OPTION_RS422D_PINOUT             147  /* OBSOLETE */
#define SV_OPTION_FIELD_DOMINANCE           148
#define SV_OPTION_ANALOGOUTPUT              149
#define SV_OPTION_ZEROLSB                   150  /* OBSOLETE */
#define SV_OPTION_LTC_TC                    151
#define SV_OPTION_LTC_UB                    152
#define SV_OPTION_VITC_TC                   153
#define SV_OPTION_VITC_UB                   154
#define SV_OPTION_VTR_TC                    155
#define SV_OPTION_VTR_UB                    156
#define SV_OPTION_VTR_INFO                  157
#define SV_OPTION_VTR_INFO2                 158
#define SV_OPTION_VTR_INFO3                 159
#define SV_OPTION_DVITC_TC                  160
#define SV_OPTION_DVITC_UB                  161
#define SV_OPTION_FILM_TC                   162
#define SV_OPTION_FILM_UB                   163
#define SV_OPTION_PROD_TC                   164
#define SV_OPTION_PROD_UB                   165
#define SV_OPTION_DLTC_TC                   166
#define SV_OPTION_DLTC_UB                   167
#define SV_OPTION_AFILM_TC                  168
#define SV_OPTION_AFILM_UB                  169
#define SV_OPTION_APROD_TC                  170
#define SV_OPTION_APROD_UB                  171
#define SV_OPTION_TICK                      172  /* INTERNAL */
#define SV_OPTION_LTCDELAY                  173
#define SV_OPTION_DVI_VIDEOMODE             174
#define SV_OPTION_DVI_RASTERID              175
#define SV_OPTION_FLUSH_TIMECODE            176
#define SV_OPTION_FORCERASTERDETECT         177
#define SV_OPTION_ANCCOMPLETE               178
#define SV_OPTION_VITCSOURCE                179
#define SV_OPTION_DVI_OUTPUT                180
#define SV_OPTION_HWWATCHDOG_TIMEOUT        181
#define SV_OPTION_HWWATCHDOG_ACTION         182
#define SV_OPTION_HWWATCHDOG_TRIGGER        183
#define SV_OPTION_MULTICHANNEL              184
#define SV_OPTION_MULTICHANNEL_LOCK         185  /* OBSOLETE */
#define SV_OPTION_HWWATCHDOG_REFRESH        186
#define SV_OPTION_DETECTION_TOLERANCE       187
#define SV_OPTION_ASSIGN_LTC                188
#define SV_OPTION_ASSIGN_VTR                189  /* OBSOLETE */
#define SV_OPTION_LINKENCRYPT               190
#define SV_OPTION_WATERMARK                 191
#define SV_OPTION_DEBUGINDEX                192
#define SV_OPTION_DETECTION_NO4K            193  /* OBSOLETE */
#define SV_OPTION_ALPHAMIXER                194
#define SV_OPTION_ALPHAGAIN                 195
#define SV_OPTION_ALPHAOFFSET               196
#define SV_OPTION_ANCGENERATOR_RP165        197
#define SV_OPTION_AUDIOAESROUTING           198
#define SV_OPTION_HWWATCHDOG_RELAY_DELAY    199
#define SV_OPTION_SWITCH_TOLERANCE          200
#define SV_OPTION_RENDER_VIDEOMODE          201
#define SV_OPTION_MAINOUTPUT                202
#define SV_OPTION_PULLDOWN_SYNCTC           203
#define SV_OPTION_PULLDOWN_DFTC             204
#define SV_OPTION_IOSPEED                   205
#define SV_OPTION_FIFO_NORENDER             206  /* OBSOLETE */
#define SV_OPTION_ROUTING                   207
#define SV_OPTION_FIFO_RENDER               208  /* OBSOLETE */
#define SV_OPTION_CROPPINGMODE              209
#define SV_OPTION_SYNCSELECT                210
#define SV_OPTION_STILLDISPLAYMODE          211  /* INTERNAL */
#define SV_OPTION_PULLDOWN_STARTPHASE       212
#define SV_OPTION_PULLDOWN_STARTLTC         213
#define SV_OPTION_PULLDOWN_STARTVTRTC       214
#define SV_OPTION_IOMODE_AUTODETECT         215
#define SV_OPTION_AUDIODRIFT_ADJUST         216


#if defined(DOCUMENTATION_SDK)
/** @} */
#endif /* DOCUMENTATION_SDK */

/*-----------------------------------------------------------------------*/

#define SV_PROXY_SYNC_AUTO              0
#define SV_PROXY_SYNC_INTERNAL          1
#define SV_PROXY_SYNC_GENLOCKED         2

#define SV_PROXY_OPTION_NTSCJAPAN       0x0001  /* INTERNAL */
#define SV_PROXY_OPTION_SDTVFULL        0x0002
#define SV_PROXY_OPTION_FREEZEFIELD     0x0004
#define SV_PROXY_OPTION_DESKTOPONLY     0x0008
#define SV_PROXY_OPTION_LOWRESOLUTION   0x0010
#define SV_PROXY_OPTION_640X480         0x0020

#define SV_PROXY_OUTPUT_UNDERSCAN       0
#define SV_PROXY_OUTPUT_LETTERBOX       1
#define SV_PROXY_OUTPUT_CROPPED         2
#define SV_PROXY_OUTPUT_ANAMORPH        3

#define SV_DVI_OUTPUT_DVI8              0
#define SV_DVI_OUTPUT_DVI12             1
#define SV_DVI_OUTPUT_DVI16             2

/*
//
//      Defines
//
*/
#define SV_ANCDATA_DEFAULT              0x0000
#define SV_ANCDATA_DISABLE              0x0001
#define SV_ANCDATA_USERDEF              0x0002
#define SV_ANCDATA_RP188                0x0003
#define SV_ANCDATA_RP201                0x0004
#define SV_ANCDATA_RP196                0x0005
#define SV_ANCDATA_RP196LTC             0x0006
#define SV_ANCDATA_RP215                0x0007
#define SV_ANCDATA_MASK                 0x00ff
#define SV_ANCDATA_FLAG_NOAIV           0x0100
#define SV_ANCDATA_FLAG_NOLTC           0x0200
#define SV_ANCDATA_FLAG_NOSMPTE352      0x0400

#define SV_ANCUSER_FLAG_VANC            0x0001

#define SV_AUDIOINPUT_AIV               0
#define SV_AUDIOINPUT_AESEBU            1


#define SV_AUDIOLOCK_OFF                0
#define SV_AUDIOLOCK_ON                 1


#define SV_AUDIOMODE_ALWAYS             0
#define SV_AUDIOMODE_ON_SPEED1          1
#define SV_AUDIOMODE_ON_MOTION          2


#define SV_AUDIOMUTE_OFF                0
#define SV_AUDIOMUTE_ON                 1

#define SV_CONNECT_OFF                  0
#define SV_CONNECT_MASTER               1
#define SV_CONNECT_SLAVE                2

#define SV_CROPPINGMODE_DEFAULT         0
#define SV_CROPPINGMODE_HEAD            1
#define SV_CROPPINGMODE_FULL            2

#define SV_CURRENTTIME_CURRENT          0
#define SV_CURRENTTIME_VSYNC_DISPLAY    1
#define SV_CURRENTTIME_VSYNC_RECORD     2
#define SV_CURRENTTIME_FRAME_DISPLAY    3
#define SV_CURRENTTIME_FRAME_RECORD     4

#define SV_EYEMODE_DEFAULT              0
#define SV_EYEMODE_LEFT                 1
#define SV_EYEMODE_RIGHT                2

#define SV_FASTMODE_DEFAULT             0 /* OBSOLETE */
#define SV_FASTMODE_ODDFIELDS           1 /* OBSOLETE */
#define SV_FASTMODE_EVENFIELDS          2 /* OBSOLETE */
#define SV_FASTMODE_BESTMATCH           3 /* OBSOLETE */
#define SV_FASTMODE_FRAMEREP_ABA        4 /* OBSOLETE */
#define SV_FASTMODE_FRAMEREP_ABAB       5 /* OBSOLETE */


#define SV_FSTYPE_FLAT                  0
#define SV_FSTYPE_PVFS                  1
#define SV_FSTYPE_CSFS                  2
#define SV_FSTYPE_OSFS                  3


#define SV_GAMMA_MODE_MASK              0xe0000000
#define SV_GAMMA_MODE_DISABLED          0x00000000
#define SV_GAMMA_MODE_LUMA              0x20000000
#define SV_GAMMA_MODE_RGB               0x40000000
#define SV_GAMMA_MODE_LUT               0x60000000
#define SV_GAMMA_FULLRANGE              0x01000000
#define SV_GAMMA_ANALOG                 0x02000000
#define SV_GAMMA_DIGITAL                0x04000000
#define SV_GAMMA_8BIT                   0x08000000
#define SV_GAMMA_VALUE_MASK             0x00ffffff

#define SV_GPI_FIRSTFRAME               0
#define SV_GPI_LASTFRAME                1

#define SV_GPIOUT_DEFAULT               0x0000  /* fifoapi / recorded */
#define SV_GPIOUT_OPTIONGPI             0x0001  /* SV_OPTION_GPI */
#define SV_GPIOUT_INOUTPOINT            0x0002  /* bit0-Inpoint / bit1-Outpoint */
#define SV_GPIOUT_PULLDOWN              0x0003  /* bit0-PhaseA bit1-Valid */
#define SV_GPIOUT_PULLDOWNPHASE         0x0004  /* 00-A 01-B 10-C 11-D */
#define SV_GPIOUT_REPEATED              0x0005  /* bit0-Valid bit1-Repeated(=!valid) */
#define SV_GPIOUT_SPIRIT                0x0006  /* bit0-Every 5 frames high bit1-Inverted */
#define SV_GPIOUT_MASK                  0x00ff

#define SV_GPIIN_IGNORE                 0x0000  /* GPI is ignored (default) */
#define SV_GPIIN_PULLDOWN               0x0000  /* 00-A 01-B 10-C 11-D, ignored not pulldown */
#define SV_GPIIN_PULLDOWNPHASE          0x0000  /* bit0-PhaseA bit1-Valid */
#define SV_GPIIN_ACTIVELOW              0x0000  /* Record all frames with bit0 low */
#define SV_GPIIN_ACTIVEHIGH             0x0000  /* Record all frames with bit0 high */
#define SV_GPIIN_WAITLOW                0x0000  /* Start recording when bit0 is low */
#define SV_GPIIN_WAITHIGH               0x0000  /* Start recording when bit0 is high */
#define SV_GPIIN_MASK                   0x00ff


#define SV_MAINOUTPUT_SDI               0x0000  /* Default, sdi has priority */
#define SV_MAINOUTPUT_DVI               0x0001  /* On cards that can not do dvi / sdi make dvi the main output */
#define SV_MAINOUTPUT_MASK              0x00ff  /* Mask for mainoutput settings */
#define SV_MAINOUTPUT_FLAG_MASK         0xff00  /* Mask for mainoutput flags */


#define SV_INPUTPORT_SDI                0
#define SV_INPUTPORT_ANALOG             1
#define SV_INPUTPORT_SDI2               2
#define SV_INPUTPORT_DVI                3
#define SV_INPUTPORT_SDI3               4


#define SV_IOSPEED_UNKNOWN              0
#define SV_IOSPEED_1GB5                 1
#define SV_IOSPEED_3GBA                 2
#define SV_IOSPEED_3GBB                 3
#define SV_IOSPEED_SDTV                 4




/*
//      IO modes
*/
#define SV_IOMODE_YUV                   0x00000000        /* OBSOLETE */
#define SV_IOMODE_YUV422                0x00000000
#define SV_IOMODE_RGB                   0x00000001
#define SV_IOMODE_YUV444                0x00000002
#define SV_IOMODE_YUV422A               0x00000003
#define SV_IOMODE_RGBA                  0x00000004
#define SV_IOMODE_YUV444A               0x00000005
#define SV_IOMODE_RGBA_SHIFT2           0x00000006
#define SV_IOMODE_YUV422_12             0x00000007
#define SV_IOMODE_YUV444_12             0x00000008
#define SV_IOMODE_RGB_12                0x00000009
#define SV_IOMODE_YUV422STEREO          0x0000000a
#define SV_IOMODE_XYZ                   0x0000000b
#define SV_IOMODE_RGB_8                 0x0000000c
#define SV_IOMODE_XYZ_12                0x0000000d
#define SV_IOMODE_YCC422                0x0000000e
#define SV_IOMODE_YCC                   0x0000000f
#define SV_IOMODE_YCC_12                0x00000010
#define SV_IOMODE_RGBA_8                0x00000011
#define SV_IOMODE_IO_MASK               0x000000ff
#define SV_IOMODE_OUTPUT_MASK           0x0000ff00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT2MODE(x)        (((x)&0x00007f00)>>8)                             /* OBSOLETE */
#define SV_IOMODE_MODE2OUTPUT(x)        ((((x)&0x0000007f)<<8) | SV_IOMODE_OUTPUT_ENABLE) /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV422         0x00008000        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_RGB            0x00008100        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV444         0x00008200        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV422A        0x00008300        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_RGBA           0x00008400        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV444A        0x00008500        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_RGBA_SHIFT2    0x00008600        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV422_12      0x00008700        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV444_12      0x00008800        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_RGB_12         0x00008900        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YUV422STEREO   0x00008a00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_XYZ            0x00008b00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_RGB_8          0x00008c00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_XYZ_12         0x00008d00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YCC422         0x00008e00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YCC            0x00008f00        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_YCC_12         0x00009000        /* OBSOLETE */
#define SV_IOMODE_OUTPUT_ENABLE         0x00008000        /* OBSOLETE */

#define SV_IOMODE_RANGE_DEFAULT         0x00000000
#define SV_IOMODE_RANGE_HEAD            0x00010000
#define SV_IOMODE_RANGE_FULL            0x00020000
#define SV_IOMODE_RANGE_MASK            0x00030000

#define SV_IOMODE_MATRIX_DEFAULT        0x00000000
#define SV_IOMODE_MATRIX_601            0x00040000
#define SV_IOMODE_MATRIX_274            0x00080000
#define SV_IOMODE_MATRIX_MASK           0x000c0000

#define SV_IOMODE_OUTPUT_RANGE_DEFAULT  0x00000000
#define SV_IOMODE_OUTPUT_RANGE_HEAD     0x00100000
#define SV_IOMODE_OUTPUT_RANGE_FULL     0x00200000
#define SV_IOMODE_OUTPUT_RANGE_MASK     0x00300000

#define SV_IOMODE_OUTPUT_MATRIX_DEFAULT 0x00000000
#define SV_IOMODE_OUTPUT_MATRIX_601     0x00400000
#define SV_IOMODE_OUTPUT_MATRIX_274     0x00800000
#define SV_IOMODE_OUTPUT_MATRIX_MASK    0x00c00000

#define SV_IOMODE_CLIP_DEFAULT          0x00000000
#define SV_IOMODE_CLIP                  0x01000000
#define SV_IOMODE_CLIP_MASK             0x01000000

#define SV_IOMODE_OUTPUT_CLIP_DEFAULT   0x00000000
#define SV_IOMODE_OUTPUT_CLIP           0x10000000
#define SV_IOMODE_OUTPUT_CLIP_MASK      0x10000000

#define SV_IOMODE_OUTPUT2RANGE(x)       (((x)&0xf0f00000)>>4)
#define SV_IOMODE_RANGE2OUTPUT(x)       (((x)&0x0f0f0000)<<4)


#define SV_LTCFILTER_ENABLE             0
#define SV_LTCFILTER_DISABLE            1
#define SV_LTCFILTER_DUPLICATE          2

#define SV_LTCDROPFRAME_DEFAULT         0
#define SV_LTCDROPFRAME_OFF             1
#define SV_LTCDROPFRAME_ON              2

#define SV_LTCSOURCE_MASK               0xffff
#define SV_LTCSOURCE_DEFAULT            0
#define SV_LTCSOURCE_DISK               0           /* OBSOLETE */
#define SV_LTCSOURCE_INTERN             1
#define SV_LTCSOURCE_PLAYLIST           2           /* OBSOLETE */
#define SV_LTCSOURCE_MASTER             3
#define SV_LTCSOURCE_FREERUNNING        4
#define SV_LTCSOURCE_LTCOFFSET          5
#define SV_LTCSOURCE_PROXY              6
#define SV_LTCSOURCE_VCOUNT             0x10000
#define SV_LTCSOURCE_PULLDOWN_ADJUST    0x20000

#define SV_LTCSOURCE_REC_INTERN         0 
#define SV_LTCSOURCE_REC_EE             1 

#define SV_VITCSOURCE_MASK               0xffff
#define SV_VITCSOURCE_DEFAULT            0
#define SV_VITCSOURCE_PULLDOWN_ADJUST    0x20000

#define SV_TCOUT_AUX_OFF                0
#define SV_TCOUT_AUX_INTERN_TC          1
#define SV_TCOUT_AUX_INTERN_F           2
#define SV_TCOUT_AUX_INTERN_TC_F        3

#define SV_LOOPMODE_FORWARD             0
#define SV_LOOPMODE_REVERSE             1
#define SV_LOOPMODE_SHUTTLE             2
#define SV_LOOPMODE_ONCE                3
#define SV_LOOPMODE_DEFAULT             4
#define SV_LOOPMODE_INFINITE            5

// The ports are numerated from 0 to 3. The ports 2 and 3 are the Aux Ports
#define SV_HOSTPROTOCOL_MIXER_DISABLE   0
#define SV_HOSTPROTOCOL_MIXER_AUX1      2
#define SV_HOSTPROTOCOL_MIXER_AUX2      3

/*
//      sv_lut() modes
*/
#define SV_LUT_LUMA                     0
#define SV_LUT_RED                      1
#define SV_LUT_GREEN                    2
#define SV_LUT_BLUE                     3
#define SV_LUT_ALPHA                    4
#define SV_LUT_RGBA                     5
#define SV_LUT_DISABLE                  6
#define SV_LUT_MASK                     0x0000ff

#define SV_LUT_BITDEPTH_10BIT           0x000000
#define SV_LUT_BITDEPTH_16BIT           0x010000
#define SV_LUT_BITDEPTH_MASK            0x0f0000

#define SV_OUTDURINGREC_DEFAULT         0           /* OBSOLETE */
#define SV_OUTDURINGREC_INPUT           1           /* OBSOLETE */
#define SV_OUTDURINGREC_OUTPUT          2           /* OBSOLETE */
#define SV_OUTDURINGREC_BLACK           3           /* OBSOLETE */
#define SV_OUTDURINGREC_BYPASS          4           /* OBSOLETE */
#define SV_OUTDURINGREC_COLORBAR        5           /* OBSOLETE */

#define SV_OUTPUTPORT_DEFAULT           0
#define SV_OUTPUTPORT_SWAPPED           1
#define SV_OUTPUTPORT_MIRROR            2

#define SV_OVERLAY_OFF                  0x0000
#define SV_OVERLAY_ON                   0x0001
#define SV_OVERLAY_TRANSPARENT          0x0002
#define SV_OVERLAY_MASK                 0x00ff
#define SV_OVERLAY_DIGITAL              0x0100


#define SV_OVERLAY_TYPE_OFF             0
#define SV_OVERLAY_TYPE_LTC_TC          1
#define SV_OVERLAY_TYPE_LTC_FRAME_ABS   2
#define SV_OVERLAY_TYPE_LTC_FRAME_REL   3
#define SV_OVERLAY_TYPE_TC              4
#define SV_OVERLAY_TYPE_FRAME_ABS       5
#define SV_OVERLAY_TYPE_FRAME_REL       6


#define SV_OVERLAY_LINE_DEVICE          0
#define SV_OVERLAY_LINE_MASTER          1
#define SV_OVERLAY_LINE_LTC             2
#define SV_OVERLAY_LINE_CLIP            3
#define SV_OVERLAY_LINE_STATUS          4
#define SV_OVERLAY_LINE_USER            5
#define SV_OVERLAY_LINE_MASK            0xff

#define SV_ROUTING_DEFAULT              0
#define SV_ROUTING_12                   12
#define SV_ROUTING_1234                 1234

#define SV_TIMECODECHASE_OFF            0
#define SV_TIMECODECHASE_VTRMASTER      1
#define SV_TIMECODECHASE_LTC            2
#define SV_TIMECODECHASE_VITC           3
#define SV_TIMECODECHASE_DVITC          4
#define SV_TIMECODECHASE_DLTC           5
#define SV_TIMECODECHASE_SLAVE          6

#define SV_TRACE_ERROR                  0x01
#define SV_TRACE_SETUP                  0x02
#define SV_TRACE_STORAGE                0x04
#define SV_TRACE_FIFOAPI                0x08
#define SV_TRACE_CAPTURE                0x10
#define SV_TRACE_VTRCONTROL             0x20
#define SV_TRACE_TCCHECK                0x40
#define SV_TRACE_RENDER                 0x80
#define SV_TRACE_ALL                    -1


#define SV_PATTERN_BLACK                0x0000
#define SV_PATTERN_COLORBAR             0x0001
#define SV_PATTERN_RANDOM               0x0002
#define SV_PATTERN_CHROMARAMP           0x0003
#define SV_PATTERN_LUMARAMP             0x0004
#define SV_PATTERN_MASK                 0x00ff
#define SV_PATTERN_FLAG_FRAMENR         0x0100

#define SV_PORTTYPE_DEFAULT             0
#define SV_PORTTYPE_MASTER              1
#define SV_PORTTYPE_SLAVE               2
#define SV_PORTTYPE_MIXER               3
#define SV_PORTTYPE_KEYCODE             4
#define SV_PORTTYPE_VDCPSLAVE           5


#define SV_POSITION_FLAG_RELATIVE       0x0001    /* The new position is relative to the actual one */
#define SV_POSITION_FLAG_PAUSE          0x0002    /* Goes to the position and Pause                 */
#define SV_POSITION_FLAG_SPEEDONE       0x0004    /* Goes to the position and Speed 1               */


#define SV_PRESET_VIDEO                 0x00000001      ///< Preset for video
#define SV_PRESET_AUDIO12               0x00000002      ///< Preset audio pair 1
#define SV_PRESET_AUDIO34               0x00000004      ///< Preset audio pair 2
#define SV_PRESET_KEY                   0x00000008      ///< Preset key channel
#define SV_PRESET_TIMECODE              0x00000010      ///< Preset Timecode
#define SV_PRESET_SECOND_VIDEO          0x00000020      ///< Preset 2nd video ch
#define SV_PRESET_AUDIO56               0x00000040      ///< Preset audio pair 3
#define SV_PRESET_AUDIO78               0x00000080      ///< Preset audio pair 4
#define SV_PRESET_AUDIO9a               0x00000100      ///< Preset audio pair 5
#define SV_PRESET_AUDIObc               0x00000200      ///< Preset audio pair 6
#define SV_PRESET_AUDIOde               0x00000400      ///< Preset audio pair 7
#define SV_PRESET_AUDIOfg               0x00000800      ///< Preset audio pair 8
#define SV_PRESET_AUDIOMASK             0x00000fc6      ///< Preset all audio channels 
#define SV_PRESET_VIDEOMASK             0x00000029      ///< Preset all video channels 


#define SV_PULLDOWN_CMD_STARTPHASE      0
#define SV_PULLDOWN_CMD_STARTLTC        1
#define SV_PULLDOWN_CMD_STARTVTRTC      2

#define SV_PULLDOWN_STARTPHASE_UNDEF    0
#define SV_PULLDOWN_STARTPHASE_A        1   /* Phase 01   */
#define SV_PULLDOWN_STARTPHASE_B        2   /* Phase 010  */
#define SV_PULLDOWN_STARTPHASE_C        3   /* Phase 10   */
#define SV_PULLDOWN_STARTPHASE_D        4   /* Phase 101  */
#define SV_PULLDOWN_STARTPHASE_AUTO     5

#define SV_PULLDOWN_STARTPHASE_A23      4   /* new definitions for          */
#define SV_PULLDOWN_STARTPHASE_B23      1   /* startphase with 2:3 pulldown */
#define SV_PULLDOWN_STARTPHASE_C23      2
#define SV_PULLDOWN_STARTPHASE_D23      3

  
#define SV_REPEAT_FRAME                 0
#define SV_REPEAT_FIELD                 1   /* OBSOLETE */
#define SV_REPEAT_FIELD1                1
#define SV_REPEAT_FIELD2                2
#define SV_REPEAT_CURRENT               3
#define SV_REPEAT_DEFAULT               4


#define SV_DROPMODE_REPEAT              0
#define SV_DROPMODE_BLACK               1


#define SV_SHOWINPUT_DEFAULT            0x00
#define SV_SHOWINPUT_BYPASS             0x01  /* OBSOLETE */
#define SV_SHOWINPUT_FRAMEBUFFERED      0x02
#define SV_SHOWINPUT_FIELDBUFFERED      0x03
#define SV_SHOWINPUT_MASK               0xff


#define SV_SLOWMOTION_FRAME             0
#define SV_SLOWMOTION_FIELD             1
#define SV_SLOWMOTION_FIELD1            2
#define SV_SLOWMOTION_FIELD2            3



#define SV_ANALOG_BLACKLEVEL_BLACK75    0x0000  /* OBSOLETE */
#define SV_ANALOG_BLACKLEVEL_BLACK0     0x0001  /* OBSOLETE */
#define SV_ANALOG_BLACKLEVEL_MASK       0x000f  /* OBSOLETE */
#define SV_ANALOG_FORCE_NONE            0x0000  /* OBSOLETE */
#define SV_ANALOG_FORCE_PAL             0x0010  /* OBSOLETE */
#define SV_ANALOG_FORCE_NTSC            0x0020  /* OBSOLETE */
#define SV_ANALOG_FORCE_MASK            0x00f0  /* OBSOLETE */
#define SV_ANALOG_SHOW_AUTO             0x0000  /* OBSOLETE */
#define SV_ANALOG_SHOW_INPUT            0x0100  /* OBSOLETE */
#define SV_ANALOG_SHOW_OUTPUT           0x0200  /* OBSOLETE */
#define SV_ANALOG_SHOW_BLACK            0x0300  /* OBSOLETE */
#define SV_ANALOG_SHOW_COLORBAR         0x0400  /* OBSOLETE */
#define SV_ANALOG_SHOW_MASK             0x0f00  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_YC             0x0000  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_YUV            0x1000  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_RGB            0x2000  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_YUVS           0x3000  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_RGBS           0x4000  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_CVBS           0x5000  /* OBSOLETE */
#define SV_ANALOG_OUTPUT_MASK           0xf000  /* OBSOLETE */


#define SV_ANALOGOUTPUT_RGBFULL         0
#define SV_ANALOGOUTPUT_RGBHEAD         1
#define SV_ANALOGOUTPUT_YUVFULL         2
#define SV_ANALOGOUTPUT_YUVHEAD         3


#define SV_VITCLINE_DISABLED            0x000000
#define SV_VITCLINE_DEFAULT             0x0000ff
#define SV_VITCLINE_MASK                0x0000ff
#define SV_VITCLINE_VCOUNT              0x000100
#define SV_VITCLINE_ARP201              0x000200
#define SV_VITCLINE_DYNAMIC             0x000400
#define SV_VITCLINE_DUPLICATE_MASK      0xff0000
#define SV_VITCLINE_DUPLICATE_GET(x)    (((x) >> 16) & 0xff)
#define SV_VITCLINE_DUPLICATE(x)        (((x) & 0xff) << 16)


#define SV_RECORDMODE_NORMAL                 0
#define SV_RECORDMODE_GPI_CONTROLLED         1
#define SV_RECORDMODE_VARIFRAME_CONTROLLED   2


#define SV_RS422_PINOUT_DEFAULT         0x0000
#define SV_RS422_PINOUT_NORMAL          0x0001 /* Not for Centaurus II, Centaurus II LT */
#define SV_RS422_PINOUT_SWAPPED         0x0002 /* Not for Centaurus II, Centaurus II LT */
#define SV_RS422_PINOUT_MASTER          0x0003
#define SV_RS422_PINOUT_SLAVE           0x0004
#define SV_RS422_PINOUT_MASK            0x000f
#define SV_RS422_TASK_DEFAULT           0x0000
#define SV_RS422_TASK_NONE              0x0010
#define SV_RS422_TASK_MASTER            0x0020
#define SV_RS422_TASK_SLAVE             0x0030
#define SV_RS422_TASK_VDCPSLAVE         0x0040
#define SV_RS422_TASK_MASK              0x00f0
#define SV_RS422_IOCHANNEL_FLAG         0x1000 /*INTERNAL*/
#define SV_RS422_IOCHANNEL_MASK         0x0f00
#define SV_RS422_IOCHANNEL_GET(x)       (((x) & SV_RS422_IOCHANNEL_MASK) >> 8)
#define SV_RS422_IOCHANNEL_SET(x)       ((((x) << 8) & SV_RS422_IOCHANNEL_MASK) | SV_RS422_IOCHANNEL_FLAG)


/*
//              Vtr Emulation Modes
*/
#define SV_SLAVE_SLAVEINFO              0
#define SV_SLAVE_DISABLED               1
#define SV_SLAVE_ENABLED                2
#define SV_SLAVE_ALWAYS                 3
#define SV_SLAVE_DRIVER                 4
#define SV_SLAVE_ODETICS                5

/*
//  Used by sv_slaveinfo_get() and sv_slaveinfo_set().
*/
typedef struct {
  int           flags;                  ///< Flags
  int           command_cmd;            ///< Command -> Command Code	
  int           command_tick;           ///< Command -> Received Tick
  int           command_length;         ///< Command -> Data Length
  unsigned char command_data[16];       ///< Command -> Data
  int           command_lost;           ///< Command -> Lost commands
  unsigned char status_data[16];        ///< VSync -> Status bits
  int           status_timecode;        ///< Status -> Timecode
  int           status_speed;           ///< Vsync -> Current Speed
  int           status_userbytes;       ///< Setup -> User bytes
  int           edit_preset;            ///< Setup -> Edit Preset
  int           edit_inpoint;           ///< Edit -> Edit Inpoint
  int           edit_outpoint;          ///< Edit -> Edit Outpoint
  int           edit_ainpoint;          ///< Edit -> Edit A Inpoint
  int           edit_aoutpoint;         ///< Edit -> Edit A Outpoint
  int           setup_devicetype;       ///< Setup -> Device Type
  int           setup_preroll;          ///< Setup -> Edit Preroll time
  int           setup_recinhibit;       ///< Setup -> Rec Inhibit
  int           setup_timermode;        ///< Setup -> Current timermode
  int           setup_postroll;         ///< Setup -> Edit Postroll time
  int           pad[15];                ///< Reserved
} sv_slaveinfo;

#define SV_SLAVEINFO_USESTATUSANDTC     0x0001

/*
//              Sync Input Mode
*/
#define SV_SYNC_INT                     0           /* OBSOLETE */
#define SV_SYNC_INTERNAL                0
#define SV_SYNC_EXT                     1           /* OBSOLETE */
#define SV_SYNC_EXTERNAL                1
#define SV_SYNC_GENLOCK_ANALOG          2
#define SV_SYNC_GENLOCK_DIGITAL         3
#define SV_SYNC_SLAVE                   4
#define SV_SYNC_AUTO                    5
#define SV_SYNC_MODULE                  6
#define SV_SYNC_BILEVEL                 7
#define SV_SYNC_TRILEVEL                8
#define SV_SYNC_HVTTL                   9
#define SV_SYNC_LTC                     10          /* OBSOLETE */
#define SV_SYNC_MASK                    0xffff

#define SV_SYNC_HVTTL_HFVF              0x00000
#define SV_SYNC_HVTTL_HRVF              0x10000
#define SV_SYNC_HVTTL_HFVR              0x20000
#define SV_SYNC_HVTTL_HRVR              0x30000
#define SV_SYNC_HVTTL_MASK              0xF0000

#define SV_SYNC_FLAG_NTSC               0x100000    /* OBSOLETE */
#define SV_SYNC_FLAG_SDTV               0x100000    /* force sync of raster to ntsc or pal source */ 
/*
//      Sync Output Mode
*/
#define SV_SYNCOUT_OFF                  0x0000
#define SV_SYNCOUT_BILEVEL              0x0001
#define SV_SYNCOUT_TRILEVEL             0x0002
#define SV_SYNCOUT_HVTTL                0x0003      /* OBSOLETE */
#define SV_SYNCOUT_HVTTL_HFVF           0x0003
#define SV_SYNCOUT_USERDEF              0x0004
#define SV_SYNCOUT_CURRENT              0x0005
#define SV_SYNCOUT_DEFAULT              0x0006
#define SV_SYNCOUT_HVTTL_HRVF           0x0007
#define SV_SYNCOUT_HVTTL_HFVR           0x0008
#define SV_SYNCOUT_HVTTL_HRVR           0x0009
#define SV_SYNCOUT_AUTOMATIC            0x000a
#define SV_SYNCOUT_MASK                 0x00ff

#define SV_SYNCOUT_OUTPUT_MAIN          0x1000
#define SV_SYNCOUT_OUTPUT_MODULE        0x2000
#define SV_SYNCOUT_OUTPUT_GREEN         0x4000
#define SV_SYNCOUT_OUTPUT_MASK          0xf000
 
#define SV_SYNCOUT_LEVEL_SET(x)         ((x) << 16)
#define SV_SYNCOUT_LEVEL_GET(x)         (((x) >> 16) & 0xff)
#define SV_SYNCOUT_LEVEL_MASK           0x00ff0000
#define SV_SYNCOUT_LEVEL_DEFAULT        0x00000000

#define SV_SYNCSELECT_LINKA             0
#define SV_SYNCSELECT_LINKB             1


#define SV_TIMELINE_FLAG_GPI            1
#define SV_TIMELINE_FLAG_GPI_LOOP       4
#define SV_TIMELINE_FLAG_LOOP           8
#define SV_TIMELINE_FLAG_BLACK          16

#define SV_VALIDTIMECODE_LTC        0x000001
#define SV_VALIDTIMECODE_DLTC       0x000002
#define SV_VALIDTIMECODE_VTR        0x000004
#define SV_VALIDTIMECODE_RP215      0x000008
#define SV_VALIDTIMECODE_VITC_F1    0x000100
#define SV_VALIDTIMECODE_DVITC_F1   0x000200
#define SV_VALIDTIMECODE_RP201_F1   0x000400
#define SV_VALIDTIMECODE_CC_F1      0x000800
#define SV_VALIDTIMECODE_ARP201_F1  0x001000
#define SV_VALIDTIMECODE_VITC_F2    0x010000
#define SV_VALIDTIMECODE_DVITC_F2   0x020000
#define SV_VALIDTIMECODE_RP201_F2   0x040000
#define SV_VALIDTIMECODE_CC_F2      0x080000
#define SV_VALIDTIMECODE_ARP201_F2  0x100000
#define SV_VALIDTIMECODE_MASK_F1    0x00ffff
#define SV_VALIDTIMECODE_MASK_F2    0xff0000


#define SV_VSYNCWAIT_DISPLAY            0
#define SV_VSYNCWAIT_RECORD             1
#define SV_VSYNCWAIT_CANCEL             2
#define SV_VSYNCWAIT_STATUS             3


#define SV_WORDCLOCK_OFF                0
#define SV_WORDCLOCK_ON                 1


#define SV_WATCHDOG_NONE                0
#define SV_WATCHDOG_BYPASS              1
#define SV_WATCHDOG_BLACK               2
#define SV_WATCHDOG_COLORBAR            3

#define SV_HWWATCHDOG_NONE              0x00
#define SV_HWWATCHDOG_RELAY             0x01
#define SV_HWWATCHDOG_GPI2              0x02
#define SV_HWWATCHDOG_MANUAL            0x10

#define SV_ZOOMFLAGS_PROGRESSIVE        0x01
#define SV_ZOOMFLAGS_INTERLACED         0x02
#define SV_ZOOMFLAGS_FIXEDFLOAT         0x04
#define SV_ZOOMFLAGS_PIXELREPETITION    0x08 /* OBSOLETE */
#define SV_ZOOMFLAGS_BILINEAR           0x10 /* OBSOLETE */

#define SV_JACK_CHANNEL_DISCONNECTED    0x00000000
#define SV_JACK_CHANNEL_OUT             0x80000000
#define SV_JACK_CHANNEL_IN              0x80000001
#define SV_JACK_CHANNEL_OUTB            0x80000100
#define SV_JACK_CHANNEL_INB             0x80000101

#define SV_ANCCOMPLETE_MODE_MASK           0x00ff
#define SV_ANCCOMPLETE_OFF                 0x0000
#define SV_ANCCOMPLETE_ON                  0x0001
#define SV_ANCCOMPLETE_STREAMER            0x0002

#define SV_ANCCOMPLETE_FLAG_MASK                0xff00
#define SV_ANCCOMPLETE_FLAG_FORCE_SWITCHINGLINE 0x0100

#define SV_MULTICHANNEL_OFF             0
#define SV_MULTICHANNEL_ON              1
#define SV_MULTICHANNEL_DEFAULT         2
#define SV_MULTICHANNEL_INPUT           3
#define SV_MULTICHANNEL_OUTPUT          4
#define SV_MULTICHANNEL_FLAG_ALLOW_DUALLINK 0x00010000
#define SV_MULTICHANNEL_FLAG_MASK           0xffff0000

#define SV_ALPHAMIXER_OFF               0
#define SV_ALPHAMIXER_AB                1
#define SV_ALPHAMIXER_AB_PREMULTIPLIED  2
#define SV_ALPHAMIXER_BA                3
#define SV_ALPHAMIXER_BA_PREMULTIPLIED  4

#define SV_LINKENCRYPT_A                1
#define SV_LINKENCRYPT_B                2
#define SV_LINKENCRYPT_TEST             4

#define SV_AUDIOAESROUTING_DEFAULT      0 //<< multichannel off: 16 0, multichannel on: 8_8
#define SV_AUDIOAESROUTING_16_0         1
#define SV_AUDIOAESROUTING_8_8          2
#define SV_AUDIOAESROUTING_4_4          3

#define SV_SWITCH_TOLERANCE_DEFAULT            0x0000
#define SV_SWITCH_TOLERANCE_OFF                0x0000
#define SV_SWITCH_TOLERANCE_DETECT             0x0001
#define SV_SWITCH_TOLERANCE_SYNC               0x0002
#define SV_SWITCH_TOLERANCE_DETECT_CYCLES_MASK 0xff00
#define SV_SWITCH_TOLERANCE_DETECT_CYCLES(x)   ((((x) << 8) & SV_SWITCH_TOLERANCE_DETECT_CYCLES_MASK) | SV_SWITCH_TOLERANCE_DETECT)

#if defined(DOCUMENTATION_SDK)
/*----------------------------------------------------------------------*/
/**
//  \weakgroup svquery
//  @{
*/
#endif /* DOCUMENTATION_SDK */

#define SV_QUERY_AUDIOSIZE              0       /* OBSOLETE */
#define SV_QUERY_AUDIOSIZE_FROMHOST     0       /* OBSOLETE */
#define SV_QUERY_VERSION_DVSOEM         1
#define SV_QUERY_VERSION_DRIVER         2
#define SV_QUERY_CARRIER                3
#define SV_QUERY_ANALOG                 4 
#define SV_QUERY_TICK                   5
#define SV_QUERY_VSYNCWAIT              6       /* OBSOLETE */
#define SV_QUERY_REPEATMODE             7       /* OBSOLETE */
#define SV_QUERY_SLOWMOTION             8       /* OBSOLETE */
#define SV_QUERY_FASTMOTION             9       /* OBSOLETE */
#define SV_QUERY_FASTMODE               9       /* OBSOLETE */
#define SV_QUERY_AUDIOMODE              10
#define SV_QUERY_LOOPMODE               11      /* OBSOLETE */
#define SV_QUERY_NDISKS                 12      /* OBSOLETE */
#define SV_QUERY_DEVTYPE                13
#define SV_QUERY_INPUTPORT              14
#define SV_QUERY_AUDIOFREQ              15
#define SV_QUERY_AUDIOBITS              16
#define SV_QUERY_TL_POSITION            17      /* OBSOLETE */
#define SV_QUERY_HDELAY                 18
#define SV_QUERY_VDELAY                 19
#define SV_QUERY_OVERLAY                20      /* OBSOLETE */
#define SV_QUERY_PULLDOWN               21
#define SV_QUERY_NCLIPDIR               22      /* OBSOLETE */
#define SV_QUERY_MAXCLIPS               23      /* OBSOLETE */
#define SV_QUERY_NCLIPS                 24      /* OBSOLETE */
#define SV_QUERY_CLIP_SUPPORT           25      /* OBSOLETE */
#define SV_QUERY_VALUE_AVAILABLE        26
#define SV_QUERY_DISKSETUP              27      /* OBSOLETE */
#define SV_QUERY_AUDIO_OFFSET           28      /* OBSOLETE */
#define SV_QUERY_OVERLAYMODE            29      /* OBSOLETE */
#define SV_QUERY_AUDIOMUTE              30
#define SV_QUERY_DEFRAG_PROGRESS        31      /* OBSOLETE */
#define SV_QUERY_STEREOMODE             32      /* OBSOLETE */
#define SV_QUERY_STEREOOFFSET           33      /* OBSOLETE */
#define SV_QUERY_DUALCHANNEL_MODE       34      /* OBSOLETE */
#define SV_QUERY_PREREAD_OFFSET         35      /* OBSOLETE */
#define SV_QUERY_AUDIOCHANNELS          36
#define SV_QUERY_AUDIOSIZE_TOHOST       37      /* OBSOLETE */
#define SV_QUERY_AUDIOINPUT             38
#define SV_QUERY_INTERLACE_ID           39      /* OBSOLETE */
#define SV_QUERY_INTERLACEID_STORAGE    39      /* OBSOLETE */
#define SV_QUERY_DISKSCAN               40      /* OBSOLETE */
#define SV_QUERY_PRONTOVISION           41      /* OBSOLETE */
#define SV_QUERY_STREAMERSIZE           42      /* OBSOLETE */
#define SV_QUERY_AUDIOSEGMENT           43      /* OBSOLETE */
#define SV_QUERY_LTCSOURCE              44
#define SV_QUERY_LTCOFFSET              45
#define SV_QUERY_OVERLAYTYPE            46      /* OBSOLETE */
#define SV_QUERY_PRESET                 47      /* OBSOLETE */
#define SV_QUERY_IOMODE                 48
#define SV_QUERY_FSTYPE                 49      /* OBSOLETE */
#define SV_QUERY_OVERLAY_FRAMEOFFSET    50      /* OBSOLETE */
#define SV_QUERY_SYNCMODE               51
#define SV_QUERY_SYNCOUT                52
#define SV_QUERY_VTRMASTER_LOCAL        53      /* OBSOLETE */
#define SV_QUERY_FEATURE                53
#define SV_QUERY_MODE_AVAILABLE         54
#define SV_QUERY_MODE_CURRENT           55
#define SV_QUERY_OUTDURINGREC           56      /* OBSOLETE */
#define SV_QUERY_NPLAYLIST              57
#define SV_QUERY_LTCSOURCE_REC          58
#define SV_QUERY_XZOOM                  59
#define SV_QUERY_YZOOM                  60
#define SV_QUERY_XPANNING               61
#define SV_QUERY_YPANNING               62
#define SV_QUERY_ZOOMFLAGS              63
#define SV_QUERY_INTERLACEID_VIDEO      64      /* OBSOLETE */
#define SV_QUERY_WORDCLOCK              65
#define SV_QUERY_SYNCOUTDELAY           66
#define SV_QUERY_AUTOPULLDOWN           67      /* internal */
#define SV_QUERY_LTCTIMECODE            68
#define SV_QUERY_LTCUSERBYTES           69
#define SV_QUERY_VITCTIMECODE           70
#define SV_QUERY_VITCUSERBYTES          71
#define SV_QUERY_GPI                    72
#define SV_QUERY_NOHSWTRANSFER          73      /* OBSOLETE */
#define SV_QUERY_GAMMA                  74      /* OBSOLETE */
#define SV_QUERY_GAMMA_RED              75      /* OBSOLETE */
#define SV_QUERY_GAMMA_GREEN            76      /* OBSOLETE */
#define SV_QUERY_GAMMA_BLUE             77      /* OBSOLETE */
#define SV_QUERY_SYNCOUTVDELAY          78
#define SV_QUERY_SYNCSTATE              79
#define SV_QUERY_RASTERID               80
#define SV_QUERY_HW_EPLDVERSION         81
#define SV_QUERY_HW_EPLDOPTIONS         82
#define SV_QUERY_HW_CARDVERSION         83
#define SV_QUERY_HW_CARDOPTIONS         84
#define SV_QUERY_PULLDOWNFPS            85      /* OBSOLETE */
#define SV_QUERY_VITCLINE               86
#define SV_QUERY_TEMPERATURE            87
#define SV_QUERY_GPIIN                  88
#define SV_QUERY_GPIOUT                 89
#define SV_QUERY_OUTPUTPORT             90
#define SV_QUERY_AUDIOSIZE_TICK         91      /* OBSOLETE */
#define SV_QUERY_TCOUT_AUX              92      /* OBSOLETE */
#define SV_QUERY_TILEFACTOR             93
#define SV_QUERY_AUDIOINERROR           94
#define SV_QUERY_VIDEOINERROR           95
#define SV_QUERY_AUDIO_SPEED_COMPENSATION 96
#define SV_QUERY_INPUTFILTER            97
#define SV_QUERY_OUTPUTFILTER           98
#define SV_QUERY_INPUTRASTER            99
#define SV_QUERY_RECORD_LINENR          100
#define SV_QUERY_DISPLAY_LINENR         101
#define SV_QUERY_GENLOCK                102
#define SV_QUERY_FILM_TC                103
#define SV_QUERY_FILM_UB                104
#define SV_QUERY_PROD_TC                105
#define SV_QUERY_PROD_UB                106
#define SV_QUERY_VITCREADERLINE         107
#define SV_QUERY_VITCINPUTLINE          108     /* OBSOLETE */
#define SV_QUERY_DVITC_TC               109
#define SV_QUERY_DVITC_UB               110
#define SV_QUERY_DLTC_TC                111
#define SV_QUERY_DLTC_UB                112
#define SV_QUERY_DLTC_SOURCE            113     /* OBSOLETE */
#define SV_QUERY_DVITC_SOURCE           114     /* OBSOLETE */
#define SV_QUERY_FILMTC_SOURCE          115     /* OBSOLETE */
#define SV_QUERY_PRODTC_SOURCE          116     /* OBSOLETE */
#define SV_QUERY_TIMECODE_OFFSET        117     /* OBSOLETE */
#define SV_QUERY_TIMECODE_DROPFRAME     118     /* OBSOLETE */
#define SV_QUERY_FREEFRAMES             119     /* OBSOLETE */
#define SV_QUERY_INPUTRASTER_GENLOCK    120
#define SV_QUERY_VOLTAGE_1V5            121
#define SV_QUERY_VOLTAGE_2V5            122
#define SV_QUERY_VOLTAGE_3V3            123
#define SV_QUERY_VOLTAGE_5V0            124
#define SV_QUERY_VOLTAGE_12V0           125
#define SV_QUERY_SERIALNUMBER           126
#define SV_QUERY_HW_PCISPEED            127
#define SV_QUERY_HW_PCIWIDTH            128
#define SV_QUERY_AUDIO_AIVCHANNELS      129
#define SV_QUERY_AUDIO_AESCHANNELS      130
#define SV_QUERY_HW_FLASHSIZE           131     /* internal */
#define SV_QUERY_HW_EPLDTYPE            132     /* internal */
#define SV_QUERY_DMAALIGNMENT           133
#define SV_QUERY_VALIDTIMECODE          134
#define SV_QUERY_INPUTRASTER_SDIA       135
#define SV_QUERY_INPUTRASTER_SDIB       136
#define SV_QUERY_INPUTRASTER_SDIC       137     /* internal */
#define SV_QUERY_INPUTRASTER_DVI        138
#define SV_QUERY_CLOSEDCAPTION          139     /* OBSOLETE */
#define SV_QUERY_PULLDOWNPHASE          140     /* internal */
#define SV_QUERY_FANSPEED               141
#define SV_QUERY_SDILINKDELAY           142
#define SV_QUERY_LTCFILTER              143
#define SV_QUERY_LTCDROPFRAME           144
#define SV_QUERY_INPUTRASTER_GENLOCK_TYPE 145
#define SV_QUERY_AFILM_TC               146
#define SV_QUERY_AFILM_UB               147
#define SV_QUERY_APROD_TC               148
#define SV_QUERY_APROD_UB               149
#define SV_QUERY_MATRIXINFO_INPUT       150
#define SV_QUERY_MATRIXINFO_OUTPUT      151
#define SV_QUERY_FEATURE_AUDIOCHANNELS  152
#define SV_QUERY_FEATURE_LICENCETYPE    153
#define SV_QUERY_RASTER_XSIZE           154
#define SV_QUERY_RASTER_YSIZE           155
#define SV_QUERY_RASTER_FPS             156
#define SV_QUERY_RASTER_INTERLACE       157
#define SV_QUERY_RASTER_SEGMENTED       158
#define SV_QUERY_RASTER_DROPFRAME       159
#define SV_QUERY_HW_MEMORYSIZE          160     /* internal */
#define SV_QUERY_HW_MAPPEDSIZE          161     /* internal */
#define SV_QUERY_PULLDOWNPHASE_RECORD   162     /* OBSOLETE */
#define SV_QUERY_PULLDOWNPHASE_DISPLAY  163     /* OBSOLETE */
#define SV_QUERY_HW_FLASHVERSION        164     /* internal */
#define SV_QUERY_STORAGE_XSIZE          165
#define SV_QUERY_STORAGE_YSIZE          166
#define SV_QUERY_ANC_MINLINENR          167
#define SV_QUERY_ANC_MAXHANCLINENR      168
#define SV_QUERY_AUDIO_MAXCHANNELS      169
#define SV_QUERY_ANC_MAXVANCLINENR      170
#define SV_QUERY_VOLTAGE_1V2            171
#define SV_QUERY_VOLTAGE_1V8            172
#define SV_QUERY_VOLTAGE_2V3            173     /* internal */
#define SV_QUERY_INPUTRASTER_SDID       174     /* internal */
#define SV_QUERY_IOSPEED_SDIA           175
#define SV_QUERY_IOSPEED_SDIB           176
#define SV_QUERY_IOSPEED_SDIC           177
#define SV_QUERY_IOSPEED_SDID           178
#define SV_QUERY_FIFO_MEMORYMODE        179
#define SV_QUERY_VOLTAGE_1V0            180
#define SV_QUERY_HW_PCIELANES           181
#define SV_QUERY_INPUTIOMODE            182
#define SV_QUERY_IOCHANNELS             183
#define SV_QUERY_IOLINKS_INPUT          184
#define SV_QUERY_IOLINKS_OUTPUT         185
#define SV_QUERY_IOLINK_MAPPING         186
#define SV_QUERY_IOSPEED                187
#define SV_QUERY_IOMODEINERROR          188
#define SV_QUERY_SMPTE352               189

#if defined(DOCUMENTATION_SDK)
/** @} */
#endif /* DOCUMENTATION_SDK */
 
#define SV_FEATURE_VTRMASTER_LOCAL      0x00000001     /* OBSOLETE */
#define SV_FEATURE_PLAYLISTMAP          0x00000002     /* OBSOLETE */
#define SV_FEATURE_LTC_RECORDOUT        0x00000004     /* OBSOLETE */
#define SV_FEATURE_MIXERSUPPORT         0x00000008     /* OBSOLETE */
#define SV_FEATURE_ZOOMSUPPORT          0x00000010
#define SV_FEATURE_RASTERLIST           0x00000020
#define SV_FEATURE_NOHSWTRANSFER        0x00000040     /* OBSOLETE */
#define SV_FEATURE_AUTOPULLDOWN         0x00000080     /* OBSOLETE */
#define SV_FEATURE_SCSISWITCH           0x00000100     /* OBSOLETE */
#define SV_FEATURE_LUTSUPPORT           0x00000200
#define SV_FEATURE_HEADERTRANSFER       0x00000400     /* OBSOLETE */
#define SV_FEATURE_CAPTURE              0x00000800
#define SV_FEATURE_ZOOMANDPAN           0x00001000     /* INTERNAL */
#define SV_FEATURE_MIXERPROCESSING      0x00002000
#define SV_FEATURE_DUALLINK             0x00004000
#define SV_FEATURE_KEYCHANNEL           0x00008000
#define SV_FEATURE_INDEPENDENT_IO       0x00010000
#define SV_FEATURE_MULTIJACK            0x00020000






#define SV_INPUTFILTER_DEFAULT                  0     //  Selects the default filter.     
#define SV_INPUTFILTER_NOFILTER                 1     //  pixel duplication
#define SV_INPUTFILTER_5TAPS                    2     //  -> (a+b)/2
#define SV_INPUTFILTER_9TAPS                    3     //  
#define SV_INPUTFILTER_13TAPS                   4     //  
#define SV_INPUTFILTER_17TAPS                   5     //  

#define SV_OUTPUTFILTER_DEFAULT                 0     //  Selects the default filter.     
#define SV_OUTPUTFILTER_NOFILTER                1     //
#define SV_OUTPUTFILTER_5TAPS                   2     //
#define SV_OUTPUTFILTER_9TAPS                   3     //  
#define SV_OUTPUTFILTER_13TAPS                  4     //  
#define SV_OUTPUTFILTER_17TAPS                  5     //  


/*-----------------------------------------------------------------------
//
//      Mixer Functions
//
*/
#define SV_MIXER_EDGE_HARD          0
#define SV_MIXER_EDGE_RAMP          1
#define SV_MIXER_EDGE_SOFT          2
#define SV_MIXER_EDGE_STEPS         3

#define SV_MIXER_INPUT_DONTCHANGE   0
#define SV_MIXER_INPUT_BLACK        1
#define SV_MIXER_INPUT_VIDEO        2
#define SV_MIXER_INPUT_INPUT        3
#define SV_MIXER_INPUT_COLORBAR     4

#define SV_MIXER_MODE_DISABLE       0
#define SV_MIXER_MODE_BOTTOMLEFT    1
#define SV_MIXER_MODE_BOTTOM2TOP    2
#define SV_MIXER_MODE_BOTTOMRIGHT   3
#define SV_MIXER_MODE_LEFT2RIGHT    4
#define SV_MIXER_MODE_CENTER        5
#define SV_MIXER_MODE_RIGHT2LEFT    6
#define SV_MIXER_MODE_TOPLEFT       7
#define SV_MIXER_MODE_TOP2BOTTOM    8
#define SV_MIXER_MODE_TOPRIGHT      9
#define SV_MIXER_MODE_BLEND         10
#define SV_MIXER_MODE_ALPHA         11
#define SV_MIXER_MODE_CURTAINH      12
#define SV_MIXER_MODE_CURTAINV      13
#define SV_MIXER_MODE_CURTAINHOPEN  14
#define SV_MIXER_MODE_CURTAINVOPEN  15
#define SV_MIXER_MODE_LINESH        16
#define SV_MIXER_MODE_LINESV        17
#define SV_MIXER_MODE_FADE          18
#define SV_MIXER_MODE_STRIPESH      19
#define SV_MIXER_MODE_STRIPESV      20
#define SV_MIXER_MODE_STRIPESHSWAP  21
#define SV_MIXER_MODE_STRIPESVSWAP  22
#define SV_MIXER_MODE_TIMERSTOP     23
#define SV_MIXER_MODE_TOBLACK       24
#define SV_MIXER_MODE_ZOOMANDPAN    25
#define SV_MIXER_MODE_FRAMENUMBER   26
#define SV_MIXER_MODE_CROPMARKS     27
#define SV_MIXER_MODE_RECT          28
#define SV_MIXER_MODE_OVERLAY       29
#define SV_MIXER_MODE_TIMECODE      30
#define SV_MIXER_MODE_CROSS         31
#define SV_MIXER_MODE_AWIPE         32
#define SV_MIXER_MODE_KEYWIPE       33
#define SV_MIXER_MODE_KEYWIPEB      34
#define SV_MIXER_MODE_ANAGLYPH_RC   35
#define SV_MIXER_MODE_ANAGLYPH_RCM  36
#define SV_MIXER_MODE_ANAGLYPH_RG   37
#define SV_MIXER_MODE_ANAGLYPH_RGM  38
#define SV_MIXER_MODE_STEREO_TOPBOTTOM    39
#define SV_MIXER_MODE_STEREO_LEFTRIGHT    40
#define SV_MIXER_MODE_STEREO_INTERLACE    41
#define SV_MIXER_MODE_STEREO_BLEND        42


#define SV_RS422_OPENFLAG_SWAPPINOUT    0x01 /* OBSOLETE */
#define SV_RS422_OPENFLAG_MASTERPINOUT  0x02
#define SV_RS422_OPENFLAG_SLAVEPINOUT   0x04

#define SV_CLOCKTYPE_GMT    0

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
typedef struct {
  int year;
  int month;
  int day;
  int weekday;
  int hours;
  int minutes;
  int seconds;
  int microseconds;
  int pad[8];
} sv_clock_info;
#endif

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
/*
//  Used by sv_mixer_status()
*/
typedef struct {
  int size;
  int mode;
  int modeparam;
  int position;
  int start;
  int end;
  int nframes;
  int edge;
  int edgeparam;
  int edgewidth;  
  int porta;
  int portb;
  int portc;
  int xzoom;
  int yzoom;
  int xpanning;
  int ypanning;
  int zoomflags;
} sv_mixer_info;
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */



/*----------------------------------------------------------------------*/

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_) && !defined(CBLIBINT_H) && !defined(_DVSDDR_H) && !defined(_DVS_CLIB_NOSVHANDLE_)
/*
//  User accessable fields of the sv_handle structure, the rest is internal to the library.
*/
typedef struct {
  int            magic;                         ///< sv_handle magic number
  int            size;                          ///< Size of structure
  int            version;                       ///< Last version read back
  int            vgui;                          ///< True if vgui is running
  int            prontovideo;                   ///< True if prontovideo
  int            debug;
  int            pad[10];
  /* The rest is private */
} sv_handle;
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
/*
//  This structure is used together with sv_status().
*/
typedef struct {
  int   size;
  int   version;
  int   config;                     ///< Current videomode
  int   colormode;                  ///< Current color mode
  int   xsize;                      ///< Current video X size
  int   ysize;                      ///< Current video Y size
  int   sync;                       ///< Current sync mode
  int   nbit;                       ///< Number of bits per pixel

  struct {
    int   timecode;                 ///< Last read timecode from VTR
    int   info;                     ///< Last read info bits from VTR
    int   error;                    ///< Last errorcode from VTRCTRL
    int   preset;                   ///< Edit preset for VTR
    int   preroll;                  ///< Current preroll time
    int   postroll;                 ///< Current postroll time
    int   editlag;                  ///< Editlag in frames
    int   parktime;                 ///< Current parktime
    int   autopark;                 ///< Autopark enabled
    int   timecodetype;             ///< LTC or ASTC timecode selected
    int   tolerance;                ///< Edit control tolerance
    int   forcedropframe;           ///< Force dropframe calc for NTSC
    int   device;                   ///< Returned device id
    int   inpoint;                  ///< Current inpoint
    int   nframes;                  ///< Number of frames
    int   recoffset;                ///< Current record delay
    int   disoffset;                ///< Current display delay
    int   framerate;                ///< Current vertical frequency
    int   debug;                    ///< Current debug
    int   flags;                    ///< Current timecode flag
    int   infobits[4];              ///< Last read info bits from VTR
  } master;

  struct {
    int   ramsize;                  /* RAM size in Megabytes            */
    int   disksize;                 /* Disk size in Megabytes           */
    int   nframes;                  /* Number of available frames       */
    int   genlock;                  /* True if genlock available        */
    int   ccube;                    /* True if ccube chip available     */
    int   audio;                    /* True if audio available          */
    int   preset;                   /* Edit preset for SV/PV            */
    int   key;                      /* Key channel available            */
    int   mono;                     /* Monochrome instead of YUV        */
    int   compression;              /* compression rate (0=uncompr.)    */
    int   rgbm;                     /* RGB modul available              */
    int   flags;                    /* settings of sv_flag()            */
    int   licence;                  /* licence flags                    */
    int   storagexsize;             /* Current storage X size           */
    int   storageysize;             /* Current storage Y size           */
    int   pad[1];                   /* reserved                         */
  } setup;

  struct {
    int   position;                 /* Currently displayed frame        */
    int   inpoint;                  /* First frame to be displayed      */
    int   outpoint;                 /* First frame not to be displayed  */
    int   speed;                    /* Current speed nominator          */
    int   speedbase;                /* Current speed denominator        */
    int   slavemode;                /* Status of VTR slave mode         */
    int   state;                    /* Current state of SV/PV           */
    int   error;                    /* Error code from SV/PV            */
    int   sofdelay;                 /* Start Of Frame Delay             */
    int   iomode;                   /* IO mode for dual link            */
    int   clip_updateflag;          /* flag for synchronisation (server)*/
    int   tl_position;              /* timeline position (playlist)     */
    int   supervisorflag;           /* flag for supervisor mode         */
    int   framerate_mHz;            /* video framerate in [mhz]         */
    int   positiontc;               /* Currently displayed frame in TC  */
    int   flags;                    /* reserved                         */
  } video;

} sv_info; 
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */


#define SV_INFO_VIDEO_FLAGS_ODETICS  0x0001 /* Odetics active, clip selected */
#define SV_INFO_VIDEO_FLAGS_PLAYLIST 0x0002 /* Playlist mapped               */
#define SV_INFO_VIDEO_FLAGS_UPDATE   0x0004 /* Change in partition/videomode */
#define SV_INFO_VIDEO_FLAGS_PULLDOWN 0x0008 /* Cur. frame is in pulldown dir */
#define SV_INFO_VIDEO_FLAGS_PROTECT  0x0010 /* Cur. frame is protected       */



/*
//  Used by the sv_vsyncwait() function
*/
#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
typedef struct {
  int size;                 ///< Size of structure (input)
  int elements;             ///< Number of elements filled in
  int tick;                 ///< Last tick
  int recordtick;           ///< Last record tick
  int displaytick;          ///< Last display tick
  int recordaddr;           ///< Last recorded offset on card
  int displayaddr;          ///< Currently displayed address on card
  int recordframe;          ///< Last record frame
  int displayframe;         ///< Currently displaying frame
  int openprogram;          ///< Program trying to open card
  int opentick;             ///< When open was last called
  int closetick;            ///< Tick when card was last closed
  int pad[4];
} sv_vsyncwait_info;
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */


/*-----------------------------------------------------------------------
//
//      sv_overlapped used by queue api and dma
//
*/



#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)

#ifndef _DVS_CLIB_H_SV_OVERLAPPED_INTERNAL_
#define _DVS_CLIB_H_SV_OVERLAPPED_INTERNAL_
# if defined(_WINBASE_) && defined(WIN32)
typedef OVERLAPPED sv_overlapped_internal;
# else
typedef struct {
    void *        Internal;
    void *        InternalHigh;
    unsigned int  Offset;
    unsigned int  OffsetHigh;
    void *        Event;
} sv_overlapped_internal;
# endif
#endif

typedef struct {
  sv_overlapped_internal overlapped;
  unsigned int           temp[64];  // need enough space for pcistudio_asyncbuffer
} sv_overlapped;


/*-----------------------------------------------------------------------*/

/*
//  Used by sv_memory_dmalist()
*/
typedef struct {
  int     addr;
  int     size;
  int     offset;
} sv_dmalist;


/*-----------------------------------------------------------------------*/

/*
//  Used by sv_timecode_feedback()
*/
typedef struct {
  int     size;         ///< Used internally (do not change).
  int     tick;         ///< Current tick.
  int     altc_tc;      ///< Analog LTC timecode without bit masking.
  int     altc_ub;      ///< Analog LTC user bytes.
  int     altc_received;///< Only important for input, shows if a ltc timecode is received in the current vsync.
  int     avitc_tc[2];  ///< Analog VITC timecode.
  int     avitc_ub[2];  ///< Analog VITC user bytes.
  int     afilm_tc[2];   ///< RP201 timecode.
  int     afilm_ub[2];   ///< User bytes of the RP201 timecode.
  int     aprod_tc[2];   ///< RP201 production code.
  int     aprod_ub[2];   ///< RP201 production code.
  int     vtr_tc;       ///< VTR timecode.
  int     vtr_ub;       ///< VTR user bytes.
  int     vtr_received; ///< Only important for input, shows if a vtr is timecode received in the current vsync.
  int     dltc_tc;      ///< Digital/ANC LTC timecode.
  int     dltc_ub;      ///< Digital/ANC LTC user bytes.
  int     dvitc_tc[2];  ///< Digital/ANC VITC timecode.
  int     dvitc_ub[2];  ///< Digital/ANC VITC user bytes.
  int     dfilm_tc[2];   ///< RP201 timecode.
  int     dfilm_ub[2];   ///< User bytes of the RP201 timecode.
  int     dprod_tc[2];   ///< RP201 production code.
  int     dprod_ub[2];   ///< RP201 production code.
  int     pad[32];      ///<Reserved.
} sv_timecode_info;


/*-----------------------------------------------------------------------
//      Storage Functions
*/

/*
//  Used with sv_storage_status().
*/
typedef struct {
  int size;               ///< Size of this structure (filled by sv_storage_status()).
  int version;            ///< Used internally.
  int cookie;             ///< Used internally.
  int pad1[5];            ///< Reserved.

  int alignment;          ///< Storage: Buffer start address alignment (bytes).
  int bigendian;          ///< Storage: Is big-endian.
  int buffersize;         ///< Storage: Size of buffer to hold 1 frame.
  int components;         ///< Storage: Number of components (planes).
  int dominance21;        ///< Storage: Second field is first temporal field.
  int dropframe;          ///< Storage: Dropframe == 1/1.001 * fps.
  int fps;                ///< Storage: Frames per second.
  int fullrange;          ///< Storage: Datarange, 0->16->235, 1->0->255
  int interlace;          ///< Storage: Interlace, 1->Frame, 2->2Fields
  int nbits;              ///< Storage: Number of bits.
  int rgbformat;          ///< Storage: Is RGB Format.
  int subsample;          ///< Subsampling factor (422|444|0=1=all 1).
  int xsize;              ///< Video: X Size
  int ysize;              ///< Video: Y Size
  int yuvmatrix;          ///< Storage: Default Matrix
  int colormode;          ///< Storage: Actual Colormode (SV_COLORMODE_XXX)
  int videomode;          ///< Actual videomode (SV_MODE_XXX)
  int vinterlace;         ///< Video: Interlaced
  int vfps;               ///< Video: Frames per second
  int filmmaterial;       ///< Storage: Both fields have the same time.
  int nbit10type;         ///< Storage: SV_NBIT10TYPE_XXX (OBSOLETE)
  int bottom2top;         ///< Storage: Data is bottom2top
  int nbittype;           ///< Storage: SV_NBITTYPE_XXX

  int pixelsize_num;      ///< Storage: Size of 1 pixel, numerator/denominator
  int pixelsize_denom;    ///< Storage: Size of 1 pixel, numerator/denominator
  int pixeloffset_num[8]; ///< Storage: Offset between pixels
  int dataoffset_num[8];  ///< Storage: Offset of first pixel
  int linesize;           ///< Storage: Size of 1 line valid pixels
  int lineoffset[2];      ///< Storage: Offset from line to line
  int fieldoffset[2];     ///< Storage: Offset of each field
  int fieldsize[2];       ///< Storage: Size of each field
  int storagexsize;       ///< Storage: X Size
  int storageysize;       ///< Storage: Y Size
  int pixelgroup;         ///< Storage: Minimum pixelgroup 
  int pad3[13];           ///< Reserved.
 
  int abits;              ///< Audio: Number of bits per sample 16|32
  int abigendian;         ///< Audio: Sample endianess big=1, little=0
  int aunsigned;          ///< Audio: Sample unsigned=1, signed=0
  int afrequency;         ///< Audio: Sampling Frequency [Hz]
  int achannels;          ///< Audio: Number of (stereo) channels
  int aoffset[2];         ///< Audio: Offset (currently not used)
  int pad4[8];            ///< Reserved.
} sv_storageinfo;

typedef struct {
  int size;
  int version;
  int channel;            ///< Hardware channel which is connected to this jack.
  int card2host;          ///< Transfer direction.
  int pad[64];            ///< Reserved.
} sv_jack_info;

#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */


#define SV_STORAGEINFO_COOKIEISMODE  0x0001
#define SV_STORAGEINFO_INPUT_XSIZE   0x0002
#define SV_STORAGEINFO_INPUT_YSIZE   0x0004
#define SV_STORAGEINFO_INPUT_NBITS   0x0008
#define SV_STORAGEINFO_COOKIEISJACK  0x0010





/*-----------------------------------------------------------------------
//
//      sv_raster_status Function
//
*/
#define SV_RASTEROPTION_CUSTOMRASTER      0x00000001
#define SV_RASTEROPTION_SEGMENTEDFRAME    0x00000002
#define SV_RASTEROPTION_TIMECODE24HZ      0x00000004
#define SV_RASTEROPTION_FRAMEREPEAT       0x00000008
#define SV_RASTEROPTION_QUADOUTPUT        0x00000010
#define SV_RASTEROPTION_SDIDUALLINK       0x00000020
#define SV_RASTEROPTION_SDIDUALLINKARRI   0x00000040
#define SV_RASTEROPTION_SDIDUALLINKHDCAM  0x00000080
#define SV_RASTEROPTION_SDIDUALLINKSTEREO 0x00000100
#define SV_RASTEROPTION_DUALOUTPUT        0x00000200
#define SV_RASTEROPTION_RGBDUALLINK       0x00000400
/* SV_RASTEROPTION_RESERVED               0x00001000 */

#define SV_RASTERGROUP_CUSTOM         0
#define SV_RASTERGROUP_SDTV           1
#define SV_RASTERGROUP_HDTV           2
#define SV_RASTERGROUP_PHDTV          3
#define SV_RASTERGROUP_FILM2K         4
#define SV_RASTERGROUP_FILM4K         5
#define SV_RASTERGROUP_OTHER          6
#define SV_RASTERGROUP_STEREO         7

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
#if defined(_RASTERMOD_H_) || defined(_DVS_DRIVER_H_)
#define sv_rasterheader RM_HEAD
#else
#define RM_VERSION      "REF 1.3" /* id verification string     */
#define RM_MAXINTERL    4         /* maximum of interlace factor*/
#define RM_MAXNAMELEN   48        /* maximum length of user name*/
#define RM_MAXSEGMENT   1024      /* maximum amount of segments */



/*
//  Raster definition, used together with sv_raster_status().
*/
typedef struct { 
  char  id[8];              ///< Magic identification word
  int   sdata;              ///< Offset to signal description list
  char  name[RM_MAXNAMELEN];///< User raster name seen by GUI
  int   lines;              ///< Total lines per frame
  int   vrate;              ///< Frame rate [mHz]
  int   ilace;              ///< Fields per frame
  int   il_id;              ///< Field order
  int   dfreq;              ///< Default sampling rate
  int   dsout;              ///< Default output signal
  int   rfreq;              ///< Reference sampling rate
  int   pels;               ///< Total pixels per line
  int   clen;               ///< Clamp pulse length
  int   cdly;               ///< Clamp pulse delay

  /*
  // active video area
  */
  int   hlen;               ///< Default active pixels per line
  int   hdly;               ///< Default active pixels start
  int   vlen[RM_MAXINTERL]; ///< Default active lines per field
  int   svlen[RM_MAXINTERL];///< Standard active lines per field
  int   vdly[RM_MAXINTERL]; ///< Default active lines start
  int   svdly[RM_MAXINTERL];///< Standard active lines start
    
  /*
  // special initialization data
  */
  int   reserved1[20];
  int   vmod;               ///< Video mode BURST/STREAMER
  int   svind;              ///< SV_MODE_xxx index
  int   shlen;              ///< Standard active pixels per line
  int   shdly;              ///< Standard active pixels start
  int   sof[RM_MAXINTERL];  ///< Start of field lines
  int   options;            ///< Raster options (e.g. drop frame)
  int   group;              ///< Raster group
  int   reserved2[2];

  int   disable;            ///< Currently disabled if != 0
  int   clormap;            ///< Color spaces available
  int   nbitmap;            ///< Quantizations available
  int   abitmap;            ///< Alpha quantizations available
  int   index;              ///< Index in ref table
  int   unused[3];          ///< Reserved for future use
} sv_rasterheader;
#endif
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */

/*-----------------------------------------------------------------------
//
//      sv_version Function
//
*/
#define SV_VERSION_COMMENT_SIZE         64
#define SV_VERSION_MODULE_SIZE          32
#define SV_VERSION_FLAG_BETA            1

#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_)
/*
//  This structure is used by sv_version_status() to return software/hardware version information.
*/
typedef struct { 
  union {
    unsigned int         version;       ///< Full version number
    struct {
      unsigned char      patch;         ///< Patch level byte.
      unsigned char      fix;           ///< Fix level byte.
      unsigned char      major;         ///< Major revision byte.
      unsigned char      minor;         ///< Minor revision byte.
    } v;
  } release;
  int           flags;                  ///< Flag release (not used).
  int           rbeta;                  ///< Beta release counter (not used).
  union {
    int         date;                   ///< BCD format YYYYMMDD.
    struct {
      short     yyyy;                   ///< Year
      char      mm;                     ///< Month
      char      dd;                     ///< Day
    } d;
  } date;
  union {
    int   time;                         ///< BCD format 00HHMMSS
    struct {
      char              nn;             ///< 00
      char              hh;             ///< hour
      char              mm;             ///< minute
      char              ss;             ///< second
    } t;
  } time;
  int     devicecount;                      ///< Number of devices.
  int     modulecount;                      ///< Number of modules.
  char    comment[SV_VERSION_COMMENT_SIZE]; ///< Version comment.
  char    module[SV_VERSION_MODULE_SIZE];   ///< Module name.
  char    res1[20];                         ///< Reserved.
} sv_version;
#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */

#define SV_MAGIC_DVSDDR                         0x43576142
#define SV_MAGIC_DVSOEM                         0x47615743
#define SV_MAGIC_DVSOEMX                        0x49625723
#define SV_MAGIC_DVSLIB                         0x834fed21

#define SV_OPENTYPE_DEFAULT               0

#define SV_OPENTYPE_VOUTPUT               0x00000001
#define SV_OPENTYPE_AOUTPUT               0x00000002
#define SV_OPENTYPE_OUTPUT                0x00000003
#define SV_OPENTYPE_VINPUT                0x00000004
#define SV_OPENTYPE_AINPUT                0x00000008
#define SV_OPENTYPE_INPUT                 0x0000000c
#define SV_OPENTYPE_RS422A                0x00000010
#define SV_OPENTYPE_MASTER                0x00000010
#define SV_OPENTYPE_RS422B                0x00000020
#define SV_OPENTYPE_SLAVE                 0x00000020
#define SV_OPENTYPE_RS422C                0x00000040 /* OBSOLETE */
#define SV_OPENTYPE_RS422D                0x00000080 /* OBSOLETE */
#define SV_OPENTYPE_RENDER                0x00000100
#define SV_OPENTYPE_MASK_ONCE             0x0000ffff /* if this is ever changed, code must be carefully reviewed */
#define SV_OPENTYPE_MASK_DEFAULT          0x000001ff
#define SV_OPENTYPE_CAPTURE               0x00010000
#define SV_OPENTYPE_WAITFORCLOSE          0x00020000
#define SV_OPENTYPE_QUERY                 0x00040000
#define SV_OPENTYPE_MASK_MULTIPLE         0x000f0000
#define SV_OPENTYPE_RESERVED              0x00f00000
#define SV_OPENTYPE_MASK                  0x000fffff
#define SV_OPENTYPE_VALID                 0x0107ffff


#define SV_OPENPROGRAM_NONE               0x00000000
#define SV_OPENPROGRAM_DEFAULT            0x00000001
#define SV_OPENPROGRAM_SVPROGRAM          0x00000002
#define SV_OPENPROGRAM_TESTPROGRAM        0x00000003
#define SV_OPENPROGRAM_DEMOPROGRAM        0x00000004
#define SV_OPENPROGRAM_VSERVER            0x00000005
#define SV_OPENPROGRAM_KERNEL             0x00000006
#define SV_OPENPROGRAM_MULTIMEDIA         0x00000007
#define SV_OPENPROGRAM_OPENML             0x00000008
#define SV_OPENPROGRAM_QUICKTIME          0x00000009
#define SV_OPENPROGRAM_DVSOEMX            0x0000000a
#define SV_OPENPROGRAM_CLIPSTER           0x0000000b
#define SV_OPENPROGRAM_DVSCONF            0x0000000c
#define SV_OPENPROGRAM_SPYCER             0x0000000d
#define SV_OPENPROGRAM_APPLICATION        0x01000000
#define SV_OPENPROGRAM_APPID(appid)       (0x01000000|appid)
#define SV_OPENPROGRAM_MASK               0x00ffffff

#define SV_CAPTURE_FLAG_ONLYFRAME     0x0001
#define SV_CAPTURE_FLAG_WAITFORNEXT   0x0002
#define SV_CAPTURE_FLAG_ONLYFIELD     0x0004
#define SV_CAPTURE_FLAG_NOCPUCONVERT  0x0008
#define SV_CAPTURE_FLAG_AUDIO         0x0010

#define SV_KDMKEY_FORMAT_MASK         0x0f
#define SV_KDMKEY_FORMAT_PLAIN_R2L    0x01
#define SV_KDMKEY_FORMAT_BASE64       0x02
#define SV_KDMKEY_FORMAT_PLAIN_L2R    0x03
#define SV_KDMKEY_FORMAT_HEXCODE_R2L  0x04
#define SV_KDMKEY_FORMAT_HEXCODE_L2R  0x05
#define SV_KDMKEY_ENCRYPTION_MASK     0xf0
#define SV_KDMKEY_ENCRYPTION_RSA      0x10
#define SV_KDMKEY_ENCRYPTION_NONE     0x20

#ifndef _DVS_CLIB_H_SV_DEFINESONLY_
typedef struct {
  int version;
  int storagemode;
  int maxbuffersize;
  int maxxsize;
  int maxysize;
  int xsize;
  int ysize;
  int lineoffset;
  int pad[7];
} sv_capture_info;

typedef struct {
  int cookie;       ///! Version:1 - cookie
  int tick;         ///! Version:1 - tick
  int clockhigh;    ///! Version:1 - upper 32 bit of clock
  int clocklow;     ///! Version:1 - lower 32 bit of clock
  int xsize;        ///! Version:1 - xsize
  int ysize;        ///! Version:1 - ysize
  int lineoffset;   ///! Version:1 - lineoffset
  int storagemode;  ///! Version:1 - storagemode
  int matrixtype;   ///! Version:1 - matrixtype
  int buffertype;   ///! Version:2 - frame=0,field1-1,field2-2
  int timecode;     ///! Version:2 - Timeline timecode
  int framenumber;  ///! Version:2 - Timeline framenumber
  int ltc;          ///! Version:2 - Analog LTC timecode
  int vitc;         ///! Version:2 - Analog VITC timecode.
  int vtrslave;     ///! Version:2 - VTR Slave timecode.
  int vtrmaster;    ///! Version:2 - VTR Master timecode.
  int dltc;         ///! Version:2 - Digital/ANC LTC timecode.
  int dvitc;        ///! Version:2 - Digital/ANC VITC timecode.
  int eyemode;      ///! Version:3 - SV_EYEMODE_xxx (INTERNAL)
  int audioformat;  ///! Version:4 - 1632 -> 16 channels, 32 bits (only if flag SV_CAPTURE_FLAG_AUDIO is set) (INTERNAL)
  int audiooffset;  ///! Version:4 - Offset to first audio sample in buffer (only if flag SV_CAPTURE_FLAG_AUDIO is set) (INTERNAL)
  int audiosize;    ///! Version:4 - Size of audio captured (only if flag SV_CAPTURE_FLAG_AUDIO is set) (INTERNAL)
  int usercookiehigh; ///! Version:5 - Avus buffermagic to synchronize capture buffers for proxy creation (INTERNAL)
  int usercookielow;  ///! Version:5 - Avus buffermagic to synchronize capture buffers for proxy creation (INTERNAL)
  int pad[8];
} sv_capturebuffer;
#endif

#define SV_JP2K_PROGRESSION_CPRL              0   // Component-Position-Resolution-Layer
#define SV_JP2K_PROGRESSION_LRCP              1   // Layer-Resolution-Component-Position
#define SV_JP2K_PROGRESSION_RLCP              2   // Resolution-Layer-Component-Position
#define SV_JP2K_PROGRESSION_RPCL              3   // Resolution-Position-Component-Layer
#define SV_JP2K_PROGRESSION_PCRL              4   // Position-Component-Resolution-Layer

#define SV_JP2K_RATECONTROL_DEFAULT           0
#define SV_JP2K_RATECONTROL_RAWMODE           1
#define SV_JP2K_RATECONTROL_TARGETSIZE        2
#define SV_JP2K_RATECONTROL_QUALITY           3
#define SV_JP2K_RATECONTROL_LAYERSTARGETSIZE  4
#define SV_JP2K_RATECONTROL_LAYERSQUALITY     5

#define SV_JP2K_CODESTYLE_J2C                 0
#define SV_JP2K_CODESTYLE_JP2                 1
#define SV_JP2K_CODESTYLE_RAW                 2

#define SV_JP2K_CODEBLOCKSIZE_128x32          0
#define SV_JP2K_CODEBLOCKSIZE_64x32           1
#define SV_JP2K_CODEBLOCKSIZE_32x32           2
#define SV_JP2K_CODEBLOCKSIZE_64x64           3

#define SV_JP2K_WAVELET_LOSSY_9X7             0
#define SV_JP2K_WAVELET_LOSSLESS_5X3          1
#define SV_JP2K_WAVELET_LOSSY_5X3             2

#define SV_JP2K_ATTRDATA_DEFAULT              0
#define SV_JP2K_ATTRDATA_RAWMODE              1
#define SV_JP2K_ATTRDATA_LOG2UNADJUSTED       2
#define SV_JP2K_ATTRDATA_LOG2ADJUSTED         3

#define SV_JP2K_PARAMS_DCIMODE                0x000001
#define SV_JP2K_PARAMS_RATECONTROL            0x000002
#define SV_JP2K_PARAMS_PROGRESSION            0x000004
#define SV_JP2K_PARAMS_WAVELET                0x000008
#define SV_JP2K_PARAMS_CODEBLOCKSIZE          0x000010
#define SV_JP2K_PARAMS_ATTRDATA               0x000020
#define SV_JP2K_PARAMS_QFACTOR                0x000040
#define SV_JP2K_PARAMS_BITDEPTH               0x000080
#define SV_JP2K_PARAMS_TRANSFORMS             0x000100
#define SV_JP2K_PARAMS_MATRIXTYPE             0x000200
#define SV_JP2K_PARAMS_STEPSIZES              0x010000
#define SV_JP2K_PARAMS_VISUALFACTORS          0x020000
#define SV_JP2K_PARAMS_RASTER                 0x040000
#define SV_JP2K_PARAMS_DCI4KMODE              0x080000

#define SV_COMPRESSION_CODE(a, b, c, d) ((unsigned int)((a)|((b)<<8)|((c)<<16)|((d)<<24)))
#define SV_COMPRESSION_JPEG2K         SV_COMPRESSION_CODE('J','P','2','K')

#define SV_ENCRYPTION_CODE(a, b, c, d) ((unsigned int)((a)|((b)<<8)|((c)<<16)|((d)<<24)))
#define SV_ENCRYPTION_NONE            0x00000000
#define SV_ENCRYPTION_AES             SV_ENCRYPTION_CODE('A','E','S',' ')

#define SV_COMPATIBLE_MIXER           0x00000001
#define SV_COMPATIBLE_BLACKINIT       0x00000002

#ifndef _DVS_CLIB_H_SV_DEFINESONLY_
typedef union {
  struct {
    int params;
    int ratecontrol;
    int progression;
    int wavelet;
    int codeblocksize;
    int attrdata;
    int qfactor;
    int bitdepth;
    int transforms;
    int dstmatrixtype;
    int srcmatrixtype;
    int rasterprolongation;
    int pad[2];
    struct {
      int      target;
      int      targetlayer[16];
      int      ntargetlayer;
      int      stepsizes[19];
      int      nstepsizes;
      int      visualfactors[19];
      int      nvisualfactors;
      int      pad[32];
    } component[3];
  } v1;
} sv_jpegencode_parameters;
#endif


#if !defined(_DVS_CLIB_H_SV_DEFINESONLY_) && !defined(_DVS_CLIB_NO_FUNCTIONS_)
/*
//           function prototypes
*/

export sv_handle * sv_open(
#ifdef __PROTOTYPES__
        char *      setup
#endif
);



export int sv_openex(
#ifdef __PROTOTYPES__
        sv_handle ** psv,
        char *      setup,
        int         openprogram,
        int         opentype,
        int         timeout,
        int         spare
#endif
);

export int  sv_asc2tc(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        char *      asc, 
        int *       tc
#endif
);

export int sv_black(
#ifdef __PROTOTYPES__
        sv_handle * sv
#endif
);


export int sv_capture(
#ifdef __PROTOTYPES__
        sv_handle *     sv, 
        char *          buffer, 
        int             buffersize, 
        int             lineoffset, 
        int *           pxsize, 
        int *           pysize, 
        int *           ptick, 
        uint32 *        pclockhigh, 
        uint32 *        pclocklow, 
        int             flags, 
        sv_overlapped * poverlapped
#endif
);

export int sv_capture_ready(
#ifdef __PROTOTYPES__
        sv_handle *     sv, 
        sv_overlapped * poverlapped, 
        int *           pxsize, 
        int *           pysize, 
        int *           ptick, 
        uint32 *        pclockhigh, 
        uint32 *        pclocklow
#endif
);


export int sv_capture_status(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        sv_capture_info * pinfo
#endif
);


export int sv_captureex(
#ifdef __PROTOTYPES__
        sv_handle *        sv, 
        char *             buffer,
        int                buffersize,
        sv_capturebuffer * pcapture, 
        int                version,
        int                flags, 
        sv_overlapped *    poverlapped
#endif
);

export int sv_captureex_ready(
#ifdef __PROTOTYPES__
        sv_handle *        sv, 
        sv_overlapped *    poverlapped, 
        sv_capturebuffer * pcapture, 
        int                version
#endif
);


export int  sv_close(
#ifdef __PROTOTYPES__
        sv_handle * sv
#endif
);

export int sv_colorbar(
#ifdef __PROTOTYPES__
        sv_handle * sv
#endif
);


export int sv_currenttime(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         type, 
        int *       ptick, 
        uint32 *    pclockhigh, 
        uint32 *    pclocklow
#endif
);


export int  sv_debugprint(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        char *      buffer, 
        int         buffer_size, 
        int *       buffer_count
#endif
);

export int  sv_debugdump(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         debugid,
        char *      buffer, 
        int         buffer_size, 
        int *       buffer_count,
        int         flags,
        int         spare
#endif
);

export int  sv_display(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        char *      buffer, 
        int         size,
        int         xsize,
        int         ysize,
        int         start,
        int         nframes,
        int         tc
#endif
);

export void sv_errorprint(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         code
#endif
);

export char * sv_errorstring(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         code
#endif
);

export char * sv_geterrortext(
#ifdef __PROTOTYPES__
        int         code
#endif
);

export int sv_getlicence(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int *       pType, 
        int *       pRevision, 
        int *       pSerial, 
        int *       pVersion, 
        int *       pRam, 
        int *       pDisk, 
        int *       pFlags, 
        int         dim,
        unsigned    aKeys[/*dim*/]
#endif
);

export int sv_goto(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         frame
#endif
);

export int  sv_host2sv(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        char *      buffer, 
        int         size, 
        int         xsize, 
        int         ysize, 
        int         start, 
        int         nframes, 
        int         mode
#endif
);

typedef struct {
 int version;
 int queryindex;
 int queryparam;
 int expectedvalue;
 int actualvalue;
 int divisor;
 int pad[3];
} sv_hwverify_rec;

export int sv_hwverify(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        sv_hwverify_rec * hwverify,
        int spare
#endif
);

export int sv_inpoint(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         frame
#endif
);

export int sv_jack_assign(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         jack, 
        int         channel
#endif
);

export int sv_jack_find(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         jack, 
        char *      pstring, 
        int         stringsize, 
        int *       presult
#endif
);

export int sv_jack_status(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         jack, 
        sv_jack_info * pinfo
#endif
);

export int sv_jack_option_get(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         jack, 
        int         option, 
        int *       pvalue
#endif
);

export int sv_jack_option_set(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         jack, 
        int         option, 
        int         pvalue
#endif
);

export int sv_jack_query(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         jack, 
        int         query, 
        int         param, 
        int *       presult
#endif
);

#define sv_license      sv_licence
export int sv_licence(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         knum,
        char *      code
#endif
);

export int sv_licenceinfo(
#ifdef __PROTOTYPES__
        sv_handle *     sv, 
        int *           pdevtype, 
        int *           pserial, 
        int *           pexpire,
        unsigned char * pfeatures, 
        int             featuresize, 
        unsigned char * pkeys, 
        int             keysize
#endif
);

export char * sv_licencebit2string(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int bitno
#endif
);


export int  sv_live(
#ifdef __PROTOTYPES__
        sv_handle * sv
#endif
);


export int sv_lut(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         command, 
        int *       ptable, 
        int         lutid
#endif
);


export int sv_matrix(
#ifdef  __PROTOTYPES__
        sv_handle *     sv, 
        int             matrixtype,
        sv_matrixinfo * pmatrix 
#endif
);


export int sv_matrixex(
#ifdef  __PROTOTYPES__
        sv_handle *       sv,
        int               matrixtype,
        sv_matrixexinfo * pmatrix,
        sv_matrixexinfo * pquery
#endif
);


export int sv_memory_delay(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         nframes, 
        int         inpoint, 
        int         outpoint
#endif
);


export int sv_memory_dma(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         btomemory, 
        char *      addr, 
        int         offset, 
        int         size, 
        sv_overlapped * overlapped
#endif
);

export int sv_memory_dmax(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         btomemory, 
        char *      addr, 
        int         offseth, 
        int         offsetl, 
        int         size, 
        sv_overlapped * overlapped
#endif
);


export int sv_memory_dmarect(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         btomemory, 
        char *      memoryaddr, 
        int         memorysize, 
        int         offset,
        int         xoffset, 
        int         yoffset, 
        int         xsize, 
        int         ysize, 
        int         lineoffset, 
        int         spare, 
        sv_overlapped * poverlapped
#endif
);


export int sv_memory_dmaex(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         btomemory, 
        char *      memoryaddr, 
        int         memorysize, 
        int         memoryoffset, 
        int         memorylineoffset, 
        int         cardoffset, 
        int         cardlineoffset, 
        int         linesize, 
        int         linecount, 
        int         spare, 
        sv_overlapped * poverlapped
#endif
);


export int sv_memory_dmalist(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         btomemory, 
        char *      memoryaddr, 
        int         memorysize, 
        int         count, 
        sv_dmalist * plist, 
        int         spare, 
        sv_overlapped * poverlapped
#endif
);


export int sv_memory_dma_ready(
#ifdef  __PROTOTYPES__
        sv_handle *     sv, 
        sv_overlapped * poverlapped,
        int             resorg
#endif
);


export int sv_memory_frameinfo(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         frame, 
        int         channel,
        int *       field1addr, 
        int *       field1size, 
        int *       field2addr, 
        int *       field2size
#endif
);


export int sv_memory_play(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         inpoint, 
        int         outpoint, 
        double      speed, 
        int         tc, 
        int         flags
#endif
);


export int sv_memory_record(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         inpoint, 
        int         outpoint, 
        double      speed, 
        int         tc, 
        int         flags
#endif
);


export int sv_mixer_edge(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         edge,
        int         edgeparam,
        int         edgewidth,
        int         spare
#endif
);

export int sv_mixer_input(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         mixa,
        int         mixb,
        int         mixc,
        int         spare
#endif
);

export int sv_mixer_mode(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         mode,
        int         param,
        int         start,
        int         end,
        int         nframes,
        int         spare
#endif
);

export int sv_mixer_status(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        sv_mixer_info * info
#endif
);

export int sv_monitorinfo(
#ifdef  __PROTOTYPES__
      sv_handle * sv, 
      int         spare, 
      char *      buffer, 
      int         buffersize
#endif
);


export int  sv_option(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         code, 
        int         value
#endif
);

export int  sv_option_available(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         code,
        int         value
#endif
);

export int sv_option_get(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         code, 
        int *       value
#endif
);

export int sv_option_set(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         code, 
        int         value
#endif
);

export int sv_option_setat(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         code,
        int         value,
        int         when
#endif
);

export int sv_option_menu(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         menu,
        int         submenu,
        int         menulabel, 
        char *      label, 
        int         labelsize, 
        int *       pvalue, 
        int *       pmask, 
        int *       spare
#endif
);

export int sv_option_load(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         code, 
        int *       value
#endif
);

export int sv_option_save(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         code, 
        int         value
#endif
);

export int sv_option_file_read(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         card,
        int         set,
        int         code,
        int *       value
#endif
);

export int sv_option_file_write(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         card,
        int         code,
        int         value
#endif
);


export char * sv_option_value2string(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         option, 
        int         value
#endif
);


export int sv_outpoint(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         frame
#endif
);


export int sv_overlay(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         entry, 
        int         xoffset, 
        int         yoffset, 
        int         xsize, 
        int         ysize, 
        int         textcolor,
        int         backcolor,
        int         flags, 
        char *      pbuffer
#endif
);


export int  sv_query(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         cmd,
        int         par,
        int *       val
#endif
);

export char * sv_query_value2string(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         option, 
        int         value
#endif
);

export int  sv_pattern(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         pattern,
        int         frame, 
        int         chan 
#endif
);

export int  sv_poll(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         task,
        int         loop
#endif
);

export int sv_position(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         frame,
        int         field,
        int         repeat,
        int         flags
#endif
);

export int sv_preset(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         preset
#endif
);

export int sv_pulldown(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         cmd,
        int         param
#endif
);

export int sv_raster_status(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         rasterid,
        sv_rasterheader * raster,
        int         rastersize,
        int *       nrasters,
        int         spare
#endif
);

export int sv_record(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        char *      buffer, 
        int         size, 
        int *       xsize, 
        int *       ysize, 
        int         start, 
        int         nframes, 
        int         tc
#endif
);

export int sv_rs422_open(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         device, 
        int         baudrate, 
        int         flags
#endif
);

export int sv_rs422_close(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         device
#endif
);

export int sv_rs422_rw(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         device, 
        int         bwrite, 
        char *      buffer, 
        int         buffersize, 
        int *       bytecount, 
        int         flags
#endif
);

export int sv_rs422_port(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         device, 
        int         baudrate, 
        int         porttype
#endif
);

export int sv_showinput(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         showinput,
        int         noblock
#endif
);


export int sv_slave(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         mode
#endif
);

export int sv_slaveinfo_set(
#ifdef __PROTOTYPES__
        sv_handle *     sv, 
        sv_slaveinfo *  slaveinfo
#endif
);

export int  sv_slaveinfo_get(
#ifdef __PROTOTYPES__
        sv_handle *     sv, 
        sv_slaveinfo *  slaveinfo,
        sv_overlapped * poverlapped
#endif
);


export int sv_vtrslave_set(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         option,
        int         value
#endif
);


export int sv_vtrslave_get(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         option,
        int *       value
#endif
);

export int sv_realtimeclock(
#ifdef  __PROTOTYPES__
        sv_handle * sv, 
        int         clocktype, 
        sv_clock_info * pclock, 
        int         flags
#endif
);


export int  sv_status(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        sv_info *   info
#endif
);

export int  sv_stop(
#ifdef __PROTOTYPES__
        sv_handle * sv
#endif
);

export int  sv_sv2host(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        char *      buffer, 
        int         size, 
        int         xsize, 
        int         ysize, 
        int         start, 
        int         nframes, 
        int         mode
#endif
);

export int sv_sync(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         sync
#endif
);

export int sv_sync_output(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         syncout
#endif
);

export int  sv_tc2asc(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         tc, 
        char *      asc, 
        int         asclen
#endif
);

export int sv_usleep(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         microsecs
#endif
);

export int sv_version_status(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        sv_version * version,
        int         versionsize,
        int         deviceid,
        int         moduleid,
        int         spare
#endif
);

export int sv_version_check(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         major,
        int         minor,
        int         patch,
        int         fix
#endif
);

export int sv_version_verify(
#ifdef  __PROTOTYPES__
        sv_handle *  sv, 
        unsigned int neededlicence,
        char *       errorstring, 
        int          errorstringsize
#endif
);

export int sv_version_check_firmware(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        char *      current,
        int         current_size,
        char *      recommended,
        int         recommended_size
#endif
);

export int  sv_videomode(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         videomode
#endif
);


export int sv_vtrcontrol(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         binput, 
        int         init, 
        int         tc, 
        int         nframes, 
        int *       pwhen, 
        int *       vtrtc,
        int         flags
#endif
);

export int sv_vtredit(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         tc,
        int         nframes
#endif
);

export int  sv_vtrmaster(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         option, 
        int         value
#endif
);

export int  sv_vtrmaster_get(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         option, 
        int *       pvalue
#endif
);

export int  sv_vtrmaster_info(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int *       ptimecode, 
        int *       puserbytes,
        char *      pinfobits,
        int *       ptick,
        int *       pclock_low,
        int *       pclock_high
#endif
);

export int sv_vtrmaster_raw(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        char *      cmdstr,
        char *      replystr
#endif
);

export int  sv_vsyncwait(
#ifdef __PROTOTYPES__
        sv_handle * sv, 
        int         operation, 
        sv_vsyncwait_info * pinfo
#endif
);

export int  sv_timecode_feedback(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        sv_timecode_info* input,
        sv_timecode_info* output
#endif
);

/* Note, sv_wait() is used by semaphores on irix, this function is now called sv_waitvtr */
export int  sv_waitvtr(
#ifdef __PROTOTYPES__
        sv_handle * sv,
        int         poll
#endif
);

export int sv_zoom(
#ifdef  __PROTOTYPES__
        sv_handle * sv,
        int         xzoom,
        int         yzoom,
        int         xpanning,
        int         ypanning,
        int         zoomflags
#endif
);




/*
//  Query image data storage format from video device
*/
export int sv_storage_status(
#ifdef  __PROTOTYPES__
        sv_handle *      sv,
        int              cookie,
        sv_storageinfo * psiin,
        sv_storageinfo * psiout,
        int              psioutsize,
        int              flags
#endif
   );


/**
//  \ingroup svclib
//  
//  \struct sv_jack_memoryinfo
//
//  The following details the memory setup structure used by sv_jack_memorysetup().
*/
typedef struct {
  struct {
    int bytes;                ///< Amount of used memory in bytes.
    int percent;              ///< Amount of used memory in percent.
    int pad[16-2];            ///< Reserved for future use.
  } usage;

  struct {
    int override;             ///< Set to TRUE if limits should be used.
    int storagemode;          ///< Maximum video storage mode (SV_MODE_*).
    int xsize;                ///< Maximum video xsize.
    int ysize;                ///< Maximum video ysize.
    int audiochannels;        ///< Maximum audio channels.
    int audiofreq;            ///< Maximum audio frequency.
    int audiobits;            ///< Maximum audio bit depth.
    int pad[16-6];            ///< Reserved for future use.
  } limit;

  struct {
    int frames;               ///< Amount of frames.
    int videosize;            ///< Size of one video frame.
    int audiosize;            ///< Size of one audio frame.
    int pad[16-3];            ///< Reserved for future use.
  } info;
  int pad[32];                ///< Reserved for future use.
} sv_jack_memoryinfo;

export int sv_jack_memorysetup(
#ifdef  __PROTOTYPES__
  sv_handle * sv,
  int bquery,
  sv_jack_memoryinfo ** info,
  int njacks,
  int * pjacks,
  int flags
#endif
);

/*
//  Check current version vs file version
*/
export int sv_version_certify(
#ifdef  __PROTOTYPES__
  sv_handle * sv,
  char *      path,
  int *       required_sw,
  int *       required_fw,
  int *       bcertified,
  void *      spare
#endif
);

export int sv_kdmkey_load(
#ifdef  __PROTOTYPES__
  sv_handle * sv,
  int         keyid,
  char *      value,
  int         size,
  int         flags
#endif
);

export int sv_kdmkey_readback(
#ifdef  __PROTOTYPES__
  sv_handle * sv,
  int *       keyid,
  char      * data,
  int         size
#endif
);

export int sv_random(
#ifdef  __PROTOTYPES__
  sv_handle * sv,
  char *      data,
  int         size
#endif
);

export int sv_publickey(
#ifdef  __PROTOTYPES__
  sv_handle * sv,
  char *      data,
  int         size
#endif
);

export int sv_jp2k_encodeparameters(
#ifdef __PROTOTYPES__
      sv_handle * sv,
      int         jack, 
      int         version, 
      int         size, 
      sv_jpegencode_parameters * params
#endif
);

#endif /* !defined(_DVS_CLIB_H_SV_DEFINESONLY_) */

#ifdef __cplusplus 
  } 
#endif 

#endif

