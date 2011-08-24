/*
 *      DVS C-Library Errors Header File, Copyright (c) 1994-2010 DVS Digital Video Systems AG
 *
 *      This file contains the error codes and it is possible
 *      to include them into an application with the following 
 *      construct:
 *
 *
 *      #undef _DVS_ERRORS_H_     // include the error header file again with
 *      #define ERRSTRUCT         // ERRSTRUCT defined to get a global error
 *      #include "dvs_errors.h"   // structure inserted here          .     
 *  
 *      char * sv_geterrortext(int errorcode)
 *      {
 *        int i;
 *         
 *        for(i = 0; (error[i].code != MAXERRCODE) && (error[i].code != errorcode); i++);
 *
 *        return error[i].text;
 *      }
 *
 */

#ifndef _DVS_ERRORS_H_
#define _DVS_ERRORS_H_

/* don't touch these lines *vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
#ifdef ERRSTRUCT
#define err(nam,val,txt)	{ nam, txt },
struct error {
	int		code;
	char	*text;
} global_dvs_errors[] = {
#elif defined(QT_GENERATE_ERROR)
#define INCLUDE_NOP(x) x
INCLUDE_NOP(#include <qstring.h>)
INCLUDE_NOP(#include <qobject.h>)

QString ErrorText( int  error_code )
{
    QString error_text;

    switch( error_code )
    {
#define err(nam,val,txt) case val: error_text = QObject::tr( txt, "sv error" );break;
#else
#define err(nam,val,txt)	nam = val,
enum {
#endif
/* don't touch these lines *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

/* max message length for on screen display is 42 chars/line
"123456789012345678901234567890123456789012"
*/

err(SV_ACTIVE,                          -1,
"SV_ACTIVE")

err(SV_OK,                              0,
"SV_OK")

/*** 0x00xx SCSIVIDEO error messages	***/

err(SV_ERROR_PARAMETER,             1,
"SV_ERROR_PARAMETER. Parameter error.")

err(SV_ERROR_MALLOC,                2,
"SV_ERROR_MALLOC. Memory allocation failed.")

err(SV_ERROR_NOTIMPLEMENTED,	    3,
"SV_ERROR_NOTIMPLEMENTED. A not implemented function was called.")

err(SV_ERROR_BUFFERSIZE,            4,
"SV_ERROR_BUFFERSIZE. Buffer too small.")

err(SV_ERROR_NOTEXTSYNC,            5,
"SV_ERROR_NOTEXTSYNC. Not in external sync.")

err(SV_ERROR_VIDEOMODE,             6,
"SV_ERROR_VIDEOMODE. Wrong video mode specified.")

err(SV_ERROR_FILEOPEN,              7,
"SV_ERROR_FILEOPEN. Couldn't open file.")

err(SV_ERROR_FILECREATE,            8,
"SV_ERROR_FILECREATE. Couldn't create file.")

err(SV_ERROR_DATALOST,              9,
"SV_ERROR_DATALOST. Data lost during transfer.")

err(SV_ERROR_SVNULL,                10,
"SV_ERROR_SVNULL. Invalid SV handle.")

err(SV_ERROR_SVMAGIC,               11,
"SV_ERROR_SVMAGIC. Invalid SV magic number.")

err(SV_ERROR_FILEREAD,              12,
"SV_ERROR_FILEREAD. Couldn't read from file.")

err(SV_ERROR_FILEWRITE,             13,
"SV_ERROR_FILEWRITE. Couldn't write to file.")

err(SV_ERROR_FILECLOSE,             14,
"SV_ERROR_FILECLOSE. Couldn't close file.")

err(SV_ERROR_FILEDIRECT,            15,
"SV_ERROR_FILEDIRECT. Couldn't set direct file access.")

err(SV_ERROR_FILESEEK,              16,
"SV_ERROR_FILESEEK. Couldn't seek in file.")

err(SV_ERROR_FILETRUNCATE,          17,
"SV_ERROR_FILETRUNCATE. Couldn't truncate file.")

err(SV_ERROR_SVJ_FRAMENR,           18,
"SV_ERROR_SVJ_FRAMENR. Frame number not in file.")

err(SV_ERROR_SVJ_NULL,              19,
"SV_ERROR_SVJ_NULL. Null svj handle.")

err(SV_ERROR_SCSI,                  20,
"SV_ERROR_SCSI. SCSI transfer error.")

err(SV_ERROR_TIMECODE,              21,
"SV_ERROR_TIMECODE. Timecode not valid, format is 01:00:00:00.")

err(SV_ERROR_MASTER,                22,
"SV_ERROR_MASTER. Master control error.")

err(SV_ERROR_MEM_NULL,              23,
"SV_ERROR_MEM_NULL. Null mem handle.")

err(SV_ERROR_MEM_BUFFERSIZE,	    24,
"SV_ERROR_MEM_BUFFERSIZE. Memory buffer too small.")

err(SV_ERROR_VIDEOPAGE,             25,
"SV_ERROR_VIDEOPAGE. Non-existing video page.")

err(SV_ERROR_TRANSFER,              26,
"SV_ERROR_TRANSFER. Transfer failed.")

err(SV_ERROR_NOCARRIER,	            27,
"SV_ERROR_NOCARRIER. No input signal detected.")

err(SV_ERROR_NOGENLOCK,             28,
"SV_ERROR_NOGENLOCK. Genlock option not available.")

err(SV_ERROR_NODRAM,                29,
"SV_ERROR_NODRAM. DRAM option not available.")

err(SV_ERROR_FIRMWARE,              30,
"SV_ERROR_FIRMWARE. The firmware version is wrong.")

err(SV_ERROR_QUANTLOSS,             31,
"SV_ERROR_QUANTLOSS. AC quant values lost during compression.")

err(SV_ERROR_RECORD,                32,
"SV_ERROR_RECORD. Error during record operation.")

err(SV_ERROR_SLAVE,                 33,
"SV_ERROR_SLAVE. Device is in remote control mode.")

err(SV_ERROR_DISKFORMAT,            34,
"SV_ERROR_DISKFORMAT. Unappropriate disk format for requested video mode.")

err(SV_ERROR_PROGRAM,               35,
"SV_ERROR_PROGRAM. Illegal internal state.")

err(SV_ERROR_TIMELINE,              36,
"SV_ERROR_TIMELINE. Illegal segment specified.")

err(SV_ERROR_SCSIDEVICE,            37,
"SV_ERROR_SCSIDEVICE. SCSI device not found.")

err(SV_ERROR_SCSIWRITE,             38,
"SV_ERROR_SCSIWRITE. SCSI write failed.")

err(SV_ERROR_SCSIREAD,              39,
"SV_ERROR_SCSIREAD. SCSI read failed.")

err(SV_ERROR_DTM_TIMEOUT,           40,
"SV_ERROR_DTM_TIMEOUT. Connection timeout.")

err(SV_ERROR_NOTSUPPORTED,          41,
"SV_ERROR_NOTSUPPORTED. Not supported feature.")

err(SV_ERROR_CLIP_NOTFOUND,         42,
"SV_ERROR_CLIP_NOTFOUND. Clip not found.")

err(SV_ERROR_CLIP_NAMEEXISTS,       43,
"SV_ERROR_CLIP_NAMEEXISTS. Clip name already exists.")

err(SV_ERROR_CLIP_NOENTRY,          44,
"SV_ERROR_CLIP_NOENTRY. Clip directory full.")

err(SV_ERROR_CLIP_OVERLAP,          45,
"SV_ERROR_CLIP_OVERLAP. Clip would overlap with directory.")

err(SV_ERROR_CLIPDIR_NOTFOUND,      46,
"SV_ERROR_CLIPDIR_NOTFOUND. Directory not found.")

err(SV_ERROR_CLIPDIR_NAMEEXISTS,    47,
"SV_ERROR_CLIPDIR_NAMEEXISTS. Directory name already exists.")

err(SV_ERROR_CLIPDIR_NOENTRY,       48,
"SV_ERROR_CLIPDIR_NOENTRY. Directory field full.")

err(SV_ERROR_CLIPDIR_OVERLAP,       49,
"SV_ERROR_CLIPDIR_OVERLAP. Directory would overlap \
with other clip or directory.")

err(SV_ERROR_NOLICENCE,             50,
"SV_ERROR_NOLICENCE. No license for this operation.")

err(SV_ERROR_FRAME_NOACCESS,        51,
"SV_ERROR_FRAME_NOACCESS. Frame not accessible.")

err(SV_ERROR_PARTITION_NOENTRY,     52,
"SV_ERROR_PATTITION_NOENTRY. No free entry in partition table.")

err(SV_ERROR_PARTITION_NOSPACE,     53,
"SV_ERROR_PARTITION_NOSPACE. Requested space exceeds available space.")

err(SV_ERROR_PARTITION_NOTLAST,     54,
"SV_ERROR_PARTITION_NOTLAST. Only the last partition can be deleted.")

err(SV_ERROR_PARTITION_NOTFOUND,    55,
"SV_ERROR_PARTITION_NOTFOUND. Invalid partition name.")

err(SV_ERROR_PARTITION_INVALID,     56,
"SV_ERROR_PARTITION_INVALID. Invalid active partition.")

err(SV_ERROR_AUDIO_SEGMENT,         57,
"SV_ERROR_AUDIO_SEGMENT. Illegal audio segment specified.")

err(SV_ERROR_POLL_TASK_ACTIVE,	    58,
"SV_ERROR_POLL_TASK_ACTIVE. Poll task active.")

err(SV_ERROR_HARDWARELOAD,          59,
"SV_ERROR_HARDWARELOAD. Hardware failed to load.")

err(SV_ERROR_CLIPDIR_NOTEMPTY,      60,
"SV_ERROR_CLIPDIR_NOTEMPTY. Directory not empty.")

err(SV_ERROR_DISABLED,              61,
"SV_ERROR_DISABLED. A disabled function has been called.")

err(SV_ERROR_WRONG_HARDWARE,        62,
"SV_ERROR_WRONG_HARDWARE. Wrong hardware for operation.")

err(SV_ERROR_SYNCMODE,              63,
"SV_ERROR_SYNCMODE. Invalid sync mode selected.")

err(SV_ERROR_IOMODE,                64,
"SV_ERROR_IOMODE. Invalid I/O mode selected.")

err(SV_ERROR_VIDEO_RASTER_TABLE,    65,
"SV_ERROR_VIDEO_RASTER_TABLE. Video raster table not initialized.")

err(SV_ERROR_VIDEO_RASTER_FILE,	    66,
"SV_ERROR_VIDEO_RASTER_FILE. Loading/checking raster definition file failed.")

err(SV_ERROR_SYNC_CALCULATION,	    67,
"SV_ERROR_SYNC_CALCULATION. Calculation of sync output signal failed.")

err(SV_ERROR_SYNC_OUTPUT,    	    68,
"SV_ERROR_SYNC_OUTPUT. Specified sync output signal not supported.")

err(SV_ERROR_FLASH_ERASETIMEOUT,    69,
"SV_ERROR_FLASH_ERASETIMEOUT. A timeout during the erase of the flash appeared.")

err(SV_ERROR_FLASH_VERIFY,          70,
"SV_ERROR_FLASH_VERIFY. Verify of the flash after programming failed.")

err(SV_ERROR_EPLD_MAGIC,            71,
"SV_ERROR_EPLD_MAGIC. An EPLD with the wrong magic number detected.")

err(SV_ERROR_EPLD_PRODUCT,          72,
"SV_ERROR_EPLD_PRODUCT. An EPLD from the wrong device detected.")

err(SV_ERROR_EPLD_CHIP,             73,
"SV_ERROR_EPLD_CHIP. An EPLD with the wrong chip ID detected.")

err(SV_ERROR_EPLD_VERSION,          74,
"SV_ERROR_EPLD_VERSION. An EPLD with the wrong version detected.")

err(SV_ERROR_NOTREADY,		    75,
"SV_ERROR_NOTREADY. Operation is not ready.")

err(SV_ERROR_NOTDEBUGDRIVER,        76,
"SV_ERROR_NOTDEBUGDRIVER. This is only supported by the debug driver.")

err(SV_ERROR_DRIVER_CONNECTIRQ,     77,
"SV_ERROR_DRIVER_CONNECTIRQ. Driver could not connect to an IRQ.")

err(SV_ERROR_DRIVER_MAPIOSPACE,     78,
"SV_ERROR_DRIVER_MAPIOSPACE. Driver could not map on-board memory into kernel memory.")

err(SV_ERROR_DRIVER_RESOURCES,      79,
"SV_ERROR_DRIVER_RESOURCES. Driver did not get resources from the kernel.")

err(SV_ERROR_DRIVER_MALLOC,         80,
"SV_ERROR_DRIVER_MALLOC. Driver could not malloc critical memory.")

err(SV_ERROR_VSYNCPASSED,           81,
"SV_ERROR_VSYNCPASSED. An operation was issued for a vsync that has already passed.")

err(SV_ERROR_VSYNCFUTURE,           82,
"SV_ERROR_VSYNCFUTURE. An operation was issued too long before it should start.")

err(SV_ERROR_IOCTL_FAILED,          83,
"SV_ERROR_IOCTL_FAILED. An ioctl operation failed.")

err(SV_ERROR_FIFO_TIMEOUT,          84,
"SV_ERROR_FIFO_TIMEOUT. The FIFO timed out.")

err(SV_ERROR_FIFO_PUTBUFFER,        85,
"SV_ERROR_FIFO_PUTBUFFER. The FIFO getbuffer/putbuffer was called incorrectly.")

err(SV_ERROR_SAMPLINGFREQ,    	    86,
"SV_ERROR_SAMPLINGFREQ. Illegal sampling frequeny specified.")

err(SV_ERROR_MMAPFAILED,    	    87,
"SV_ERROR_MMAPFAILED. Memory mapping function failed.")

err(SV_ERROR_TIMEOUT,		    88,
"SV_ERROR_TIMEOUT. Operation timed out.")

err(SV_ERROR_CANCELED,		    89,
"SV_ERROR_CANCELED. Operation was cancelled.")

err(SV_ERROR_UNKNOWNFLASH,          90,
"SV_ERROR_UNKNOWNFLASH. This flash chip is not supported by the software.")

err(SV_ERROR_WRONG_COLORMODE,       91,
"SV_ERROR_WRONG_COLORMODE. This color mode or operation for this color mode is not supported.")

err(SV_ERROR_DRIVER_HWPATH,         92,
"SV_ERROR_DRIVER_HWPATH. Path to driver hardware files is missing.")

err(SV_ERROR_DISPLAYONLY,           93,
"SV_ERROR_DISPLAYONLY. This video raster can only be used for display.")

err(SV_ERROR_VTR_OFFLINE,           94,
"SV_ERROR_VTR_OFFLINE. There is no VTR connected.")

err(SV_ERROR_VTR_LOCAL,             95,
"SV_ERROR_VTR_LOCAL. The VTR is in local mode. Please check front panel switch.")

err(SV_ERROR_VTR_SERIAL,            96,
"SV_ERROR_VTR_SERIAL. Error from serial driver.")

err(SV_ERROR_VTR_NAK,               97,
"SV_ERROR_VTR_NAK. Received NAK (not acknowledged) from VTR.")

err(SV_ERROR_VTR_GOTOERROR,         98,
"SV_ERROR_VTR_GOTOERROR. Goto operation on VTR did not complete.")

err(SV_ERROR_VTR_NOSTATUS,          99,
"SV_ERROR_VTR_NOSTATUS. Status reply from VTR is missing.")

err(SV_ERROR_VTR_NOACK,             100,
"SV_ERROR_VTR_NOACK. Acknowledge from VTR is missing.")

err(SV_ERROR_VTR_NOTIMECODE,        101,
"SV_ERROR_VTR_NOTIMECODE. Timecode reply is wrong.")

err(SV_ERROR_VTR_NOTIMECODECHANGE,  102,
"SV_ERROR_VTR_NOTCHANGE. Timecode did not change during edit.")

err(SV_ERROR_VTR_TCORDER,           103,
"SV_ERROR_VTR_TCORDER. Timecode order during VTR edit is wrong.")

err(SV_ERROR_VTR_TICKORDER,         104,
"SV_ERROR_VTR_TICKORDER. Tick order during VTR edit is wrong.")

err(SV_ERROR_VTR_EDIT,              105,
"SV_ERROR_VTR_EDIT. Master control during edit failed.")

err(SV_ERROR_BUFFER_NOTALIGNED,     106,
"SV_ERROR_BUFFER_NOTALIGNED. The buffer does not have the needed alignment.")

err(SV_ERROR_BUFFER_NULL,           107,
"SV_ERROR_BUFFER_NULL. A buffer does not point to anything.")

err(SV_ERROR_BUFFER_TOLARGE,        108,
"SV_ERROR_BUFFER_TOLARGE. A buffer is too large.")

err(SV_ERROR_NOTFRAMESTORAGE,       109,
"SV_ERROR_NOTFRAMESTORAGE. This can only be done in frame storage mode.")

err(SV_ERROR_NOTRUNNING,	    110,
"SV_ERROR_NOTRUNNING. A polled operation is no longer active.")

err(SV_ERROR_NOHSWTRANSFER,	    111,
"SV_ERROR_NOHSWTRANSFER. Host-software transfer has been disabled.")

err(SV_ERROR_INPUT_VIDEO_NOSIGNAL,  112,
"SV_ERROR_INPUT_VIDEO_NOSIGNAL. Cannot detect an input video signal.")

err(SV_ERROR_INPUT_VIDEO_RASTER,    113,
"SV_ERROR_INPUT_VIDEO_RASTER. The input video signal does not match the video raster.")

err(SV_ERROR_INPUT_KEY_NOSIGNAL,    114,
"SV_ERROR_INPUT_KEY_NOSIGNAL. There is no key input detected.")

err(SV_ERROR_INPUT_KEY_RASTER,      115,
"SV_ERROR_INPUT_KEY_RASTER. The key signal input does not match the video raster.")

err(SV_ERROR_INPUT_AUDIO_NOAESEBU,  116,
"SV_ERROR_INPUT_AUDIO_NOAESEBU. There is no AES/EBU audio input detected.")

err(SV_ERROR_TRANSFER_NOAUDIO,      117,
"SV_ERROR_TRANSFER_NOAUDIO. There is no audio configured on the device.")

err(SV_ERROR_FLASH_ERASEVERIFY,     118,
"SV_ERROR_FLASH_ERASEVERIFY. Verification of the flash erase failed.")

err(SV_ERROR_INPUT_AUDIO_FREQUENCY, 119,
"SV_ERROR_INPUT_AUDIO_FREQUENCY. The audio input has the wrong frequency.")

err(SV_ERROR_INPUT_AUDIO_NOAIV,     120,
"SV_ERROR_INPUT_AUDIO_NOAIV. There is no AIV audio signal in video input detected.")

err(SV_ERROR_CHECKWORD,             121,
"SV_ERROR_CHECKWORD. Checkword is wrong.")

err(SV_ERROR_CLIPDIR_NOTSELECT,     122,
"SV_ERROR_CLIPDIR_NOTSELECT. A directory cannot be selected in this file system.")

err(SV_ERROR_EPLD_NOTFOUND,         123,
"SV_ERROR_EPLD_NOTFOUND. Could not find the PLD files needed to program the hardware.")

err(SV_ERROR_PARAMETER_NEGATIVE,    124,
"SV_ERROR_PARAMETER_NEGATIVE. A negative parameter is not valid.")

err(SV_ERROR_PARAMETER_TOLARGE,     125,
"SV_ERROR_PARAMETER_TOLARGE. A parameter is too large.")

err(SV_ERROR_ALREADY_RUNNING,       126,
"SV_ERROR_ALREADY_RUNNING. You tried to start an operation that was already started.")

err(SV_ERROR_WRONG_OS,       	    127,
"SV_ERROR_WRONG_OS. This function is not supported on the current operating system.")

err(SV_ERROR_TOMANYAUDIOCHANNELS,   128,
"SV_ERROR_TOMANYAUDIOCHANNELS. You tried to set too many audio channels.")

err(SV_ERROR_LICENCE_AUDIO,         129,
"SV_ERROR_LICENCE_AUDIO. You tried to use more audio channels than licensed.")

err(SV_ERROR_LICENCE_STREAMER,      130,
"SV_ERROR_LICENCE_STREAMER. You tried to use the streamer mode without license.")

err(SV_ERROR_LICENCE_RGB,           131,
"SV_ERROR_LICENCE_RGB. You tried to use RGB without license.")

err(SV_ERROR_LICENCE_KEYCHANNEL,    132,
"SV_ERROR_LICENCE_KEYCHANNEL. You tried to use key channel without license.")

err(SV_ERROR_LICENCE_MIXER,         133,
"SV_ERROR_LICENCE_MIXER. You tried to use the mixer without license.")

err(SV_ERROR_LICENCE_DUALLINK,      134,
"SV_ERROR_LICENCE_DUALLINK. You tried to use dual link without license.")

err(SV_ERROR_LICENCE_SDTV,          135,
"SV_ERROR_LICENCE_SDTV. You tried to use an SDTV raster without license.")

err(SV_ERROR_LICENCE_FILM2K,        136,
"SV_ERROR_LICENCE_FILM2K. You tried to use a FILM2K raster without license.")

err(SV_ERROR_CLIP_BLOCKED,          137,
"SV_ERROR_CLIP_BLOCKED. The selected clip is blocked by another process.")

err(SV_ERROR_CLIP_INVALID,          138,
"SV_ERROR_CLIP_INVALID. The clip is not valid.")

err(SV_ERROR_FILEFORMAT,            139,
"SV_ERROR_FILEFORMAT. The file format is not valid.")

err(SV_ERROR_VTR_UNDEFINEDCOMMAND,  140,
"SV_ERROR_VTR_UNDEFINEDCOMMAND. VTR returned an undefined command.")

err(SV_ERROR_LICENCE_HD360,         141,
"SV_ERROR_LICENCE_HD360. You tried to use HD360 without license.")

err(SV_ERROR_NOTASYNCCALL,          142,
"SV_ERROR_NOTASYNCCALL. This function cannot be called in an asynchronous mode.")

err(SV_ERROR_ASYNCNOTFOUND,         143,
"SV_ERROR_ASYNCNOTFOUND. This asynchronous call is no longer available.")

err(SV_ERROR_LICENCE_HSDL,          144,
"SV_ERROR_LICENCE_HSDL. You tried to use an HSDL raster without license.")

err(SV_ERROR_LICENCE_FILM2KPLUS,    145,
"SV_ERROR_LICENCE_FILM2KPLUS. You tried to use a FILM2Kplus feature without license.")

err(SV_ERROR_OBSOLETE,              146,
"SV_ERROR_OBSOLETE. A function was called that is obsolete.")

err(SV_ERROR_DRIVER_MISMATCH,       147,
"SV_ERROR_DRIVER_MISMATCH. Driver and library version mismatch.")

err(SV_ERROR_TOLERANCE,             148,
"SV_ERROR_TOLERANCE. Tolerance value exceeded.")

err(SV_ERROR_NOTAVAILABLE,          149,
"SV_ERROR_NOTAVAILABLE. Value is not in input stream.")

err(SV_ERROR_DATARATE,              150,
"SV_ERROR_DATARATE. Data rate for this raster is too high.")

err(SV_ERROR_WRONGMODE,             151,
"SV_ERROR_WRONGMODE. Currently this command is not possible.")

err(SV_ERROR_FIFOOPENED,            152,
"SV_ERROR_FIFOOPENED. This command cannot be done while the FIFO is opened.")

err(SV_ERROR_NOINPUTANDOUTPUT,      153,
"SV_ERROR_NOINPUTANDOUTPUT. In this mode you cannot do both input and output.")

err(SV_ERROR_FIFOCLOSED,            154,
"SV_ERROR_FIFOCLOSED. This command cannot be done while the FIFO is closed.")

err(SV_ERROR_ALREADY_OPENED,        155,
"SV_ERROR_ALREADY_OPENED. The resource is already opened.")

err(SV_ERROR_ALREADY_CLOSED,        156,
"SV_ERROR_ALREADY_CLOSED. The resource is already closed.")

err(SV_ERROR_ANCONSWITCHINGLINE,    157,
"SV_ERROR_ANCONSWITCHINGLINE. You cannot put ANC data on the switching line.")

err(SV_ERROR_WRONG_BITDEPTH,        158,
"SV_ERROR_WRONG_BITDEPTH. Not supported bit depth or operation for this bit depth was selected.")

err(SV_ERROR_NOTFORDDR,             159,
"SV_ERROR_NOTFORDDR. The function is not supported by the DDR.")

err(SV_ERROR_SVOPENSTRING,          160,
"SV_ERROR_SVOPENSTRING. There is a syntax error in the sv_open() string.")

err(SV_ERROR_DEVICEINUSE,           161,
"SV_ERROR_DEVICEINUSE. The device is in use.")

err(SV_ERROR_DEVICENOTFOUND,        162,
"SV_ERROR_DEVICENOTFOUND. The device could not be found.")

err(SV_ERROR_FLASH_WRITE,           163,
"SV_ERROR_FLASH_WRITE. Flash write failed.")

err(SV_ERROR_CLIP_NOTCREATED,       164,
"SV_ERROR_CLIP_NOTCREATED. The clip cannot be created (possibly because of an unsupported format).")

err(SV_ERROR_CLIP_TOOBIG,           165,
"SV_ERROR_CLIP_TOOBIG. There is not enough free space to create that clip.")

err(SV_ERROR_INTERNALMAGIC,         166,
"SV_ERROR_INTERNALMAGIC. An internal library check failed.")

err(SV_ERROR_OPENTYPE,              167,
"SV_ERROR_OPENTYPE. You have not opened this resource.")

err(SV_ERROR_DRIVER_MEMORY,         168,
"SV_ERROR_DRIVER_MEMORY. All memory modules could not be found.")

err(SV_ERROR_DRIVER_MEMORYMATCH,    169,
"SV_ERROR_DRIVER_MEMORYMATCH. Mounted memory modules does not match.")

err(SV_ERROR_CLIP_PROTECTED,        170,
"SV_ERROR_CLIP_PROTECTED. This clip is protected and cannot be deleted.")

err(SV_ERROR_SYNCDELAY,             171,
"SV_ERROR_SYNCDELAY. Invalid sync H-/V-delay.")

err(SV_ERROR_DRIVER_BADPCIMAPPING,  172,
"SV_ERROR_DRIVER_BADPCIMAPPING. The PCI mapping has an overlap.")

err(SV_ERROR_LICENCE_FILM4K,        173,
"SV_ERROR_LICENCE_FILM4K. You tried to use a FILM4K raster without license.")

err(SV_ERROR_LICENCE_HSDLRT,        174,
"SV_ERROR_LICENCE_HSDLRT. You tried to use HSDLRT without license.")

err(SV_ERROR_LICENCE_12BITS,        175,
"SV_ERROR_LICENCE_12BITS. You tried to use the 12-bit I/O mode without license.")

err(SV_ERROR_SERIALNUMBER,          176,
"SV_ERROR_SERIALNUMBER. Missing serial number.")

err(SV_ERROR_FILEEXISTS,            177,
"SV_ERROR_FILEEXISTS. File already exists.")

err(SV_ERROR_DIRCREATE,             178,
"SV_ERROR_DIRCREATE. Couldn't create directory.")

err(SV_ERROR_USERNOTALLOWED,        179,
"SV_ERROR_USERNOTALLOWED. No privileges to interact with VServer.")

err(SV_ERROR_LICENCE_HDTV,          180,
"SV_ERROR_LICENCE_HDTV. You tried to use an HDTV raster without license.")

err(SV_ERROR_LICENCE_HIRES,         181,
"SV_ERROR_LICENCE_HIRES. You tried to use a HiRes raster without license.")

err(SV_ERROR_LICENCE_MULTIDEVICE,   182,
"SV_ERROR_LICENCE_MULTIDEVICE. You tried to use multi-device without license.")

err(SV_ERROR_MALLOC_FRAGMENTED,     183,
"SV_ERROR_MALLOC_FRAGMENTED. Memory allocation failed due to memory fragmentation.")

err(SV_ERROR_HIGH_MEMORY,           184,
"SV_ERROR_HIGH_MEMORY. Operation cannot be performed on high memory.")

err(SV_ERROR_LICENCE_CUSTOMRASTER,  185,
"SV_ERROR_LICENCE_CUSTOMRASTER. You tried to use a custom raster without license.")

err(SV_ERROR_INPUT_VIDEO_DETECTING, 186,
"SV_ERROR_INPUT_VIDEO_DETECTING. Video input detection not yet ready.")

err(SV_ERROR_LICENCE_PHDTV,         187,
"SV_ERROR_LICENCE_PHDTV. You tried to use a PHDTV raster without license.")

err(SV_ERROR_LICENCE_SLOWPAL,       188,
"SV_ERROR_LICENCE_SLOWPAL. You tried to use the SLOWPAL raster without license.")

err(SV_ERROR_WRONG_PCISPEED,        189,
"SV_ERROR_WRONG_PCISPEED. Card is running at a not supported PCI speed.")

err(SV_ERROR_FIFO_STOPPED,          190,
"SV_ERROR_FIFO_STOPPED. This command cannot be done while the FIFO is stopped.")

err(SV_ERROR_LICENCE_SDTVFF,        191,
"SV_ERROR_LICENCE_SDTVFF. You tried to use an SDTV FF raster without license.")

err(SV_ERROR_LICENCE_EUREKA,        192,
"SV_ERROR_LICENCE_EUREKA. You tried to use an Eureka raster without license.")

err(SV_ERROR_LICENCE_DVIINPUT,      193,
"SV_ERROR_LICENCE_DVIINPUT. You tried to use the DVI input without license.")

err(SV_ERROR_INF_MISMATCH,          194,
"SV_ERROR_INF_MISMATCH. The INF file does not match the driver binary.")

err(SV_ERROR_SYNC_MISSING,          195,
"SV_ERROR_SYNC_MISSING. The sync signal is either bad or missing.")

err(SV_ERROR_JACK_INVALID,          196,
"SV_ERROR_JACK_INVALID. Invalid jack name/ID for this operation.")

err(SV_ERROR_JACK_ASSIGNMENT,       197,
"SV_ERROR_JACK_ASSIGNMENT. This channel is already assigned to another jack.")

err(SV_ERROR_JACK_NOTASSIGNED,      198,
"SV_ERROR_JACK_NOTASSIGNED. No channels are assigned to this jack.")

err(SV_ERROR_JPEG2K_CODESTREAM,     199,
"SV_ERROR_JPEG2K_CODESTREAM. Error in the JPEG2000 codestream.")

err(SV_ERROR_NODATA,                200,
"SV_ERROR_NODATA. No data provided.")

err(SV_ERROR_JACK_NOBYPASS,         201,
"SV_ERROR_JACK_NOBYPASS. The jack does not have a bypass jack assigned.")

err(SV_ERROR_VERSION,               202,
"SV_ERROR_VERSION. Version mismatch detected.")

err(SV_ERROR_VOLTAGE_TOLOW,         203,
"SV_ERROR_VOLTAGE_TOLOW. Voltage measured is too low.")

err(SV_ERROR_VOLTAGE_TOHIGH,        204,
"SV_ERROR_VOLTAGE_TOHIGH. Voltage measured is too high.")

err(SV_ERROR_FANSPEED,              205,
"SV_ERROR_FANSPEED. A fan is not turning fast enough.")

err(SV_ERROR_TEMPERATURE,           206,
"SV_ERROR_TEMPERATURE. Temperature measured is too high.")

err(SV_ERROR_LICENCE_DVI16,         207,
"SV_ERROR_LICENCE_DVI16. You tried to use DVI 16 without license.")

err(SV_ERROR_IOCHANNEL_INVALID,     208,
"SV_ERROR_IOCHANNEL_INVALID. You tried to use an invalid I/O channel.")

err(SV_ERROR_DRIVER_MAPPEDSIZE,     209,
"SV_ERROR_DRIVER_MAPPEDSIZE. Mapped size is too small.")

err(SV_ERROR_LICENCE_EXPIRED,       210,
"SV_ERROR_LICENCE_EXPIRED, The licence has expired.")

err(SV_ERROR_LICENCE_LINKENCRYPT,   211,
"SV_ERROR_LICENCE_LINKENCRYPT, You tried to use link encryption without license.")

err(SV_ERROR_MULTICHANNEL_RASTER,   212,
"SV_ERROR_MULTICHANNEL_RASTER, The current multichannel raster configuration is not supported by this board.")

err(SV_ERROR_LICENCE_RENDER,        213,
"SV_ERROR_LICENCE_RENDER, You tried to use the render api without license.")

err(SV_ERROR_LICENCE_JPEG2000RAW,   214,
"SV_ERROR_LICENCE_JPEG2000RAW, You tried to use JPEG2000 raw decompression without license.")

err(SV_ERROR_DRIVER_HWCHECK,        215,
"SV_ERROR_DRIVER_HWCHECK, Driver hardware check failed, run dvs test utility.")

err(SV_ERROR_DRIVER_MEMORYINIT,     216,
"SV_ERROR_DRIVER_MEMORYINIT, On card memory init failed.")

err(SV_ERROR_LICENCE_MULTICHANNEL,  217,
"SV_ERROR_LICENCE_MULTICHANNEL. You tried to use multi-channel without license.")

err(SV_ERROR_LICENCE_STEREO,        218,
"SV_ERROR_LICENCE_STEREO. You tried to use Stereo without license.")

err(SV_ERROR_SLEEPING,              219,
"SV_ERROR_SLEEPING. Driver is still in sleep mode. Please reopen video board.")

err(SV_ERROR_LICENCE_JPEG2000CODEC4K,  220,
"SV_ERROR_LICENCE_JPEG2000CODEC4K, You tried to use JPEG2000 4k without license.")

err(SV_ERROR_LICENCE_WATERMARK,     221,
"SV_ERROR_LICENCE_WATERMARK, Trying to use watermark without licence.")

err(SV_ERROR_MISSING_SLAVE_TASK,    222,
"SV_ERROR_MISSING_SLAVE_TASK, No slave task has been assigned for current iochannel.")

err(SV_ERROR_RESOURCENOTAVAIBLE,    223,
"SV_ERROR_RESOURCENOTAVAIBLE, The needed resource for this mode is not currently available.")

err(SV_ERROR_WRONGMODE_QUADMODE_DUALLINK_1GB5,  224,
"SV_ERROR_WRONGMODE_QUADMODE_DUALLINK_1GB5, Quadmode raster in combination with duallink iomode and iospeed 1.5GB not possible on this hardware configuration.")

/* don't touch these lines *vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
#ifdef ERRSTRUCT
err(MAXERRCODE,                     MAXERRCODE,
"Sorry, no error text available.")
} ;
#elif defined(QT_GENERATE_ERROR)
        default: error_text = QObject::tr("Unknown error.");
    }

    return error_text;
}
#else
	MAXERRCODE
} ;
#endif
/* don't touch these lines *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

#undef err

#endif /*_DVS_ERRORS_H_*/

/**** E O F *****************************************************************/
