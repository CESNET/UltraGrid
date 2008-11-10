/*
 *      DVS C-Library Errors Header File, Copyright (c) 1994-2008 DVS Digital Video Systems AG
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
"SV_ERROR_PARAMETER, Parameter error")

err(SV_ERROR_MALLOC,                2,
"SV_ERROR_MALLOC, Memory allocation failed")

err(SV_ERROR_NOTIMPLEMENTED,	    3,
"SV_ERROR_NOTIMPLEMENTED Not implemented function called")

err(SV_ERROR_BUFFERSIZE,            4,
"SV_ERROR_BUFFERSIZE, Buffer to small")

err(SV_ERROR_NOTEXTSYNC,            5,
"SV_ERROR_NOTEXTSYNC, Not in external sync")

err(SV_ERROR_VIDEOMODE,             6,
"SV_ERROR_VIDEOMODE, Wrong video mode specified")

err(SV_ERROR_FILEOPEN,              7,
"SV_ERROR_FILEOPEN, Couldn't open file")

err(SV_ERROR_FILECREATE,            8,
"SV_ERROR_FILECREATE, Couldn't create file")

err(SV_ERROR_DATALOST,              9,
"SV_ERROR_DATALOST Data lost during transfer")

err(SV_ERROR_SVNULL,                10,
"SV_ERROR_SVNULL, Invalid SV handle")

err(SV_ERROR_SVMAGIC,               11,
"SV_ERROR_SVMAGIC, Invalid SV magic number")

err(SV_ERROR_FILEREAD,              12,
"SV_ERROR_FILEREAD, Couldn't read from file")

err(SV_ERROR_FILEWRITE,             13,
"SV_ERROR_FILEWRITE, Couldn't write to file")

err(SV_ERROR_FILECLOSE,             14,
"SV_ERROR_FILECLOSE, Couldn't close file")

err(SV_ERROR_FILEDIRECT,            15,
"SV_ERROR_FILEDIRECT, Couldn't set direct file access")

err(SV_ERROR_FILESEEK,              16,
"SV_ERROR_FILESEEK, Couldn't seek in file")

err(SV_ERROR_FILETRUNCATE,          17,
"SV_ERROR_FILETRUNCATE, Couldn't truncate file")

err(SV_ERROR_SVJ_FRAMENR,           18,
"SV_ERROR_SVJ_FRAMENR, Framenumber not in file")

err(SV_ERROR_SVJ_NULL,              19,
"SV_ERROR_SVJ_NULL, Null svj handle")

err(SV_ERROR_SCSI,                  20,
"SV_ERROR_SCSI, SCSI transfer error")

err(SV_ERROR_TIMECODE,              21,
"SV_ERROR_TIMECODE, Illegal timecode, format is 01:00:00:00")

err(SV_ERROR_MASTER,                22,
"SV_ERROR_MASTER, Master control error")

err(SV_ERROR_MEM_NULL,              23,
"SV_ERROR_MEM_NULL, Null mem handle")

err(SV_ERROR_MEM_BUFFERSIZE,	    24,
"SV_ERROR_MEM_BUFFERSIZE, Memory buffer too small")

err(SV_ERROR_VIDEOPAGE,             25,
"SV_ERROR_VIDEOPAGE, Non existing video page")

err(SV_ERROR_TRANSFER,              26,
"SV_ERROR_TRANSFER, Transfer failed")

err(SV_ERROR_NOCARRIER,	            27,
"SV_ERROR_NOCARRIER, No input signal detected")

err(SV_ERROR_NOGENLOCK,             28,
"SV_ERROR_NOGENLOCK, Genlock option not available")

err(SV_ERROR_NODRAM,                29,
"SV_ERROR_NODRAM, DRAM option not available")

err(SV_ERROR_FIRMWARE,              30,
"SV_ERROR_FIRMWARE, The firmware version is wrong")

err(SV_ERROR_QUANTLOSS,             31,
"SV_ERROR_QUANTLOSS, AC quant values lost during compression")

err(SV_ERROR_RECORD,                32,
"SV_ERROR_RECORD, Error during record operation")

err(SV_ERROR_SLAVE,                 33,
"SV_ERROR_SLAVE, Device in remote control mode")

err(SV_ERROR_DISKFORMAT,            34,
"SV_ERROR_DISKFORMAT, Unappropriate disk format for requested videomode")

err(SV_ERROR_PROGRAM,               35,
"SV_ERROR_PROGRAM, Illegal internal state")

err(SV_ERROR_TIMELINE,              36,
"SV_ERROR_TIMELINE, Illegal segment specification")

err(SV_ERROR_SCSIDEVICE,            37,
"SV_ERROR_SCSIDEVICE, SCSI device not found")

err(SV_ERROR_SCSIWRITE,             38,
"SV_ERROR_SCSIWRITE, SCSI write failed")

err(SV_ERROR_SCSIREAD,              39,
"SV_ERROR_SCSIREAD, SCSI read failed")

err(SV_ERROR_DTM_TIMEOUT,           40,
"SV_ERROR_DTM_TIMEOUT, Connection timeout")

err(SV_ERROR_NOTSUPPORTED,          41,
"SV_ERROR_NOTSUPPORTED, Not supported feature")

err(SV_ERROR_CLIP_NOTFOUND,         42,
"SV_ERROR_CLIP_NOTFOUND, Clip not found")

err(SV_ERROR_CLIP_NAMEEXISTS,       43,
"SV_ERROR_CLIP_NAMEEXISTS, Clip name already exists")

err(SV_ERROR_CLIP_NOENTRY,          44,
"SV_ERROR_CLIP_NOENTRY, Clip directory full")

err(SV_ERROR_CLIP_OVERLAP,          45,
"SV_ERROR_CLIP_OVERLAP, Clip would overlap with directory")

err(SV_ERROR_CLIPDIR_NOTFOUND,      46,
"SV_ERROR_CLIPDIR_NOTFOUND, Directory not found")

err(SV_ERROR_CLIPDIR_NAMEEXISTS,    47,
"SV_ERROR_CLIPDIR_NAMEEXISTS, Directory name already exists")

err(SV_ERROR_CLIPDIR_NOENTRY,       48,
"SV_ERROR_CLIPDIR_NOENTRY, Directory field full")

err(SV_ERROR_CLIPDIR_OVERLAP,       49,
"SV_ERROR_CLIPDIR_OVERLAP, Directory would overlap \
with other clip or directory")

err(SV_ERROR_NOLICENCE,             50,
"SV_ERROR_NOLICENCE, No licence for this operation.")

err(SV_ERROR_FRAME_NOACCESS,        51,
"SV_ERROR_FRAME_NOACCESS, Frame not accessible")

err(SV_ERROR_PARTITION_NOENTRY,     52,
"SV_ERROR_PATTITION_NOENTRY, No free entry in partition table")

err(SV_ERROR_PARTITION_NOSPACE,     53,
"SV_ERROR_PARTITION_NOSPACE, Requested space exceeds available space")

err(SV_ERROR_PARTITION_NOTLAST,     54,
"SV_ERROR_PARTITION_NOTLAST, Only the last partition can be deleted")

err(SV_ERROR_PARTITION_NOTFOUND,    55,
"SV_ERROR_PARTITION_NOTFOUND, Invalid partition name")

err(SV_ERROR_PARTITION_INVALID,     56,
"SV_ERROR_PARTITION_INVALID, Invalid active partition")

err(SV_ERROR_AUDIO_SEGMENT,         57,
"SV_ERROR_AUDIO_SEGMENT, Illegal audio segment specified")

err(SV_ERROR_POLL_TASK_ACTIVE,	    58,
"SV_ERROR_POLL_TASK_ACTIVE, Poll task active")

err(SV_ERROR_HARDWARELOAD,          59,
"SV_ERROR_HARDWARELOAD, Hardware Failed to Load")

err(SV_ERROR_CLIPDIR_NOTEMPTY,      60,
"SV_ERROR_CLIPDIR_NOTEMPTY, Directory not empty")

err(SV_ERROR_DISABLED,              61,
"SV_ERROR_DISABLED, A disabled function called")

err(SV_ERROR_WRONG_HARDWARE,        62,
"SV_ERROR_WRONG_HARDWARE, Wrong Hardware for operation")

err(SV_ERROR_SYNCMODE,              63,
"SV_ERROR_SYNCMODE, Invalid sync mode selected")

err(SV_ERROR_IOMODE,                64,
"SV_ERROR_IOMODE, Invalid iomode selected")

err(SV_ERROR_VIDEO_RASTER_TABLE,    65,
"SV_ERROR_VIDEO_RASTER_TABLE, Video raster table not initialized")

err(SV_ERROR_VIDEO_RASTER_FILE,	    66,
"SV_ERROR_VIDEO_RASTER_FILE, Loading/Checking raster definition file failed")

err(SV_ERROR_SYNC_CALCULATION,	    67,
"SV_ERROR_SYNC_CALCULATION, Calculation of sync output signal failed")

err(SV_ERROR_SYNC_OUTPUT,    	    68,
"SV_ERROR_SYNC_OUTPUT, Specified sync output signal not supported")

err(SV_ERROR_FLASH_ERASETIMEOUT,    69,
"SV_ERROR_FLASH_ERASETIMEOUT, A timeout during erase of the flash appeared")

err(SV_ERROR_FLASH_VERIFY,          70,
"SV_ERROR_FLASH_VERIFY, Verify of the flash after programming failed")

err(SV_ERROR_EPLD_MAGIC,            71,
"SV_ERROR_EPLD_MAGIC, A EPLD with the wrong magic")

err(SV_ERROR_EPLD_PRODUCT,          72,
"SV_ERROR_EPLD_PRODUCT, An EPLD from the wrong device")

err(SV_ERROR_EPLD_CHIP,             73,
"SV_ERROR_EPLD_CHIP, An EPLD with the wrong chipid")

err(SV_ERROR_EPLD_VERSION,          74,
"SV_ERROR_EPLD_VERSION, An EPLD with the wrong version")

err(SV_ERROR_NOTREADY,		    75,
"SV_ERROR_NOTREADY, Operation is not ready")

err(SV_ERROR_NOTDEBUGDRIVER,        76,
"SV_ERROR_NOTDEBUGDRIVER, This is only supported by the debug driver")

err(SV_ERROR_DRIVER_CONNECTIRQ,     77,
"SV_ERROR_DRIVER_CONNECTIRQ, Driver could not connect to an IRQ")

err(SV_ERROR_DRIVER_MAPIOSPACE,     78,
"SV_ERROR_DRIVER_MAPIOSPACE, Driver could not map onboard memory into kernel memory")

err(SV_ERROR_DRIVER_RESOURCES,      79,
"SV_ERROR_DRIVER_RESOURCES, Driver did not get resources from the kernel")

err(SV_ERROR_DRIVER_MALLOC,         80,
"SV_ERROR_DRIVER_MALLOC, Driver could not malloc critical memory")

err(SV_ERROR_VSYNCPASSED,           81,
"SV_ERROR_VSYNCPASSED, An operation was issued for a vsync that has already passed")

err(SV_ERROR_VSYNCFUTURE,           82,
"SV_ERROR_VSYNCFUTURE, An operation was issued to long time before it should start")

err(SV_ERROR_IOCTL_FAILED,          83,
"SV_ERROR_IOCTL_FAILED, An ioctl operation failed")

err(SV_ERROR_FIFO_TIMEOUT,          84,
"SV_ERROR_FIFO_TIMEOUT, The fifo timed out.")

err(SV_ERROR_FIFO_PUTBUFFER,        85,
"SV_ERROR_FIFO_PUTBUFFER, The fifo getbuffer/putbuffer was called incorrectly.")

err(SV_ERROR_SAMPLINGFREQ,    	    86,
"SV_ERROR_SAMPLINGFREQ, Illegal sampling frequeny specified")

err(SV_ERROR_MMAPFAILED,    	    87,
"SV_ERROR_MMAPFAILED, Memory mapping function failed")

err(SV_ERROR_TIMEOUT,		    88,
"SV_ERROR_TIMEOUT, Operation timed out")

err(SV_ERROR_CANCELED,		    89,
"SV_ERROR_CANCELED, Operation was canceled")

err(SV_ERROR_UNKNOWNFLASH,          90,
"SV_ERROR_UNKNOWNFLASH, Flash chip that this software does not support.")

err(SV_ERROR_WRONG_COLORMODE,       91,
"SV_ERROR_WRONG_COLORMODE, Not supported colormode or operation for this colormode was selected.")

err(SV_ERROR_DRIVER_HWPATH,         92,
"SV_ERROR_DRIVER_HWPATH, Path to driver hardware files missing.")

err(SV_ERROR_DISPLAYONLY,           93,
"SV_ERROR_DISPLAYONLY, This video raster can only do display.")

err(SV_ERROR_VTR_OFFLINE,           94,
"SV_ERROR_VTR_OFFLINE, There is no VTR connected.")

err(SV_ERROR_VTR_LOCAL,             95,
"SV_ERROR_VTR_LOCAL, The VTR is in local mode, please check front panel switch.")

err(SV_ERROR_VTR_SERIAL,            96,
"SV_ERROR_VTR_SERIAL, Error from serial driver.")

err(SV_ERROR_VTR_NAK,               97,
"SV_ERROR_VTR_NAK, Recieved NAK (Not Acknowledge) from VTR.")

err(SV_ERROR_VTR_GOTOERROR,         98,
"SV_ERROR_VTR_GOTOERROR, Goto operation on VTR did not complete.")

err(SV_ERROR_VTR_NOSTATUS,          99,
"SV_ERROR_VTR_NOSTATUS, Status reply from VTR missing.")

err(SV_ERROR_VTR_NOACK,             100,
"SV_ERROR_VTR_NOACK, Acknowledge from VTR missing.")

err(SV_ERROR_VTR_NOTIMECODE,        101,
"SV_ERROR_VTR_NOTIMECODE, Timecode reply is wrong.")

err(SV_ERROR_VTR_NOTIMECODECHANGE,  102,
"SV_ERROR_VTR_NOTCHANGE, Timecode did not change during edit.")

err(SV_ERROR_VTR_TCORDER,           103,
"SV_ERROR_VTR_TCORDER, Timecode order during VTR edit is wrong.")

err(SV_ERROR_VTR_TICKORDER,         104,
"SV_ERROR_VTR_TICKORDER, Tick order during VTR edit is wrong.")

err(SV_ERROR_VTR_EDIT,              105,
"SV_ERROR_VTR_EDIT, Master control during Edit failed.")

err(SV_ERROR_BUFFER_NOTALIGNED,     106,
"SV_ERROR_BUFFER_NOTALIGNED, The buffer does not have the needed alignment")

err(SV_ERROR_BUFFER_NULL,           107,
"SV_ERROR_BUFFER_NULL, A buffer not pointing to anything.")

err(SV_ERROR_BUFFER_TOLARGE,        108,
"SV_ERROR_BUFFER_TOLARGE, A buffer was to large.")

err(SV_ERROR_NOTFRAMESTORAGE,       109,
"SV_ERROR_NOTFRAMESTORAGE, This can only be done in frame storage mode.")

err(SV_ERROR_NOTRUNNING,	    110,
"SV_ERROR_NOTRUNNING, A polled operation is no longer active.")

err(SV_ERROR_NOHSWTRANSFER,	    111,
"SV_ERROR_NOHSWTRANSFER, Hostsoftware transfer has been disabled.")

err(SV_ERROR_INPUT_VIDEO_NOSIGNAL,  112,
"SV_ERROR_INPUT_VIDEO_NOSIGNAL, There is no video input detected.")

err(SV_ERROR_INPUT_VIDEO_RASTER,    113,
"SV_ERROR_INPUT_VIDEO_RASTER, The input video signal does not match the video raster.")

err(SV_ERROR_INPUT_KEY_NOSIGNAL,    114,
"SV_ERROR_INPUT_KEY_NOSIGNAL, There is no key input detected.")

err(SV_ERROR_INPUT_KEY_RASTER,      115,
"SV_ERROR_INPUT_KEY_RASTER, The input key signal does not match the video raster.")

err(SV_ERROR_INPUT_AUDIO_NOAESEBU,  116,
"SV_ERROR_INPUT_AUDIO_NOAESEBU, There is no AES/EBU audio input detected.")

err(SV_ERROR_TRANSFER_NOAUDIO,      117,
"SV_ERROR_TRANSFER_NOAUDIO, There is no audio configured on the device.")

err(SV_ERROR_FLASH_ERASEVERIFY,     118,
"SV_ERROR_FLASH_ERASEVERIFY, Verify of the flash erase failed.")

err(SV_ERROR_INPUT_AUDIO_FREQUENCY, 119,
"SV_ERROR_INPUT_AUDIO_FREQUENCY, The audio input has the wrong frequency.")

err(SV_ERROR_INPUT_AUDIO_NOAIV,     120,
"SV_ERROR_INPUT_AUDIO_NOAIV, There is no AiV audio signal in video input detected.")

err(SV_ERROR_CHECKWORD,             121,
"SV_ERROR_CHECKWORD, Checkword was wrong.")

err(SV_ERROR_CLIPDIR_NOTSELECT,     122,
"SV_ERROR_CLIPDIR_NOTSELECT, A directory cannot be selected in this filesystem.")

err(SV_ERROR_EPLD_NOTFOUND,         123,
"SV_ERROR_EPLD_NOTFOUND, Could not find the pld files needed to program the hardware.")

err(SV_ERROR_PARAMETER_NEGATIVE,    124,
"SV_ERROR_PARAMETER_NEGATIVE, A negative parameter is not valid.")

err(SV_ERROR_PARAMETER_TOLARGE,     125,
"SV_ERROR_PARAMETER_TOLARGE, A parameter is to large.")

err(SV_ERROR_ALREADY_RUNNING,       126,
"SV_ERROR_ALREADY_RUNNING, Tried to start an operation that was already started.")

err(SV_ERROR_WRONG_OS,       	    127,
"SV_ERROR_WRONG_OS, This function is not supported on the current operating system.")

err(SV_ERROR_TOMANYAUDIOCHANNELS,   128,
"SV_ERROR_TOMANYAUDIOCHANNELS, Trying to set to many audiochannels.")

err(SV_ERROR_LICENCE_AUDIO,         129,
"SV_ERROR_LICENCE_AUDIO, Trying to use more audio channel than licenced.")

err(SV_ERROR_LICENCE_STREAMER,      130,
"SV_ERROR_LICENCE_STREAMER, Trying to use streamer without licence.")

err(SV_ERROR_LICENCE_RGB,           131,
"SV_ERROR_LICENCE_RGB, Trying to use rgb without licence.")

err(SV_ERROR_LICENCE_KEYCHANNEL,    132,
"SV_ERROR_LICENCE_KEYCHANNEL, Trying to use keychannel without licence.")

err(SV_ERROR_LICENCE_MIXER,         133,
"SV_ERROR_LICENCE_MIXER, Trying to use mixer without licence.")

err(SV_ERROR_LICENCE_DUALLINK,      134,
"SV_ERROR_LICENCE_DUALLINK, Trying to use duallink without licence.")

err(SV_ERROR_LICENCE_SDTV,          135,
"SV_ERROR_LICENCE_SDTV, Trying to use sdtv raster without licence.")

err(SV_ERROR_LICENCE_FILM2K,        136,
"SV_ERROR_LICENCE_FILM2K, Trying to use film2k raster without licence.")

err(SV_ERROR_CLIP_BLOCKED,          137,
"SV_ERROR_CLIP_BLOCKED, At that time the clip is blocked by another process.")

err(SV_ERROR_CLIP_INVALID,          138,
"SV_ERROR_CLIP_INVALID, The clip is not valid.")

err(SV_ERROR_FILEFORMAT,            139,
"SV_ERROR_FILEFORMAT, The file format is not valid.")

err(SV_ERROR_VTR_UNDEFINEDCOMMAND,  140,
"SV_ERROR_VTR_UNDEFINEDCOMMAND, VTR return undefined command.")

err(SV_ERROR_LICENCE_HD360,         141,
"SV_ERROR_LICENCE_HD360, Trying to use HD360 without licence.")

err(SV_ERROR_NOTASYNCCALL,          142,
"SV_ERROR_NOTASYNCCALL, Function can not be called async.")

err(SV_ERROR_ASYNCNOTFOUND,         143,
"SV_ERROR_ASYNCNOTFOUND, This async call is no longer available.")

err(SV_ERROR_LICENCE_HSDL,          144,
"SV_ERROR_LICENCE_HSDL, Trying to use hsdl raster without licence.")

err(SV_ERROR_LICENCE_FILM2KPLUS,    145,
"SV_ERROR_LICENCE_FILM2KPLUS, Trying to use film2kplus feature without licence.")

err(SV_ERROR_OBSOLETE,              146,
"SV_ERROR_OBSOLETE, A function that has been obsoleted has been called.")

err(SV_ERROR_DRIVER_MISMATCH,       147,
"SV_ERROR_DRIVER_MISMATCH, Driver and library version mismatch.")

err(SV_ERROR_TOLERANCE,             148,
"SV_ERROR_TOLERANCE, Tolerance value exceeded.")

err(SV_ERROR_NOTAVAILABLE,          149,
"SV_ERROR_NOTAVAILABLE, Value is not in input stream.")

err(SV_ERROR_DATARATE,              150,
"SV_ERROR_DATARATE, Datarate for this raster is to high.")

err(SV_ERROR_WRONGMODE,             151,
"SV_ERROR_WRONGMODE, This command is not currently possible.")

err(SV_ERROR_FIFOOPENED,            152,
"SV_ERROR_FIFOOPENED, This command can not be done while the fifo is opened.")

err(SV_ERROR_NOINPUTANDOUTPUT,      153,
"SV_ERROR_NOINPUTANDOUTPUT, In this mode you can not do both input and output.")

err(SV_ERROR_FIFOCLOSED,            154,
"SV_ERROR_FIFOCLOSED, This command can not be done while the fifo is closed.")

err(SV_ERROR_ALREADY_OPENED,        155,
"SV_ERROR_ALREADY_OPENED, The resource is already opened.")

err(SV_ERROR_ALREADY_CLOSED,        156,
"SV_ERROR_ALREADY_OPENED, The resource is already closed.")

err(SV_ERROR_ANCONSWITCHINGLINE,    157,
"SV_ERROR_ANCONSWITCHINGLINE, You can not put ANC data on the switching line.")

err(SV_ERROR_WRONG_BITDEPTH,        158,
"SV_ERROR_WRONG_BITDEPTH, Not supported bitdepth or operation for this bitdepth was selected.")

err(SV_ERROR_NOTFORDDR,             159,
"SV_ERROR_NOTFORDDR, The function is not supported by the DDR.")

err(SV_ERROR_SVOPENSTRING,          160,
"SV_ERROR_SVOPENSTRING, There is a syntax error in the sv_open string.")

err(SV_ERROR_DEVICEINUSE,           161,
"SV_ERROR_DEVICEINUSE, The device is in use.")

err(SV_ERROR_DEVICENOTFOUND,        162,
"SV_ERROR_DEVICENOTFOUND, The device is not found.")

err(SV_ERROR_FLASH_WRITE,           163,
"SV_ERROR_FLASH_WRITE, Flash write failed.")

err(SV_ERROR_CLIP_NOTCREATED,       164,
"SV_ERROR_CLIP_NOTCREATED, The clip cannot be created (possibly because unsupported format).")

err(SV_ERROR_CLIP_TOOBIG,           165,
"SV_ERROR_CLIP_TOOBIG, There is not enough free space to create that clip")

err(SV_ERROR_INTERNALMAGIC,         166,
"SV_ERROR_INTERNALMAGIC, An library internal check failed.")

err(SV_ERROR_OPENTYPE,              167,
"SV_ERROR_OPENTYPE, You have not opened this resource.")

err(SV_ERROR_DRIVER_MEMORY,         168,
"SV_ERROR_DRIVER_MEMORY, All memory modules not found.")

err(SV_ERROR_DRIVER_MEMORYMATCH,    169,
"SV_ERROR_DRIVER_MEMORYMATCH, Mounted memory modules does not match.")

err(SV_ERROR_CLIP_PROTECTED,        170,
"SV_ERROR_CLIP_PROTECTED, This clip is protected and cannot be deleted.")

err(SV_ERROR_SYNCDELAY,             171,
"SV_ERROR_SYNCDELAY, Invalid sync h/v-delay.")

err(SV_ERROR_DRIVER_BADPCIMAPPING,  172,
"SV_ERROR_DRIVER_BADPCIMAPPING, The pci mapping has an overlapp.")

err(SV_ERROR_LICENCE_FILM4K,        173,
"SV_ERROR_LICENCE_FILM4K, Trying to use film4k raster without licence.")

err(SV_ERROR_LICENCE_HSDLRT,        174,
"SV_ERROR_LICENCE_HSDLRT, Trying to use HSDLRT without licence.")

err(SV_ERROR_LICENCE_12BITS,        175,
"SV_ERROR_LICENCE_12BITS, Trying to use 12bit iomode without licence.")

err(SV_ERROR_SERIALNUMBER,          176,
"SV_ERROR_SERIALNUMBER, Missing serialnumber.")

err(SV_ERROR_FILEEXISTS,            177,
"SV_ERROR_FILEEXISTS, File already exists.")

err(SV_ERROR_DIRCREATE,             178,
"SV_ERROR_DIRCREATE, Couldn't create directory.")

err(SV_ERROR_USERNOTALLOWED,        179,
"SV_ERROR_USERNOTALLOWED, No privilegies to interact with VServer.")

err(SV_ERROR_LICENCE_HDTV,          180,
"SV_ERROR_LICENCE_HDTV, Trying to use hdtv raster without licence.")

err(SV_ERROR_LICENCE_HIRES,         181,
"SV_ERROR_LICENCE_HIRES, Trying to use hires raster without licence.")

err(SV_ERROR_LICENCE_MULTIDEVICE,   182,
"SV_ERROR_LICENCE_MULTIDEVICE, Trying to use multidevice without licence.")

err(SV_ERROR_MALLOC_FRAGMENTED,     183,
"SV_ERROR_MALLOC_FRAGMENTED, Memory allocation failed due to memory fragmentation.")

err(SV_ERROR_HIGH_MEMORY,           184,
"SV_ERROR_HIGH_MEMORY, Operation cannot be performed on high memory.")

err(SV_ERROR_LICENCE_CUSTOMRASTER,  185,
"SV_ERROR_LICENCE_CUSTOMRASTER, Trying to use custom raster without licence.")

err(SV_ERROR_INPUT_VIDEO_DETECTING, 186,
"SV_ERROR_INPUT_VIDEO_DETECTING, Video input detection not yet ready.")

err(SV_ERROR_LICENCE_PHDTV,         187,
"SV_ERROR_LICENCE_PHDTV, Trying to use PHDTV raster without licence.")

err(SV_ERROR_LICENCE_SLOWPAL,       188,
"SV_ERROR_LICENCE_SLOWPAL, Trying to use SLOWPAL raster without licence.")

err(SV_ERROR_WRONG_PCISPEED,        189,
"SV_ERROR_WRONG_PCISPEED, Card is running at a not supported pci speed.")

err(SV_ERROR_FIFO_STOPPED,          190,
"SV_ERROR_FIFO_STOPPED, This command can not be done while the fifo is stopped.")

err(SV_ERROR_LICENCE_SDTVFF,        191,
"SV_ERROR_LICENCE_SDTVFF, Trying to use SDTV FF raster without licence.")

err(SV_ERROR_LICENCE_EUREKA,        192,
"SV_ERROR_LICENCE_EUREKA, Trying to use Eureka raster without licence.")

err(SV_ERROR_LICENCE_DVIINPUT,      193,
"SV_ERROR_LICENCE_DVIINPUT, Trying to use dvi input when not licenced.")

err(SV_ERROR_INF_MISMATCH,          194,
"SV_ERROR_INF_MISMATCH, The inf file does not match the driver binary.")

err(SV_ERROR_SYNC_MISSING,          195,
"SV_ERROR_SYNC_MISSING, The sync signal is either bad or missing.")

err(SV_ERROR_JACK_INVALID,          196,
"SV_ERROR_JACK_INVALID, Invalid jack name/id for this operation.")

err(SV_ERROR_JACK_ASSIGNMENT,       197,
"SV_ERROR_JACK_ASSIGNMENT, Channel is already assigned to another jack.")

err(SV_ERROR_JACK_NOTASSIGNED,      198,
"SV_ERROR_JACK_NOTASSIGNED, No channels are assigned to this jack.")

err(SV_ERROR_JPEG2K_CODESTREAM,     199,
"SV_ERROR_JPEG2K_CODESTREAM, Error in the jpeg2000 codestream.")

err(SV_ERROR_NODATA,                200,
"SV_ERROR_NODATA, No data provided.")

err(SV_ERROR_JACK_NOBYPASS,         201,
"SV_ERROR_JACK_NOBYPASS, The jack does not have a bypass jack assigned.")

err(SV_ERROR_VERSION,               202,
"SV_ERROR_VERSION, Version mismatch.")

err(SV_ERROR_VOLTAGE_TOLOW,         203,
"SV_ERROR_VOLTAGE_TOLOW, Voltage measured is to low.")

err(SV_ERROR_VOLTAGE_TOHIGH,        204,
"SV_ERROR_VOLTAGE_TOHIGH, Voltage measured is to high.")

err(SV_ERROR_FANSPEED,              205,
"SV_ERROR_FANSPEED, A fan is not turning fast enough.")

err(SV_ERROR_TEMPERATURE,           206,
"SV_ERROR_TEMPERATURE, Temperature measured is to high.")

err(SV_ERROR_LICENCE_DVI16,         207,
"SV_ERROR_LICENCE_DVI16, Trying to use dvi 16 when not licenced.")

err(SV_ERROR_IOCHANNEL_INVALID,     208,
"SV_ERROR_IOCHANNEL_INVALID, Trying to use invalid iochannel.")

err(SV_ERROR_DRIVER_MAPPEDSIZE,     209,
"SV_ERROR_DRIVER_MAPPEDSIZE, Mapped size to small.")



/* don't touch these lines *vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
#ifdef ERRSTRUCT
err(MAXERRCODE,                     MAXERRCODE,
"Sorry, no error text available")
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
