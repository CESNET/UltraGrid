/*
//	Definitions for the File Handle module.
*/


#ifndef FM_H
#define FM_H
/** \file */

/*
 *	Check for definitions of uint, ushort and ubyte
 *	most machines defines these in <sys/types.h>
 */

#if defined(sun)
# if defined(__sys_types_h) || defined(_SYS_TYPES_H)
#  define DEFINED_UINT
#  define DEFINED_USHORT
# endif
# elif !defined(_WIN32)
# if defined(_SYS_TYPES_H) || defined(_SYS_TYPES_H_) || defined(_SYS_TYPES_INCLUDED)
#  define DEFINED_UINT
#  define DEFINED_USHORT
# endif
#endif

/*
//	Windows/dos binary flag, if not defined define it to 0
*/
#ifndef O_BINARY
#define O_BINARY 0
#endif

#if defined(_WIN32) && defined(DLL)
#define export  __declspec(dllexport)
#else
#define export
#endif

#define FM_MODE_MONO				0
#define FM_MODE_RGB					1
#define FM_MODE_YUV422			2
#define FM_MODE_JPEG				3
#define FM_MODE_RGBA				4
#define FM_MODE_YUV422A			5
#define FM_MODE_YUV444 			6
#define FM_MODE_YUV444A			7

#define FM_YUV_CCIR601                  0
#define FM_YUV_CCIR709                  1
#define FM_YUV_SMPTE240                 2
#define FM_YUV_CCIR601_DIGITAL          3
#define FM_YUV_CCIR601_CGR              4
/* New matrices with more accurate coefficients */
#define FM_YUV_CCIR601_DIGITAL_10BIT    5
#define FM_YUV_CCIR601_CGR_10BIT        6
#define FM_YUV_SMPTE240_DIGITAL_10BIT   7
#define FM_YUV_SMPTE240_CGR_10BIT       8
#define FM_YUV_SMPTE274_DIGITAL_10BIT	9
#define FM_YUV_SMPTE274_CGR_10BIT       10
#define FM_YUV_CCIR709_DIGITAL_10BIT    11
#define FM_YUV_CCIR709_CGR_10BIT        12
#define FM_YUV_COUNT			13

#define FM_YUV_Y2R			0
#define FM_YUV_V2R			1
#define FM_YUV_U2R			2
#define FM_YUV_Y2G			3
#define FM_YUV_V2G			4
#define FM_YUV_U2G			5
#define FM_YUV_Y2B			6
#define FM_YUV_V2B			7
#define FM_YUV_U2B			8

#define FM_YUV_R2Y			0
#define FM_YUV_G2Y			1
#define FM_YUV_B2Y			2
#define FM_YUV_R2V			3
#define FM_YUV_G2V			4
#define FM_YUV_B2V			5
#define FM_YUV_R2U			6
#define FM_YUV_G2U			7
#define FM_YUV_B2U			8

#define FM_YUV_ZERO_Y			9
#define FM_YUV_ZERO_C			10


#define FM_PLANE_MONO			0

#define FM_PLANE_RGB_R			0
#define FM_PLANE_RGB_G			1
#define FM_PLANE_RGB_B			2
#define FM_PLANE_RGB_A			3

#define FM_PLANE_YUV_Y0			0
#define FM_PLANE_YUV_Y1			1				/* NOT used for YUV444 */
#define FM_PLANE_YUV_U			2
#define FM_PLANE_YUV_V			3
#define FM_PLANE_YUV_Y			FM_PLANE_YUV_Y0	/* used for YUV444 */

#define FM_PLANE_CMYK_C			0
#define FM_PLANE_CMYK_M			1
#define FM_PLANE_CMYK_Y			2
#define FM_PLANE_CMYK_K			3

#define FM_PLANE_JPEG_QUANT		0
#define FM_PLANE_JPEG_HUFF		1
#define FM_PLANE_JPEG_F1		2
#define FM_PLANE_JPEG_F2		3


#define FM_PLANE_A			0
#define FM_PLANE_B			1
#define FM_PLANE_C			2
#define FM_PLANE_D			3
#define FM_PLANE_COUNT			4

#define FM_CALLBACK_CREATE		0
#define FM_CALLBACK_OPEN		1
#define FM_CALLBACK_CLOSE		2
#define FM_CALLBACK_READ		3
#define FM_CALLBACK_WRITE		4
#define FM_CALLBACK_CHECK		5
#define FM_CALLBACK_GETSIZE		6
#define FM_CALLBACK_FHEAD		7
#define FM_CALLBACK_SHEAD		8
#define FM_CALLBACK_UNPACK		9
#define FM_CALLBACK_TERMINATE	        10


#define FM_OK			0
#define FM_ERROR		1	/* Undefined error		*/
#define FM_ERROR_MALLOC		2	/* Out of memory		*/
#define FM_ERROR_PROGRAM	3	/* Programmer error		*/
#define FM_ERROR_PROGRAM_FF	4	/* Null ff handle		*/
#define FM_ERROR_PROGRAM_NOCONV	5	/* No conversion available	*/
#define FM_ERROR_PROGRAM_MODE	6	/* Illegal mode			*/
#define FM_ERROR_PROGRAM_CB    	7	/* No callback available	*/
#define FM_ERROR_PROGRAM_BUFFER	8	/* No buffer allocated		*/
#define FM_ERROR_BUFFER_SIZE	9	/* Buffer is to small		*/
#define FM_ERROR_PROGRAM_HANDLE	10	/* Null handle *		*/
#define FM_ERROR_FILE_CREATE	11	/* A file create failed		*/
#define FM_ERROR_FILE_OPEN	12	/* A file open failed		*/
#define FM_ERROR_FILE_SEEK	13	/* A file seek failed		*/
#define FM_ERROR_FILE_READ	14	/* A file read failed		*/
#define FM_ERROR_FILE_WRITE	15	/* A file write failed		*/
#define FM_ERROR_FILE_NAME	16	/* Illegal file name		*/
#define FM_ERROR_FORMAT		17	/* Illegal file forma		*/
#define FM_ERROR_FORMAT_PAGE	18	/* Illegal page nr		*/
#define FM_ERROR_PROGRAM_FM	19	/* Null fm handle		*/
#define FM_ERROR_FILE_STAT	20	/* A file stat failed  		*/
#define FM_ERROR_FILE_TRUNCATE	21	/* A file truncate failed	*/


#define FM_CONVERT_READY		0
#define FM_CONVERT_MONO_MONO		1
#define FM_CONVERT_MONO_RGB		2
#define FM_CONVERT_MONO_YUV422		3
#define FM_CONVERT_RGB_MONO		4
#define FM_CONVERT_RGB_RGB		5
#define FM_CONVERT_RGB_YUV422		6
#define FM_CONVERT_YUV422_MONO		7
#define FM_CONVERT_YUV422_RGB		8
#define FM_CONVERT_YUV422_YUV422	9
#define FM_CONVERT_KEY_KEY		10
#define FM_CONVERT_YUV444_MONO	        11
#define FM_CONVERT_YUV444_RGB	        12
#define FM_CONVERT_YUV444_YUV422	13
#define FM_CONVERT_YUV444_YUV444	14
#define FM_CONVERT_MONO_YUV444	        15
#define FM_CONVERT_RGB_YUV444	        16
#define FM_CONVERT_YUV422_YUV444	17

#define FM_CONVERSION_COUNT		18

/*
 *	Is word data in little or big endian format
 */
#define FM_ENDIAN_BIG			0
#define FM_ENDIAN_LITTLE		1
#define FM_ENDIAN_COUNT			2

/*
 *	What do we describe the data in for types.
 */
#define FM_TYPE_BYTE			0
#define FM_TYPE_BITS			1
#define FM_TYPE_SHORT			2
#define FM_TYPE_COUNT			3

/*
 *	If data is aligned to the left or the right if not 
 *	the data is completly filled into the pixel.
 */

#define FM_ALIGN_LEFT			0
#define FM_ALIGN_RIGHT			1
#define FM_ALIGN_COUNT			2


/*
 *	Lets define uint, ushort and ubyte
 */

#ifndef DEFINED_UBYTE
#define DEFINED_UBYTE
typedef unsigned char   ubyte;
#endif

#ifndef DEFINED_USHORT
#define DEFINED_USHORT
#ifndef _AIX
typedef unsigned short  ushort;
#endif
#endif

#ifndef DEFINED_UINT
#define DEFINED_UINT
#ifndef _AIX
typedef unsigned int	uint;
#endif
#endif

#ifndef NULL
#define NULL 0
#endif



#define FM_STRING_COUNT 	80


typedef struct _fm_rec {
  struct 
    _fm_rec  *next;
  
  char        name[FM_STRING_COUNT];	/* Fileformat name		*/

  char        extension[FM_STRING_COUNT];/*Default extension		*/
  
  int         single;			/* File supports only 1 frame	*/

  struct {
    void *      (*create)();
    void *      (*open)();
    int         (*close)();
    int         (*write)();
    int         (*read)();
    char *      (*check)();
    int         (*getbuffersize)();
    int         (*fillheader)();
    int         (*scanheader)();
    int         (*unpack)();
    int         (*terminate)();		/* shutdown prop. libs          */
  } callback;

  int	      defaultx;			/* default horizontal size      */
  int         headersize_max;           /* max. headersize in bytes     */ 
  int         ffrec_create_support;     /* flags if the ffrec structure */
                                        /* could be filled with desired */
                                        /* values for mode, nbit etc.   */  
} fm_rec;

typedef struct {
  fm_rec     *fm;			/* File mode we are using       */

  void       *handle;			/* Opaque file handle		*/

  int         xsize;			/* Rows in frame		*/
  int         ysize;			/* Lines in frame		*/
  int         pages;			/* Pages in the file		*/

  ubyte      *buffer;			/* Frame buffer			*/
  int         size;			/* Size of frame buffer		*/
  
  int         mode;			/* FM_MODE_xxx			*/
  int	      yuv;			/* YUV conversion option	*/

  int         byteorder;		/* Little or bigendian data	*/
  int         pixelvalidbits;		/* Bits per pixel		*/
  int         pixelallocbits;		/* Allocated bits per pixel	*/
  int         datatype;			/* bits, byte, word		*/
  int         alignment;		/* Left or right aligned data	*/
  
  int         offset   [FM_PLANE_COUNT];/* Offset of first pixel	*/
  int 	      increment[FM_PLANE_COUNT];/* Offset to next pixel 	*/

  int         offset_odd;		/* Offsets between odd  -> even */
  int         offset_even;		/* Offsets between even -> odd	*/

  int         offset_key;               /* Offset for 1. pixel of key ch*/
  int         increment_key;            /* Increment for key channel    */

  int         offset_odd_key;		/* Offsets between odd  -> even */
  int         offset_even_key;		/* Offsets between even -> odd	*/

  int         headersize_current;       /* current headersize in bytes  */
  int         packed;			/* packed representation	*/
  int         prefill_flag;             /* flags that the ff_rec has    */
                                        /* been filled with desired     */
                                        /* values for mode, nbit etc.   */
  /* expansion for arbitrary additions */
  int         ffMagic;          /* FM_FF_MAGIC */
  int         ffVersion;        /* version of this struct definition - starts at 1 */
  int         ffSize;           /* size actually allocated for this struct */
  char       *ffParam;          /* pointer to optional parameter string */
  /* end of expansion version 1; note that parameter string might follow here! */
  void       *ffCookie;         /* pointer to optional cookie (struct) */
  /* end of expansion version 2; note that parameter string or cookie struct might follow here! */
} ff_rec;

#define FM_FF_MAGIC 0xF1FF1BEE

/*
//	Macros for support of little endian machines
*/

#if defined(WORDS_BIGENDIAN)
#define fm_ntohl(x)	(x)
#define fm_htonl(x)	(x)
#define fm_ntohs(x)	(x)
#define fm_htons(x)	(x)
#else
#define fm_ntohl(x)	fm_swap_uint(x)
#define fm_htonl(x)	fm_swap_uint(x)
#define fm_ntohs(x)	fm_swap_ushort(x)
#define fm_htons(x)	fm_swap_ushort(x)


export uint fm_swap_uint
   (
   uint x
   );

export ushort fm_swap_ushort
   (
   ushort x
   );
#endif

   
/*
 * Function:	Initialize the FileManager, must be called before usage
 */
export int fm_initialize(void);

/*
 * Function:	Deinitalize the FileManager
 */
export int fm_deinitialize(void);

/*
 * Function:	Register a file reader to the filemanager
 * 
 */
fm_rec * fm_register(
   char   *name				/* File format name		*/
);

/*
 * Function:	Add a callback to a fileformat	
 *
 */
void fm_callback(
   fm_rec *fm,				/* File format handle		*/
   int     code,			/* FM_CALLBACK_XXX		*/
   void   *(*function)()		/* callback function		*/
);


/*
 * Function:	Match a fileformat name to a ff handle;
 */

export int fm_fileformat_find(
   ff_rec *ff, 
   char   *name				/* File format name		*/
);

/*
 * Function:	Get a fileformat name;
 */

export int fm_fileformat_name(
   char   *name,			/* File format name		*/
   int    len,				/* max. size for name		*/
   int    position			/* index of file format		*/
);

/*
 * Function:	Allocate a fileformat and return a ff handle
 */
export ff_rec *fm_fileformat_allocate(
    char  *name				/* File format name		*/
);

/*
 * Function:	Release a fileformat handle;
 */

export int fm_fileformat_free(
   ff_rec *ff			  	/* File format handle		*/
);


/* 
 *  Function:	Create a new file
 */
export int fm_create(
   ff_rec    *ff,			/* Fileformat handle		*/
   char	     *name,			/* File name			*/
   ff_rec    *format			/* Format to read info from	*/
);


/*
 *  Function:	Open an old file
 */
export int fm_open(
   ff_rec    *ff,			/* Fileformat handle		*/
   char	     *name			/* File name			*/
);


/*
 *  Function:	Close a file
 */
export int  fm_close(
   ff_rec *ff				/* Fileformat handle		*/
);


/*
 *  Function:	Read a page from a file
 */
export int  fm_read(
   ff_rec *ff,				/* Fileformat handle		*/
   uint    page				/* Page number to read		*/
);

/*
 *  Function:	Read a page from a file
 */
export int  fm_write(
   ff_rec *ff,				/* Fileformat handle		*/
   uint    page				/* Page number to read		*/
);


/*
 *  Function:	Check if a file is of a file type
 */
export char * fm_check(
   char *filename
);


/*
 *  Function:	Read header from image file
 */
export int  fm_rheader(
   ff_rec *ff, 				/* Fileformat handle		*/
   char   *filename
);


/*
 *  Function:	get size of image data buffer for a particular format
 */
export int  fm_getbuffersize(
   ff_rec *ff  				/* Fileformat handle		*/
);


/*
 *  Function:	Fill header buffer with file header data
 */
export int  fm_fillheader(
   ff_rec *ff, 				/* Fileformat handle		*/
   char   *header,			/* header buffer		*/
   int     size				/* size of header buffer	*/
);


/*
 *  Function:	Scan buffer for valid file header 
 */
export int  fm_scanheader(
   ff_rec *ff, 				/* Fileformat handle		*/
   char   *header,			/* header buffer		*/
   int     size				/* size of header buffer	*/
);


/*
 *  Function:	Scan buffer for valid file header 
 */
export int  fm_unpack(
   ff_rec *ff,				/* Fileformat handle		*/
   char   *dbuf,			/* destination buffer		*/
   int     dsize,			/* size of destination buffer	*/
   char   *sbuf,			/* source buffer		*/
   int     ssize,			/* size of source buffer	*/
   char   *hbuf,			/* header buffer		*/
   int     hsize			/* size of header buffer	*/
);


/*
 * Function:	Convert frame pointed to by ff_from into frame
 *		in ff_to. Data reordering and colorspace conversion
 *		is done.
 */
export int  fm_convert(
   ff_rec *ff_from,			/* Fileformat handle		*/
   ff_rec *ff_to			/* Fileformat handle		*/
);


/*
 *  Function:	Allocate buffer space for file
 */
export int   fm_malloc_buffer(ff_rec * ff);

/*
 *  Function:	Free buffer space for file
 */
export int   fm_free_buffer(ff_rec * ff);


/*
 *  Function:	enable error messages in GUI-style 
 */
export void  fm_set_guiflag();


/*
 *  Function:	Print an error message 
 */
export void  fm_errorprint(int code, char * str);



/*
 *  Function:	Return a string describing the error code 
 */
#undef fm_errorstr
export char * fm_errorstr(int errorcode);


/*
 *  Function:	Call the register functions to registrate needed fileformats
 */
void fm_config(void);

export extern int fm_errno;

#ifdef FM_ERRORPRINT
#if defined(__STDC__) || defined(_WIN32)
#define fm_error(code, str)    \
  if(!fm_errno) fm_errno=code; \
  fprintf(stderr, "Error %d in file %s, line %d\n", code, __FILE__, __LINE__); \
  fm_errorprint(code, str);
#else
#define fm_error(code, str)	\
  if(!fm_errno) fm_errno=code; \
  fm_errorprint(code, str);
#endif
#else
#define fm_error(code, str)	if(!fm_errno) fm_errno=code;
#endif

#if 1
#define fm_malloc(size)	malloc((uint) size)
#define fm_free(ptr)	free((char *) ptr)
#else
#define fm_malloc(size)	fm_malloc_debug((uint)   size, __FILE__, __LINE__)
#define fm_free(ptr)	fm_free_debug  ((char *) ptr,  __FILE__, __LINE__)
int debugmalloc = 0;

static void *fm_malloc_debug(uint size, char *file, int line)
{
  void *res = malloc(size);
  fprintf(stderr, "fm_malloc %08x : %7d %4d: %s @ %d\n", res, size, 
		debugmalloc++, file, line);
  return res;
}

static void fm_free_debug(void *ptr, char *file, int line)
{
  free((char *) ptr);
  fprintf(stderr, "fm_free   %08x :         %4d: %s @ %d\n", ptr, 
		debugmalloc++, file, line);
}
#endif


/*
 *  Definitions for the function fm_converter()
 */

#define FM_CONVERTER_NOP	    0x00
#define FM_CONVERTER_FILELIST	    0x01
#define FM_CONVERTER_VERBOSE	    0x02

typedef struct {
  ff_rec  *ff;
  char    *filelist;
  char    *basename;
  char     devisor;	    /* Character to add pagenumber to */
  int      start;
  int      nframes;
  int      increment;
  int      xsize;	    /* Prefered xsize		    */
  int      ysize;	    /* Prefered ysize		    */
} fm_converter_rec;


export int fm_converter(
    fm_converter_rec *fp_in, 
    fm_converter_rec *fp_out, 
    int               flags
);

/*
 *	Functions that register a fileformat
 */
export void fm_tiff_register(void);

export void fm_abekas_register(void);

export void fm_bmp_register(void);

export void fm_sgi_register(void);

export void fm_ppm_register(void);

export void fm_dvs_register(void);

export void fm_fts_register(void);

export void fm_soft_register(void);

export void fm_tga_register(void);

export void fm_als_register(void);

export void fm_avi_register(void);

export void fm_yuv16_register(void);

export void fm_raw_register(void);

export void fm_ras_register(void);

export void fm_cineon_register(void);

export void fm_dpx_register(void);

export void fm_yuv10_register(void);

export void fm_jpg_register(void);

export void fm_user_register(int num);


export int am_check(
    char     *filename,
    char    **format,
    int      *nsample,
    int      *nchan,
    int      *nbit 
);

export int am_aiff_open(
    char *filename
);

export int am_aiff_create(
    char  *filename
);

export void am_aiff_update(
    int      fd,
    int      nsample
);

export void am_aiff_close(
  int	  fd
);



export int am_direct_open(
    char    *buffer,
    int      flags,
    int      mode
);


export void am_direct_close(
    int      fd
);


export int am_direct_read(
    int      fd,
    char    *buffer,
    int      nbyte
);

export int am_direct_write(
    int      fd,
    char    *buffer,
    int      nbyte
);

#endif

