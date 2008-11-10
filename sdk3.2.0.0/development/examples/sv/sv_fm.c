/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv/svram program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/


#include "svprogram.h"

#if defined(WIN32)
# define fm_errorstr(a)	""
#endif

#if !defined(WIN32) && !defined(__CYGWIN__)
# define strnicmp(a,b,c)	strncasecmp(a,b,c)
#endif


/*
 *	Variable used to signal an abort to a transfer
 */
static int jpeg_abort;


void jpeg_transfer_abort(int signal)
{
  jpeg_abort = 1;
}


#if defined(WIN32)
#include        <windows.h>
 
int jpeg_createmapping(int  ProcessId )
{
  char    szName[128];
 
  sprintf(szName,MV_SHM_TEMPLATE,ProcessId);
  hMvShm = CreateFileMapping(INVALID_HANDLE_VALUE,NULL,PAGE_READWRITE,0,sizeof(MVSHM),szName);
  if( hMvShm==NULL ) {
    printf("Failed to create mapping for %s - error code 0x%08X\n",szName,GetLastError());
    pMvShm = NULL;
    return -1;
  }
  pMvShm = MapViewOfFile(hMvShm,FILE_MAP_WRITE,0,0,sizeof(MVSHM));
  if( pMvShm==NULL ) {
    printf("Failed to map mapping - error code 0x%08X\n",GetLastError());
    if( !CloseHandle(hMvShm) ) {
      printf("Failed to close mapping - error code 0x%08X\n",GetLastError());
    }
    hMvShm = NULL;
    return -1;
  }
  memset(pMvShm,0,sizeof(MVSHM));
  pMvShm->size   = sizeof(MVSHM);
  pMvShm->major  = 0;
  pMvShm->minor  = 0;
  pMvShm->magic  = 0xFEEDBEEF;
         
  return 0;
}
 
int jpeg_closemapping(void)
{
  if( pMvShm && !UnmapViewOfFile(pMvShm) ) {
    printf("Failed to unmap mapping - error code 0x%08X\n",GetLastError());
  }
  pMvShm = NULL;
  if( hMvShm && !CloseHandle(hMvShm) ) {
    printf("Failed to close mapping - error code 0x%08X\n",GetLastError());
  }
  hMvShm = NULL;
  return 0;
}        
#endif


static int jpeg_getchannels(sv_handle * sv, char * channels)
{
  int   tmp[16];
  int   i;
  int   ret = -1;
  int   count = 0;
  int   pos;

  memset(tmp, 0, sizeof(tmp));
  for (i = 0; i < (int)strlen(channels); i++) {
    if((channels[i] >= 'a') && (channels[i] <= 'g')) {
      pos = channels[i] - 'a' + 9;
    } else {
      pos = channels[i] - '1';
    }
    if((pos >= 0) && (pos < sizeof(tmp)/sizeof(tmp[0]))) {
      tmp[pos] = 1; count++;
    }
  }

  if(count == 2) {
    if (tmp[0] && tmp[1]) {
      ret = SV_TYPE_AUDIO12;
    } else if (tmp[2] & tmp[3]) {
      ret = SV_TYPE_AUDIO34;
    } else if (tmp[4] & tmp[5]) {
      ret = SV_TYPE_AUDIO56;
    } else if (tmp[6] & tmp[7]) {
      ret = SV_TYPE_AUDIO78;
    } else if (tmp[8] & tmp[9]) {
      ret = SV_TYPE_AUDIO9a;
    } else if (tmp[10] & tmp[11]) {
      ret = SV_TYPE_AUDIObc;
    } else if (tmp[12] & tmp[13]) {
      ret = SV_TYPE_AUDIOde;
    } else if (tmp[14] & tmp[15]) {
      ret = SV_TYPE_AUDIOfg;
    }  
  } else {
    jpeg_errorprintf(sv, "Error: no audio channel specified\n");
    ret = 0;
  }

  if (ret == -1) {
    jpeg_errorprintf(sv, "Error: one audio channel pair must be specified\n");
    ret = 0;
  }

  return ret;
}


int jpeg_convertimage(sv_handle * sv, int loadimage, char * file, char * type, int page, int xsize, int ysize, ubyte * buffer, int size, int bpc, int cmode, int fieldmode, int nbit, int fieldcount, int bottom2top, int xsizevideo, int *with_key)
{
  ff_rec *ff_host;
  ff_rec  ff_sv, ff_sv_tmp;
  int     res;
  uint   *p1, *p2;
  uint    i, j, k;
  uint	  offset;
  uint    size_field1;
  uint    size_field2;
  uint    size_line;
  int     fieldmode_orig = fieldmode;

  memset(&ff_sv, 0, sizeof(ff_rec));

  ff_host = fm_fileformat_allocate(type);

  if(ff_host == NULL) {
    jpeg_errorprintf(sv, "Couldn't open file converter %s\n", type);
    return SV_ERROR_FILEOPEN;
  }

  ff_sv.xsize	= xsize;
  if (fieldmode == 1) {
    ff_sv.ysize	= ysize>>1;
  } else {
    ff_sv.ysize	= ysize;
  }
  ff_sv.pages	= page + 1;
  ff_sv.buffer	= buffer;
  ff_sv.size	= size;
  ff_sv.mode	= FM_MODE_YUV422;
  ff_sv.yuv	= FM_YUV_CCIR601_CGR;
  ff_sv.yuv	= FM_YUV_CCIR601;

  if(bpc == 1) {
    ff_sv.datatype       = FM_TYPE_BYTE;
  } else {
    ff_sv.datatype       = FM_TYPE_SHORT;
  }

#ifdef WORDS_BIGENDIAN
  ff_sv.byteorder = FM_ENDIAN_BIG;
#else
  ff_sv.byteorder = FM_ENDIAN_LITTLE;
#endif

  switch(cmode) {
  case SV_COLORMODE_YUV2QT:
  case SV_COLORMODE_YUV422:
  case SV_COLORMODE_YUV422_YUYV:
    ff_sv.mode  = FM_MODE_YUV422;
    break;
  case SV_COLORMODE_YUV422A:
    ff_sv.mode  = FM_MODE_YUV422A;
    break;
  case SV_COLORMODE_RGB_BGR:
  case SV_COLORMODE_RGB_RGB:
    ff_sv.mode  = FM_MODE_RGB;
    break;
  case SV_COLORMODE_RGBVIDEO:
    ff_sv.mode  = FM_MODE_RGB;
    ff_sv.yuv	= FM_YUV_CCIR601;
    break;
  case SV_COLORMODE_ABGR:
  case SV_COLORMODE_ARGB:
  case SV_COLORMODE_BGRA:
  case SV_COLORMODE_RGBA:
    ff_sv.mode  = FM_MODE_RGBA;
    break;
  case SV_COLORMODE_RGBAVIDEO:
    ff_sv.mode  = FM_MODE_RGBA;
    ff_sv.yuv	= FM_YUV_CCIR601;
    break;
  case SV_COLORMODE_BAYER_RGB:
  case SV_COLORMODE_MONO:
  case SV_COLORMODE_CHROMA:
    ff_sv.mode  = FM_MODE_MONO;
    break;
  case SV_COLORMODE_YUV444:
    ff_sv.mode  = FM_MODE_YUV444;
    break;
  case SV_COLORMODE_YUV444A:
    ff_sv.mode  = FM_MODE_YUV444A;
    break;
  default:
    jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
    return SV_ERROR_PROGRAM;
  }

  /*
  //  Size of first field, take of of odd sized images.
  */
  if(ysize == 1035) {
    size_field1 = xsize * ((ysize    ) >> 1);
    size_field2 = xsize * ((ysize + 1) >> 1);
  } else {
    size_field1 = xsize * ((ysize + 1) >> 1);
    size_field2 = xsize * ((ysize    ) >> 1);
  }

  switch(cmode) {
   case SV_COLORMODE_YUV2QT:
   case SV_COLORMODE_YUV422:
   case SV_COLORMODE_YUV422_YUYV:
    ff_sv.mode  = FM_MODE_YUV422;
    size_field1 = 2 * size_field1;
    size_field2 = 2 * size_field2;
    size_line   = 2 * xsize;
    break;
   case SV_COLORMODE_YUV422A:
    ff_sv.mode  = FM_MODE_YUV422A;
    size_field1 = 3 * size_field1;
    size_field2 = 3 * size_field2;
    size_line   = 3 * xsize;
    break;
   case SV_COLORMODE_RGB_BGR:
   case SV_COLORMODE_RGB_RGB:
   case SV_COLORMODE_RGBVIDEO:
    ff_sv.mode  = FM_MODE_RGB;
    size_field1 = 3 * size_field1;
    size_field2 = 3 * size_field2;
    size_line   = 3 * xsize;
    break;
   case SV_COLORMODE_ABGR:
   case SV_COLORMODE_ARGB:
   case SV_COLORMODE_BGRA:
   case SV_COLORMODE_RGBA:
   case SV_COLORMODE_RGBAVIDEO:
    ff_sv.mode  = FM_MODE_RGBA;
    size_field1 = 4 * size_field1;
    size_field2 = 4 * size_field2;
    size_line   = 4 * xsize;
    break;
   case SV_COLORMODE_BAYER_RGB:
   case SV_COLORMODE_MONO:
   case SV_COLORMODE_CHROMA:
    ff_sv.mode  = FM_MODE_MONO;
    size_field1 = size_field1;
    size_field2 = size_field2;
    size_line   = xsize;
    break;
   case SV_COLORMODE_YUV444:
    ff_sv.mode  = FM_MODE_YUV444;
    size_field1 = 3 * size_field1;
    size_field2 = 3 * size_field2;
    size_line   = 3 * xsize;
    break;
   case SV_COLORMODE_YUV444A:
    ff_sv.mode  = FM_MODE_YUV444A;
    size_field1 = 4 * size_field1;
    size_field2 = 4 * size_field2;
    size_line   = 4 * xsize;
    break;
   default:
    jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
    return SV_ERROR_PROGRAM;
  }

  if ((fieldmode == 1) || (fieldmode == 2)) {
    // set size_fieldx in order to do filling with black correctly
    size_field1 = size;
    size_field2 = 0;
  }

  if (ysize == 496) {            /*  NTSC with SCSIVideo    */
     ff_sv.buffer = &buffer[10 * xsize];
     ff_sv.ysize  = 486;
     ff_sv.size  -= 20 * xsize;
  }

  if(!sv->prontovideo) {
    if(fieldcount == 1) {
      fieldmode = 1;
    }
  }

  if(((ysize == 486)  ||           /*  NTSC with ProntoVideo  */
      (ysize == 496)  ||           /*  NTSC with SCSIVideo    */
      (ysize == 1035)) &&     
     (fieldmode == 0)) {           /*  frame transfer         */
    switch (cmode) {
     case SV_COLORMODE_YUV2QT:
      ff_sv.offset[FM_PLANE_YUV_Y0]   = bpc * (    size_field1);
      ff_sv.offset[FM_PLANE_YUV_Y1]   = bpc * (2 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_U]    = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_V]    = bpc * (3 + size_field1);
      break;
     case SV_COLORMODE_YUV422:
      ff_sv.offset[FM_PLANE_YUV_Y0]   = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_Y1]   = bpc * (3 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_U]    = bpc * (    size_field1);
      ff_sv.offset[FM_PLANE_YUV_V]    = bpc * (2 + size_field1);
      break;
     case SV_COLORMODE_YUV422_YUYV:
      ff_sv.offset[FM_PLANE_YUV_Y0]   = bpc * (   size_field1);
      ff_sv.offset[FM_PLANE_YUV_Y1]   = bpc * (2 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_U]    = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_V]    = bpc * (3 + size_field1);
      break;
     case SV_COLORMODE_YUV422A:
      if ((nbit == 8) || (!sv->prontovideo)) {
        ff_sv.offset[FM_PLANE_YUV_Y0] = bpc * (1 + size_field1);
        ff_sv.offset[FM_PLANE_YUV_Y1] = bpc * (4 + size_field1);
        ff_sv.offset[FM_PLANE_YUV_U]  = bpc * (    size_field1);
        ff_sv.offset[FM_PLANE_YUV_V]  = bpc * (3 + size_field1);
      } else {
        ff_sv.offset[FM_PLANE_YUV_Y0] = bpc * (    size_field1);
        ff_sv.offset[FM_PLANE_YUV_Y1] = bpc * (3 + size_field1);
        ff_sv.offset[FM_PLANE_YUV_U]  = bpc * (1 + size_field1);
        ff_sv.offset[FM_PLANE_YUV_V]  = bpc * (4 + size_field1);
      }
      ff_sv.offset_key                = bpc * (2 + size_field1);
      break;
     case SV_COLORMODE_RGB_BGR:
      if(sv->prontovideo) {
        ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (    size_field1);
        ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (1 + size_field1);
        ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (2 + size_field1);
      } else {
        ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (    size_field1);
        ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (1 + size_field1);
        ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (2 + size_field1);
      }
      break;
     case SV_COLORMODE_ABGR:
      ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (2 + size_field1);
      ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (3 + size_field1);
      ff_sv.offset_key              = bpc * (    size_field1);
      break;
     case SV_COLORMODE_ARGB:
      ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (3 + size_field1);
      ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (2 + size_field1);
      ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (1 + size_field1);
      ff_sv.offset_key              = bpc * (    size_field1);
      break;
     case SV_COLORMODE_BGRA:
      ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (    size_field1);
      ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (2 + size_field1);
      ff_sv.offset_key              = bpc * (3 + size_field1);
      break;
     case SV_COLORMODE_RGB_RGB:
     case SV_COLORMODE_RGBVIDEO:
      ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (    size_field1);
      ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (2 + size_field1);
      break;
     case SV_COLORMODE_RGBA:
     case SV_COLORMODE_RGBAVIDEO:
      if(sv->prontovideo) {
        ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (    size_field1);
        ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (1 + size_field1);
        ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (2 + size_field1);
        ff_sv.offset_key              = bpc * (3 + size_field1);
      } else {
        ff_sv.offset[FM_PLANE_RGB_R]  = bpc * (    size_field1);
        ff_sv.offset[FM_PLANE_RGB_G]  = bpc * (1 + size_field1);
        ff_sv.offset[FM_PLANE_RGB_B]  = bpc * (2 + size_field1);
        ff_sv.offset_key              = bpc * (3 + size_field1);
      }
      break;
     case SV_COLORMODE_BAYER_RGB:
     case SV_COLORMODE_MONO:
     case SV_COLORMODE_CHROMA:
      ff_sv.offset[FM_PLANE_MONO]     = bpc * (    size_field1);
      break;
     case SV_COLORMODE_YUV444:
      ff_sv.offset[FM_PLANE_YUV_U]    = bpc * (0 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_Y]    = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_V]    = bpc * (2 + size_field1);
      break;
     case SV_COLORMODE_YUV444A:
      ff_sv.offset[FM_PLANE_YUV_U]    = bpc * (0 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_Y]    = bpc * (1 + size_field1);
      ff_sv.offset[FM_PLANE_YUV_V]    = bpc * (2 + size_field1);
      ff_sv.offset_key                = bpc * (3 + size_field1);
      break;
     default:
      jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
      return SV_ERROR_PROGRAM;
    }
  } else {
    switch (cmode) {
     case SV_COLORMODE_YUV2QT:
      ff_sv.offset[FM_PLANE_YUV_Y0]   = 0;
      ff_sv.offset[FM_PLANE_YUV_Y1]   = 2 * bpc;
      ff_sv.offset[FM_PLANE_YUV_U]    = bpc;
      ff_sv.offset[FM_PLANE_YUV_V]    = 3 * bpc;
      break;
     case SV_COLORMODE_YUV422:
      ff_sv.offset[FM_PLANE_YUV_Y0]   =     bpc;
      ff_sv.offset[FM_PLANE_YUV_Y1]   = 3 * bpc;
      ff_sv.offset[FM_PLANE_YUV_U]    = 0;
      ff_sv.offset[FM_PLANE_YUV_V]    = 2 * bpc;
      break;
     case SV_COLORMODE_YUV422_YUYV:
      ff_sv.offset[FM_PLANE_YUV_Y0]   = 0;
      ff_sv.offset[FM_PLANE_YUV_Y1]   = 2 * bpc;
      ff_sv.offset[FM_PLANE_YUV_U]    = 1 * bpc;
      ff_sv.offset[FM_PLANE_YUV_V]    = 3 * bpc;
      break;
     case SV_COLORMODE_YUV422A:
      if ((nbit == 8) || (!sv->prontovideo)) {
        ff_sv.offset[FM_PLANE_YUV_Y0] = bpc;
        ff_sv.offset[FM_PLANE_YUV_Y1] = bpc * 4;
        ff_sv.offset[FM_PLANE_YUV_U]  = 0;
        ff_sv.offset[FM_PLANE_YUV_V]  = bpc * 3;
      } else {
        ff_sv.offset[FM_PLANE_YUV_Y0] = 0;
        ff_sv.offset[FM_PLANE_YUV_Y1] = bpc * 3;
        ff_sv.offset[FM_PLANE_YUV_U]  = bpc;
        ff_sv.offset[FM_PLANE_YUV_V]  = bpc * 4;
      }
      ff_sv.offset_key                = bpc * 2;
      break;
     case SV_COLORMODE_RGB_RGB:
     case SV_COLORMODE_RGBVIDEO:
       ff_sv.offset[FM_PLANE_RGB_R]  = 0;
       ff_sv.offset[FM_PLANE_RGB_G]  = bpc;
       ff_sv.offset[FM_PLANE_RGB_B]  = 2 * bpc;
       break;
     case SV_COLORMODE_RGB_BGR:
      if(sv->prontovideo) {
        ff_sv.offset[FM_PLANE_RGB_G]  = 0;
        ff_sv.offset[FM_PLANE_RGB_B]  = bpc;
        ff_sv.offset[FM_PLANE_RGB_R]  = 2 * bpc;
      } else {
        ff_sv.offset[FM_PLANE_RGB_B]  = 0;
        ff_sv.offset[FM_PLANE_RGB_G]  = bpc;
        ff_sv.offset[FM_PLANE_RGB_R]  = 2 * bpc;
      }
      break;
     case SV_COLORMODE_ABGR:
      ff_sv.offset[FM_PLANE_RGB_B]  = 1 * bpc;
      ff_sv.offset[FM_PLANE_RGB_G]  = 2 * bpc;
      ff_sv.offset[FM_PLANE_RGB_R]  = 3 * bpc;
      ff_sv.offset_key              = 0;
      break;
     case SV_COLORMODE_ARGB:
      ff_sv.offset[FM_PLANE_RGB_B]  = 3 * bpc;
      ff_sv.offset[FM_PLANE_RGB_G]  = 2 * bpc;
      ff_sv.offset[FM_PLANE_RGB_R]  = 1 * bpc;
      ff_sv.offset_key              = 0;
      break;
     case SV_COLORMODE_BGRA:
      ff_sv.offset[FM_PLANE_RGB_B]  = 0;
      ff_sv.offset[FM_PLANE_RGB_G]  = bpc;
      ff_sv.offset[FM_PLANE_RGB_R]  = 2 * bpc;
      ff_sv.offset_key              = 3 * bpc;
      break;
     case SV_COLORMODE_RGBA:
     case SV_COLORMODE_RGBAVIDEO:
      if(sv->prontovideo) {
        ff_sv.offset[FM_PLANE_RGB_G]  = 0;
        ff_sv.offset[FM_PLANE_RGB_B]  = bpc;
        ff_sv.offset[FM_PLANE_RGB_R]  = 2 * bpc;
        ff_sv.offset_key              = 3 * bpc;
      } else {
        ff_sv.offset[FM_PLANE_RGB_R]  = 0;
        ff_sv.offset[FM_PLANE_RGB_G]  = bpc;
        ff_sv.offset[FM_PLANE_RGB_B]  = 2 * bpc;
        ff_sv.offset_key              = 3 * bpc;
      }
      break;
     case SV_COLORMODE_BAYER_RGB:
     case SV_COLORMODE_MONO:
     case SV_COLORMODE_CHROMA:
      ff_sv.offset[FM_PLANE_MONO]     = 0;
      break;
     case SV_COLORMODE_YUV444:
     case SV_COLORMODE_YUV444A:
      ff_sv.offset[FM_PLANE_YUV_U]    = 0;
      ff_sv.offset[FM_PLANE_YUV_Y]    = bpc;
      ff_sv.offset[FM_PLANE_YUV_V]    = 2 * bpc;
      ff_sv.offset_key                = 3 * bpc;
      break;
     default:
      jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
      return SV_ERROR_PROGRAM;
    }
  }

  switch (cmode) {
   case SV_COLORMODE_YUV2QT:
   case SV_COLORMODE_YUV422:
   case SV_COLORMODE_YUV422_YUYV:
    ff_sv.increment[FM_PLANE_YUV_Y0] = 4 * bpc;
    ff_sv.increment[FM_PLANE_YUV_Y1] = 4 * bpc;
    ff_sv.increment[FM_PLANE_YUV_U]  = 4 * bpc;
    ff_sv.increment[FM_PLANE_YUV_V]  = 4 * bpc;
    break;
   case SV_COLORMODE_YUV422A:
    ff_sv.increment[FM_PLANE_YUV_Y0] = 6 * bpc;
    ff_sv.increment[FM_PLANE_YUV_Y1] = 6 * bpc;
    ff_sv.increment[FM_PLANE_YUV_U]  = 6 * bpc;
    ff_sv.increment[FM_PLANE_YUV_V]  = 6 * bpc;
    ff_sv.increment_key              = 3 * bpc;
    break;
   case SV_COLORMODE_RGB_BGR:
   case SV_COLORMODE_RGB_RGB:
   case SV_COLORMODE_RGBVIDEO:
    ff_sv.increment[FM_PLANE_RGB_G]  = 3 * bpc;
    ff_sv.increment[FM_PLANE_RGB_B]  = 3 * bpc;
    ff_sv.increment[FM_PLANE_RGB_R]  = 3 * bpc;
    break;
   case SV_COLORMODE_ABGR:
   case SV_COLORMODE_ARGB:
   case SV_COLORMODE_BGRA:
   case SV_COLORMODE_RGBA:
   case SV_COLORMODE_RGBAVIDEO:
    ff_sv.increment[FM_PLANE_RGB_G]  = 4 * bpc;
    ff_sv.increment[FM_PLANE_RGB_B]  = 4 * bpc;
    ff_sv.increment[FM_PLANE_RGB_R]  = 4 * bpc;
    ff_sv.increment_key              = 4 * bpc;
    break;
   case SV_COLORMODE_BAYER_RGB:
   case SV_COLORMODE_MONO:
   case SV_COLORMODE_CHROMA:
    ff_sv.increment[FM_PLANE_MONO]   = bpc;
    break;
   case SV_COLORMODE_YUV444:
    ff_sv.increment[FM_PLANE_YUV_U]  = 3 * bpc;
    ff_sv.increment[FM_PLANE_YUV_Y]  = 3 * bpc;
    ff_sv.increment[FM_PLANE_YUV_V]  = 3 * bpc;
    break;
   case SV_COLORMODE_YUV444A:
    ff_sv.increment[FM_PLANE_YUV_U]  = 4 * bpc;
    ff_sv.increment[FM_PLANE_YUV_Y]  = 4 * bpc;
    ff_sv.increment[FM_PLANE_YUV_V]  = 4 * bpc;
    ff_sv.increment_key              = 4 * bpc;
    break;
   default:
    jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
    return SV_ERROR_PROGRAM;
  }

  if((ysize == 486)  ||          /*  NTSC with ProntoVideo  */
     (ysize == 496)  ||          /*  NTSC with SCSIVideo    */
     (ysize == 1035)) {          /*  SMPTE240               */    
    switch (fieldmode) {
     case 0:
      switch (cmode) {
       case SV_COLORMODE_YUV2QT:
       case SV_COLORMODE_YUV422:
       case SV_COLORMODE_YUV422_YUYV:
        ff_sv.offset_even              = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd               =   bpc * (size_field1            );
        break;
       case SV_COLORMODE_YUV422A:
        ff_sv.offset_even              = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd               =   bpc * (size_field1            );
        ff_sv.offset_even_key          = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd_key           =   bpc * (size_field1            );
        break;
       case SV_COLORMODE_RGB_BGR:
       case SV_COLORMODE_RGB_RGB:
       case SV_COLORMODE_RGBVIDEO:
       case SV_COLORMODE_YUV444:
        ff_sv.offset_even              = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd               =   bpc * (size_field1            );
        break;
       case SV_COLORMODE_ABGR:
       case SV_COLORMODE_ARGB:
       case SV_COLORMODE_BGRA:
       case SV_COLORMODE_RGBA:
       case SV_COLORMODE_RGBAVIDEO:
       case SV_COLORMODE_YUV444A:
        ff_sv.offset_even              = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd               =   bpc * (size_field1            );
        ff_sv.offset_even_key          = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd_key           =   bpc * (size_field1            );
        break;
       case SV_COLORMODE_BAYER_RGB: 
       case SV_COLORMODE_MONO:
       case SV_COLORMODE_CHROMA:
        ff_sv.offset_even              = - bpc * (size_field1 + size_line);
        ff_sv.offset_odd               =   bpc * (size_field1            );
        break;
       default:
        jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
        return SV_ERROR_PROGRAM;
      }
      break;
     case 1:
      if ((fieldcount == 1) && (fieldmode_orig == 1)) {
        // field wise loading with progressive storage
        ff_sv.offset_even              =   bpc * size_line;
        ff_sv.offset_odd               =   bpc * size_line;
      } else {
        ff_sv.offset_even              =   0;
        ff_sv.offset_odd               =   0;
      }
      break;
     case 2: 
      switch (cmode) {
       case SV_COLORMODE_YUV2QT:
       case SV_COLORMODE_YUV422:
       case SV_COLORMODE_YUV422_YUYV:
        ff_sv.offset_even              = - bpc * size_line;
        break;
       case SV_COLORMODE_YUV422A:
        ff_sv.offset_even              = - bpc * size_line;
        ff_sv.offset_even_key          = - bpc * size_line;
        break;
       case SV_COLORMODE_RGB_BGR:
       case SV_COLORMODE_RGB_RGB:
       case SV_COLORMODE_RGBVIDEO:
       case SV_COLORMODE_YUV444:
        ff_sv.offset_even              = - bpc * size_line;
        break;
       case SV_COLORMODE_ABGR:
       case SV_COLORMODE_ARGB:
       case SV_COLORMODE_BGRA:
       case SV_COLORMODE_RGBA:
       case SV_COLORMODE_RGBAVIDEO:
       case SV_COLORMODE_YUV444A:
        ff_sv.offset_even              = - bpc * size_line;
        ff_sv.offset_even_key          = - bpc * size_line;
        break;
       case SV_COLORMODE_BAYER_RGB: 
       case SV_COLORMODE_MONO:
       case SV_COLORMODE_CHROMA:
        ff_sv.offset_even              = - bpc * size_line;
        break;
       default:
        jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
        return SV_ERROR_PROGRAM;
      }
      ff_sv.offset_odd               =   0;
      ff_sv.offset_odd_key           =   0;
      break;
     default:
      jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Fieldmode");
      return SV_ERROR_PROGRAM;
    }
  } else {
    switch (fieldmode) {
     case 0:
      switch (cmode) {
       case SV_COLORMODE_YUV2QT:
       case SV_COLORMODE_YUV422:
       case SV_COLORMODE_YUV422_YUYV:
        ff_sv.offset_even             =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd              = - bpc * (size_field1            );
        break;
       case SV_COLORMODE_YUV422A:
        ff_sv.offset_even             =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd              = - bpc * (size_field1            );
        ff_sv.offset_even_key         =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd_key          = - bpc * (size_field1            );
        break;
       case SV_COLORMODE_RGB_BGR:
       case SV_COLORMODE_RGB_RGB:
       case SV_COLORMODE_RGBVIDEO:
       case SV_COLORMODE_YUV444:
        ff_sv.offset_even             =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd              = - bpc * (size_field1            );
        break;
       case SV_COLORMODE_ABGR:
       case SV_COLORMODE_ARGB:
       case SV_COLORMODE_BGRA:
       case SV_COLORMODE_RGBA:
       case SV_COLORMODE_RGBAVIDEO:
       case SV_COLORMODE_YUV444A:
        ff_sv.offset_even             =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd              = - bpc * (size_field1            );
        ff_sv.offset_even_key         =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd_key          = - bpc * (size_field1            );
        break;
       case SV_COLORMODE_BAYER_RGB: 
       case SV_COLORMODE_MONO:
       case SV_COLORMODE_CHROMA:
        ff_sv.offset_even             =   bpc * (size_field1 - size_line);
        ff_sv.offset_odd              = - bpc * (size_field1            );
        break;
       default:
        jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
        return SV_ERROR_PROGRAM;
      }
      break;
     case 1:
      if ((fieldcount == 1) && (fieldmode_orig == 1)) {
        // field wise loading with progressive storage
        ff_sv.offset_even              =   bpc * size_line;
        ff_sv.offset_odd               =   bpc * size_line;
      } else {
        ff_sv.offset_even              =   0;
        ff_sv.offset_odd               =   0;
      }
      break;
     case 2: 
      switch (cmode) {
       case SV_COLORMODE_YUV2QT:
       case SV_COLORMODE_YUV422:
       case SV_COLORMODE_YUV422_YUYV:
        ff_sv.offset_even              = - bpc * size_line;
        break;
       case SV_COLORMODE_YUV422A:
        ff_sv.offset_even              = - bpc * size_line;
        ff_sv.offset_even_key          = - bpc * size_line;
        break;
       case SV_COLORMODE_RGB_BGR:
       case SV_COLORMODE_RGB_RGB:
       case SV_COLORMODE_RGBVIDEO:
       case SV_COLORMODE_YUV444:
        ff_sv.offset_even              = - bpc * size_line;
        break;
       case SV_COLORMODE_ABGR:
       case SV_COLORMODE_ARGB:
       case SV_COLORMODE_BGRA:
       case SV_COLORMODE_RGBA:
       case SV_COLORMODE_RGBAVIDEO:
       case SV_COLORMODE_YUV444A:
        ff_sv.offset_even              = - bpc * size_line;
        ff_sv.offset_even_key          = - bpc * size_line;
        break;
       case SV_COLORMODE_BAYER_RGB: 
       case SV_COLORMODE_MONO:
       case SV_COLORMODE_CHROMA:
        ff_sv.offset_even              = - bpc * size_line;
        break;
       default:
        jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
        return SV_ERROR_PROGRAM;
      }
      ff_sv.offset_odd               =   0;
      ff_sv.offset_odd_key           =   0;
      break;
     default:
      jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Fieldmode");
      return SV_ERROR_PROGRAM;
    }
  }

  if(loadimage) {
    int black_flag = 0;

    if(fm_open(ff_host, file) != FM_OK) {
      jpeg_errorprintf(sv, "Couldn't open %s file %s : %s\n", type, file, fm_errorstr(fm_errno));
      return SV_ERROR_FILEOPEN;
    }

    if(((getenv("SCSIVIDEO_TEMPORAL_FIELD_SWAP") != NULL) && (fieldmode_orig == 0)) ^ ((ysize == 1080) && (ff_host->ysize == 1035))) { /* Normal XOR is ok */
      /* we want to swap fields in temporal direction */
      /* therefore we insert one black line in a      */
      /* tricky way                                   */
      int tmp  = ff_sv.offset_even;
      int tmp1 = ff_sv.offset_even_key;
      sv_storageinfo    psi;
      
      res = sv_storage_status(sv, 0, NULL, &psi, sizeof(psi), 0);
      switch( psi.interlace ) {
      default:
      case 1:
        ff_sv.offset[FM_PLANE_A] += bpc * size_line;
        ff_sv.offset[FM_PLANE_B] += bpc * size_line;
        ff_sv.offset[FM_PLANE_C] += bpc * size_line; 
        ff_sv.offset[FM_PLANE_D] += bpc * size_line;
        ff_sv.offset_key         += bpc * size_line;
        ff_sv.ysize -= 1;
        ff_sv.size  -= bpc * size_line;
        black_flag = 1;
        break;
      case 2:
        ff_sv.offset_even     = ff_sv.offset_odd; 
        ff_sv.offset_even_key = ff_sv.offset_odd_key; 
        ff_sv.offset_odd      = tmp; 
        ff_sv.offset_odd_key  = tmp1; 
        if((ysize == 486) || (ysize == 496) || (ysize == 1036)) {
          ff_sv.offset[FM_PLANE_A] -= bpc * size_field1;
          ff_sv.offset[FM_PLANE_B] -= bpc * size_field1;
          ff_sv.offset[FM_PLANE_C] -= bpc * size_field1;
          ff_sv.offset[FM_PLANE_D] -= bpc * size_field1;
          ff_sv.offset_key         -= bpc * size_field1;
        } else {
          ff_sv.offset[FM_PLANE_A] += bpc * size_field1;
          ff_sv.offset[FM_PLANE_B] += bpc * size_field1;
          ff_sv.offset[FM_PLANE_C] += bpc * size_field1; 
          ff_sv.offset[FM_PLANE_D] += bpc * size_field1;
          ff_sv.offset_key         += bpc * size_field1;
        }
        ff_sv.ysize -= 1;
        ff_sv.size  -= bpc * size_line;
        black_flag = 1;
        break;
      }
    }

    if(fm_read(ff_host, page) != FM_OK) {
      jpeg_errorprintf(sv, "Error reading image from %s file %s : %s\n", type, file, fm_errorstr(fm_errno));
      return SV_ERROR_FILEREAD;
    }

    if((xsizevideo > ff_host->xsize) || (ff_sv.ysize > ff_host->ysize) || black_flag) {
    
      /*
       *    Fill buffer first with black
       */

      if (fieldmode) {
        if(ysize != 720) {
          ysize     >>= 1;
        }
      }

      p1 = (uint*) &buffer[0];
      switch (cmode) {
       case SV_COLORMODE_YUV2QT:
       case SV_COLORMODE_YUV422:
       case SV_COLORMODE_YUV422_YUYV:
        j = bpc * (size_field1 + size_field2) >> 2;
        break;
       case SV_COLORMODE_YUV422A:
        j = bpc * (size_field1 + size_field2) >> 2;
        break;
       case SV_COLORMODE_RGB_BGR:
       case SV_COLORMODE_RGB_RGB:
       case SV_COLORMODE_RGBVIDEO:
       case SV_COLORMODE_YUV444:
        j = bpc * (size_field1 + size_field2) >> 2;
        break;
       case SV_COLORMODE_ABGR:
       case SV_COLORMODE_ARGB:
       case SV_COLORMODE_BGRA:
       case SV_COLORMODE_RGBA:
       case SV_COLORMODE_RGBAVIDEO:
       case SV_COLORMODE_YUV444A:
        j = bpc * (size_field1 + size_field2) >> 2;
        break;
       case SV_COLORMODE_BAYER_RGB: 
       case SV_COLORMODE_MONO:
       case SV_COLORMODE_CHROMA:
        j = bpc * (size_field1 + size_field2) >> 2;
        break;
       default:
        jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
        return SV_ERROR_PROGRAM;
      }

      if(bpc == 1) {
        switch (cmode) {
         case SV_COLORMODE_BAYER_RGB: 
         case SV_COLORMODE_YUV2QT:
           for(i = 0; i < j; i++) {
#ifdef WORDS_BIGENDIAN
            *p1++ = 0x00000000;
#else
            *p1++ = 0x00000000;
#endif
           }
           break;
         case SV_COLORMODE_YUV422:
         case SV_COLORMODE_YUV444A:
           for(i = 0; i < j; i++) {
#ifdef WORDS_BIGENDIAN
            *p1++ = 0x80108010;
#else
            *p1++ = 0x10801080;
#endif
           }
           break;
         case SV_COLORMODE_YUV422_YUYV:
           for(i = 0; i < j; i++) {
#ifdef WORDS_BIGENDIAN
            *p1++ = 0x10801080;
#else
            *p1++ = 0x80108010;
#endif
           }
           break;
         case SV_COLORMODE_YUV422A:
           for(i = 2; i < j; i+=3) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x80101080;
             *p1++ = 0x10108010;
             *p1++ = 0x10801010;
#else
             *p1++ = 0x80101080;
             *p1++ = 0x10801010;
             *p1++ = 0x10108010;
#endif
           }
           break;
         case SV_COLORMODE_YUV444:
           for(i = 2; i < j; i+=3) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x80108080;
             *p1++ = 0x10808010;
             *p1++ = 0x80801080;
#else
             *p1++ = 0x80801080;
             *p1++ = 0x10808010;
             *p1++ = 0x80108080;
#endif
           }
           break;
         case SV_COLORMODE_RGB_BGR:
         case SV_COLORMODE_ABGR:
         case SV_COLORMODE_ARGB:
         case SV_COLORMODE_BGRA:
         case SV_COLORMODE_RGB_RGB:
         case SV_COLORMODE_RGBA:
           for(i = 0; i < j; i++) {
             *p1++ = 0x01010101;
           }
           break;
         case SV_COLORMODE_MONO:
         case SV_COLORMODE_RGBVIDEO:
         case SV_COLORMODE_RGBAVIDEO:
           for(i = 0; i < j; i++) {
             *p1++ = 0x10101010;
           }
           break;
         case SV_COLORMODE_CHROMA:
           for(i = 0; i < j; i++) {
             *p1++ = 0x80808080;
           }
           break;
         default:
           jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
           return SV_ERROR_PROGRAM;
        }
      } else {
        switch (cmode) {
        case SV_COLORMODE_BAYER_RGB: 
		case SV_COLORMODE_YUV2QT:
           for(i = 0; i < j; i++) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x00000000;
#else
             *p1++ = 0x00000000;
#endif
           }
           break;
         case SV_COLORMODE_YUV422:
         case SV_COLORMODE_YUV444A:
           for(i = 0; i < j; i++) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x80001000;
#else
             *p1++ = 0x10008000;
#endif
           }
           break;
         case SV_COLORMODE_YUV422_YUYV:
           for(i = 0; i < j; i++) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x10008000;
#else
             *p1++ = 0x80001000;
#endif
           }
           break;
         case SV_COLORMODE_YUV422A:
           for(i = 5; i < j; i+=6) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x80001000;
             *p1++ = 0x10008000;
             *p1++ = 0x10001000;
             *p1++ = 0x80001000;
             *p1++ = 0x10008000;
             *p1++ = 0x10001000;
#else
             *p1++ = 0x10008000;
             *p1++ = 0x80001000;
             *p1++ = 0x10001000;
             *p1++ = 0x10008000;
             *p1++ = 0x80001000;
             *p1++ = 0x10001000;
#endif
           }
           break;
         case SV_COLORMODE_YUV444:
           for(i = 5; i < j; i+=6) {
#ifdef WORDS_BIGENDIAN
             *p1++ = 0x80001000;
             *p1++ = 0x80008000;
             *p1++ = 0x10008000;
             *p1++ = 0x80001000;
             *p1++ = 0x80008000;
             *p1++ = 0x10008000;
#else
             *p1++ = 0x10008000;
             *p1++ = 0x80008000;
             *p1++ = 0x80001000;
             *p1++ = 0x10008000;
             *p1++ = 0x80008000;
             *p1++ = 0x80001000;
#endif
           }
           break;
         case SV_COLORMODE_ABGR:
         case SV_COLORMODE_ARGB:
         case SV_COLORMODE_RGB_BGR:
         case SV_COLORMODE_BGRA:
         case SV_COLORMODE_RGB_RGB:
         case SV_COLORMODE_RGBA:
           for(i = 0; i < j; i++) {
             *p1++ = 0x00000000;
           }
           break;
         case SV_COLORMODE_RGBVIDEO:
         case SV_COLORMODE_RGBAVIDEO:
           for(i = 0; i < j; i++) {
            *p1++ = 0x10001000;
           }
           break;
         case SV_COLORMODE_MONO:
           for(i = 0; i < j; i++) {
             *p1++ = 0x01000100;
           }
           break;
         case SV_COLORMODE_CHROMA:
           for(i = 0; i < j; i++) {
             *p1++ = 0x80008000;
           }
           break;
         default:
           jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
           return SV_ERROR_PROGRAM;
        }
      }
 
      
      offset = 0;

      /*
       *    Center picture
       */
      if((xsizevideo > ff_host->xsize) || (ff_sv.ysize > ff_host->ysize)) {
        
        /*
         *    offset = pixelsize * (xsize - xsize_pic)/2
         */
        if(xsizevideo > ff_host->xsize) {
          int diff = (xsizevideo - ff_host->xsize)>>1;
          switch (cmode) {
           case SV_COLORMODE_YUV2QT:
           case SV_COLORMODE_YUV422:
           case SV_COLORMODE_YUV422_YUYV:
            offset += 2 * bpc * (diff & ~1);
            break;
           case SV_COLORMODE_YUV422A:
            offset += 3 * bpc * (diff & ~1);
            break;
           case SV_COLORMODE_RGB_BGR:
           case SV_COLORMODE_RGB_RGB:
           case SV_COLORMODE_RGBVIDEO:
           case SV_COLORMODE_YUV444:
            offset += 3 * bpc * diff;
            break;
           case SV_COLORMODE_ABGR:
           case SV_COLORMODE_ARGB:
           case SV_COLORMODE_BGRA:
           case SV_COLORMODE_RGBA:
           case SV_COLORMODE_RGBAVIDEO:
           case SV_COLORMODE_YUV444A:
            diff = 4 * (diff / 4);
            offset += 4 * bpc * diff;
            break;
					 case SV_COLORMODE_BAYER_RGB: 
           case SV_COLORMODE_MONO:
           case SV_COLORMODE_CHROMA:
            offset += bpc * diff;
            break;
           default:
            jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
            return SV_ERROR_PROGRAM;
          }
        }

        /*
         *    offset = linesize * (ysize - ysize_pic)/2
         *
         *    (&~1) -> discard odd offsets
         */
        if(ff_sv.ysize > ff_host->ysize) {
          int diff;
          int sv_ysize   = ff_sv.ysize;
          int host_ysize = ff_host->ysize;

          if (fieldmode==1) {
            sv_ysize   <<= 1;
            host_ysize <<= 1;
          }

          diff = ((sv_ysize - host_ysize)>>1) & ~1;
          switch (cmode) {
           case SV_COLORMODE_YUV2QT:
           case SV_COLORMODE_YUV422:
           case SV_COLORMODE_YUV422_YUYV:
            offset += (2 * bpc * diff * ff_sv.xsize)>>1;
            break;
           case SV_COLORMODE_YUV422A:
            offset += (3 * bpc * diff * ff_sv.xsize)>>1;
            break;
           case SV_COLORMODE_RGB_BGR:
           case SV_COLORMODE_RGB_RGB:
           case SV_COLORMODE_RGBVIDEO:
           case SV_COLORMODE_YUV444:
            offset += (3 * bpc * diff * ff_sv.xsize)>>1;
            break;
           case SV_COLORMODE_ABGR:
           case SV_COLORMODE_ARGB:
           case SV_COLORMODE_BGRA:
           case SV_COLORMODE_RGBA:
           case SV_COLORMODE_RGBAVIDEO:
           case SV_COLORMODE_YUV444A:
            offset +=  (4 * bpc * diff * ff_sv.xsize)>>1;
            break;
	         case SV_COLORMODE_BAYER_RGB: 
           case SV_COLORMODE_MONO:
           case SV_COLORMODE_CHROMA:
            offset += (bpc * diff * ff_sv.xsize)>>1;
            break;
           default:
            jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
            return SV_ERROR_PROGRAM;
          }
        }

        switch (cmode) {
        case SV_COLORMODE_YUV2QT:
        case SV_COLORMODE_YUV422:
        case SV_COLORMODE_YUV422_YUYV:
          offset -= (bpc * (offset & 3));
          break;
        case SV_COLORMODE_YUV422A:
          offset -= (bpc * (offset % 6));
          break;
        }

        if(offset) {
          switch (cmode) {
           case SV_COLORMODE_YUV2QT:
           case SV_COLORMODE_YUV422:
           case SV_COLORMODE_YUV422_YUYV:
            ff_sv.offset[FM_PLANE_YUV_Y0] += offset;
            ff_sv.offset[FM_PLANE_YUV_Y1] += offset;
            ff_sv.offset[FM_PLANE_YUV_U]  += offset;
            ff_sv.offset[FM_PLANE_YUV_V]  += offset;
            break;
           case SV_COLORMODE_YUV422A:
            ff_sv.offset[FM_PLANE_YUV_Y0] += offset;
            ff_sv.offset[FM_PLANE_YUV_Y1] += offset;
            ff_sv.offset[FM_PLANE_YUV_U]  += offset;
            ff_sv.offset[FM_PLANE_YUV_V]  += offset;
            ff_sv.offset_key              += offset;
            break;
           case SV_COLORMODE_RGB_BGR:
           case SV_COLORMODE_RGB_RGB:
           case SV_COLORMODE_RGBVIDEO:
            ff_sv.offset[FM_PLANE_RGB_G] += offset;
            ff_sv.offset[FM_PLANE_RGB_B] += offset;
            ff_sv.offset[FM_PLANE_RGB_R] += offset;
            break;
           case SV_COLORMODE_ABGR:
           case SV_COLORMODE_ARGB:
           case SV_COLORMODE_BGRA:
           case SV_COLORMODE_RGBA:
           case SV_COLORMODE_RGBAVIDEO:
            ff_sv.offset[FM_PLANE_RGB_G] += offset;
            ff_sv.offset[FM_PLANE_RGB_B] += offset;
            ff_sv.offset[FM_PLANE_RGB_R] += offset;
            ff_sv.offset_key             += offset;
            break;
           case SV_COLORMODE_YUV444:
            ff_sv.offset[FM_PLANE_YUV_Y] += offset;
            ff_sv.offset[FM_PLANE_YUV_U] += offset;
            ff_sv.offset[FM_PLANE_YUV_V] += offset;
            break;
           case SV_COLORMODE_YUV444A:
            ff_sv.offset[FM_PLANE_YUV_Y] += offset;
            ff_sv.offset[FM_PLANE_YUV_U] += offset;
            ff_sv.offset[FM_PLANE_YUV_V] += offset;
            ff_sv.offset_key             += offset;
            break;
	         case SV_COLORMODE_BAYER_RGB: 
           case SV_COLORMODE_MONO:
           case SV_COLORMODE_CHROMA:
            ff_sv.offset[FM_PLANE_MONO] += offset;
            break;
           default:
            jpeg_errorprintf(sv, "jpeg_convertimage: Wrong Colormode");
            return SV_ERROR_PROGRAM;
          }
        }
      }
    }
    
    if (ysize == 496) {
      p1 = (uint*) &buffer[0];
      p2 = (uint*) &buffer[bpc * ysize * xsize];
      j = (10 * xsize * bpc) / 4;
      k = 0x80108010;
      for (i=0; i<j; i++) {
        *p1++ = k;
        *p2++ = k;
      }
    }
  } else {
    memcpy(&ff_sv_tmp, &ff_sv, sizeof(ff_sv_tmp));
    ff_sv_tmp.xsize = xsizevideo;
    if(with_key && *with_key) {
      ff_sv_tmp.prefill_flag = 1;
    }
    if(fm_create(ff_host, file, &ff_sv_tmp) != FM_OK) {
      jpeg_errorprintf(sv, "Error creating %s file %s : %s\n", type, file, fm_errorstr(fm_errno)); 
      return SV_ERROR_FILECREATE;
    }
  }
 
  if(bottom2top) {
    int yoffset[4], yoffset_key;

    // remember ypadding
    yoffset[0]  = ff_sv.offset[0]  / (bpc * size_line) * (bpc * size_line);
    yoffset[1]  = ff_sv.offset[1]  / (bpc * size_line) * (bpc * size_line);
    yoffset[2]  = ff_sv.offset[2]  / (bpc * size_line) * (bpc * size_line);
    yoffset[3]  = ff_sv.offset[3]  / (bpc * size_line) * (bpc * size_line);
    yoffset_key = ff_sv.offset_key / (bpc * size_line) * (bpc * size_line);

    // remove positive ypadding
    ff_sv.offset[0]  -= yoffset[0];
    ff_sv.offset[1]  -= yoffset[1];
    ff_sv.offset[2]  -= yoffset[2];
    ff_sv.offset[3]  -= yoffset[3];
    ff_sv.offset_key -= yoffset_key;

    // add offset to last line (bottom2top)
    ff_sv.offset[0]     += bpc * (size_field1 + size_field2 - size_line); 
    ff_sv.offset[1]     += bpc * (size_field1 + size_field2 - size_line); 
    ff_sv.offset[2]     += bpc * (size_field1 + size_field2 - size_line); 
    ff_sv.offset[3]     += bpc * (size_field1 + size_field2 - size_line); 
    ff_sv.offset_key    += bpc * (size_field1 + size_field2 - size_line);

    // add ypadding (in negative direction due to bottom2top)
    ff_sv.offset[0]  -= yoffset[0];
    ff_sv.offset[1]  -= yoffset[1];
    ff_sv.offset[2]  -= yoffset[2];
    ff_sv.offset[3]  -= yoffset[3];
    ff_sv.offset_key -= yoffset_key;

    if(fieldmode == 1) {
      ff_sv.offset_even     = -2 * bpc * size_line;
      ff_sv.offset_odd      = -2 * bpc * size_line;
      ff_sv.offset_even_key = -2 * bpc * size_line;
      ff_sv.offset_odd_key  = -2 * bpc * size_line;
    } else {
      ff_sv.offset_even     = - bpc * (size_field1 +     size_line);
      ff_sv.offset_odd      =   bpc * (size_field1 - 2 * size_line);
      ff_sv.offset_even_key = - bpc * (size_field1 +     size_line);
      ff_sv.offset_odd_key  =   bpc * (size_field1 - 2 * size_line);
    }
  }

  if(loadimage) {
    res = fm_convert(ff_host, &ff_sv);
  } else {
    res = fm_convert(&ff_sv, ff_host);
  }


  if(res != FM_OK) {
    jpeg_errorprintf(sv, "Error converting from %s file %s : %s\n", type, file, fm_errorstr(fm_errno)); 
    return SV_ERROR_FILEREAD;
  }
  

  if(!loadimage) {
    if(fm_write(ff_host, page) != FM_OK) {
      jpeg_errorprintf(sv, "Error writing %s file %s : %s\n", type, file, fm_errorstr(fm_errno));
      return SV_ERROR_FILEWRITE;
    }
  } else {
    if(with_key) {
      *with_key = ((ff_host->mode==FM_MODE_YUV422A)||(ff_host->mode==FM_MODE_YUV444A)||(ff_host->mode==FM_MODE_RGBA))?1:0;
    }
  }

  if(fm_close(ff_host) != FM_OK) {
    jpeg_errorprintf(sv, "Error closing %s file %s : %s\n", type, file, fm_errorstr(fm_errno));
    return SV_ERROR_FILECLOSE;
  }

  fm_fileformat_free(ff_host);

  return SV_OK;
}




int jpeg_filestreamer(sv_handle * sv, int write, char * name, char * buffer, int size, int field1size, int field2size, int xsize, int ysize, int nbits, int nbits10dvs)
{
  char   tmp[64];
  FILE * fp;

  if(write) {
#if defined WIN32 || defined __CYGWIN__
    fp = fopen(name, "wb");
#else
    fp = fopen(name, "w");
#endif

    if(fp == NULL) {
      return SV_ERROR_FILECREATE;
    }

    fprintf(fp, "STREAMER\n");
    fprintf(fp, "size %d\n", size);
    fprintf(fp, "xsize %d\n", xsize);
    fprintf(fp, "ysize %d\n", ysize);
    if(field1size) {
      fprintf(fp, "field1size %d\n", field1size);
    }
    if(field2size) {
      fprintf(fp, "field2size %d\n", field2size);
    }
    fprintf(fp, "nbits %d\n", nbits);
    if(nbits10dvs) {
      fprintf(fp, "nbits10dvs\n");
    }
    fprintf(fp, "\t");

    fwrite(buffer, 1, size, fp);
 
    fclose(fp);
  } else {
#if defined WIN32 || defined __CYGWIN__
    fp = fopen(name, "rb");
#else
    fp = fopen(name, "r");
#endif

    fread(&tmp[0], 8, 1, fp);
    if(strncmp(tmp, "STREAMER", 8) != 0) {
      jpeg_errorprintf(sv, "Error: Streamer mode magic (STREAMER) wrong\n");
      return SV_ERROR_PARAMETER;
    }

    while(!feof(fp) && (fgetc(fp) != '\t'));

    if(!feof(fp)) {
      fread(buffer, 1, size, fp);
    }

    fclose(fp);
  }

  return SV_OK;
}

void jpeg_exchange_channel(ubyte * to_buffer, ubyte * from_buffer, int size, int colormode, int bpc, int channel)
{
  int i;
  int to_start = 0;
  int from_start = 0;
  int length = 0;
  int offset = 0;

  switch(colormode) {
  case SV_COLORMODE_YUV422A:
    offset = 3;
    switch(channel) {
    case 0:  // video to video
      to_start   = 0;
      from_start = 0;
      length     = 2;
      break;
    case 1:  // key to key
      to_start   = 2;
      from_start = 2;
      length     = 1;
      break;
    case 2:  // video to key
      to_start   = 2;
      from_start = 1;
      length     = 1;
      break;
    }
    break;
  case SV_COLORMODE_ABGR:
  case SV_COLORMODE_ARGB:
    offset = 4;
    switch(channel) {
    case 0:  // video to video
      to_start   = 1;
      from_start = 1;
      length     = 3;
      break;
    case 1:  // key to key
      to_start   = 0;
      from_start = 0;
      length     = 1;
      break;
    case 2:  // video to key
      to_start   = 0;
      from_start = 2;
      length     = 1;
      break;
    }
    break;
  case SV_COLORMODE_BGRA:
  case SV_COLORMODE_RGBA:
  case SV_COLORMODE_RGBAVIDEO:
  case SV_COLORMODE_YUV444A:
    offset = 4;
    switch(channel) {
    case 0:  // video to video
      to_start   = 0;
      from_start = 0;
      length = 3;
      break;
    case 1:  // key to key
      to_start   = 3;
      from_start = 3;
      length = 1;
      break;
    case 2:  // video to key
      to_start   = 3;
      from_start = 1;
      length = 1;
      break;
    }
    break;
  }

  to_start   *= bpc;
  from_start *= bpc;
  length     *= bpc;
  offset     *= bpc;

  to_buffer   += to_start;
  from_buffer += from_start;
  i = to_start;

  switch(length) {
  case 1:
    while(i < size) {
      *to_buffer = *from_buffer;
      to_buffer += offset;
      from_buffer += offset;
      i += offset;
    }
    break;
  case 2:
    while(i < size) {
      to_buffer[0] = from_buffer[0];
      to_buffer[1] = from_buffer[1];
      to_buffer += offset;
      from_buffer += offset;
      i += offset;
    }
    break;
  case 3:
    while(i < size) {
      to_buffer[0] = from_buffer[0];
      to_buffer[1] = from_buffer[1];
      to_buffer[2] = from_buffer[2];
      to_buffer += offset;
      from_buffer += offset;
      i += offset;
    }
    break;
  case 4:
    while(i < size) {
      to_buffer[0] = from_buffer[0];
      to_buffer[1] = from_buffer[1];
      to_buffer[2] = from_buffer[2];
      to_buffer[3] = from_buffer[3];
      to_buffer += offset;
      from_buffer += offset;
      i += offset;
    }
    break;
  case 6:
    while(i < size) {
      to_buffer[0] = from_buffer[0];
      to_buffer[1] = from_buffer[1];
      to_buffer[2] = from_buffer[2];
      to_buffer[3] = from_buffer[3];
      to_buffer[4] = from_buffer[4];
      to_buffer[5] = from_buffer[5];
      to_buffer += offset;
      from_buffer += offset;
      i += offset;
    }
    break;
  }
}


void jpeg_host2isp(sv_handle * sv, char * cmd, char * type, char * filename, int page, int start, int nframes, char * channels)
{ 
  int      res      = SV_OK;
  int      size     = 0;
  ubyte   *buffer   = NULL;
  ubyte   *buffer_org = NULL;
  sv_info  info;
  char    *name;
  char    *form;
  int      i,j;
  int      mode;
  int      bpc;
  char *   yuvmode;
  char *   cgrmode;
  int      fieldmode = 0;
  int      compare  = FALSE;
  int      with_key = FALSE;
  
  if (strcmp(cmd, "load/field") == 0) {
    fieldmode = 1;
  } else if (strcmp(cmd, "load/dfield") == 0) {
    fieldmode = 2;
  } else if (strncmp(cmd, "cmp", 3) == 0) {
    compare = TRUE;
  } else if (strcmp(cmd, "load") != 0) {
    jpeg_errorprintf(sv, "Error: Illegal command specifier\n");
    return;
  }

  res = sv_status(sv, &info);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  if (strncmp(type, "raw", 3) == 0) {
    int fd, chan;
    ubyte  buffer[32000];
    int   nsample;
    int   bits;
    int   freq;
    int   big, mode;

    if (fieldmode) {
      jpeg_errorprintf(sv, "Error: Fieldtransfer not allowed for audio data\n");
      return;
    }

    freq = 48000;
    bits = 32;
    big  = 1;

    /* parse parameters: raw,freq,bits,little  */

    if (type[3] == ',') {
      char *tok;
      int val;

      tok = strtok (&type[4],",");
      if (tok != NULL) {
        val = atoi(tok);
        if (val < 1000) {
          jpeg_errorprintf(sv, "Error: frequency must be greater\n");
          return;       
        }
        freq = val;

        sv_option (sv, SV_OPTION_AUDIOFREQ, freq);

        tok = strtok (NULL, ",");
        if (tok != NULL) {
          val = atoi(tok);
          if (val != 8 && val != 16 && val != 32) {
            jpeg_errorprintf(sv, "Error: number of bits must be 8, 16 or 32\n");
            return;       
          }
          bits = val;

          /*
          //sv_option (sv, SV_OPTION_AUDIOBITS, bits);
          // Audoio is always transfered as 32 bit
          */

          tok = strtok (NULL,",");
          if (tok != NULL) {
            if (tok[0] == 'b' || tok[0] == 'B') {
              big = 1;
            } else if (tok[0] == 'l' || tok[0] == 'L') {
              big = 0;
            } else {
              jpeg_errorprintf(sv, "Error: endian wrong, it should be Big or Little endian\n");
              return;       
            }
          }
        }
      }
    }

    switch(bits) {
    case 32:
      mode = big ? SV_DATASIZE_32BIT_BIG : SV_DATASIZE_32BIT_LITTLE;
      break;
    case 16:
      mode = big ? SV_DATASIZE_16BIT_BIG : SV_DATASIZE_16BIT_LITTLE;
      break;
    case 8:
      mode = SV_DATASIZE_8BIT;
      break;
    default:
      mode = SV_DATASIZE_8BIT;
    }

    if (page != 0) {
      jpeg_errorprintf(sv, "Error: Startpage must be zero\n");
      return;
    }

    chan = jpeg_getchannels(sv, channels);
    if (!chan) {
      return;
    }

    /*
    // It is needed to open the file under NT, do not change this
    // to an open function.
    */
    fd = am_direct_open(filename, O_RDONLY | O_BINARY, 0644);    

    if (fd <=0) {
      jpeg_errorprintf(sv, "Error: can't open audio file %s\n", filename);
      return;
    }

    for (i=0; i<nframes; i++) {
      res = sv_query(sv, SV_QUERY_AUDIOSIZE_FROMHOST, start, &nsample);

      if(res != SV_OK) {
        jpeg_errorprint(sv, res);
        am_direct_close(fd);
        return;
      }
 
      res = am_direct_read(fd, (char *)&buffer[0], nsample*bits/8);

      if (res < (nsample * bits/8)) {
        jpeg_errorprintf(sv, "Error: can't read frame %d (end of file ?) res=%d errno=%d\n", i, res, errno);
        am_direct_close(fd);
        return;
      }

      res = sv_host2sv(sv, (char*)buffer, sizeof(buffer), nsample, 0, start++, 1, chan | mode );
 
      if(res != SV_OK) {
        jpeg_errorprint(sv, res);
        am_direct_close(fd);
        return;
      }
 
      jpeg_vgui_feedback(sv, '.');
    }

    am_direct_close(fd);
    return;
  } else if (strncmp(type, "aiff", 4) == 0) {
    int fd, chan;
    ubyte buffer[32000];
    int   nsample;
    int   nchan;
    int   nbit;
    char *type;
    int   nbyte;
    int chunk;
    unsigned char  *p1;
    unsigned char  *p2;

    if (fieldmode) {
      jpeg_errorprintf(sv, "Error: Fieldtransfer not allowed for audio data\n");
      return;
    }

    if (page != 0) {
      jpeg_errorprintf(sv, "Error: Startpage must be zero\n");
      return;
    }

    chan = jpeg_getchannels(sv, channels);
    if (!chan) {
      return;
    }

    
    if ((!am_check(filename, &type, &nsample, &nchan, &nbit)) ||
     (strncmp(type, "aiff", 4) != 0)) {
      jpeg_errorprintf(sv, "Error: can't open audio file %s\n", filename);
      return;
    }

    switch (nbit) {
     case 16:
      nbyte = 2;
      size  = SV_DATASIZE_16BIT_BIG;
      break;
     case 24:
      nbyte = 3;
      size  = SV_DATASIZE_32BIT_BIG;
      break;
     case 32:
      nbyte = 4;
      size  = SV_DATASIZE_32BIT_BIG;
      break;
     default:
      jpeg_errorprintf(sv, "Error: illegal number of bits per sample");
      return;
    }  
   
    fd = am_aiff_open(filename);    

    if (fd <=0) {
      jpeg_errorprintf(sv, "Error: can't open audio file %s\n", filename);
      return;
    }

    for (i=0; i<nframes; i++) {
      
      sv_query(sv, SV_QUERY_AUDIOSIZE_FROMHOST, start, &nsample);

      chunk = (nsample*nbyte*nchan)/2;
      res = am_direct_read(fd, (char*)&buffer[0], chunk);

      if (res < chunk) {
        jpeg_errorprintf(sv, "Error: can't read frame %d (end of file ?)\n", i);
        am_aiff_close(fd);
        return;
      }

      switch(nchan) {
       case 1:
        /* we need two channels of audio, so each sample will be doubled */
        switch (nbyte) {
         case 2:
          p1 = &buffer[chunk-1];
          p2 = &buffer[2*chunk-1];
          for (j=0; j<nsample; j+=2) {
            *p2-- = *p1--;
            *p2-- = *p1;
            *p2-- = p1[1];
            *p2-- = *p1--;
          }
          break;
         case 3:
          p1 = &buffer[chunk-1];
          p2 = &buffer[4*nsample-1];
          for (j=0; j<nsample; j+=2) {
            *p2-- = 0;
            *p2-- = *p1--;
            *p2-- = *p1--;
            *p2-- = *p1;
            *p2-- = 0;
            *p2-- = p1[2];
            *p2-- = p1[1];
            *p2-- = *p1--;
          }
          break;
         case 4:
          p1 = &buffer[chunk-1];
          p2 = &buffer[2*chunk-1];
          for (j=0; j<nsample; j+=2) {
            *p2-- = *p1--;
            *p2-- = *p1--;
            *p2-- = *p1--;
            *p2-- = *p1;
            *p2-- = p1[3];
            *p2-- = p1[2];
            *p2-- = p1[1];
            *p2-- = *p1--;
          }
          break;
        }
        break;
       case 2:
        switch (nbyte) {
         case 2:
         case 4:
          break;
         case 3:
          p1 = &buffer[chunk-1];
          p2 = &buffer[4*nsample-1];
          for (j=0; j<nsample; j++) {
            *p2-- = 0;
            *p2-- = *p1--;
            *p2-- = *p1--;
            *p2-- = *p1--;
          }
          break;
        }
        break;
      }

      res = sv_host2sv(sv, (char*)buffer, sizeof(buffer), nsample, 0, start++, 1, chan | size );
 
      if(res != SV_OK) {
        jpeg_errorprint(sv, res);
        am_aiff_close(fd);
        return;
      }
 
      jpeg_vgui_feedback(sv, '.');
    }

    am_aiff_close(fd);
    return;
  } else if (strncmp(type, "wave", 4) == 0) {
    jpeg_errorprintf(sv, "Error: format not implemented\n");
    return;
  }

  fm_initialize();
  
  size   = (int)strlen(filename)+32;
  form   = (char *) malloc(size);

  if(form == NULL) {
    jpeg_errorprintf(sv, "jpeg_host2isp: Malloc(%d) failed for file name format\n", size);
    return; 
  }
  memset(form,'\0',size);
  name = strchr(filename,'#');
  if( name!=NULL ) {
    if( name!=filename ) {
      strncpy(form,filename,name-filename);
    }
    strcat(form,"%");
    ++name;
    if( '0'<=*name && *name<='9' ) {
      i = *name-'0';
      ++name;
    } else {
      i = 0;
    }
    if( i>0 ) {
      strcat(form,"0");
      form[strlen(form)] = '0'+i;
    }
    strcat(form,"d");
    strcat(form,name);
  } else if( (name=strchr(filename,'%'))!=NULL ) {
    ++name;
    strncpy(form,filename,name-filename);
    if( '1'<=*name && *name<='9' ) {
      strcat(form,"0");
    }
    strcat(form,name);
  } else {
    strcpy(form,filename);
  }
  
  name   = (char *) malloc(size);

  if(name == NULL) {
    jpeg_errorprintf(sv, "jpeg_host2isp: Malloc(%d) failed for file name\n", size);
    free(form);
    return; 
  }

  if(strcmp(type, "streamer") == 0) {
    res = sv_query(sv, SV_QUERY_STREAMERSIZE, 0, &size);
    if(res != SV_OK) {
      jpeg_errorprintf(sv, "jpeg_host2isp: Can not read streamersize\n");
      return;
    }
    if(size == 0) {
      jpeg_errorprintf(sv, "jpeg_host2isp: Not in streamer mode\n");
      return;
    }
    mode = SV_TYPE_STREAMER;
    bpc  = 1;
    fieldmode = -1;
  } else {
    switch(info.colormode) {
    case SV_COLORMODE_YUV2QT:
      mode = SV_TYPE_YUV2QT;
      size = 2;
      break;
    case SV_COLORMODE_YUV422:
      mode = SV_TYPE_YUV422;
      size = 2;
      break;
    case SV_COLORMODE_YUV422_YUYV:
      mode = SV_TYPE_YUV422_YUYV;
      size = 2;
      break;
    case SV_COLORMODE_YUV422A:
      mode = SV_TYPE_YUV422A;
      size = 3;
      break;
    case SV_COLORMODE_RGB_BGR:
      mode = SV_TYPE_RGB_BGR;
      size = 3;
      break;
    case SV_COLORMODE_ABGR:
      mode = SV_TYPE_RGBA_ABGR;
      size = 4;
      break;
    case SV_COLORMODE_ARGB:
      mode = SV_TYPE_RGBA_ARGB;
      size = 4;
      break;
    case SV_COLORMODE_BGRA:
      mode = SV_TYPE_RGBA_BGRA;
      size = 4;
      break;
    case SV_COLORMODE_RGB_RGB:
    case SV_COLORMODE_RGBVIDEO:
      mode = SV_TYPE_RGB_RGB;
      size = 3;
      break;
    case SV_COLORMODE_RGBA:
    case SV_COLORMODE_RGBAVIDEO:
      mode = SV_TYPE_RGBA_RGBA;
      size = 4;
      break;
    case SV_COLORMODE_BAYER_RGB:
      mode = SV_TYPE_BAYER_RGB;
      size = 1;
      break;
    case SV_COLORMODE_MONO:
    case SV_COLORMODE_CHROMA:
      mode = SV_TYPE_MONO;
      size = 1;
      break;
    case SV_COLORMODE_YUV444:
      mode = SV_TYPE_YUV444;
      size = 3;
      break;
    case SV_COLORMODE_YUV444A:
      mode = SV_TYPE_YUV444A;
      size = 4;
      break;
    default:
      jpeg_errorprintf(sv, "jpeg_host2isp: Unknown colormode\n");
      return;
    }
    if(info.nbit > 8) {
      size  *= 2 * info.setup.storagexsize * info.setup.storageysize;
#ifdef WORDS_BIGENDIAN
      mode  |= SV_DATASIZE_16BIT_BIG;
#else
      mode  |= SV_DATASIZE_16BIT_LITTLE;
#endif
      bpc    = 2;
    } else {
      size  *= info.setup.storagexsize * info.setup.storageysize;
      mode  |= SV_DATASIZE_8BIT;
      bpc    = 1;
    }
      
    if(((yuvmode = getenv("SCSIVIDEO_YUV")) != NULL)) {
      if(strcmp(yuvmode, "clip") == 0) {
        mode |= SV_DATA_YUV_CLIP;
      } else if(strcmp(yuvmode, "scale") == 0) {
        mode |= SV_DATA_YUV_SCALE; 
      }
    }
  }

  buffer_org = (ubyte *) malloc(size + 0x1f);
  buffer = (ubyte *)((uintptr)(buffer_org + 0x1f) & ~(uintptr)0x1f);

  if(buffer == NULL) {
    jpeg_errorprintf(sv, "jpeg_host2isp: Malloc(%d) failed for image buffer\n", size);
    free(name);
    free(form);
    return; 
  }

  jpeg_abort = 0;

#if defined(WIN32)
  jpeg_createmapping(getpid());
#else
  signal(SIGHUP,  jpeg_transfer_abort);
  signal(SIGTERM, jpeg_transfer_abort);
#endif

  if(sv->vgui) {
    fprintf(stdout, "%d\n", getpid());
    fflush(stdout);
  }

  /*RJ*011017* handle CGR -> digital video scaling */
  cgrmode = getenv( "SCSIVIDEO_YUVMATRIX" );
  if( cgrmode &&
      strstr( cgrmode, "cgr" ) &&
      strnicmp( type, "dvr", 3 ) &&
      strnicmp( type, "yuv", 3 ) &&
      strnicmp( type, "dvs_yuv", 7 )) { /* CGR but not YUV file */
      switch( info.colormode ) {
      case SV_COLORMODE_YUV444A:
      case SV_COLORMODE_YUV422A:
          switch( mode & SV_TYPE_MASK ) {
          case SV_TYPE_YUV444A:
          case SV_TYPE_YUV422A:
              mode |= SV_DATARANGE_SCALE_KEY;
              break;
          }
          break;
      case SV_COLORMODE_ABGR:
      case SV_COLORMODE_ARGB:
      case SV_COLORMODE_BGRA:
      case SV_COLORMODE_RGB_BGR:
      case SV_COLORMODE_RGB_RGB:
      case SV_COLORMODE_RGBVIDEO:
      case SV_COLORMODE_RGBA:
      case SV_COLORMODE_RGBAVIDEO:
      case SV_COLORMODE_MONO:
          switch( mode & SV_TYPE_MASK ) {
          case SV_TYPE_RGBA_ABGR:
          case SV_TYPE_RGBA_ARGB:
          case SV_TYPE_RGB_BGR:
          case SV_TYPE_RGBA_BGRA:
          case SV_TYPE_RGB_RGB:
          case SV_TYPE_RGBA_RGBA:
          case SV_TYPE_MONO:
          case SV_TYPE_KEY:
              mode |= SV_DATARANGE_SCALE;
          }
      }
  }
  
  for(i = 0; (i < nframes) && (!jpeg_abort); ++i ) {
#if defined(WIN32)
    if(pMvShm && (pMvShm->sigcode != 0)) {
      pMvShm->sigcode = 0;
      jpeg_transfer_abort(999);          /* fake abort signal    */
      break;
    }
#endif

    sprintf(name,form,page+i);
    switch (fieldmode) {
     case -1: /* Streamer mode */
      res = jpeg_filestreamer(sv, FALSE, name, (char*)buffer, size, 0, 0, info.xsize, info.ysize, info.nbit, ((info.config & SV_MODE_NBIT_MASK)==SV_MODE_NBIT_10BDVS));
      break;
     case 0:
      res = jpeg_convertimage(sv, TRUE, name, type, page + i, 
			  info.setup.storagexsize, info.setup.storageysize, buffer, size, bpc,
	    		  info.colormode, fieldmode, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, &with_key);
      break;
     case 1:
     case 2:
      if (fieldmode == 1)
        strcat(name, ".f1");
      else
        strcat(name, ".df1");
      res = jpeg_convertimage(sv, TRUE, name, type, page + i, 
			  info.setup.storagexsize, info.setup.storageysize, buffer, size, bpc,
                          info.colormode, fieldmode, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, NULL);
      name[strlen(name)-1] = '2';
      res = jpeg_convertimage(sv, TRUE, name, type, page + i, 
        info.setup.storagexsize, info.setup.storageysize, (info.config&SV_MODE_STORAGE_FRAME?&buffer[size/info.setup.storageysize]:&buffer[size>>1]), 0, 
	    		  bpc, info.colormode, fieldmode, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, &with_key);
      break;
    }

    if(res != SV_OK) {
      break;
    }

    if (info.colormode == SV_COLORMODE_YUV2QT) {
      // minor workaround to fix zero-point
      int index;
      for (index = 0; index < size; index+=4) {
        buffer[index + 1] = buffer[index + 1] - 128;
        buffer[index + 3] = buffer[index + 3] - 128;
      }
    }

    if(!compare) {
      int rmw = 0;
      int rmw_channel = 0;

      switch(info.colormode) {
      case SV_COLORMODE_YUV422A:
      case SV_COLORMODE_ARGB:
      case SV_COLORMODE_ABGR:
      case SV_COLORMODE_BGRA:
      case SV_COLORMODE_RGBA:
      case SV_COLORMODE_RGBAVIDEO:
      case SV_COLORMODE_YUV444A:
        if((tolower(channels[0]) == 'v') && (channels[1] == 0)) {
          rmw = 1;
          rmw_channel = 0;
        } else if((tolower(channels[0]) == 'k') && (channels[1] == 0)) {
          rmw = 1;
          rmw_channel = with_key?1:2;
        }
        break;
      default:
        break;
      }

      if(rmw) {
        ubyte * rmw_buffer = malloc(size);
        if(!rmw_buffer) {
        } else {
          res = sv_sv2host(sv, (char *) rmw_buffer, size, info.setup.storagexsize, info.setup.storageysize, start+i, 1, mode);
          jpeg_exchange_channel(rmw_buffer, buffer, size, info.colormode, bpc, rmw_channel);
          res = sv_host2sv(sv, (char *) rmw_buffer, size, info.setup.storagexsize, info.setup.storageysize, start+i, 1, mode);
          free(rmw_buffer);
        }
      } else {
        res = sv_host2sv(sv, (char *) buffer, size, info.setup.storagexsize, info.setup.storageysize, start+i, 1, mode);
      }
    } else {
      res = jpeg_compare(sv, cmd, buffer, size, info.xsize, info.setup.storagexsize, info.setup.storageysize, start+i, 1, mode);
      if(res != SV_OK) {
        break;
      }
    }

    jpeg_vgui_feedback(sv, '.');
  }

#if defined(WIN32)
  jpeg_closemapping();
#else
  signal(SIGTERM, SIG_DFL);
  signal(SIGHUP, SIG_DFL);
#endif
  
  free(buffer_org);
  free(name);
  free(form);
  
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  fm_deinitialize();
}




void jpeg_isp2host(sv_handle * sv, char * cmd, char * type, char * filename, int page, int start, int nframes, char * channels)
{
  int      res      = SV_OK;
  int      size     = 0;
  ubyte   *buffer   = NULL;
  ubyte   *buffer_org = NULL;
  sv_info  info;
  char    *name;
  char    *form;
  int      mode;
  int      i;
  int      bpc;
  char *   yuvmode;
  char *   cgrmode;
  int      fieldmode = 0;
  int      field1size = 0;
  int      field2size = 0;
  int      features;
  int      colormode;
  int      with_key_tmp;
  int      with_key = 0;
  int      key_to_video = 0;

  if (strcmp(cmd, "save/field") == 0) {
    fieldmode = 1;
  } else if (strcmp(cmd, "save/dfield") == 0) {
    fieldmode = 2;
  } else if (strcmp(cmd, "save") != 0) {
    jpeg_errorprintf(sv, "Error: Illegal command specifier\n");
    return;
  }

  res = sv_status(sv, &info);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  if (strncmp(type, "raw", 3) == 0) {
    int fd, chan;
    char  buffer[32000];
    int   nsample;
    int   bits;
    int   freq;
    int   big, mode;

    if (fieldmode) {
      jpeg_errorprintf(sv, "Error: Fieldtransfer not allowed for audio data\n");
      return;
    }

    freq = 48000;
    bits = 32;
    big  = 1;

    /* parse parameters: raw,freq,bits,little  */

    if (type[3] == ',') {
      char *tok;
      int val;

      tok = strtok (&type[4],",");
      if (tok != NULL) {
        val = atoi(tok);
        if (val < 1000) {
          jpeg_errorprintf(sv, "Error: frequency must be greater\n");
          return;       
        }
        freq = val;

        sv_option (sv, SV_OPTION_AUDIOFREQ, freq);

        tok = strtok (NULL, ",");
        if (tok != NULL) {
          val = atoi(tok);
          if (val != 8 && val != 16 && val != 32) {
            jpeg_errorprintf(sv, "Error: number of bits must be 8, 16 or 32\n");
            return;       
          }
          bits = val;

          /*
          // sv_option (sv, SV_OPTION_AUDIOBITS, 32);
          // Audio is always transfered as 32 bits
          */

          tok = strtok (NULL,",");
          if (tok != NULL) {
            if (tok[0] == 'b' || tok[0] == 'B') {
              big = 1;
            } else if (tok[0] == 'l' || tok[0] == 'L') {
              big = 0;
            } else {
              jpeg_errorprintf(sv, "Error: endian wrong, it should be Big or Little endian\n");
              return;       
            }
          }
        }
      }
    }

    switch(bits) {
    case 32:
      mode = big ? SV_DATASIZE_32BIT_BIG : SV_DATASIZE_32BIT_LITTLE;
      break;
    case 16:
      mode = big ? SV_DATASIZE_16BIT_BIG : SV_DATASIZE_16BIT_LITTLE;
      break;
    case 8:
      mode = SV_DATASIZE_8BIT;
      break;
    default:
      jpeg_errorprintf(sv, "Error: unknown bitdepth : %d\n", bits);
      return;
    }

    if (page != 0) {
      jpeg_errorprintf(sv, "Error: Startpage must be zero\n");
      return;
    }

    chan = jpeg_getchannels(sv, channels);
    if (!chan) {
      return;
    }

    fd = am_direct_open(filename, O_RDWR | O_CREAT | O_BINARY, 0644);

    if (fd <=0) {
      jpeg_errorprintf(sv, "can't open audio file %s\n", filename);
      return;
    }

    for (i=0; i<nframes; i++) {
    
      sv_query(sv, SV_QUERY_AUDIOSIZE_TOHOST, start, &nsample);

      res = sv_sv2host(sv, &buffer[0], sizeof(buffer),
          nsample, 0, start++, 1, chan | mode );

      if(res != SV_OK) {
        jpeg_errorprint(sv, res);
        am_direct_close(fd);
        return;
      }
 
      res = am_direct_write(fd, buffer, nsample*bits/8);

      if (res < (nsample * bits/8)) {
        jpeg_errorprintf(sv, "can't write frame %d (disk full ?)\n", i);
        am_direct_close(fd);
        return;
      }
 
      jpeg_vgui_feedback(sv, '.');
    }

    am_direct_close(fd);
    return;
  } else if (strncmp(type, "aiff", 4) == 0) {
    int chan, fd;
    char  buffer_org[32000 + 0x1f];
    char *buffer = (char *)((uintptr)(buffer_org + 0x1f) & ~(uintptr)0x1f);
    int   nsample;
    int   total = 0;	

    if (fieldmode) {
      jpeg_errorprintf(sv, "Error: Fieldtransfer not allowed for audio data\n");
      return;
    }

    if (page != 0) {
      jpeg_errorprintf(sv, "Error: Startpage must be zero\n");
      return;
    }

    chan = jpeg_getchannels(sv, channels);
    if (!chan) {
      return;
    }

    fd = am_aiff_create(filename);    

    if (fd <=0) {
      jpeg_errorprintf(sv, "can't open audio file %s\n", filename);
      return;
    }

    for (i=0; i<nframes; i++) {
    
      sv_query(sv, SV_QUERY_AUDIOSIZE_TOHOST, start, &nsample);

      res = sv_sv2host(sv, &buffer[0], sizeof(buffer),
          nsample, 0, start++, 1, chan | SV_DATASIZE_16BIT_BIG );
      total += nsample / 2;

      if(res != SV_OK) {
        jpeg_errorprint(sv, res);
        close(fd);
        return;
      }
 
      res = am_direct_write(fd, buffer, nsample*2);

      if (res < (nsample * 2)) {
        jpeg_errorprintf(sv, "can't write frame %d (disk full ?)\n", i);
        am_aiff_close(fd);
        return;
      }
 
      jpeg_vgui_feedback(sv, '.');
    }

    am_aiff_update(fd, total);
    am_aiff_close(fd);
    return;
  } else if (strncmp(type, "wave", 4) == 0) {
    jpeg_errorprintf(sv, "Error: format not implemented\n");
    return;
  } 


  fm_initialize();
  
  size   = (int)strlen(filename)+32;
  form   = (char *) malloc(size);

  if(form == NULL) {
    jpeg_errorprintf(sv, "jpeg_host2isp: Malloc(%d) failed for file name format\n", size);
    return; 
  }
  memset(form,'\0',size);
  name = strchr(filename,'#');
  if( name!=NULL ) {
    if( name!=filename ) {
      strncpy(form,filename,name-filename);
    }
    strcat(form,"%");
    ++name;
    if( '0'<=*name && *name<='9' ) {
      i = *name-'0';
      ++name;
    } else {
      i = 0;
    }
    if( i>0 ) {
      strcat(form,"0");
      form[strlen(form)] = '0'+i;
    }
    strcat(form,"d");
    strcat(form,name);
  } else if( (name=strchr(filename,'%'))!=NULL ) {
    ++name;
    strncpy(form,filename,name-filename);
    if( '1'<=*name && *name<='9' ) {
      strcat(form,"0");
    }
    strcat(form,name);
  } else {
    strcpy(form,filename);
  }
  
  name   = (char *) malloc(size);

  if(name == NULL) {
    free(form);
    jpeg_errorprintf(sv, "jpeg_host2isp: Malloc(%d) failed for file name\n", size);
    return; 
  }
 

  if(strcmp(type, "streamer") == 0) {
    res = sv_query(sv, SV_QUERY_STREAMERSIZE, 0, &size);
    if(res != SV_OK) {
      jpeg_errorprintf(sv, "jpeg_isp2host: Can not read streamersize\n");
      return;
    }
    res = sv_query(sv, SV_QUERY_STREAMERSIZE, 1, &field1size);
    if(res != SV_OK) {
      jpeg_errorprintf(sv, "jpeg_isp2host: Can not read streamersize\n");
      return;
    }
    res = sv_query(sv, SV_QUERY_STREAMERSIZE, 2, &field2size);
    if(res != SV_OK) {
      jpeg_errorprintf(sv, "jpeg_isp2host: Can not read streamersize\n");
      return;
    }
    if(size == 0) {
      jpeg_errorprintf(sv, "jpeg_isp2host: Not in streamer mode\n");
      return;
    }
    mode = SV_TYPE_STREAMER;
    bpc  = 1;
    fieldmode = -1;
  } else {
    switch(info.colormode) {
    case SV_COLORMODE_YUV2QT:
      mode = SV_TYPE_YUV2QT;
      size = 2;
      break;
    case SV_COLORMODE_YUV422:
      mode = SV_TYPE_YUV422;
      size = 2;
      break;
    case SV_COLORMODE_YUV422_YUYV:
      mode = SV_TYPE_YUV422_YUYV;
      size = 2;
      break;
    case SV_COLORMODE_YUV422A:
      mode = SV_TYPE_YUV422A;
      size = 3;
      break;
    case SV_COLORMODE_RGB_BGR:
      mode = SV_TYPE_RGB_BGR;
      size = 3;
      break;
    case SV_COLORMODE_ABGR:
      mode = SV_TYPE_RGBA_ABGR;
      size = 4;
      break;
    case SV_COLORMODE_ARGB:
      mode = SV_TYPE_RGBA_ARGB;
      size = 4;
      break;
    case SV_COLORMODE_BGRA:
      mode = SV_TYPE_RGBA_BGRA;
      size = 4;
      break;
    case SV_COLORMODE_BAYER_RGB:
      mode = SV_TYPE_BAYER_RGB;
      size = 1;
      break;
    case SV_COLORMODE_RGB_RGB:
    case SV_COLORMODE_RGBVIDEO:
      mode = SV_TYPE_RGB_RGB;
      size = 3;
      break;
    case SV_COLORMODE_RGBA:
    case SV_COLORMODE_RGBAVIDEO:
      mode = SV_TYPE_RGBA_RGBA;
      size = 4;
      break;
    case SV_COLORMODE_MONO:
    case SV_COLORMODE_CHROMA:
      mode = SV_TYPE_MONO;
      size = 1;
      break;
    case SV_COLORMODE_YUV444:
      mode = SV_TYPE_YUV444;
      size = 3;
      break;
    case SV_COLORMODE_YUV444A:
      mode = SV_TYPE_YUV444A;
      size = 4;
      break;
    default:
      jpeg_errorprintf(sv, "jpeg_isp2host: Unknown colormode\n");
      return;
    }

    switch(info.colormode) {
    case SV_COLORMODE_YUV422A:
    case SV_COLORMODE_ARGB:
    case SV_COLORMODE_ABGR:
    case SV_COLORMODE_BGRA:
    case SV_COLORMODE_RGBA:
    case SV_COLORMODE_RGBAVIDEO:
    case SV_COLORMODE_YUV444A:
      if((tolower(channels[0]) == 'v') && (tolower(channels[1]) == 'k')) {
        with_key = 1;
      } else if((tolower(channels[0]) == 'k') && (channels[1] == 0)) {
        key_to_video = 1;
      }
      break;
    default:
      break;
    }
    
    if(info.nbit > 8) {
      size  *= 2 * info.setup.storagexsize * info.setup.storageysize;
#ifdef WORDS_BIGENDIAN
      mode  |= SV_DATASIZE_16BIT_BIG;
#else
      mode  |= SV_DATASIZE_16BIT_LITTLE;
#endif
      bpc    = 2;
    } else {
      size  *= info.setup.storagexsize * info.setup.storageysize;
      mode  |= SV_DATASIZE_8BIT;
      bpc    = 1;
    }

    if(((yuvmode = getenv("SCSIVIDEO_YUV")) != NULL)) {
      if(strcmp(yuvmode, "clip") == 0) {
        mode |= SV_DATA_YUV_CLIP;
      } else if(strcmp(yuvmode, "scale") == 0) {
        mode |= SV_DATA_YUV_SCALE;
      }
    }

  }

  buffer_org = (ubyte *) malloc(size + 0x1f);
  buffer = (ubyte *)((uintptr)(buffer_org + 0x1f) & ~(uintptr)0x1f);

  if(buffer == NULL) {
    jpeg_errorprintf(sv, "jpeg_isp2host: Malloc(%d) failed for image buffer\n", size);
    free(name);
    free(form);
    return; 
  }

  jpeg_abort = 0;

#if defined(WIN32)
  jpeg_createmapping(getpid());
#else
  signal(SIGHUP,  jpeg_transfer_abort);
  signal(SIGTERM, jpeg_transfer_abort);
#endif

  if(sv->vgui) {
    printf("%d\n", getpid());
    fflush(stdout);
  }

  /*RJ*011017* handle digital video -> CGR scaling */
  cgrmode = getenv( "SCSIVIDEO_YUVMATRIX" );
  if( cgrmode &&
      strstr( cgrmode, "cgr" ) &&
      strnicmp( type, "dvr", 3 ) &&
      strnicmp( type, "yuv", 3 ) &&
      strnicmp( type, "dvs_yuv", 7 )) { /* CGR but not YUV file */
      switch( info.colormode ) {
      case SV_COLORMODE_YUV444A:
      case SV_COLORMODE_YUV422A:
          switch( mode & SV_TYPE_MASK ) {
          case SV_TYPE_YUV444A:
          case SV_TYPE_YUV422A:
              mode |= SV_DATARANGE_SCALE_KEY;
              break;
          }
          break;
      case SV_COLORMODE_RGB_BGR:
      case SV_COLORMODE_ABGR:
      case SV_COLORMODE_ARGB:
      case SV_COLORMODE_BGRA:
      case SV_COLORMODE_RGB_RGB:
      case SV_COLORMODE_RGBVIDEO:
      case SV_COLORMODE_RGBA:
      case SV_COLORMODE_RGBAVIDEO:
      case SV_COLORMODE_MONO:
          switch( mode & SV_TYPE_MASK ) {
          case SV_TYPE_RGB_BGR:
          case SV_TYPE_RGBA_ABGR:
          case SV_TYPE_RGBA_ARGB:
          case SV_TYPE_RGBA_BGRA:
          case SV_TYPE_RGB_RGB:
          case SV_TYPE_RGBA_RGBA:
          case SV_TYPE_MONO:
          case SV_TYPE_KEY:
              mode |= SV_DATARANGE_SCALE;
          }
      }
  }
  
  sv_query(sv, SV_QUERY_FEATURE, 0, &features);

  for(i = 0; (i < nframes) && (!jpeg_abort); ++i ) {

#if defined(WIN32)
    if( pMvShm && pMvShm->sigcode!=0 ) {
      pMvShm->sigcode = 0;
      jpeg_transfer_abort(999);          /* fake abort signal    */
      break;
    }
#endif

    sprintf(name,form,page+i);
    res = sv_sv2host(sv, (char *) buffer, size, info.setup.storagexsize, info.setup.storageysize, start+i, 1, mode);

    if(res != SV_OK) {
      break;
    }

    if (info.colormode == SV_COLORMODE_YUV2QT) {
      // minor workaround to fix zero-point
      int index;
      for (index = 0; index < size; index+=4) {
        buffer[index + 1] = buffer[index + 1] + 128;
        buffer[index + 3] = buffer[index + 3] + 128;
      }
    }

    if (key_to_video) {
      int j, offset, key, yuv;
      switch(info.colormode) {
      case SV_COLORMODE_YUV422A:
        offset = 3;
        key    = 2;
        yuv    = 1;
        break;
      case SV_COLORMODE_YUV444A:
        offset = 4;
        key    = 2;
        yuv    = 1;
        break;
      case SV_COLORMODE_ARGB:
        offset = 4;
        key    = 0;
        yuv    = 0;
        break;
      case SV_COLORMODE_ABGR:
        offset = 4;
        key    = 0;
        yuv    = 0;
        break;
      case SV_COLORMODE_BGRA:
        offset = 4;
        key    = 3;
        yuv    = 0;
        break;
      case SV_COLORMODE_RGBA:
      case SV_COLORMODE_RGBAVIDEO:
        offset = 4;
        key    = 3;
        yuv    = 0;
        break;
      default:
        offset = 0;
        key    = 0;
        yuv    = 0;
      }
      if(offset) {
        unsigned char  * p1;
        unsigned char  * p2;
        unsigned short * q1;
        unsigned short * q2;

        if(yuv) {
          switch (bpc) {
          case 1:
            p1 = (unsigned char *) buffer;
            p2 = (unsigned char *) buffer;

            for (j = 0; j < size; j += offset) {
              *p1++ = 0x80;
              *p1++ = p2[j+key];
            }
            break;
          case 2:
            q1 = (unsigned short *) buffer;
            q2 = (unsigned short *) buffer;

            for (j = 0; j < size/2; j += offset) {
              *q1++ = 0x8000;
              *q1++ = q2[j+key];
            }
            break;
          }
          colormode = SV_COLORMODE_YUV422;
        } else {
          switch (bpc) {
          case 1:
            p1 = (unsigned char *) buffer;
            p2 = (unsigned char *) buffer;

            for (j = 0; j < size; j += offset) {
              *p1++ = p2[j+key];
              *p1++ = p2[j+key];
              *p1++ = p2[j+key];
            }
            break;
          case 2:
            q1 = (unsigned short *) buffer;
            q2 = (unsigned short *) buffer;

            for (j = 0; j < size/2; j += offset) {
              *q1++ = q2[j+key];
              *q1++ = q2[j+key];
              *q1++ = q2[j+key];
            }
            break;
          }
          colormode = SV_COLORMODE_RGB;
        }
      }
    } else {
      colormode = info.colormode;
    }

    with_key_tmp = with_key;
    switch (fieldmode) {
     case -1: /* Streamer mode */
      res = jpeg_filestreamer(sv, TRUE, name, (char*)buffer, size, field1size, field2size, info.xsize, info.ysize, info.nbit,((info.config & SV_MODE_NBIT_MASK)==SV_MODE_NBIT_10BDVS));
      break;
     case 0:
      res = jpeg_convertimage(sv, FALSE, name, type, page + i, 
			  info.setup.storagexsize, info.setup.storageysize, buffer, size, bpc,
	    		  info.colormode, fieldmode, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, &with_key_tmp);
      break;
     case 1:
     case 2:
      if (fieldmode == 1)
        strcat(name, ".f1");
      else
        strcat(name, ".df1");
      res = jpeg_convertimage(sv, FALSE, name, type, page + i, 
			  info.setup.storagexsize, info.setup.storageysize, buffer, size, bpc,
	    		  info.colormode, fieldmode, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, &with_key_tmp);
      name[strlen(name)-1] = '2';
      res = jpeg_convertimage(sv, FALSE, name, type, page + i, 
			  info.setup.storagexsize, info.setup.storageysize, (info.config&SV_MODE_STORAGE_FRAME?&buffer[size/info.setup.storageysize]:&buffer[size>>1]), 0, 
	    		  bpc, info.colormode, fieldmode, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, &with_key_tmp);
      break;
    }
#if 0
#define DEBUG_DVSFORMAT_SAVE
#endif
#ifdef DEBUG_DVSFORMAT_SAVE
fprintf(stderr, "jpeg_isp2host-I: \t page = %d\n", page+i);
#endif

    if(res != SV_OK) {
      break;
    }

    jpeg_vgui_feedback(sv, '.');
  }

#if defined(WIN32)
  jpeg_closemapping();
#else
  signal(SIGTERM, SIG_DFL);
  signal(SIGHUP, SIG_DFL);
#endif
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  free(buffer_org);
  free(name);
  free(form);

  fm_deinitialize();
}



