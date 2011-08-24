/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxformat - Routines to read/write dpx files
//
*/

#include "dpxformat.h"
#include "fileops.h"

#include "../../header/dvs_clib.h"

#if 0
# define VERBOSE
#endif

#ifdef VERBOSE
char * debug_colormode2string(int storagemode)
{
  switch(storagemode & SV_MODE_COLOR_MASK) {
  case SV_MODE_COLOR_YUV422:
    return "YUV422";
  case SV_MODE_COLOR_RGBA:
    return "RGBA";
  case SV_MODE_COLOR_LUMA:
    return "LUMA";
  case SV_MODE_COLOR_CHROMA:
    return "CHROMA";
  case SV_MODE_COLOR_RGB_BGR:
    return "BGR";
  case SV_MODE_COLOR_YUV422A:
    return "YUV422A";
  case SV_MODE_COLOR_YUV444:
    return "YUV444";
  case SV_MODE_COLOR_YUV444_VYU:
    return "YUV444_VYU";
  case SV_MODE_COLOR_YUV444A:
    return "YUV444A";
  case SV_MODE_COLOR_RGB_RGB:
    return "RGB";
  case SV_MODE_COLOR_BAYER_BGGR:
    return "BAYER_BGGR";
  case SV_MODE_COLOR_BAYER_GBRG:
    return "BAYER_GBRG";
  case SV_MODE_COLOR_BAYER_GRBG:
    return "BAYER_GRBG";
  case SV_MODE_COLOR_BAYER_RGGB:
    return "BAYER_RGGB";
  case SV_MODE_COLOR_BGRA:
    return "BGRA";
  case SV_MODE_COLOR_YUV422_YUYV:
    return "YUV422_YUYV";
  case SV_MODE_COLOR_ARGB:
    return "ARGB";
  case SV_MODE_COLOR_ABGR:
    return "ABGR";
  case SV_MODE_COLOR_ALPHA:
    return "ALPHA";
  case SV_MODE_COLOR_ALPHA_422A:
    return "ALPHA_422A";
  case SV_MODE_COLOR_ALPHA_444A:
    return "ALPHA_444A";
  case SV_MODE_COLOR_ALPHA_A444:
    return "ALPHA_A444";
  case SV_MODE_COLOR_XYZ:
    return "XYZ";
  case SV_MODE_COLOR_YCC:
    return "YCC";
  case SV_MODE_COLOR_YCC422:
    return "YCC422";
  }

  return "?";
}

char * debug_bitsmode2string(int storagemode)
{
  switch(storagemode & SV_MODE_NBIT_MASK) {
  case SV_MODE_NBIT_8B:
    return "8B";
  case SV_MODE_NBIT_8BSWAP:
    return "8BSWAP";
  case SV_MODE_NBIT_10BDVS:
    return "10BDVS";
  case SV_MODE_NBIT_10BLABE:
    return "10BLABE";
  case SV_MODE_NBIT_10BLALE:
    return "10BLALE";
  case SV_MODE_NBIT_10BRABE:
    return "10BRABE";
  case SV_MODE_NBIT_10BRALE:
    return "10BRALE";
  case SV_MODE_NBIT_10BLABEV2:
    return "10BLABEV2";
  case SV_MODE_NBIT_10BLALEV2:
    return "10BLALEV2";
  case SV_MODE_NBIT_10BRABEV2:
    return "10BRABEV2";
  case SV_MODE_NBIT_10BRALEV2:
    return "10BRALEV2";
  case SV_MODE_NBIT_12B:
    return "12B";
  case SV_MODE_NBIT_12BDPX:
    return "12BDPX";
  case SV_MODE_NBIT_12BDPXLE:
    return "12BDPXLE";
  case SV_MODE_NBIT_16BBE:
    return "16BBE";
  case SV_MODE_NBIT_16BLE:
    return "16BLE";
  }

  return "?";
}
#endif

#if defined(__sun)
#  define PAGESIZE 0x2000
#elif defined(sgi)
#  define PAGESIZE 0x4000
#elif defined (__alpha__)
#  define PAGESIZE 0x2000
#elif defined(__ia64__)
#  define PAGESIZE 0x4000
#elif defined(__x86_64__)
#  define PAGESIZE 0x1000
#else
#  define PAGESIZE 0x1000
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

static void set16(unsigned char * pbuffer, unsigned short value)
{
#if defined(sgi) || defined(sun)
  memcpy(pbuffer, (void*)&value, 2);
#else
  *((unsigned short *)pbuffer) = ((value >> 8) & 0x00ff) | ((value << 8) & 0xff00);
#endif
}

static void set32(unsigned char * pbuffer, int value)
{
#if defined(sgi) || defined(sun)
  memcpy(pbuffer, (void*)&value, 4);
#else
  *((unsigned int*)pbuffer) = (((unsigned int)value >> 24) & 0x000000ff) |
                              (((unsigned int)value >>  8) & 0x0000ff00) |
                              (((unsigned int)value <<  8) & 0x00ff0000) |
                              (((unsigned int)value << 24) & 0xff000000);
#endif
}

static int get32(unsigned char * pbuffer, int swap)
{
  int value;

  memcpy((void*)&value, pbuffer, 4);

  if(swap) {
    value = (((unsigned int)value >> 24) & 0x000000ff) |
            (((unsigned int)value >>  8) & 0x0000ff00) |
            (((unsigned int)value <<  8) & 0x00ff0000) |
            (((unsigned int)value << 24) & 0xff000000);
  }

  return value;
}

int dpxformat_file_exists(char * filename)
{
  void * fp;

  fp = file_open(filename, O_RDONLY|O_BINARY, 0666, FALSE);
  if(fp) {
    file_close(fp);
  }

  return fp ? TRUE : FALSE;
}

int dpxformat_getstoragemode(int dpxtype, int nbits)
{
  int storagemode  = SV_MODE_STORAGE_FRAME;
  int bigendian    = (dpxtype & 0x100) != 0;
  int dpxv2format  = (dpxtype & 0x200) != 0;
  int bayer        = (dpxtype & 0x400) != 0;

  switch(nbits) {
  case 8:
#if 0
    if(!(dpxv2format && bigendian) {
      storagemode  |= SV_MODE_NBIT_8BSWAP;
    } else {
#endif
      storagemode  |= SV_MODE_NBIT_8B;
#if 0
    }
#endif
    break;
  case 12:
    storagemode  |= SV_MODE_NBIT_12BDPX;
    break;
  case 16:
    storagemode  |= SV_MODE_NBIT_16BBE;
    break;
  default:
#if 0
    if(dpxv2format) {
      if(bigendian) {
        storagemode |= SV_MODE_NBIT_10BLABEV2;
      } else {
        storagemode |= SV_MODE_NBIT_10BLALEV2;
      }
    } else {
#endif
      if(bigendian) {
        storagemode |= SV_MODE_NBIT_10BLABE;
      } else {
        storagemode |= SV_MODE_NBIT_10BLALE;
      }
#if 0
    }
#endif
  }

  switch(dpxtype & 0xff) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 6:
    storagemode  |= SV_MODE_COLOR_LUMA;
    break;
  case 51:
    storagemode  |= SV_MODE_COLOR_RGBA;
    break;
  case 52:
    storagemode  |= SV_MODE_COLOR_ABGR;
    break;
  case 100:
    storagemode  |= SV_MODE_COLOR_YUV422;
    break;
  case 101:
    storagemode  |= SV_MODE_COLOR_YUV422A;
    break;
  case 102:
    storagemode  |= SV_MODE_COLOR_YUV444;
    break;
  case 103:
    storagemode  |= SV_MODE_COLOR_YUV444A;
    break;
  default:
    if(nbits == 10) {
      if(!bigendian || !dpxv2format) {
        storagemode  |= SV_MODE_COLOR_RGB_RGB; 
      } else {
        storagemode  |= SV_MODE_COLOR_RGB_BGR;
      }
    } else {
      if(!dpxv2format) {
        storagemode  |= SV_MODE_COLOR_RGB_RGB; 
      } else {
        storagemode  |= SV_MODE_COLOR_RGB_BGR;
      }
    }
  }
  if(bayer) {
    switch(dpxtype & 0xf000) {
    case 0x1000:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_BAYER_BGGR;
      break;
    case 0x2000:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_BAYER_GBRG;
      break;
    case 0x3000:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_BAYER_GRBG;
      break;
    case 0x4000:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_BAYER_RGGB;
      break;
    }
  }

#ifdef VERBOSE
  printf("dpxtype      %d\n", dpxtype & 0xff);
  printf("nbits        %d\n", nbits);
  printf("bigendian    %d\n", bigendian);
  printf("dpxv2format  %d\n", dpxv2format);
  printf("colormode    %s\n", debug_colormode2string(storagemode));
  printf("nbitsmode    %s\n", debug_bitsmode2string(storagemode));
#endif


  return storagemode;
}


int dpxformat_fillheader(void * vpheader, int offset, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode)
{
  unsigned char * pheader = vpheader;
  int             size;
  int i;

  memset(pheader, 0xff, offset);

  size = dpxformat_framesize(xsize, ysize, dpxtype, nbits, NULL, NULL);
  
  set32 (&pheader[   0], 0x53445058);
  set32 (&pheader[   4], offset);
  memset(&pheader[   8], 0, 8);
  strcpy((char*)&pheader[   8], "V1.0");
  set32 (&pheader[  16], size);
  set32 (&pheader[  24], 768 + 640 + 256 );
  set32 (&pheader[  28], 256 + 128 );
  set32 (&pheader[  32], offset - 2048);
  memset(&pheader[  36], 0, 100);
  memset(&pheader[ 136], 0,  24 );
  memset(&pheader[ 160], 0, 100 );
  sprintf((char*)&pheader[ 160], "DVS SDK dpxformat %d.%d.%d.%d", DVS_VERSION_MAJOR, DVS_VERSION_MINOR, DVS_VERSION_PATCH, DVS_VERSION_FIX);
  memset(&pheader[ 260], 0, 200 );
  memset(&pheader[ 460], 0, 200 );

  set16 (&pheader[ 768], 0);
  set16 (&pheader[ 770], 1);
  set32 (&pheader[ 772], xsize);
  set32 (&pheader[ 776], ysize);
  set32 (&pheader[ 780], 0);
  pheader[800] = dpxtype;
  pheader[801] = 0;
  pheader[802] = 0;
  pheader[803] = nbits;
  set16 (&pheader[ 804], (nbits == 10)?1:0);
  set16 (&pheader[ 806], 0);
  set32 (&pheader[ 808], offset);
  set32 (&pheader[ 816], padding);
  for(i = 0; i < 8; i++ ) {
    memset(&pheader[820 + i * 72], 0, 32);
  }

  memset(&pheader[1432], 0, 100);
  memset(&pheader[1532], 0, 24);
  memset(&pheader[1556], 0, 32);
  sprintf((char*)&pheader[1556], "DVS SDK dpxformat %d.%d.%d.%d", DVS_VERSION_MAJOR, DVS_VERSION_MINOR, DVS_VERSION_PATCH, DVS_VERSION_FIX);
  memset(&pheader[1588], 0, 32);
  
  memset(&pheader[1664], 0, 2+2+2+6+4+32);

  memset(&pheader[1732], 0, 32);
  memset(&pheader[1764], 0, 100);

  set32 (&pheader[1920], timecode);

  return size;
}


int dpxformat_getoffset(void)
{
  return 0x2000;
}


int dpxformat_readheader(void * vpheader, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset)
{
  unsigned char * pheader = vpheader;
  union {
    unsigned char byte;
    unsigned int  word;
  } test;
  int swap = FALSE;
  int linesize;

  test.word = 1;

  if(test.byte == 1) {
    if(get32(&pheader[0], 0) == 0x58504453) {
      swap = TRUE;
    }
  } else {
    if(get32(&pheader[0], 0) == 0x58504453) {
      swap = TRUE;
    }
  }

  if(get32(&pheader[0], swap) == 0x53445058) {
    if(offset) {
      *offset   = get32 (&pheader[4], swap);
    }
    if(xsize) {
      *xsize    = get32 (&pheader[772], swap);
    }
    if(ysize) {
      *ysize    = get32 (&pheader[776], swap);
    }
    if(nbits) {
      *nbits    = pheader[803];
    }
    if(dpxtype) {
      *dpxtype  = pheader[800];
      if(pheader[0] == 0x53) {
        *dpxtype |= 0x100;
      }
      if(pheader[9] == '2') {
        *dpxtype |= 0x200;
      }
      if((pheader[800] == 0) && (get32(&pheader[20], swap) == 1)) {
        *dpxtype |= 0x400;
      }
    }

    linesize = (*xsize);

    switch(pheader[800]) {
    case 0: // Unknown
    case 6: // Luma:
      break;
    case 100: //YUV422
      linesize *= 2;
      break;
    case 50:  // RGB 
    case 101: // YUV422A
    case 102: // YUV444
      linesize *= 3;
      break;
    case 51:  // RGBA
    case 52:  // ARGB
    case 103: // YUV444A
      linesize *= 4;
      break;
    default:
      printf("ERROR: dpxformat_readheader descriptor == %d\n", pheader[800]);
    }

    switch(pheader[803]) {
    case 8:
      linesize = (linesize + 3) & ~3;
      break;
    case 10:
      linesize = 4 * linesize / 3;
      break;
    case 12:
      linesize = 3 * linesize / 2;
      break;
    case 16:
      linesize = 2 * linesize;
      break;
    default:
      printf("ERROR: dpxformat_readheader nbits == %d\n", pheader[803]);
    }

    // Round to 32 bit
    linesize = (linesize + 3) & ~3;

    if(lineoffset) {
      *lineoffset = linesize;
    }

    if(offset) {
      return (*offset) + (*ysize) * linesize;
    } else {
      return (*ysize) * linesize;
    }
  } else {
    printf("ERROR: dpxformat_readheader magic wrong == %08x\n", get32(&pheader[0], swap));
  }

  return 0;
}


int dpxformat_framesize(int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding)
{
  int offset = 0x2000;
  int size   = offset;
  
  size += dpxformat_framesize_noheader(xsize, ysize, dpxmode, nbits, ppadding);

  if(poffset) {
    *poffset = offset;
  }

  return size;
}


int dpxformat_framesize_noheader(int xsize, int ysize, int dpxmode, int nbits, int * ppadding)
{
  int size    = 0;
  int padding = 0;
  
  switch(dpxmode & 0xff) {
  case 6: // Luma
    switch(nbits) {
    case 10:
      size += 4 * xsize * ysize / 3;
      break;
    case 12:
      size += 3 * xsize * ysize / 2;
      break;
    case 16:
      size += 2 * xsize * ysize;
      break;
    default: // 8
      size += xsize * ysize;
    }
    break;
  case 50:  // RGB
  case 101: // YUV422A
  case 102: // YUV444
    switch(nbits) {
    case 10:
      size += 4 * xsize * ysize;
      break;
    case 12:
      size += 9 * xsize * ysize / 2;
      break;
    case 16:
      size += 6 * xsize * ysize;
      break;
    default: // 8
      size += 3 * xsize * ysize;
    }
    break;
  case 51:  // RGBA
  case 103: // YUV444A
    switch(nbits) {
    case 10:
      size += 4 * 4 * xsize * ysize / 3;
      break;
    case 12:
      size += 6 * xsize * ysize;
      break;
    case 16:
      size += 8 * xsize * ysize;
      break;
    default: // 8
      size += 4 * xsize * ysize;
    }
    break;
  case 100: // YUV422
    switch(nbits) {
    case 10:
      size += 8 * xsize * ysize / 3;
      break;
    case 12:
      size += 3 * xsize * ysize;
      break;
    case 16:
      size += 4 * xsize * ysize;
      break;
    default:
      size += 2 * xsize * ysize;
    }
    break;
  }

  if(size & (PAGESIZE-1)) {
    padding = ((size + PAGESIZE - 1) & ~(PAGESIZE-1)) - size;
  }

  if(ppadding) {
    *ppadding = padding;
  }

  return size + padding;
}


int dpxformat_dpxtype(int storagemode, int * pdpxtype, int * pnbits, int * pdittokey)
{
  if(pdittokey) {
    switch(storagemode & SV_MODE_COLOR_MASK) {
    case SV_MODE_COLOR_BAYER_BGGR:
      *pdittokey = 1;
      break;
    case SV_MODE_COLOR_BAYER_GBRG:
      *pdittokey = 2;
      break;
    case SV_MODE_COLOR_BAYER_GRBG:
      *pdittokey = 3;
      break;
    case SV_MODE_COLOR_BAYER_RGGB:
      *pdittokey = 4;
      break;
    default:
      *pdittokey = 0;
    }
  }

  if(pdpxtype) {
    switch(storagemode & SV_MODE_COLOR_MASK) {
    case SV_MODE_COLOR_LUMA:
    case SV_MODE_COLOR_CHROMA:
    case SV_MODE_COLOR_ALPHA:
    case SV_MODE_COLOR_BAYER_GBRG:
    case SV_MODE_COLOR_BAYER_GRBG:
    case SV_MODE_COLOR_BAYER_BGGR:
    case SV_MODE_COLOR_BAYER_RGGB:
      *pdpxtype = 6;
      break;
    case SV_MODE_COLOR_RGB_RGB:
    case SV_MODE_COLOR_RGB_BGR:
    case SV_MODE_COLOR_XYZ:
      *pdpxtype = 50;
      break;
    case SV_MODE_COLOR_RGBA:
    case SV_MODE_COLOR_BGRA:
    case SV_MODE_COLOR_ARGB:
    case SV_MODE_COLOR_ABGR:
      *pdpxtype = 51;
      break;
    case SV_MODE_COLOR_YUV422:
    case SV_MODE_COLOR_YUV422_YUYV:
    case SV_MODE_COLOR_YCC422:
      *pdpxtype = 100;
      break;
    case SV_MODE_COLOR_YUV422A:
      *pdpxtype = 101;
      break;
    case SV_MODE_COLOR_YUV444:
    case SV_MODE_COLOR_YUV444_VYU:
    case SV_MODE_COLOR_YCC:
      *pdpxtype = 102;
      break;
    case SV_MODE_COLOR_YUV444A:
      *pdpxtype = 103;
      break;
    default:
      return FALSE;
    }
  }

  if(pnbits) {
    switch(storagemode & SV_MODE_NBIT_MASK) {
    case SV_MODE_NBIT_8B:
    case SV_MODE_NBIT_8BSWAP:
      *pnbits = 8;
      break;
    case SV_MODE_NBIT_10BDVS:
    case SV_MODE_NBIT_10BLABE:
    case SV_MODE_NBIT_10BLALE:
    case SV_MODE_NBIT_10BRABE:
    case SV_MODE_NBIT_10BRALE:
    case SV_MODE_NBIT_10BLABEV2:
    case SV_MODE_NBIT_10BLALEV2:
    case SV_MODE_NBIT_10BRABEV2:
    case SV_MODE_NBIT_10BRALEV2:
      *pnbits = 10;
      break;
    case SV_MODE_NBIT_12B:
    case SV_MODE_NBIT_12BDPX:
    case SV_MODE_NBIT_12BDPXLE:
      *pnbits = 12;
      break;
    case SV_MODE_NBIT_16BBE:
    case SV_MODE_NBIT_16BLE:
      *pnbits = 16;
      break;
    default:
      return FALSE;
    }
  }

  return TRUE;
}

int dpxformat_readframe(char * filename, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset)
{
  void * fp;
  int    size = 0;

  fp = file_open(filename, O_RDONLY|O_BINARY, 0666, FALSE);
  if(fp) {
    size = file_read(fp, buffer, buffersize);
    file_close(fp);

    if(!size) {
      printf("ERROR: dpxformat_readframe: Read only 0 bytes '%s'\n", filename);
      return 0;
    }
  } else {
    printf("ERROR: dpxformat_readframe: Could not open file '%s' = %d\n", filename, file_errno());
    return 0;
  }

  return dpxformat_readheader(buffer, offset, xsize, ysize, dpxtype, nbits, lineoffset);
}


int dpxformat_readframe_noheader(char * filename, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset)
{
  int headersize = 0x2000;
  char header[0x2000];
  void * fp;
  int size = 0;
  int framesize = 0;
  int offset_local = 0;

  fp = file_open(filename, O_RDONLY|O_BINARY, 0666, FALSE);
  if(fp) {
    size = file_read(fp, header, headersize);
    if(!size) {
      file_close(fp);
      printf("ERROR: dpxformat_readframe_noheader: Read only 0 bytes '%s'\n", filename);
      return 0;
    }

    framesize = dpxformat_readheader(header, &offset_local, xsize, ysize, dpxtype, nbits, lineoffset);
    if(!framesize) {
      file_close(fp);
      printf("ERROR: dpxformat_readframe_noheader: Seek failed '%s'\n", filename);
      return 0;
    }

    if(file_lseek(fp, offset_local, SEEK_SET) < 0) {
      file_close(fp);
      printf("ERROR: dpxformat_readframe_noheader: Seek failed '%s'\n", filename);
      return 0;
    }

    size = file_read(fp, buffer, buffersize);
    file_close(fp);

    if(!size) {
      printf("ERROR: dpxformat_readframe_noheader: Read only 0 bytes '%s'\n", filename);
      return 0;
    }
  } else {
    printf("ERROR: dpxformat_readframe_noheader: Could not open file '%s' = %d\n", filename, file_errno());
    return 0;
  }

  if(offset) {
    *offset = 0;
  }

  return framesize - offset_local;
}


int dpxformat_writeframe(char * filename, char * buffer, int buffersize, int offset, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode)
{
  void * fp;
  int framesize = dpxformat_fillheader(buffer, offset, xsize, ysize, dpxtype, nbits, padding, timecode);

  if(framesize > buffersize) {
    printf("ERROR: dpxformat_writeframe: Buffer too small (offset:%08x framesize:%08x buffersize:%08x/%d)\n", offset, framesize, buffersize, buffersize);
    return FALSE;
  }

  fp = file_open(filename, O_WRONLY|O_BINARY|O_CREAT, 0666, FALSE);
  if(fp) {
    file_write(fp, buffer, framesize);
    file_close(fp);
  } else {
    printf("ERROR: dpxformat_writeframe: Could not create file '%s'\n", filename);
    return FALSE;
  }

  return buffersize;
}


int dpxformat_writeframe_noheader(char * filename, char * buffer, int buffersize, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode)
{
  int headersize = 0x2000;
  char header[0x2000];
  void * fp;
  int framesize = dpxformat_fillheader(header, headersize, xsize, ysize, dpxtype, nbits, padding, timecode) - headersize;

  if(framesize > buffersize) {
    printf("ERROR: dpxformat_writeframe_noheader: Buffer too small (framesize:%08x buffersize:%08x/%d)\n", framesize, buffersize, buffersize);
    return FALSE;
  }

  fp = file_open(filename, O_WRONLY|O_BINARY|O_CREAT, 0666, FALSE);
  if(fp) {
    file_write(fp, header, headersize);
    file_write(fp, buffer, framesize);
    file_close(fp);
  } else {
    printf("ERROR: dpxformat_writeframe_noheader: Could not create file '%s'\n", filename);
    return FALSE;
  }

  return buffersize;
}
