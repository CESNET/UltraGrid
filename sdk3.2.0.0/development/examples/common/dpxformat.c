/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxformat - Routines to read/write dpx files
//
*/

#include "dpxformat.h"
#include "fileops.h"

#include "../../header/dvs_clib.h"

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

static int get32(unsigned char * pbuffer)
{
  int value;

  memcpy((void*)&value, pbuffer, 4);
#if !defined(sgi) && !defined(sun)
  value = (((unsigned int)value >> 24) & 0x000000ff) |
          (((unsigned int)value >>  8) & 0x0000ff00) |
          (((unsigned int)value <<  8) & 0x00ff0000) |
          (((unsigned int)value << 24) & 0xff000000);
#endif

  return value;
}

int dpxformat_getstoragemode(int dpxtype, int nbits)
{
  int storagemode  = SV_MODE_STORAGE_FRAME;
  switch(nbits) {
  case 8:
    storagemode  |= SV_MODE_NBIT_8B;
    break;
  case 12:
    storagemode  |= SV_MODE_NBIT_12BDPX;
    break;
  case 16:
    storagemode  |= SV_MODE_NBIT_16BBE;
    break;
  default:
    storagemode  |= SV_MODE_NBIT_10BDPX;
  }
  switch(dpxtype) {
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
    storagemode  |= SV_MODE_COLOR_RGB_RGB;
  }

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
  set16 (&pheader[ 804], 1);
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


int dpxformat_readheader(void * vpheader, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits)
{
  unsigned char * pheader = vpheader;

  if(get32(&pheader[0]) == 0x53445058) {
    *offset   = get32 (&pheader[4]);
    *xsize    = get32 (&pheader[772]);
    *ysize    = get32 (&pheader[776]);
    *nbits    = pheader[803];
    *dpxtype  = pheader[800];

    switch(pheader[800]) {
    case 50:  // RGB 
    case 101: // YUV422A
    case 102: // YUV444
      switch(pheader[803]) {
      case 8:
        return *offset + 3 * (*xsize) * (*ysize);
      case 10:
        return *offset + 4 * (*xsize) * (*ysize);
      case 12:
        return *offset + 9 * (*xsize) * (*ysize) / 2;
      case 16:
        return *offset + 6 * (*xsize) * (*ysize);
      default:
        printf("ERROR: dpxformat_readheader RGB, nbits == %d\n", pheader[803]);
      }
      break;
    case 51:  // RGBA
    case 52:  // ARGB
    case 103: // YUV444A
      switch(pheader[803]) {
      case 8:
        return *offset +  4 * (*xsize) * (*ysize);
      case 10:
        return *offset + 16 * (*xsize) * (*ysize) / 3;
      case 12:
        return *offset +  6 * (*xsize) * (*ysize);
      case 16:
        return *offset +  8 * (*xsize) * (*ysize);
      default:
        printf("ERROR: dpxformat_readheader RGBA, nbits == %d\n", pheader[803]);
      }
      break;
    case 100: //YUV422
      switch(pheader[803]) {
      case 8:
        return *offset + 2 * (*xsize) * (*ysize);
      case 10:
        return *offset + 8 * (*xsize) * (*ysize) / 3;
      case 12:
        return *offset + 3 * (*xsize) * (*ysize);
      case 16:
        return *offset + 4 * (*xsize) * (*ysize);
      default:
        printf("ERROR: dpxformat_readheader YUV422, nbits == %d\n", pheader[803]);
      }
      break;
    default:
      printf("ERROR: dpxformat_readheader descriptor == %d\n", pheader[800]);
    }
  } else {
    printf("ERROR: dpxformat_readheader magic wrong == %08x\n", get32(&pheader[0]));
  }

  return 0;
}


int dpxformat_framesize(int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding)
{
  int size;
  int offset = 0x2000;
  int padding = 0;
  
  size = offset;

  switch(dpxmode) {
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
  case 51: // RGBA
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

  if(poffset) {
    *poffset = offset;
  }
  if(ppadding) {
    *ppadding = padding;
  }

  return size + padding;
}


int dpxformat_readframe(char * filename, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits)
{
  void * fp;

  fp = file_open(filename, O_RDONLY|O_BINARY, 0666, FALSE);
  if(fp) {
    file_read(fp, buffer, buffersize);
    file_close(fp);
  } else {
    printf("ERROR: Could not open file '%s'\n", filename);
    return 0;
  }

  return dpxformat_readheader(buffer, offset, xsize, ysize, dpxtype, nbits);
}


int dpxformat_writeframe(char * filename, char * buffer, int buffersize, int offset, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode)
{
  void * fp;
  int framesize = dpxformat_fillheader(buffer, offset, xsize, ysize, dpxtype, nbits, padding, timecode);

  if(offset + framesize > buffersize) {
    printf("ERROR: Buffer to small (offset:%08x framesize:%08x buffersize:%08x/%d\n", offset, framesize, buffersize, buffersize);
    return FALSE;
  }

  fp = file_open(filename, O_WRONLY|O_BINARY|O_CREAT, 0666, FALSE);
  if(fp) {
    file_write(fp, buffer, offset + framesize);
    file_close(fp);
  } else {
    printf("ERROR: Could not create file '%s'\n", filename);
    return FALSE;
  }

  return buffersize;
}

