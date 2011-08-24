/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    Implements a BMP file format reader used in some sdk examples.
*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#ifdef WIN32
# pragma warning ( disable : 4244 )
# pragma warning ( disable : 4100 )
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"

#ifndef BI_RGB
#define BI_RGB  0
#endif

#define uint32 unsigned int
#define uint16 unsigned short
#define uint8  unsigned char

typedef struct {
  uint16 pad;		/* needed for padding words to 32 bit addresses */
  uint16 magic;
  uint32 bfsize;
  uint16 reserved1;
  uint16 reserved2;
  uint32 bfoffset;
} bmppreheader;

typedef struct {
  uint32 size;
  uint32 width;
  uint32 height;
  uint16 planes;
  uint16 bitcount;
  uint32 compression;
  uint32 imagesize;
  uint32 xpelspermeter;
  uint32 ypelspermeter;
  uint32 clrused;
  uint32 clrimportant;
} bmpheader;

typedef struct {
  uint32 size;
  uint16 width;
  uint16 height;
  uint16 planes;
  uint16 bitcount;
} bmpheaderold;


static uint32 ltohl (uint32 * value)
{
  uint8 * p   = (uint8 *) value;
  uint32  res = 0;

  res = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);

  return res;
}


static uint16 ltohs (uint16 * value)
{
  uint8 * p   = (uint8 *) value;
  uint16  res = 0;

  res = p[0] | (p[1] << 8);

  return res;
}


int dvslib_readbmpfile(char * filename, sv_storageinfo * pstorage, char * buffer, int buffersize, int * pxsize, int * pysize)
{
  bmppreheader  preheader;
  bmpheader     header;
  FILE *        fp;
  uint32        clut[256];
  uint8 * p;
  int x,y,i;
  size_t tmp;
  
  fp = fopen(filename, "r");
  if(fp == NULL) {
    printf("dvslib_readbmpfile: Could not open file '%s'\n", filename);
    return SV_ERROR_FILEOPEN;
  }
  
  tmp = fread(&preheader.magic, 1, sizeof(preheader) - 2, fp); 
  if(tmp != sizeof(bmppreheader)-2) {
    fclose(fp);
    return SV_ERROR_FILEREAD;
  }
  preheader.magic      = ltohs(&preheader.magic);
  preheader.bfsize     = ltohl(&preheader.bfsize);
  preheader.reserved1  = ltohs(&preheader.reserved1);
  preheader.reserved2  = ltohs(&preheader.reserved2);
  preheader.bfoffset   = ltohl(&preheader.bfoffset);

 
  tmp = fread(&header, 1, sizeof(bmpheader), fp); 
  if(tmp != sizeof(bmpheader)) {
    fclose(fp);
    return SV_ERROR_FILEREAD;
  }

  header.size          = ltohl (&header.size);
  if(header.size >= 40) {
    header.width         = ltohl(&header.width);
    header.height        = ltohl(&header.height);
    header.planes        = ltohs(&header.planes);  
    header.bitcount      = ltohs(&header.bitcount);
    header.compression   = ltohl(&header.compression);
    header.imagesize     = ltohl(&header.imagesize);
    header.xpelspermeter = ltohl(&header.xpelspermeter);
    header.ypelspermeter = ltohl(&header.ypelspermeter);
    header.clrused       = ltohl(&header.clrused);
    header.clrimportant  = ltohl(&header.clrimportant);
  } else if(header.size == 12) {
    bmpheaderold * h     = (bmpheaderold *) &header;
    header.bitcount      = ltohs(&h->bitcount);
    header.planes        = ltohs(&h->planes);  
    header.height        = ltohs(&h->height);
    header.width         = ltohs(&h->width);
    header.compression   = BI_RGB;
    header.imagesize     = header.width * header.bitcount * header.planes * header.height / 8;
    header.xpelspermeter = 0;
    header.ypelspermeter = 0;
    header.clrused       = 0;
    header.clrimportant  = 0;    
  } else {
    fclose(fp);
    return SV_ERROR_FILESEEK;
  }

#ifdef DEBUG_BMP
  printf("BMP header\n");
  printf("size:             %d\n", header.size);
  printf("height:           %d\n", header.height);
  printf("width:            %d\n", header.width);
  printf("planes:           %d\n", header.planes);
  printf("bitcount:         %d\n", header.bitcount);
  printf("compression:      %d\n", header.compression);
  printf("imagesize:        %d\n", header.imagesize);
  printf("xpels:            %d\n", header.xpelspermeter);
  printf("ypels:            %d\n", header.ypelspermeter);
  printf("clrused:          %d\n", header.clrused);
  printf("clrimportant:     %d\n", header.clrimportant);
#endif

  if(header.compression != BI_RGB) {
    return SV_ERROR_FILEFORMAT; 
  }

  if(!((header.bitcount == 24) ||
       (header.bitcount == 8)  ||
       (header.bitcount == 4)  ||
       (header.bitcount == 1))) {
    return SV_ERROR_FILEFORMAT; 
  }
   
  if(pxsize) {
    *pxsize = (header.width + 1) & ~1;
  }
  if(pysize) {
    *pysize = header.height;
  }

  if(!buffer) {
    fclose(fp);
    return SV_OK;
  }
    
  if(header.bitcount <= 8) {
    //int clut_start;
    size_t clut_size;

    if(header.clrused == 0) {
      header.clrused = 1 << header.bitcount;
    }

    //clut_start = sizeof(bmppreheader) - 2 + header.size;
    clut_size  = header.clrused * sizeof(uint32);
    
    tmp = fread(&clut, 1, clut_size, fp); 
    if(tmp != clut_size) {
      fclose(fp);
      return SV_ERROR_FILEREAD;
    } 

    if(header.size == 12) {
      uint8 * ptmp = (uint8 *) &clut[0];
      int  from = (int)clut_size * 3 / 4 - 3 ;
      int  to   = (int)clut_size - 4;
      for(; (to >= 0); from-=3, to-=4) {
        ptmp[to + 3] = 0;
        ptmp[to + 2] = ptmp[from + 2];
        ptmp[to + 1] = ptmp[from + 1];
        ptmp[to    ] = ptmp[from    ];
      }
    } 
  }

  for(y = 0; y < pstorage->storageysize; y++) {
    int r1,g1,b1,r2,g2,b2;
    uint32 temp, bit;
    p = (uint8*)&buffer[pstorage->fieldoffset[0] + (pstorage->storageysize - y - 1) * pstorage->lineoffset[0]];
    for(temp = bit = x = 0; x < pstorage->storagexsize; x+=2) {
      if(x < (int)header.width) {
        if(header.bitcount != 24) {
          switch(header.bitcount) {
          case 1:
            if(bit == 0) {
              temp = fgetc(fp) & 0xff;
            }
            b1 = clut[temp&(1<<bit)?1:0];
            g1 = clut[temp&(1<<bit)?1:0]>>8;
            r1 = clut[temp&(1<<bit)?1:0]>>16;
            bit++;
            b2 = clut[temp&(1<<bit)?1:0];
            g2 = clut[temp&(1<<bit)?1:0]>>8;
            r2 = clut[temp&(1<<bit)?1:0]>>16;
            bit++;
            if(bit >= 8) {
              bit = 0;
            }
            break;
          case 4:
            temp = fgetc(fp) & 0xff;
            b1 = clut[temp>>4];
            g1 = clut[temp>>4]>>8;
            r1 = clut[temp>>4]>>16;
            b2 = clut[temp&15];
            g2 = clut[temp&15]>>8;
            r2 = clut[temp&15]>>16;
	    break;
          default:
            temp = fgetc(fp) & 0xff;
            b1 = clut[temp];
            g1 = clut[temp]>>8;
            r1 = clut[temp]>>16;
            temp = fgetc(fp) & 0xff;
            b2 = clut[temp];
            g2 = clut[temp]>>8;
            r2 = clut[temp]>>16;
          }
        } else {
          b1 = fgetc(fp); g1 = fgetc(fp); r1 = fgetc(fp);
          if(x + 1 < (int)header.width) {
            b2 = fgetc(fp); g2 = fgetc(fp); r2 = fgetc(fp);
          } else {
            b2 = 0; g2 = 0; r2 = 0;
          }
        }
      } else {
        b1 = 0; g1 = 0; r1 = 0;
        b2 = 0; g2 = 0; r2 = 0;
      }

      switch(pstorage->colormode) {
      case SV_COLORMODE_MONO:
        *p++ = 0x10 + ( 67 * r1      + 131 * g1      +  25 * b1) / 256;
        *p++ = 0x10 + ( 67 * r2      + 131 * g2      + 112 * b2) / 256;
        break;
      case SV_COLORMODE_CHROMA:
        *p++ = 0x80 + (112 * (r1+r2) -  93 * (g1+g2) -  18 * (b1+b2)) / 512;
        *p++ = 0x80 + (-38 * (r1+r2) -  73 * (g1+g2) + 131 * (b1+b2)) / 512;
        break;
      case SV_COLORMODE_YUV422:
        /*
        // lum =>  16 + 224 * (0.299 * R + 0.587 * G + 0.114 * B);
	// rmy => 128 + 224 * (0.500 * R - 0.419 * G - 0.081 * B);
	// bmy => 128 + 224 * (-.169 * R - 0.331 * G + 0.500 * B);
        */
        *p++ = 0x80 + (112 * (r1+r2) -  93 * (g1+g2) -  18 * (b1+b2)) / 512;
        *p++ = 0x10 + ( 67 * r1      + 131 * g1      +  25 * b1) / 256;
        *p++ = 0x80 + (-38 * (r1+r2) -  73 * (g1+g2) + 131 * (b1+b2)) / 512;
        *p++ = 0x10 + ( 67 * r2      + 131 * g2      + 112 * b2) / 256;
        break;
      case SV_COLORMODE_YUV2QT:
        *p++ = 0x10 + ( 67 * r1      + 131 * g1      +  25 * b1) / 256;
        *p++ =        (112 * (r1+r2) -  93 * (g1+g2) -  18 * (b1+b2)) / 512;
        *p++ = 0x10 + ( 67 * r2      + 131 * g2      + 112 * b2) / 256;
        *p++ =        (-38 * (r1+r2) -  73 * (g1+g2) + 131 * (b1+b2)) / 512;
       break;
      case SV_COLORMODE_YUV422A:
        *p++ = 0x80 + (112 * (r1+r2) -  93 * (g1+g2) -  18 * (b1+b2)) / 512;
        *p++ = 0x10 + ( 67 * r1      + 131 * g1      +  25 * b1) / 256;
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = 0x80 + (-38 * (r1+r2) -  73 * (g1+g2) + 131 * (b1+b2)) / 512;
        *p++ = 0x10 + ( 67 * r2      + 131 * g2      + 112 * b2) / 256;
        *p++ = (r2 + g2 + b2) / 3;
        break;
      case SV_COLORMODE_RGB_BGR:
        *p++ = b1;
        *p++ = g1;
        *p++ = r1;
        *p++ = b2;
        *p++ = g2;
        *p++ = r2;
        break;
      case SV_COLORMODE_ABGR:
      case SV_COLORMODE_ARGB:
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = r1;
        *p++ = g1;
        *p++ = b1;
        *p++ = (r2 + g2 + b2) / 3;
        *p++ = r2;
        *p++ = g2;
        *p++ = b2;
        break;
      case SV_COLORMODE_BGRA:
        *p++ = b1;
        *p++ = g1;
        *p++ = r1;
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = b2;
        *p++ = g2;
        *p++ = r2;
        *p++ = (r2 + g2 + b2) / 3;
        break;
      case SV_COLORMODE_RGB_RGB:
        *p++ = r1;
        *p++ = g1;
        *p++ = b1;
        *p++ = r2;
        *p++ = g2;
        *p++ = b2;
        break;
      case SV_COLORMODE_RGBA:
        *p++ = r1;
        *p++ = g1;
        *p++ = b1;
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = r2;
        *p++ = g2;
        *p++ = b2;
        *p++ = (r2 + g2 + b2) / 3;
        break;
      case SV_COLORMODE_YUV444:
        *p++ = 0x80;
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = 0x80;
        *p++ = 0x80;
        *p++ = (r2 + g2 + b2) / 3; 
        *p++ = 0x80;
        break;
      case SV_COLORMODE_YUV444A:
        *p++ = 0x80;
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = 0x80;
        *p++ = (r1 + g1 + b1) / 3;
        *p++ = 0x80;
        *p++ = (r2 + g2 + b2) / 3; 
        *p++ = 0x80;
        *p++ = (r2 + g2 + b2) / 3;
        break;
      default:
        return SV_ERROR_PROGRAM;
      }
    }

    i = (((header.width * header.bitcount + 31) & ~31) - header.width * header.bitcount)/8;
    for(; i > 0; i--) {
      fgetc(fp);
    } 
  }

  fclose(fp);

  return SV_OK;
}
