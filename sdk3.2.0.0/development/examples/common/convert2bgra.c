/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    convert2bgra - converts a memory buffer to bgra 8 bit
*/

#include "convert2bgra.h"

#define convert_yuv2rgb(pdst, luma, cb, cr) \
        red = ((luma)&255) + 443 * (((cb)&255) - 128) / 256; \
        grn = ((luma)&255) -  86 * (((cb)&255) - 128) / 256 - 179 * (((cr)&255) - 128) / 256; \
        blu = ((luma)&255) + 351 * (((cr)&255) - 128) / 256; \
        if(red > 255) red = 255; else if(red < 0) red = 0; \
        if(grn > 255) grn = 255; else if(grn < 0) grn = 0; \
        if(blu > 255) blu = 255; else if(blu < 0) blu = 0; \
        pdst[0] = red; \
        pdst[1] = grn; \
        pdst[2] = blu; \
        pdst[3] = 0xff;

/*
//  Converts the buffer to 8 bit RGBA
*/
int convert_tobgra(unsigned char * psource, unsigned char * pdest, int xsize, int ysize, int interlace, int lineoffset, int storagemode, int bottom2top, int bviewalpha, int fields)
{
  unsigned char * psrc;
  unsigned char * pdst;
  int yline,y,i; 
  int red,grn,blu;

  if(bviewalpha) {
    switch(storagemode & SV_MODE_COLOR_MASK) {
    case SV_MODE_COLOR_BGRA:
    case SV_MODE_COLOR_RGBA:
    case SV_MODE_COLOR_YUV444A:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_ALPHA_444A;
      break;
    case SV_MODE_COLOR_ARGB:
    case SV_MODE_COLOR_ABGR:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_ALPHA_A444;
      break;
    case SV_MODE_COLOR_YUV422A:
      storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_ALPHA_422A;
      break;
    }
  }

  switch(storagemode & SV_MODE_COLOR_MASK) {
  case SV_MODE_COLOR_XYZ:
    storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_RGB_RGB;
    break;
  case SV_MODE_COLOR_YCC:
  case SV_MODE_COLOR_YCC422:
    storagemode = (storagemode & ~SV_MODE_COLOR_MASK) | SV_MODE_COLOR_YUV444;
    break;
  }

  for(y = 0; y < ysize; y++) {
    if(bottom2top) {
      pdst = &(((unsigned char *)pdest)[4 * xsize * y]);
    } else {
      pdst = &(((unsigned char *)pdest)[4 * xsize * (ysize - y - 1)]);
    }
    switch(fields) {
    case 1:
      yline = y & ~1;
      break;
    case 2:
      yline = y | 1;
      break;
    default:
      yline = y;
    }
    if(interlace == 2) {
      if(y & 1) {
        psrc  = &psource[(yline / 2) * lineoffset + (lineoffset * ysize / 2)];
      } else{ 
        psrc  = &psource[(yline / 2) * lineoffset];
      }
    } else {
      psrc = &psource[lineoffset * yline];
    }
    switch(storagemode & SV_MODE_NBIT_MASK) {
    case SV_MODE_NBIT_8B:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=1) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[0];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=3) {
          pdst[0] = psrc[2];
          pdst[1] = psrc[1];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=3) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[1];
          pdst[2] = psrc[2];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[1];
          pdst[2] = psrc[2];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[2];
          pdst[1] = psrc[1];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[3];
          pdst[1] = psrc[2];
          pdst[2] = psrc[1];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[1];
          pdst[1] = psrc[2];
          pdst[2] = psrc[3];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=2,psrc+=4) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[2]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[3], psrc[0], psrc[2]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=6) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[3]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[4], psrc[0], psrc[3]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=3) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[2]);
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[2]);
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=6) {
          pdst[0] = psrc[2];
          pdst[1] = psrc[2];
          pdst[2] = psrc[2];
          pdst[3] = 0xff;
          pdst[4] = psrc[5];
          pdst[5] = psrc[5];
          pdst[6] = psrc[5];
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[3];
          pdst[1] = psrc[3];
          pdst[2] = psrc[3];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[0];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_10BDVS:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=4) {
          pdst[ 0] = psrc[0];
          pdst[ 1] = psrc[0];
          pdst[ 2] = psrc[0];
          pdst[ 3] = 0xff;
          pdst[ 4] = psrc[1];
          pdst[ 5] = psrc[1];
          pdst[ 6] = psrc[1];
          pdst[ 7] = 0xff;
          pdst[ 8] = psrc[2];
          pdst[ 9] = psrc[2];
          pdst[10] = psrc[2];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[2];
          pdst[1] = psrc[1];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[1];
          pdst[2] = psrc[2];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[0];
          pdst[1]  = psrc[1];
          pdst[2]  = psrc[2];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[5];
          pdst[5]  = psrc[6];
          pdst[6]  = psrc[8];
          pdst[7]  = 0xff;
          pdst[8]  = psrc[10];
          pdst[9]  = psrc[12];
          pdst[10] = psrc[13];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[2];
          pdst[1]  = psrc[1];
          pdst[2]  = psrc[0];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[8];
          pdst[5]  = psrc[6];
          pdst[6]  = psrc[5];
          pdst[7]  = 0xff;
          pdst[8]  = psrc[13];
          pdst[9]  = psrc[12];
          pdst[10] = psrc[10];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[4];
          pdst[1]  = psrc[2];
          pdst[2]  = psrc[1];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[9];
          pdst[5]  = psrc[8];
          pdst[6]  = psrc[6];
          pdst[7]  = 0xff;
          pdst[8]  = psrc[14];
          pdst[9]  = psrc[13];
          pdst[10] = psrc[12];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[1];
          pdst[1]  = psrc[2];
          pdst[2]  = psrc[4];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[6];
          pdst[5]  = psrc[8];
          pdst[6]  = psrc[9];
          pdst[7]  = 0xff;
          pdst[8]  = psrc[12];
          pdst[9]  = psrc[13];
          pdst[10] = psrc[14];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=6,psrc+=16) {
          convert_yuv2rgb(pdst, psrc[ 1], psrc[ 0], psrc[ 2]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[ 4], psrc[ 0], psrc[ 2]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[ 6], psrc[ 5], psrc[ 8]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[ 9], psrc[ 5], psrc[ 8]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[12], psrc[10], psrc[13]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[14], psrc[10], psrc[13]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=8) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[4]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[5], psrc[0], psrc[4]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,psrc+=4) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[2]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i+=3,psrc+=16) {
          convert_yuv2rgb(pdst, psrc[1], psrc[0], psrc[2]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[6], psrc[5], psrc[8]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[12], psrc[10], psrc[13]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=8) {
          pdst[0] = psrc[2];
          pdst[1] = psrc[2];
          pdst[2] = psrc[2];
          pdst[3] = 0xff;
          pdst[4] = psrc[6];
          pdst[5] = psrc[6];
          pdst[6] = psrc[6];
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[4];
          pdst[1]  = psrc[4];
          pdst[2]  = psrc[4];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[9];
          pdst[5]  = psrc[9];
          pdst[6]  = psrc[9];
          pdst[7]  = 0xff;
          pdst[8]  = psrc[14];
          pdst[9]  = psrc[14];
          pdst[10] = psrc[14];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[0];
          pdst[1]  = psrc[0];
          pdst[2]  = psrc[0];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[5];
          pdst[5]  = psrc[5];
          pdst[6]  = psrc[5];
          pdst[7]  = 0xff;
          pdst[8]  = psrc[10];
          pdst[9]  = psrc[10];
          pdst[10] = psrc[10];
          pdst[11] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_10BLABE:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=4) {
          pdst[ 0] = psrc[0];
          pdst[ 1] = psrc[0];
          pdst[ 2] = psrc[0];
          pdst[ 3] = 0xff;
          pdst[ 4] = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[ 5] = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[ 6] = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[ 7] = 0xff;
          pdst[ 8] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[ 9] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[10] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[1] = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[0];
          pdst[1] = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[2] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[0];
          pdst[1]  = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[2]  = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[5] << 2) | (psrc[6] >> 6);
          pdst[5]  = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[6]  = psrc[8];
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[10] << 4) | (psrc[11] >> 4);
          pdst[9]  = psrc[12];
          pdst[10] = (psrc[13] << 2) | (psrc[14] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[1]  = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[2]  = psrc[0];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[8];
          pdst[5]  = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[6]  = (psrc[5] << 2) | (psrc[6] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[13] << 2) | (psrc[14] >> 6);
          pdst[9]  = psrc[12];
          pdst[10] = (psrc[10] << 4) | (psrc[11] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[4];
          pdst[1]  = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[2]  = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[9] << 2) | (psrc[10] >> 6);
          pdst[5]  = psrc[8];
          pdst[6]  = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[14] << 4) | (psrc[15] >> 4);
          pdst[9]  = (psrc[13] << 2) | (psrc[14] >> 6);
          pdst[10] = psrc[12];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[1] << 2) | (psrc[2] >> 6);
          pdst[1]  = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[2]  = psrc[4];
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[5]  = psrc[8];
          pdst[6]  = (psrc[9] << 2) | (psrc[10] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = psrc[12];
          pdst[9]  = (psrc[13] << 2) | (psrc[14] >> 6);
          pdst[10] = (psrc[14] << 4) | (psrc[15] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=6,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[1] << 2) | (psrc[2] >> 6), psrc[0], (psrc[2] << 4) | (psrc[3] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, psrc[4], psrc[0], (psrc[2] << 4) | (psrc[3] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[6] << 4) | (psrc[7] >> 4), (psrc[5] << 2) | (psrc[6] >> 6), psrc[8]); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[9] << 2) | (psrc[10] >> 6), (psrc[5] << 2) | (psrc[6] >> 6), psrc[8]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[12], (psrc[10] << 4) | (psrc[11] >> 4), (psrc[13] << 2) | (psrc[14] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[14] << 4) | (psrc[15] >> 4), (psrc[10] << 4) | (psrc[11] >> 4), (psrc[13] << 2) | (psrc[14] >> 6)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=8) {
          convert_yuv2rgb(pdst, (psrc[1] << 2) | (psrc[2] >> 6), psrc[0], psrc[4]); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[5] << 2) | (psrc[6] >> 6), psrc[0], psrc[4]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,psrc+=4) {
          convert_yuv2rgb(pdst, (psrc[1] << 2) | (psrc[2] >> 6), psrc[0], (psrc[2] << 4) | (psrc[3] >> 4)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i+=3,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[1] << 2) | (psrc[2] >> 6), psrc[0], (psrc[2] << 4) | (psrc[3] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[6] << 4) | (psrc[7] >> 4), (psrc[5] << 2) | (psrc[6] >> 6), psrc[8]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[12], (psrc[10] << 4) | (psrc[11] >> 4), (psrc[13] << 2) | (psrc[14] >> 6)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=8) {
          pdst[0] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[1] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[2] = (psrc[2] << 4) | (psrc[3] >> 4);
          pdst[3] = 0xff;
          pdst[4] = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[5] = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[6] = (psrc[6] << 4) | (psrc[7] >> 4);
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[4];
          pdst[1]  = psrc[4];
          pdst[2]  = psrc[4];
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[9] << 2) | (psrc[10] >> 6);
          pdst[5]  = (psrc[9] << 2) | (psrc[10] >> 6);
          pdst[6]  = (psrc[9] << 2) | (psrc[10] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[14] << 4) | (psrc[15] >> 4);
          pdst[9]  = (psrc[14] << 4) | (psrc[15] >> 4);
          pdst[10] = (psrc[14] << 4) | (psrc[15] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[0];
          pdst[1]  = psrc[0];
          pdst[2]  = psrc[0];
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[5] << 2) | (psrc[6] >> 6);
          pdst[5]  = (psrc[5] << 2) | (psrc[6] >> 6);
          pdst[6]  = (psrc[5] << 2) | (psrc[6] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[10] << 4) | (psrc[11] >> 4);
          pdst[9]  = (psrc[10] << 4) | (psrc[11] >> 4);
          pdst[10] = (psrc[10] << 4) | (psrc[11] >> 4);
          pdst[11] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_10BLALE:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=4) {
          pdst[ 0] = psrc[3];
          pdst[ 1] = psrc[3];
          pdst[ 2] = psrc[3];
          pdst[ 3] = 0xff;
          pdst[ 4] = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[ 5] = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[ 6] = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[ 7] = 0xff;
          pdst[ 8] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[ 9] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[10] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[1] = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[2] = psrc[3];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = psrc[3];
          pdst[1] = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[2] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[3];
          pdst[1]  = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[2]  = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[6] << 2) | (psrc[5] >> 6);
          pdst[5]  = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[6]  = psrc[11];
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[9] << 4) | (psrc[8] >> 4);
          pdst[9]  = psrc[15];
          pdst[10] = (psrc[14] << 2) | (psrc[13] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[1]  = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[2]  = psrc[3];
          pdst[3]  = 0xff;
          pdst[4]  = psrc[11];
          pdst[5]  = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[6]  = (psrc[6] << 2) | (psrc[5] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[14] << 2) | (psrc[13] >> 6);
          pdst[9]  = psrc[15];
          pdst[10] = (psrc[9] << 4) | (psrc[8] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[7];
          pdst[1]  = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[2]  = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[10] << 2) | (psrc[9] >> 6);
          pdst[5]  = psrc[11];
          pdst[6]  = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[13] << 4) | (psrc[12] >> 4);
          pdst[9]  = (psrc[14] << 2) | (psrc[13] >> 6);
          pdst[10] = psrc[15];
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[2] << 2) | (psrc[1] >> 6);
          pdst[1]  = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[2]  = psrc[7];
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[5]  = psrc[11];
          pdst[6]  = (psrc[10] << 2) | (psrc[9] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = psrc[15];
          pdst[9]  = (psrc[14] << 2) | (psrc[13] >> 6);
          pdst[10] = (psrc[13] << 4) | (psrc[12] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=6,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[2] << 2) | (psrc[1] >> 6), psrc[3], (psrc[1] << 4) | (psrc[0] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, psrc[7], psrc[3], (psrc[1] << 4) | (psrc[0] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[5] << 4) | (psrc[4] >> 4), (psrc[6] << 2) | (psrc[5] >> 6), psrc[11]); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[10] << 2) | (psrc[9] >> 6), (psrc[6] << 2) | (psrc[5] >> 6), psrc[11]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[15], (psrc[9] << 4) | (psrc[8] >> 4), (psrc[14] << 2) | (psrc[13] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[13] << 4) | (psrc[12] >> 4), (psrc[9] << 4) | (psrc[8] >> 4), (psrc[14] << 2) | (psrc[13] >> 6)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=8) {
          convert_yuv2rgb(pdst, (psrc[2] << 2) | (psrc[1] >> 6), psrc[3], psrc[7]); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[6] << 2) | (psrc[5] >> 6), psrc[3], psrc[7]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,psrc+=4) {
          convert_yuv2rgb(pdst, (psrc[2] << 2) | (psrc[1] >> 6), psrc[3], (psrc[1] << 4) | (psrc[0] >> 4)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i+=3,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[2] << 2) | (psrc[1] >> 6), psrc[3], (psrc[1] << 4) | (psrc[0] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[5] << 4) | (psrc[4] >> 4), (psrc[6] << 2) | (psrc[5] >> 6), psrc[11]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[15], (psrc[9] << 4) | (psrc[8] >> 4), (psrc[14] << 2) | (psrc[13] >> 6)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=8) {
          pdst[0] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[1] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[2] = (psrc[1] << 4) | (psrc[0] >> 4);
          pdst[3] = 0xff;
          pdst[4] = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[5] = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[6] = (psrc[5] << 4) | (psrc[4] >> 4);
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[7];
          pdst[1]  = psrc[7];
          pdst[2]  = psrc[7];
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[10] << 2) | (psrc[9] >> 6);
          pdst[5]  = (psrc[10] << 2) | (psrc[9] >> 6);
          pdst[6]  = (psrc[10] << 2) | (psrc[9] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[13] << 4) | (psrc[12] >> 4);
          pdst[9]  = (psrc[13] << 4) | (psrc[12] >> 4);
          pdst[10] = (psrc[13] << 4) | (psrc[12] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = psrc[3];
          pdst[1]  = psrc[3];
          pdst[2]  = psrc[3];
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[6] << 2) | (psrc[5] >> 6);
          pdst[5]  = (psrc[6] << 2) | (psrc[5] >> 6);
          pdst[6]  = (psrc[6] << 2) | (psrc[5] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[9] << 4) | (psrc[8] >> 4);
          pdst[9]  = (psrc[9] << 4) | (psrc[8] >> 4);
          pdst[10] = (psrc[9] << 4) | (psrc[8] >> 4);
          pdst[11] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_10BRABE:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=4) {
          pdst[ 0] = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[ 1] = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[ 2] = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[ 3] = 0xff;
          pdst[ 4] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[ 5] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[ 6] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[ 7] = 0xff;
          pdst[ 8] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[ 9] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[10] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[1] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2] = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[1] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[1]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2]  = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[5] << 4) | (psrc[6] >> 4);
          pdst[5]  = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[6]  = (psrc[10] << 6) | (psrc[11] >> 2);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[8] << 2) | (psrc[9] >> 6);
          pdst[9]  = (psrc[14] << 6) | (psrc[15] >> 2);
          pdst[10] = (psrc[13] << 4) | (psrc[14] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[1]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2]  = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[10] << 6) | (psrc[11] >> 2);
          pdst[5]  = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[6]  = (psrc[5] << 4) | (psrc[6] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[13] << 4) | (psrc[14] >> 4);
          pdst[9]  = (psrc[14] << 6) | (psrc[15] >> 2);
          pdst[10] = (psrc[8] << 2) | (psrc[9] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[6] << 6) | (psrc[7] >> 2);
          pdst[1]  = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[2]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[9] << 4) | (psrc[10] >> 4);
          pdst[5]  = (psrc[10] << 6) | (psrc[11] >> 2);
          pdst[6]  = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[12] << 2) | (psrc[13] >> 6);
          pdst[9]  = (psrc[13] << 4) | (psrc[14] >> 4);
          pdst[10] = (psrc[14] << 6) | (psrc[15] >> 2);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[1]  = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[2]  = (psrc[6] << 6) | (psrc[7] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[5]  = (psrc[10] << 6) | (psrc[11] >> 2);
          pdst[6]  = (psrc[9] << 4) | (psrc[10] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[14] << 6) | (psrc[15] >> 2);
          pdst[9]  = (psrc[13] << 4) | (psrc[14] >> 4);
          pdst[10] = (psrc[12] << 2) | (psrc[13] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=6,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[1] << 4) | (psrc[2] >> 4), (psrc[2] << 6) | (psrc[3] >> 2), (psrc[0] << 2) | (psrc[1] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[6] << 6) | (psrc[7] >> 2), (psrc[2] << 6) | (psrc[3] >> 2), (psrc[0] << 2) | (psrc[1] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[4] << 2) | (psrc[5] >> 6), (psrc[5] << 4) | (psrc[6] >> 4), (psrc[10] << 6) | (psrc[11] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[9] << 4) | (psrc[10] >> 4), (psrc[5] << 4) | (psrc[6] >> 4), (psrc[10] << 6) | (psrc[11] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[14] << 6) | (psrc[15] >> 2), (psrc[8] << 2) | (psrc[9] >> 6), (psrc[13] << 4) | (psrc[14] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[12] << 2) | (psrc[13] >> 6), (psrc[8] << 2) | (psrc[9] >> 6), (psrc[13] << 4) | (psrc[14] >> 4)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=8) {
          convert_yuv2rgb(pdst, (psrc[1] << 4) | (psrc[2] >> 4), (psrc[2] << 6) | (psrc[3] >> 2), (psrc[6] << 6) | (psrc[7] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[5] << 4) | (psrc[6] >> 4), (psrc[2] << 6) | (psrc[3] >> 2), (psrc[6] << 6) | (psrc[7] >> 2)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,psrc+=4) {
          convert_yuv2rgb(pdst, (psrc[1] << 4) | (psrc[2] >> 4), (psrc[2] << 6) | (psrc[3] >> 2), (psrc[0] << 2) | (psrc[1] >> 6)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i+=3,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[1] << 4) | (psrc[2] >> 4), (psrc[2] << 6) | (psrc[3] >> 2), (psrc[0] << 2) | (psrc[1] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[4] << 2) | (psrc[5] >> 6), (psrc[5] << 4) | (psrc[6] >> 4), (psrc[10] << 6) | (psrc[11] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[14] << 6) | (psrc[15] >> 2), (psrc[8] << 2) | (psrc[9] >> 6), (psrc[13] << 4) | (psrc[14] >> 4)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=8) {
          pdst[0] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[1] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[2] = (psrc[0] << 2) | (psrc[1] >> 6);
          pdst[3] = 0xff;
          pdst[4] = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[5] = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[6] = (psrc[4] << 2) | (psrc[5] >> 6);
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[6] << 6) | (psrc[7] >> 2);
          pdst[1]  = (psrc[6] << 6) | (psrc[7] >> 2);
          pdst[2]  = (psrc[6] << 6) | (psrc[7] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[9] << 4) | (psrc[10] >> 4);
          pdst[5]  = (psrc[9] << 4) | (psrc[10] >> 4);
          pdst[6]  = (psrc[9] << 4) | (psrc[10] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[12] << 2) | (psrc[13] >> 6);
          pdst[9]  = (psrc[12] << 2) | (psrc[13] >> 6);
          pdst[10] = (psrc[12] << 2) | (psrc[13] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[1]  = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[2]  = (psrc[2] << 6) | (psrc[3] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[5] << 4) | (psrc[6] >> 4);
          pdst[5]  = (psrc[5] << 4) | (psrc[6] >> 4);
          pdst[6]  = (psrc[5] << 4) | (psrc[6] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[8] << 2) | (psrc[9] >> 6);
          pdst[9]  = (psrc[8] << 2) | (psrc[9] >> 6);
          pdst[10] = (psrc[8] << 2) | (psrc[9] >> 6);
          pdst[11] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_10BRALE:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=4) {
          pdst[ 0] = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[ 1] = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[ 2] = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[ 3] = 0xff;
          pdst[ 4] = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[ 5] = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[ 6] = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[ 7] = 0xff;
          pdst[ 8] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[ 9] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[10] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[1] = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[2] = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=4) {
          pdst[0] = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[1] = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[2] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[1]  = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[2]  = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[6] << 4) | (psrc[5] >> 4);
          pdst[5]  = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[6]  = (psrc[9] << 6) | (psrc[8] >> 2);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[11] << 2) | (psrc[10] >> 6);
          pdst[9]  = (psrc[13] << 6) | (psrc[12] >> 2);
          pdst[10] = (psrc[14] << 4) | (psrc[13] >> 4);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[1]  = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[2]  = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[9] << 6) | (psrc[8] >> 2);
          pdst[5]  = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[6]  = (psrc[6] << 4) | (psrc[5] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[14] << 4) | (psrc[13] >> 4);
          pdst[9]  = (psrc[13] << 6) | (psrc[12] >> 2);
          pdst[10] = (psrc[11] << 2) | (psrc[10] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[5] << 6) | (psrc[4] >> 2);
          pdst[1]  = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[2]  = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[10] << 4) | (psrc[9] >> 4);
          pdst[5]  = (psrc[9] << 6) | (psrc[8] >> 2);
          pdst[6]  = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[15] << 2) | (psrc[14] >> 6);
          pdst[9]  = (psrc[14] << 4) | (psrc[13] >> 4);
          pdst[10] = (psrc[13] << 6) | (psrc[12] >> 2);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[2] << 4) | (psrc[1] >> 4);
          pdst[1]  = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[2]  = (psrc[5] << 6) | (psrc[4] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[5]  = (psrc[9] << 6) | (psrc[8] >> 2);
          pdst[6]  = (psrc[10] << 4) | (psrc[9] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[13] << 6) | (psrc[12] >> 2);
          pdst[9]  = (psrc[14] << 4) | (psrc[13] >> 4);
          pdst[10] = (psrc[15] << 2) | (psrc[14] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=6,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[2] << 4) | (psrc[1] >> 4), (psrc[1] << 6) | (psrc[0] >> 2), (psrc[3] << 2) | (psrc[2] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[5] << 6) | (psrc[4] >> 2), (psrc[1] << 6) | (psrc[0] >> 2), (psrc[3] << 2) | (psrc[2] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[7] << 2) | (psrc[6] >> 6), (psrc[6] << 4) | (psrc[5] >> 4), (psrc[9] << 6) | (psrc[8] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[10] << 4) | (psrc[9] >> 4), (psrc[6] << 4) | (psrc[5] >> 4), (psrc[9] << 6) | (psrc[8] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[13] << 6) | (psrc[12] >> 2), (psrc[11] << 2) | (psrc[10] >> 6), (psrc[14] << 4) | (psrc[13] >> 4)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[15] << 2) | (psrc[14] >> 6), (psrc[11] << 2) | (psrc[10] >> 6), (psrc[14] << 4) | (psrc[13] >> 4)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=8) {
          convert_yuv2rgb(pdst, (psrc[2] << 4) | (psrc[1] >> 4), (psrc[1] << 6) | (psrc[0] >> 2), (psrc[5] << 6) | (psrc[4] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[6] << 4) | (psrc[5] >> 4), (psrc[1] << 6) | (psrc[0] >> 2), (psrc[5] << 6) | (psrc[4] >> 2)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,psrc+=4) {
          convert_yuv2rgb(pdst, (psrc[2] << 4) | (psrc[1] >> 4), (psrc[1] << 6) | (psrc[0] >> 2), (psrc[3] << 2) | (psrc[2] >> 6)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i+=3,psrc+=16) {
          convert_yuv2rgb(pdst, (psrc[2] << 4) | (psrc[1] >> 4), (psrc[1] << 6) | (psrc[0] >> 2), (psrc[3] << 2) | (psrc[2] >> 6)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[7] << 2) | (psrc[6] >> 6), (psrc[6] << 4) | (psrc[5] >> 4), (psrc[9] << 6) | (psrc[8] >> 2)); pdst+=4;
          convert_yuv2rgb(pdst, (psrc[13] << 6) | (psrc[12] >> 2), (psrc[11] << 2) | (psrc[10] >> 6), (psrc[14] << 4) | (psrc[13] >> 4)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=8) {
          pdst[0] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[1] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[2] = (psrc[3] << 2) | (psrc[2] >> 6);
          pdst[3] = 0xff;
          pdst[4] = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[5] = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[6] = (psrc[7] << 2) | (psrc[6] >> 6);
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[5] << 6) | (psrc[4] >> 2);
          pdst[1]  = (psrc[5] << 6) | (psrc[4] >> 2);
          pdst[2]  = (psrc[5] << 6) | (psrc[4] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[10] << 4) | (psrc[9] >> 4);
          pdst[5]  = (psrc[10] << 4) | (psrc[9] >> 4);
          pdst[6]  = (psrc[10] << 4) | (psrc[9] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[15] << 2) | (psrc[14] >> 6);
          pdst[9]  = (psrc[15] << 2) | (psrc[14] >> 6);
          pdst[10] = (psrc[15] << 2) | (psrc[14] >> 6);
          pdst[11] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i+=3,pdst+=12,psrc+=16) {
          pdst[0]  = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[1]  = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[2]  = (psrc[1] << 6) | (psrc[0] >> 2);
          pdst[3]  = 0xff;
          pdst[4]  = (psrc[6] << 4) | (psrc[5] >> 4);
          pdst[5]  = (psrc[6] << 4) | (psrc[5] >> 4);
          pdst[6]  = (psrc[6] << 4) | (psrc[5] >> 4);
          pdst[7]  = 0xff;
          pdst[8]  = (psrc[11] << 2) | (psrc[10] >> 6);
          pdst[9]  = (psrc[11] << 2) | (psrc[10] >> 6);
          pdst[10] = (psrc[11] << 2) | (psrc[10] >> 6);
          pdst[11] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_12B:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=3) {
          pdst[0]  = psrc[0];
          pdst[1]  = psrc[0];
          pdst[2]  = psrc[0];
          pdst[3]  = 0xff;
          pdst[0]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[1]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2]  = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[3]  = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=2,psrc+=6) {
          convert_yuv2rgb(pdst, ((psrc[1]&0x0f) << 4) | ((psrc[2]&0xf0) >> 4), (psrc[ 0]&0xff), (psrc[3]&0xff)); pdst+=4;
          convert_yuv2rgb(pdst, ((psrc[4]&0x0f) << 4) | ((psrc[5]&0xf0) >> 4), (psrc[ 0]&0xff), (psrc[3]&0xff)); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=9) {
          pdst[0] = psrc[3];
          pdst[1] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
          pdst[4] = (psrc[7] << 4) | (psrc[8] >> 4);
          pdst[5] = psrc[6];
          pdst[6] = (psrc[4] << 4) | (psrc[5] >> 4);
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=9) {
          pdst[0] = psrc[0];
          pdst[1] = (psrc[1] << 4) | (psrc[2] >> 4);
          pdst[2] = psrc[3];
          pdst[3] = 0xff;
          pdst[4] = (psrc[4] << 4) | (psrc[5] >> 4);
          pdst[5] = psrc[6];
          pdst[6] = (psrc[7] << 4) | (psrc[8] >> 4);
          pdst[7] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    case SV_MODE_NBIT_16BLE:
      psrc++;
      /* FALLTHROUGH */
    case SV_MODE_NBIT_16BBE:
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
      case SV_MODE_COLOR_ALPHA:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=2) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[0];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=6) {
          pdst[0] = psrc[4];
          pdst[1] = psrc[2];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=6) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[2];
          pdst[2] = psrc[4];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_BGRA:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[2];
          pdst[2] = psrc[4];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_RGBA:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          pdst[0] = psrc[4];
          pdst[1] = psrc[2];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ARGB:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          pdst[0] = psrc[6];
          pdst[1] = psrc[4];
          pdst[2] = psrc[2];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ABGR:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          pdst[0] = psrc[2];
          pdst[1] = psrc[4];
          pdst[2] = psrc[6];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_YUV422:
        for(i = 0; i < xsize; i+=2,psrc+=8) {
          convert_yuv2rgb(pdst, psrc[2], psrc[0], psrc[4]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[6], psrc[0], psrc[4]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(i = 0; i < xsize; i+=2,psrc+=12) {
          convert_yuv2rgb(pdst, psrc[2], psrc[0], psrc[6]); pdst+=4;
          convert_yuv2rgb(pdst, psrc[8], psrc[0], psrc[6]); pdst+=4;
        }
        break;
      case SV_MODE_COLOR_YUV444:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=6) {
          convert_yuv2rgb(pdst, psrc[2], psrc[0], psrc[4]);
        }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          convert_yuv2rgb(pdst, psrc[2], psrc[0], psrc[4]);
        }
        break;
      case SV_MODE_COLOR_ALPHA_422A:
        for(i = 0; i < xsize; i+=2,pdst+=8,psrc+=12) {
          pdst[0] = psrc[4];
          pdst[1] = psrc[4];
          pdst[2] = psrc[4];
          pdst[3] = 0xff;
          pdst[4] = psrc[10];
          pdst[5] = psrc[10];
          pdst[6] = psrc[10];
          pdst[7] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_444A:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          pdst[0] = psrc[6];
          pdst[1] = psrc[6];
          pdst[2] = psrc[6];
          pdst[3] = 0xff;
        }
        break;
      case SV_MODE_COLOR_ALPHA_A444:
        for(i = 0; i < xsize; i++,pdst+=4,psrc+=8) {
          pdst[0] = psrc[0];
          pdst[1] = psrc[0];
          pdst[2] = psrc[0];
          pdst[3] = 0xff;
        }
        break;
      default:
        return FALSE;
      }
      break;
    default:
      return FALSE;
    }
  }

  return TRUE;
}
