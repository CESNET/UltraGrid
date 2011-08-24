#include "analyzer.h"

#include <math.h>

#ifndef TRUE
# define TRUE 1
#endif

#ifndef FALSE
# define FALSE 0
#endif

int analyzer_histogram(analyzer_options * panalyzer, analyzer_buffer * pdest, analyzer_buffer * psource)
{
  unsigned char * p;
  int color[3][256];
  int x,y,c;
  int maxvalue;
  int pixels;
  int subpixels;
  int value;
  memset(color, 0, sizeof(color));

  p = psource->pbuffer;
  if(panalyzer->lineselect) {
    p += 4 * (panalyzer->lineselect-1) * psource->xsize;
    for(x = 0; x < psource->xsize; x++) {
      color[0][*p++]++;
      color[1][*p++]++;
      color[2][*p++]++;
      p++;
    }
    maxvalue = (int)(256 * sqrt(pdest->xsize));
  } else {
    for(y = 0; y < psource->ysize; y++) {
      for(x = 0; x < psource->xsize; x++) {
        color[0][*p++]++;
        color[1][*p++]++;
        color[2][*p++]++;
        p++;
      }
    }
    maxvalue = (int)(256 * sqrt(pdest->xsize * pdest->ysize));
  }

  if(!panalyzer->transparent) {
    memset(pdest->pbuffer, 0, 4*pdest->xsize * pdest->ysize);
  }

  for(c = 0; c < 3; c++) {
    color[c][0] = 0;
    for(x = 1; x < 256; x++) {
      color[c][x] = (int)(256 * sqrt(color[c][x]));
    }
  }

  p = pdest->pbuffer;
  for(c = 0; c < 3; c++) {
    int xsize   = pdest->xsize - 10;
    int ysize   = pdest->ysize / 3 - 10;
    int xoffset = 5;
    int yoffset = 5 + c * pdest->ysize / 3;
 
    for(x = 0; x < xsize; x++) {
      int xx;
      value = 0;
      for(xx = -2; xx < 3; xx++) {
        int pos   = (256 * (x + xx)) / xsize;
         
        if(pos < 0) {
          pos = 0;
        } else if(pos > 255) {
          pos = 255;
        }
        value += color[c][pos];
      }
      value /= 5;

      pixels    = ysize * value / maxvalue;
      subpixels = ysize * value % maxvalue;

      for(y = 0; y < pixels-1; y++) {
        p[4*(pdest->xsize * (yoffset+y) + xoffset + x) + c] = 255;
      } 
      p[4*(pdest->xsize * (yoffset+y) + xoffset + x) + c] = (unsigned char)(256 * subpixels / maxvalue);
    }
  }

  return TRUE;
}


#define PARADEXSIZE 256
int analyzer_parade(analyzer_options * panalyzer, analyzer_buffer * pdest, analyzer_buffer * psource)
{
  unsigned char * p;
  int color[3][PARADEXSIZE][256];
  int x,y,c;
  int value;
  int xpos,ypos;

  memset(color, 0, sizeof(color));

  p = psource->pbuffer;
  for(y = 0; y < psource->ysize; y++) {
    for(x = 0; x < psource->xsize; x++) {
      xpos = x * PARADEXSIZE / psource->xsize;
      color[0][xpos][*p++]++;
      color[1][xpos][*p++]++;
      color[2][xpos][*p++]++;
      p++;
    }
  }

  if(!panalyzer->transparent) {
    memset(pdest->pbuffer, 0, 4*pdest->xsize * pdest->ysize);
  }

  p = pdest->pbuffer;
  for(c = 0; c < 3; c++) {
    int xsize   = pdest->xsize/3 - 10;
    int ysize   = pdest->ysize - 10;
    int xoffset = 5 + (2-c) * pdest->xsize / 3;
    int yoffset = 5;
 
    for(y = 0; y < ysize; y++) {
      ypos = y * 256 / ysize;
      for(x = 0; x < xsize; x++) {
        xpos  = x * PARADEXSIZE / xsize;
        value = color[c][xpos][ypos] * 512 / xsize;
      
        if(value > 255) {
          value = 255;
        }

        p[4*(pdest->xsize * (yoffset+y) + xoffset + x) + c] = (unsigned char)value;
      }
    }
  }

  return TRUE;
}


int analyzer(analyzer_options * panalyzer, analyzer_buffer * pdest, analyzer_buffer * psource)
{
  switch(panalyzer->operation) {
  case ANALYZER_NOP:
    return TRUE;
  case ANALYZER_HISTOGRAM:
    return analyzer_histogram(panalyzer, pdest, psource);
  case ANALYZER_RGBPARADE:
    return analyzer_parade(panalyzer, pdest, psource);
  }

  return FALSE;
}
