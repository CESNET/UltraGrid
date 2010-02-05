#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"

int generate_image(char * buffer, sv_storageinfo * info)
{
  static int luma = 16;
  static int offs = 1;
  int x, y;
  int pos;
  unsigned int y0, y1, cr, cb, a0, a1;
  int res = SV_OK;
  int border = (info->storagexsize * luma) / 255;

  if (info->nbittype == SV_NBITTYPE_8B && info->colormode == SV_COLORMODE_YUV422A) {
    for(y = 0, pos = 0; y < info->storageysize; y++) {
      // fade video A -> B vertically  depending on wipe direction
      if (offs > 0) {
        a0 = a1 = (255 * y) / info->storageysize;
      } else {
        a0 = a1 = (255 * (info->storageysize - y)) / info->storageysize;
      }
      cb = luma;
      cr = (255 - luma);

      for(x = 0; x < info->storagexsize; x+=2, pos+=6) {
        y0 = y1 = (x < border) ? luma : 255 - luma;

        buffer[pos + 0] = cb; // cb0
        buffer[pos + 1] = y0; // y0
        buffer[pos + 2] = a0; // alpha0
        buffer[pos + 3] = cr; // cr0
        buffer[pos + 4] = y1; // y1
        buffer[pos + 5] = a1; // alpha1
      }
    }
  }
  else if(info->nbittype == SV_NBITTYPE_10BLABE  && info->colormode == SV_COLORMODE_YUV422A) {
    for(y = 0, pos = 0; y < info->storageysize; y++) {
      // fade video A -> B vertically depending on wipe direction
      if (offs > 0) {
        a0 = a1 = (1023 * y) / info->storageysize;
      } else {
        a0 = a1 = (1023 * (info->storageysize - y)) / info->storageysize;
      }
      cb = luma * 4;
      cr = (255 - luma) * 4;

      for(x = 0; x < info->storagexsize; x+=2, pos+=8) {
        y0 = y1 = ((x < border) ? luma : 255 - luma) * 4;

        buffer[pos + 0] = (cb >> 2);
        buffer[pos + 1] = (cb << 8) | (y0 >> 4);
        buffer[pos + 2] = (y0 << 4) | (a0 >> 6);
        buffer[pos + 3] = (a0 << 2);
        buffer[pos + 4] = (cr >> 2);
        buffer[pos + 5] = (cr << 8) | (y1 >> 4);
        buffer[pos + 6] = (y1 << 4) | (a1 >> 6);
        buffer[pos + 7] = (a1 << 2);
      }
    }
  } else {
    if (info->colormode != SV_COLORMODE_YUV422A) {
      res = SV_ERROR_WRONG_COLORMODE;
    } else {
      res = SV_ERROR_WRONG_BITDEPTH;
    }
  }

  luma += offs;
  if (luma >= 235) {
   offs = -1;
  }
  else if (luma <= 16) {
   offs = 1;
  }

  return res;
}
