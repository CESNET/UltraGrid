/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv/svram program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/

#include "svprogram.h"



int jpeg_compare(sv_handle * sv, char * cmd, unsigned char * buffer, int size, int xsizevideo, int xsize, int ysize, int frame, int nframes, int mode)
{
  sv_storageinfo storage;
  unsigned char * ddrbuffer;
  unsigned char * ddrbuffer_org;
  int res;
  int error,position,x,y,pixel;
  int bps       = 1;
  int lastpixel = -1;
  int b16bit    = FALSE;
  int b16bitbe  = FALSE;
  int bbinary   = FALSE;
  int bhistogram = FALSE;
  int bdumphost = FALSE;
  int bdumpdevice = FALSE;
  int history[32];
  int mindiff   = 0;
  int tmpa,tmpb;
  int wrong;
  int tmp;
  char * p;

  sv_storage_status(sv, 0, NULL, &storage, sizeof(storage), 0);

  ddrbuffer_org = malloc(size + 0xff);
  ddrbuffer = (unsigned char *)((uintptr)(ddrbuffer_org + 0xff) & ~(uintptr)0xff);

  if(!ddrbuffer) {
    printf("jpeg_compare: malloc() failed\n");
    return SV_ERROR_MALLOC;
  }

  printf("Compare frame %d\n", frame);

  res = sv_sv2host(sv, (char*)ddrbuffer, size, xsize, ysize, frame, nframes, mode);

  switch(mode & SV_TYPE_MASK) {
  case SV_TYPE_MONO:
  case SV_TYPE_CHROMA:
    bps = 1;
    break;
  case SV_TYPE_YUV422:
  case SV_TYPE_YUV2QT:
    bps = 2;
    break;
  case SV_TYPE_RGB_RGB:
  case SV_TYPE_RGB_BGR:
  case SV_TYPE_YUV422A:
  case SV_TYPE_YUV444:
    bps = 3;
    break;
  case SV_TYPE_RGBA_ARGB:
  case SV_TYPE_RGBA_ABGR:
  case SV_TYPE_RGBA_RGBA:
  case SV_TYPE_RGBA_BGRA:
  case SV_TYPE_YUV444A:
    bps = 4;
    break;
  default:
    printf("unknown colormode\n");
    res = SV_ERROR_PROGRAM;
  }

  switch(mode & SV_DATASIZE_MASK) {
  case SV_DATASIZE_8BIT:
    b16bit = FALSE;
    break;
  case SV_DATASIZE_16BIT_BIG:
    bps       *= 2;
    b16bit    = TRUE;
    b16bitbe  = TRUE;
    break;
  case SV_DATASIZE_16BIT_LITTLE:
    bps       *= 2;
    b16bit    = TRUE;
    b16bitbe  = FALSE;
    break;
  default:
    printf("unknown bitdepth\n");
    res = SV_ERROR_PROGRAM;
  }


  
  p = cmd;
  while(p) {
    p = strchr(p, '/');
    if(p) {
      if(!strncmp(p, "/?", 2) || !strncmp(p, "/help", 5)) {
        printf("sv cmp/help\n");
        printf("\tcmp/diff=# Set min value that should cause error\n");
        printf("\tcmp/bin    Do not mask out sync word values\n");
        printf("\tcmp/hist   Make error histogram, does not check errors\n");
        printf("\tcmp/show   Show data to be compared into device\n");
        printf("\tcmp/dump   Dump data from device\n");
        return SV_OK;
      } else if(!strncmp(p, "/bin", 4)) {
        bbinary = TRUE;
      } else if(!strncmp(p, "/diff=", 5)) {
        mindiff = atoi(p + 6);
        if(!b16bit) {
          mindiff <<= 2;
        }
      } else if(!strncmp(p, "/hist", 5)) {
        bhistogram = TRUE;
        memset(&history, 0, sizeof(history));
      } else if(!strncmp(p, "/dump", 5)) {
        bdumpdevice = TRUE;
      } else if(!strncmp(p, "/show", 5)) {
        bdumphost = TRUE;
      } else {
        printf("sv %s: Switch '%s' unknown\n", cmd, p);
        printf("\tadd switch /help to get help\n");
        return SV_ERROR_PARAMETER;
      }
      
      p++;
    }
  }
  
  error = 0;
  for(position = 0; (position < size); ) {
    if(b16bit) {
      if(b16bitbe) {
        tmpa = (ddrbuffer[position] << 2) | (ddrbuffer[position+1] >> 6);
        tmpb = (buffer[position] << 2)    | (buffer[position+1] >> 6);
      } else {
        tmpa = (ddrbuffer[position+1] << 2) | (ddrbuffer[position] >> 6);
        tmpb = (buffer[position+1] << 2)    | (buffer[position] >> 6);
      }
    } else {
      tmpa = ddrbuffer[position] << 2;
      tmpb = buffer[position] << 2;
    }

#if 0
    if(position < 10) {
      printf("%d %03x %03x\n", position, tmpa, tmpb);
    }
#endif

    if(tmpa != tmpb) {
      wrong = TRUE;

      if(xsize != xsizevideo) {
        pixel = position / bps;
        if(pixel % xsize >= xsizevideo) {
          tmpa = tmpb;
        }
      }


      if(wrong && !bbinary) {
        if(b16bit) {
          if(tmpa < 0x004) {
            tmpa = 0x004;
          } else if(tmpa >= 0x3fc) {
            tmpa = 0x3fb;
          }
          if(tmpb < 0x004) {
            tmpb = 0x004;
          } else if(tmpb >= 0x3fc) {
            tmpb = 0x3fb;
          }
        } else {
          if(tmpa < 0x004) {
            tmpa = 0x004;
          } else if(tmpa >= 0x3f8) {
            tmpa = 0x3fb;
          }
          if(tmpb < 0x004) {
            tmpb = 0x004;
          } else if(tmpb >= 0x3f8) {
            tmpb = 0x3fb;
          }
        }

        if(tmpa == tmpb) {
          wrong = FALSE;
        }
      }
        
      if(wrong) {
        if(mindiff) {
          if(tmpa > tmpb) {
            tmp = tmpa - tmpb;
          } else {
            tmp = tmpb - tmpa;
          } 
          if(tmp < mindiff) {
            wrong = FALSE;
          }
        }
      }
    } else {
      wrong = FALSE;
    }
    
    pixel = position / bps;
    x     = pixel % storage.xsize;
    y     = pixel / storage.xsize;
  
    if(bhistogram) {
      if(b16bit) {
        tmp = tmpa - tmpb + 10;
      } else {
        tmp = ((tmpa - tmpb)>>2) + 10;
      }
      if(tmp < 0) {
        history[0]++; 
      } else if(tmp > 20) {
        history[1]++; 
      } else {
        history[tmp+2]++; 
      }
      wrong = FALSE;
    }

    if(bdumphost || bdumpdevice) {
      if(bdumpdevice) {
        if((position >= bdumpdevice) || ((bdumpdevice == 1) && (position == 0))) {
          if(lastpixel != pixel) {
            printf("'\nPixel @ %08x %03dx%-3d %03x/%02x", position, x, y, tmpa, tmpa>>2);  
          } else {
            printf(" %03x/%02x", tmpa, tmpa>>2);  
          }
        }
        lastpixel = pixel;
      } else {
        if((position >= bdumphost) || ((bdumphost == 1) && (position == 0))) {
          if(lastpixel != pixel) {
            printf("\nPixel @ %08x %3dx%-3d %03x/%02x", position, x, y, tmpb, tmpb>>2);  
          } else {
            printf(" %03x/%02x", tmpb, tmpb>>2);  
          }
          lastpixel = pixel;
        }
      }
      wrong = FALSE;
    }

    if(wrong) {
      if(storage.interlace == 2) {
        if(y >= storage.ysize/2) {
          y = (y-storage.ysize/2)<<1;
        } else {
          y = y<<1;
        }
      }

      if(b16bit) {
        tmp = tmpa - tmpb;
      } else {
        tmp = ((tmpa - tmpb)>>2);
      }

      printf("Error at %08x %dx%d %03x/%02x != %03x/%02x %08x!=%08x %d\n", position, x, y, tmpa, tmpa>>2, tmpb, tmpb>>2, *(uint32*)&buffer[position&~3], *(uint32*)&ddrbuffer[position&~3], tmp);

      if(++error > 50) {
        position = size;
      }
    }

    if(b16bit) {
      position += 2;
    } else {
      position ++;
    }
  }  

  free(ddrbuffer_org);

  if(bhistogram) {
    printf("Error Histogram\n");
    printf("<= -10 %d\n", history[0]);
    for(tmp = 0; tmp < 20; tmp++) {
      printf("%3d  %d\n", tmp-10, history[tmp+2]);
    }
    printf(">= +10 %d\n", history[1]);
  } 

  return res;
}

