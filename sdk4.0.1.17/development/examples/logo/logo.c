/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
*/


#ifdef WIN32
#  include <windows.h>
#  include <windowsx.h>
#  include "resource.h"
#endif

#ifndef WIN32
#  include <stdio.h>
#  include <stdarg.h>
#  include <string.h>
#  include <stdlib.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/stat.h>
#  include <sys/time.h>
#  include <unistd.h>
#  ifdef sgi
#    include <sys/syssgi.h>
#  endif
#  ifdef linux
#    include <string.h>
#  endif

#  define DWORD       uint32
#  define wvsprintf   vsprintf
#  ifndef __cdecl
#    define __cdecl
#  endif
#  define IDABORT     'a'
#  define IDRETRY     'r'
#  define IDIGNORE    'i'
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"


#define LOGO_OPERATION_VLINE            0
#define LOGO_OPERATION_HLINE            1  
#define LOGO_OPERATION_LOGO             2
#define LOGO_OPERATION_LOGOSMALL        3
#define LOGO_OPERATION_LOGO_MOVING      4
#define LOGO_OPERATION_LOGOSMALL_MOVING 5

#define LOGO_XSIZE                      120
#define LOGO_YSIZE                      68


typedef struct {
#ifdef WIN32
  HINSTANCE hInstance;
  HWND	    hWnd;
  HANDLE    hThreadPaint;
#endif

  sv_storageinfo storage;

  int	xsize;
  int	ysize;
  int   fieldcount;
  int   interlace;
  int   fps;
  int   colormode;
  int   bps_nom;
  int   bps_denom;

  int   running;
  struct {
    int   delay;
    int   delay_clocked;
    int   startdelay;
    int   operation;
    struct {
      int flush;
      int vsyncwait;
    } input;
    struct {
      int flush;
      int vsyncwait;
    } output;
  } setup;
  struct {
    int   count;
  } hline;
  struct {
    int   count;
  } vline;
  struct {
    unsigned char  buffer[LOGO_YSIZE*2][2*LOGO_XSIZE*8];
    unsigned char  bsmall[LOGO_YSIZE][2*LOGO_XSIZE*4];
    int   loaded;
    int   xoffset;
    int   yoffset;
    int   ydelta;
    int   xdelta;
    int   xsize;
    int   ysize;
    int   bytes;
    int   sizesmall;
    int   sizelarge;
  } logo;

  struct {
    int input;
    int output;
    int output_getbuffer;
  } dropped;
} logo_handle;


logo_handle global_preview;


int __cdecl Error(char * string, ...)
{
  va_list va;
  char    ach[512];
  int res = IDABORT;
#ifndef WIN32
  char c;
#endif
  
  va_start(va, string);

  wvsprintf(ach, string, va);

  va_end(va);

#ifdef WIN32
  OutputDebugString(ach);
  res = MessageBox(NULL, ach, "Logo Error", MB_TASKMODAL | MB_ABORTRETRYIGNORE);
#else
  if (global_preview.running) {	// print no errors when program terminating
    fprintf(stderr, "Logo Error: %s\n", ach);
    fprintf(stderr, "(a)bort - (r)etry - (i)gnore : ");
    do {
      c = getchar();
    } while (c != 'a' && c != 'r' && c != 'i');
    res = (int)c;
  }
#endif

  switch(res) {
  case IDABORT:
#ifdef WIN32
    PostQuitMessage(0);
#endif
    global_preview.running = FALSE;
    break;
  case IDRETRY:
    return TRUE;
  case IDIGNORE:
  default:;
  }

  return FALSE;
}


void convertbuffer(sv_storageinfo * pstorage, unsigned char * q, int inputsize, unsigned char * p, int * outputsize)
{
#ifdef sgi
  unsigned char * porg = p;
  unsigned char * qorg = q;
#endif
  int x;

  if(pstorage->nbits == 10) {
    x = inputsize;
    while(4 * x % 3) {
      x += inputsize;
    }
    inputsize = x;
    switch(pstorage->nbittype) {
    case SV_NBITTYPE_10BDVS:
      for(x = 0; x < inputsize; x+=3) {
        p[0] = q[0];
        p[1] = q[1];
        p[2] = q[2];
        p[3] = 0;
        p+=4;
        q+=3;
      }
      break;
    case SV_NBITTYPE_10BRALE:
      for(x = 0; x < inputsize; x+=3) {
        p[0] =             (q[0]<<2);
        p[1] = (q[0]>>6) | (q[1]<<4);
        p[2] = (q[1]>>4) | (q[2]<<6);
        p[3] = (q[2]>>2);
        p+=4;
        q+=3;
      }
      break;
    case SV_NBITTYPE_10BLABE:
      for(x = 0; x < inputsize; x+=3) {
        p[0] =              q[0];
        p[1] =              q[1]>>2;
        p[2] = (q[1]<<6) | (q[2]>>4);
        p[3] = (q[2]<<4);
        p+=4;
        q+=3;
      }
      break;
   case SV_NBITTYPE_10BRABE:
      for(x = 0; x < inputsize; x+=3) {
        p[0] =             (q[2]>>2);
        p[1] = (q[2]<<6) | (q[1]>>4);
        p[2] = (q[1]<<4) | (q[0]>>6);
        p[3] = (q[0]<<2);
        p+=4;
        q+=3;
      }
      break;
   case SV_NBITTYPE_10BLALE:
      for(x = 0; x < inputsize; x+=3) {
        p[0] =             (q[2]<<4);
        p[1] = (q[2]>>4) | (q[1]<<6);
        p[2] = (q[1]>>2            );
        p[3] = (q[0]               );
        p+=4;
        q+=3;
      }
      break;
    default:
      Error("Error unknown 10 bit mode\n");
      exit(-1);
    }
    if(outputsize) {
      *outputsize = 4 * inputsize / 3;
    }
  } else {
    memcpy(p, q, inputsize);
    if(outputsize) {
      *outputsize = inputsize;
    }
  }

#ifdef sgi
  if(outputsize) {
    inputsize = *outputsize;
  }
  for(p = porg, q = qorg, x = 3; x < inputsize; x+=4,p+=4,q+=4) {
    char t;
    t = p[0]; p[0] = p[3]; p[3] = t;
    t = p[1]; p[1] = p[2]; p[2] = t;
  }
#endif
}

   
int logo_operation_hline(logo_handle * handle, char * pbufferfield1,char * pbufferfield2, int interlaced )
{
  int line;
  int count;

  if((handle->hline.count < 0) || (handle->hline.count >= (handle->storage.linesize - 8))) {
    handle->hline.count = 0;
  }
  count = handle->hline.count++;

  for(line = 0; line < handle->ysize/handle->fieldcount; line++) {
   /* Avoid usage of memset - as for example under linux, for small amounts
      of data, memset operates with single-bytes, what is NOT supported by
      the HDBoard */
    *(uint32 *)(pbufferfield1 + line*handle->storage.lineoffset[0] + count * 8    ) = 0x80808080;
    *(uint32 *)(pbufferfield1 + line*handle->storage.lineoffset[0] + count * 8 + 4) = 0x80808080;
    if(interlaced) {
      *(uint32 *)(pbufferfield2 + line*handle->storage.lineoffset[0] + count * 8    ) = 0x80808080;
      *(uint32 *)(pbufferfield2 + line*handle->storage.lineoffset[0] + count * 8 + 4) = 0x80808080;
    }
  }

  return TRUE;
}


void* malloc_aligned( int size, int alignment, char**	orgptr ) 
{
  if( !size )    return NULL;
  
  if(alignment > 8) 
  {
    *orgptr = (char*)malloc(size + alignment);
    return (void*)(((uintptr)*orgptr + alignment - 1) & ~(alignment-1));
  }

  *orgptr = (char*)malloc(size);
  return *orgptr;
}


int logo_operation_vline(logo_handle * handle,  char * pbufferfield1,char * pbufferfield2, int interlaced)
{
  int line;

  if((handle->vline.count < 0) || (handle->vline.count >= handle->ysize/handle->fieldcount)) {
    handle->vline.count = 0;
  }
  line = handle->vline.count++;

  memset(pbufferfield1 + line*handle->storage.lineoffset[0], 0x80, handle->storage.linesize);
  if(interlaced) {
    memset(pbufferfield2 + line*handle->storage.lineoffset[0],  0x80, handle->storage.linesize);
  }

  return TRUE;
}



int logo_operation_logo(logo_handle * handle, char* pbufferfield1, char* pbufferfield2, int bsmall, int moving, int interlaced)
{
  unsigned char  bufferl[LOGO_YSIZE*2][2*LOGO_XSIZE*8];
  unsigned char  buffers[LOGO_YSIZE][2*LOGO_XSIZE*4];
  int line;
  int x,y;
  unsigned char * p;
#ifdef WIN32
  HBITMAP hBitmap;
#else
  int file;
  struct stat stat;
  unsigned char * p_base;
#endif
  char * addr1;
  char * addr2;
  unsigned char black[32];
  unsigned char white[32];
  unsigned char blue[32];
  int bytes = 1;
  int i;

  
  if(!handle->logo.loaded) {

#ifdef WIN32
    HANDLE hXXX = FindResource(handle->hInstance, MAKEINTRESOURCE(IDB_DVS), RT_BITMAP);
    hBitmap = LoadResource(handle->hInstance, hXXX);

    p = LockResource(hBitmap);
#else
    file = open("dvs.bmp", O_RDONLY);
    if (file < 0) {
      printf("could not open file 'dvs.bmp'\n");
      return FALSE;
    }
    fstat(file, &stat);

    p_base = p = (unsigned char *)malloc(stat.st_size);
    read(file, p, stat.st_size);
#endif
    

    /*
    //  Hard coded to image size 120x64 and the offset in the bmp file,
    //  the file was created with pbrush in 8 bit mode.
    */
    if(p) {
#ifdef WIN32
      p += 40;
#else
      p += 54;
#endif

      switch(handle->colormode) {
      case SV_COLORMODE_MONO:
        white[1] = white[0] = 0xf0;
        black[1] = black[0] = 0x10;
        blue[1]  = blue[0]  = 0x29;
        bytes    = 1;
        break;
      case SV_COLORMODE_CHROMA:
        white[1] = white[0] = 0x70;
        black[1] = black[0] = 0x90;
        blue[0]  = 0xf0;
        blue[1]  = 0x6e;
        bytes    = 1;
        break;
      case SV_COLORMODE_YUV422:
        white[2] = white[0] = 0x80;
        white[3] = white[1] = 0xf0;
        black[2] = black[0] = 0x80;
        black[3] = black[1] = 0x10;
        blue[0]  = 0xcd;
        blue[1]  = 0x77;
        blue[2]  = 0x2b;
        blue[3]  = 0x77;
        bytes    = 2;
        break;
      case SV_COLORMODE_YUV422A:
        white[3] = white[0] = 0x80;
        white[4] = white[1] = 0xf0;
        white[5] = white[2] = 0xf0;
        black[3] = black[0] = 0x80;
        black[4] = black[1] = 0x10;
        black[5] = black[2] = 0x10;
        blue[0]  = 0xcd;
        blue[1]  = 0x77;
        blue[2]  = 0x29;
        blue[3]  = 0x2b;
        blue[4]  = 0x77;
        blue[5]  = 0x29;
        bytes    = 3;
        break;
      case SV_COLORMODE_RGB_BGR:
        memset(black, 0x00, sizeof(black));
        memset(white, 0xff, sizeof(white));
        blue[3] = blue[0] = 0xff;
        blue[4] = blue[1] = 0x99;
        blue[5] = blue[2] = 0x00;
        bytes   = 3;
        break;
      case SV_COLORMODE_RGB_RGB:
        memset(black, 0x00, sizeof(black));
        memset(white, 0xff, sizeof(white));
        blue[3] = blue[0] = 0x00;
        blue[4] = blue[1] = 0x99;
        blue[5] = blue[2] = 0xff;
        bytes    = 3;
        break;
      case SV_COLORMODE_ABGR:
        memset(black, 0x00, sizeof(black));
        memset(white, 0xff, sizeof(white));
        blue[4] = blue[0] = 0x29;
        blue[5] = blue[1] = 0xff;
        blue[6] = blue[2] = 0x99;
        blue[7] = blue[3] = 0x00;
        bytes    = 4;
        break;
      case SV_COLORMODE_ARGB:
        memset(black, 0x00, sizeof(black));
        memset(white, 0xff, sizeof(white));
        blue[4] = blue[0] = 0x29;
        blue[5] = blue[1] = 0x00;
        blue[6] = blue[2] = 0x99;
        blue[7] = blue[3] = 0xff;
        bytes    = 4;
        break;
      case SV_COLORMODE_BGRA:
        memset(black, 0x00, sizeof(black));
        memset(white, 0xff, sizeof(white));
        blue[4] = blue[0] = 0xff;
        blue[5] = blue[1] = 0x99;
        blue[6] = blue[2] = 0x00;
        blue[7] = blue[3] = 0x29;
        bytes    = 4;
        break;
      case SV_COLORMODE_RGBA:
        memset(black, 0x00, sizeof(black));
        memset(white, 0xff, sizeof(white));
        blue[4] = blue[0] = 0x00;
        blue[5] = blue[1] = 0x99;
        blue[6] = blue[2] = 0xff;
        blue[7] = blue[3] = 0x29;
        bytes    = 4;
        break;
      case SV_COLORMODE_YUV444:
        white[3] = white[0] = 0x80;
        white[4] = white[1] = 0xf0;
        white[5] = white[2] = 0x80;
        black[3] = black[0] = 0x80;
        black[4] = black[1] = 0x10;
        black[5] = black[2] = 0x80;
        blue[0]  = 0xcd;
        blue[1]  = 0x77;
        blue[2]  = 0x2b;
        blue[3]  = 0xcd;
        blue[4]  = 0x77;
        blue[5]  = 0x2b;
        bytes    = 3;
        break;
      case SV_COLORMODE_YUV444A:
        white[4] = white[0] = 0x80;
        white[5] = white[1] = 0xf0;
        white[6] = white[2] = 0x80;
        white[7] = white[3] = 0xf0;
        black[4] = black[0] = 0x80;
        black[5] = black[1] = 0x10;
        black[6] = black[2] = 0x80;
        black[7] = black[3] = 0x10;
        blue[0]  = 0xcd;
        blue[1]  = 0x77;
        blue[2]  = 0x2b;
        blue[3]  = 0x29;
        blue[4]  = 0xcd;
        blue[5]  = 0x77;
        blue[6]  = 0x2b;
        blue[7]  = 0x29;
        bytes    = 4;
        break;
      default:
        Error("Colormode %d not supported\n", handle->colormode);
      }

      for(y = 0; y < LOGO_YSIZE; y++) {
        for(x = 0; x < LOGO_XSIZE; x++) {
          if((p[0] == 0xff) && (p[1] == 0xff) && (p[2] == 0xff)) {
            for(i = 0; i < bytes; i++) {
              buffers[LOGO_YSIZE - 1 - y][x*bytes + i]    = white[i];
            }

            for(i = 0; i < 2 * bytes; i++) {
              bufferl[LOGO_YSIZE - 1 - y][x*2*bytes + i]  = white[i%bytes];
            }
          } else {
            for(i = 0; i < bytes; i++) {
              buffers[LOGO_YSIZE - 1 - y][x*bytes + i]    = black[i];
            }

            for(i = 0; i < 2 * bytes; i++) {
              bufferl[LOGO_YSIZE - 1 - y][x*2*bytes + i]  = blue[i];
            }
          }
          convertbuffer(&handle->storage, &buffers[LOGO_YSIZE - 1 - y][0],     LOGO_XSIZE * bytes, &handle->logo.bsmall[LOGO_YSIZE - 1 - y][0], NULL);
          convertbuffer(&handle->storage, &bufferl[LOGO_YSIZE - 1 - y][0], 2 * LOGO_XSIZE * bytes, &handle->logo.buffer[LOGO_YSIZE - 1 - y][0], NULL);

          p += 3;
        }
      }
    }

#ifdef WIN32
    DeleteObject(hBitmap);
#else
    if (p_base) {
      free(p_base);
    }
    close(file);
#endif


    handle->logo.loaded = TRUE;

    handle->logo.xsize     = LOGO_XSIZE;
    handle->logo.ysize     = LOGO_YSIZE;
    handle->logo.sizesmall = handle->bps_nom * LOGO_XSIZE / handle->bps_denom;
    handle->logo.sizelarge = 2 * handle->bps_nom * LOGO_XSIZE / handle->bps_denom;

    handle->logo.xdelta    = handle->bps_nom;
    if(handle->logo.xdelta & 3) {
      handle->logo.xdelta *= 2;
    }
    if(handle->logo.xdelta & 3) {
      handle->logo.xdelta *= 2;
    }
    handle->logo.ydelta    = 1;
  }

  if(moving) {
    int size = bsmall?1:2;

    if(handle->logo.yoffset >= (handle->ysize - size * LOGO_YSIZE - size) / handle->fieldcount) {
      if(handle->logo.ydelta > 0) {
        handle->logo.ydelta = -handle->logo.ydelta;
      }
    } else if(handle->logo.yoffset <= abs(handle->logo.ydelta)) {
      if(handle->logo.ydelta < 0) {
        handle->logo.ydelta = -handle->logo.ydelta;
      }
    }

    if(handle->logo.xoffset >= handle->bps_nom * (handle->xsize - size * LOGO_XSIZE - size) / handle->bps_denom) {
      if(handle->logo.xdelta > 0) {
        handle->logo.xdelta = -handle->logo.xdelta;
      }
    } else if(handle->logo.xoffset <= abs(handle->logo.xdelta)) {
      if(handle->logo.xdelta < 0) {
        handle->logo.xdelta = -handle->logo.xdelta;
      }
    }

    handle->logo.yoffset += 2*handle->logo.ydelta;
    handle->logo.xoffset += handle->logo.xdelta;
  }

  if(bsmall) {
    if(handle->fieldcount == 1) {
      for(line = 0; line < 2 * handle->logo.ysize; line++) {
        addr1 = pbufferfield1 + (line + handle->logo.yoffset)*(handle->storage.lineoffset[0]) + handle->logo.xoffset;
        memcpy(addr1, &handle->logo.bsmall[line/2][0], handle->logo.sizesmall);
      }
    } else {
      for(line = 0; line < handle->logo.ysize; line+=2) {
        addr1 = pbufferfield1 + (line/2+handle->logo.yoffset)*(handle->storage.lineoffset[0]) + handle->logo.xoffset;
        addr2 = pbufferfield2 + (line/2+handle->logo.yoffset)*(handle->storage.lineoffset[0]) + handle->logo.xoffset;
        if(moving) {
          addr2 += handle->logo.ydelta * handle->storage.lineoffset[0] + handle->logo.xdelta;
        }
        memcpy(addr1, &handle->logo.bsmall[line  ][0], handle->logo.sizesmall);
        if(interlaced) {
          memcpy(addr2, &handle->logo.bsmall[line+1][0], handle->logo.sizesmall);
        }
      }
    }
  } else {
    if(handle->fieldcount == 1) {
      for(line = 0; line < 2 * handle->logo.ysize; line++) {
        addr1 = pbufferfield1 + (line + handle->logo.yoffset)*handle->storage.lineoffset[0] + handle->logo.xoffset;
        memcpy(addr1, &handle->logo.buffer[line/2][0], handle->logo.sizelarge);
      }
    } else {
      for(line = 0; line < handle->logo.ysize; line++) {
        addr1 = pbufferfield1 + (line + handle->logo.yoffset) * handle->storage.lineoffset[0] + handle->logo.xoffset;
        addr2 = pbufferfield2 + (line + handle->logo.yoffset) * handle->storage.lineoffset[0] + handle->logo.xoffset;
        if(moving) {
          addr2 += handle->logo.ydelta * handle->storage.lineoffset[0] + handle->logo.xdelta;
        }
        memcpy(addr1, &handle->logo.buffer[line][0], handle->logo.sizelarge);
        if(interlaced) {
          memcpy(addr2, &handle->logo.buffer[line][0], handle->logo.sizelarge);
        }
      }
    }
  }

  return TRUE;
}


DWORD logo_paint_thread(void * voidhandle)
{
  logo_handle *       handle = voidhandle;
  sv_fifo_buffer *    pbuffer = 0;

  sv_fifo_configinfo  mConfigInfo; //Buffersize and dmaalignment

  sv_fifo_info        info;
  sv_fifo *           pinput  = NULL;
  sv_fifo *           poutput = NULL;
  sv_handle *         sv;
  int                 res     = SV_OK;
  int                 running = TRUE;
  int                 started = FALSE;
  int                 startdelay;
  int                 when    = 0;
  int                 flags;

  char *mpVBuffer;              //aligned Pointer to Videobuffer for DMA Transfer
  char *mpVBuffer_org;          //notaligned original Pointer, need to free Videobuffer


  memset( (void*)&mConfigInfo,   0, sizeof(sv_fifo_configinfo) );

  /*
  //    Open the device, for the direct access dll the syntax is
  //    "PCI,card:0" for the first card (this is the default) and 
  //    "PCI,card:n" for the next card in the system.
  */  
  res = sv_openex(&sv, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_DEFAULT, 0, 0);
  if(res != SV_OK) {
    Error("logo: Error '%s' opening video device", sv_geterrortext(res));
    running = FALSE;
  } 

  /*
  //  Initialize the input fifo, note that this is using the shared flag,
  //  thus the input and output fifo will use the same memory. In this case
  //  we will do the manipulation directly on the card, without transfering the
  //  video data to memory.
  */
  if(running) {
    res = sv_fifo_init(sv, &pinput, TRUE, TRUE, TRUE, SV_FIFO_FLAG_VIDEOONLY, 0);
    if(res != SV_OK)  {
      running = Error("logo_paint_thread: sv_fifo_init(pinput) failed %d '%s'", res, sv_geterrortext(res));
    } 
  }

  //Get Buffer size
  if(running)
  {
    res = sv_fifo_configstatus( sv, pinput, &mConfigInfo );
    if(res != SV_OK){
      running = Error("logo_paint_thread: sv_fifo_configstatus() failed");
    }
  }

  //Allocate FifoBuffer
  mpVBuffer = (char*) malloc_aligned( mConfigInfo.vbuffersize + mConfigInfo.abuffersize, mConfigInfo.dmaalignment , &mpVBuffer_org );
  if( !mpVBuffer ){
      running = Error("logo_paint_thread: malloc_aligned() failed");
  }

  
  /*
  //  Initialize the output fifo.
  */
  if(running) {
    res = sv_fifo_init(sv, &poutput, FALSE, TRUE, TRUE, FALSE, 0);
    if(res != SV_OK)  {
      running = Error("logo_paint_thread: sv_fifo_init(poutput) failed %d '%s'", res, sv_geterrortext(res));
    } 
  }

  /*
  //  Reset the position of the input fifo.
  */
  if(running) {
    res = sv_fifo_start(sv, pinput);
    if(res != SV_OK)  {
      running = Error("logo_paint_thread: sv_fifo_start(pinput) failed %d '%s'", res, sv_geterrortext(res));
    }
  }
  
  startdelay = handle->setup.startdelay;
  while(running && handle->running) {
    
    flags = 0;

    if(handle->setup.input.vsyncwait) {
      flags |= SV_FIFO_FLAG_VSYNCWAIT;
    }
    if(handle->setup.input.flush) {
      flags |= SV_FIFO_FLAG_FLUSH;
    }
    /*
    //  Get the next buffer recorded.
    */
    res = sv_fifo_getbuffer(sv, pinput, &pbuffer, NULL, flags);
    if(res != SV_OK){
      running = Error("logo_paint_thread: sv_fifo_getbuffer(pinput) failed %d '%s'", res, sv_geterrortext(res));
    }

    //WriteDmaAdresses
    if( running && pbuffer && mpVBuffer )
    {
      pbuffer->dma.addr = mpVBuffer;
      pbuffer->dma.size = mConfigInfo.vbuffersize;
    }


    /*
    //  Here would be code that processes the captured video data
    //  when using the mapped transfer operations, the paint code
    //  that is in the display loop could as well be here.
    */

    if (running && res == SV_OK) {
      when = pbuffer->control.tick;
    }

    if(running) {
      /*
      //  Give back the buffer to be queued again for input, 
      */
      res = sv_fifo_putbuffer(sv, pinput, pbuffer, NULL);
      if(res != SV_OK)  {
        running = Error("logo_paint_thread: sv_fifo_putbuffer(pinput) failed %d '%s'", res, sv_geterrortext(res));
      }

      res = sv_fifo_status(sv, pinput, &info);
      if(res != SV_OK)  {
        running = Error("logo_paint_thread: sv_fifo_status(pinput) failed %d '%s'", res, sv_geterrortext(res));
      }
      handle->dropped.input = info.dropped;
    }


    if(running) {
      if(startdelay <= 0) {
        
       if(running && pbuffer) {
          /*
          //  Here the operation on the recorded video is done, 
          //  Replace this with your own code.
          */
          switch(handle->setup.operation) {
          case LOGO_OPERATION_HLINE:
            running = logo_operation_hline(handle, mpVBuffer + (uintptr)pbuffer->video[0].addr, mpVBuffer + (uintptr)pbuffer->video[1].addr,pbuffer->video[1].size);
            break;
          case LOGO_OPERATION_VLINE:
            running = logo_operation_vline(handle, mpVBuffer + (uintptr)pbuffer->video[0].addr, mpVBuffer + (uintptr)pbuffer->video[1].addr,pbuffer->video[1].size);
            break;
        
          case LOGO_OPERATION_LOGO_MOVING:
            running = logo_operation_logo(handle, mpVBuffer + (uintptr)pbuffer->video[0].addr, mpVBuffer + (uintptr)pbuffer->video[1].addr, FALSE, TRUE,pbuffer->video[1].size );
            break;
        
          case LOGO_OPERATION_LOGO:
            running = logo_operation_logo(handle, mpVBuffer + (uintptr)pbuffer->video[0].addr, mpVBuffer + (uintptr)pbuffer->video[1].addr, FALSE, FALSE,pbuffer->video[1].size );
            break;
          case LOGO_OPERATION_LOGOSMALL:
            running = logo_operation_logo(handle, mpVBuffer + (uintptr)pbuffer->video[0].addr, mpVBuffer + (uintptr)pbuffer->video[1].addr, TRUE, FALSE,pbuffer->video[1].size );
            break;
          case LOGO_OPERATION_LOGOSMALL_MOVING:
            running = logo_operation_logo(handle, mpVBuffer + (uintptr)pbuffer->video[0].addr, mpVBuffer + (uintptr)pbuffer->video[1].addr, TRUE, TRUE,pbuffer->video[1].size );
            break;
          default:
            running = Error("logo_paint_thread: operation == %d not implemented", handle->setup.operation);
          }
        }

        if(running) {
          /*
          //  Fetch the buffer to the next frame to be filled in.
          */
          flags = 0;
          if(handle->setup.output.vsyncwait) {
            flags |= SV_FIFO_FLAG_VSYNCWAIT;
          }
          if(handle->setup.output.flush) {
            flags |= SV_FIFO_FLAG_FLUSH;
          }
          
          res = sv_fifo_getbuffer(sv, poutput, &pbuffer, 0, flags);
          if(res == SV_ERROR_VSYNCPASSED) {
            handle->dropped.output_getbuffer++;
          }else if(res != SV_OK)  {
            running = Error("logo_paint_thread: sv_fifo_getbuffer(poutput) failed %d '%s'", res, sv_geterrortext(res));
          }
        }

        //WriteDmaAdresses
        if( mpVBuffer )
        {
          pbuffer->dma.addr = mpVBuffer;
          pbuffer->dma.size = mConfigInfo.vbuffersize;
        }

        /*
        //  Release the buffer to be queued for display
        */
        if(running && pbuffer) {
          res = sv_fifo_putbuffer(sv, poutput, pbuffer, NULL);
          if(res != SV_OK)  {
            running = Error("logo_paint_thread: sv_fifo_putbuffer(poutput) failed %d '%s'", res, sv_geterrortext(res));
          }
        }

        res = sv_fifo_status(sv, poutput, &info);
        if(res != SV_OK)  {
          running = Error("logo_paint_thread: sv_fifo_status(poutput) failed %d '%s'", res, sv_geterrortext(res));
        }
        handle->dropped.output = info.dropped;       

        if(!started) {
          /*
          //  Start the output fifo.
          */
          res = sv_fifo_start(sv, poutput);
          if(res != SV_OK)  {
            running = Error("logo_paint_thread: sv_fifo_start(poutput) failed %d '%s'", res, sv_geterrortext(res));
          }
          started = TRUE;
        }
      } else {
        startdelay--;
      }
    }
  }


#ifndef WIN32
  printf("Dropped on input:%d output:%d output_getbuffer:%d\n", handle->dropped.input, handle->dropped.output, handle->dropped.output_getbuffer);
#endif

  if(poutput) {
    res = sv_fifo_free(sv, poutput);
    if(running && (res != SV_OK))  {
      Error("logo_paint_thread: sv_fifo_exit(poutput) failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  if(pinput) {
    res = sv_fifo_free(sv, pinput);
    if(running && (res != SV_OK))  {
      Error("logo_paint_thread: sv_fifo_exit(pinput) failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  if(sv) {
    res = sv_close(sv);
    if(running && (res != SV_OK)) {
      running = Error("logo_paint_thread: sv_close(sv) failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  handle->running = FALSE;

#ifdef WIN32
  ExitThread(res); 
#else
  return 0;
#endif
}



void logo_exit(logo_handle * handle)
{  
#ifdef WIN32
  int res;
  DWORD dwExitCode;
#endif

  handle->running = FALSE;

#ifdef WIN32
  do {
    res = GetExitCodeThread(handle->hThreadPaint, &dwExitCode);
    if(!res) {
      Error("logo_exit: GetExitCodeThread(handle->hThreadRecord,) failed = %d", GetLastError());
    }
    if(dwExitCode == STILL_ACTIVE) {
      Sleep(50);
    }
  } while(res && (dwExitCode == STILL_ACTIVE)); 
#endif
}


int logo_init(logo_handle * handle) 
{
  sv_handle *	  sv;
  int		  res;

  res = sv_openex(&sv, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_DEFAULT, 0, 0);
  if(res != SV_OK) {
    Error("logo: Error '%s' opening video device", sv_geterrortext(res));
    return FALSE;
  } 

  res = sv_storage_status(sv, 0, NULL, &handle->storage, sizeof(handle->storage), 0);
  if(res != SV_OK) {
    Error("logo_init: sv_storage_status(sv, &info) failed %d '%s'", res, sv_geterrortext(res));
    return FALSE;
  }

  sv_close(sv);

  handle->xsize       = handle->storage.xsize;
  handle->ysize       = handle->storage.ysize;
  handle->fieldcount  = handle->storage.interlace;
  handle->interlace   = handle->storage.vinterlace;
  handle->fps         = handle->storage.vfps;
  handle->colormode   = handle->storage.colormode;
  handle->bps_nom     = handle->storage.pixelsize_num;
  handle->bps_denom   = handle->storage.pixelsize_denom;

  handle->running    = TRUE;

  if (handle->setup.delay == -1) {	// not yet set
    if(handle->interlace == 2) {
      handle->setup.delay = 3;
    } else {
      handle->setup.delay = 4;
    }
  }

#ifdef WIN32 
  handle->hThreadPaint = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)logo_paint_thread, handle, 0, NULL);
  if(handle->hThreadPaint == NULL) {
    Error("logo_init: CreateThread(record) failed = %d", GetLastError());
    handle->running = FALSE;
  }
  
  if(!handle->running) {
    logo_exit(handle);
  }
#endif

  return TRUE;
}


void logo_inithandle(logo_handle * handle)
{
  handle->logo.yoffset = 0;
  handle->logo.xoffset = 0;

  handle->setup.operation = LOGO_OPERATION_LOGO_MOVING;
  handle->setup.startdelay        = 0;
  handle->setup.delay             = 0;
  handle->setup.input.flush       = 0;
  handle->setup.output.flush      = 0;
}


#ifdef WIN32
void logo_update(logo_handle * handle, HWND hWnd)
{
  HMENU hMenu = GetMenu(hWnd);

  CheckMenuItem(hMenu, ID_DELAY_X,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_1,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_2,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_3,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_4,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_5,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_6,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_7,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_8,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_9,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_10, MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_DELAY_20, MF_BYCOMMAND | MF_UNCHECKED);
  switch(handle->setup.delay) {
  case 0:
    CheckMenuItem(hMenu, ID_DELAY_X,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 1:
    CheckMenuItem(hMenu, ID_DELAY_1,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 2:
    CheckMenuItem(hMenu, ID_DELAY_2,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 3:
    CheckMenuItem(hMenu, ID_DELAY_3,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 4:
    CheckMenuItem(hMenu, ID_DELAY_4,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 5:
    CheckMenuItem(hMenu, ID_DELAY_5,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 6:
    CheckMenuItem(hMenu, ID_DELAY_6,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 7:
    CheckMenuItem(hMenu, ID_DELAY_7,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 8:
    CheckMenuItem(hMenu, ID_DELAY_8,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 9:
    CheckMenuItem(hMenu, ID_DELAY_9,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 10:
    CheckMenuItem(hMenu, ID_DELAY_10, MF_BYCOMMAND | MF_CHECKED);
    break;
  case 20:
    CheckMenuItem(hMenu, ID_DELAY_20, MF_BYCOMMAND | MF_CHECKED);
    break;
  }



  CheckMenuItem(hMenu, ID_STARTDELAY_0,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_1,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_2,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_3,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_4,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_5,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_6,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_7,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_8,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_9,  MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_10, MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_STARTDELAY_20, MF_BYCOMMAND | MF_UNCHECKED);
  switch(handle->setup.startdelay) {
  case 0:
    CheckMenuItem(hMenu, ID_STARTDELAY_0,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 1:
    CheckMenuItem(hMenu, ID_STARTDELAY_1,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 2:
    CheckMenuItem(hMenu, ID_STARTDELAY_2,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 3:
    CheckMenuItem(hMenu, ID_STARTDELAY_3,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 4:
    CheckMenuItem(hMenu, ID_STARTDELAY_4,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 5:
    CheckMenuItem(hMenu, ID_STARTDELAY_5,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 6:
    CheckMenuItem(hMenu, ID_STARTDELAY_6,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 7:
    CheckMenuItem(hMenu, ID_STARTDELAY_7,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 8:
    CheckMenuItem(hMenu, ID_STARTDELAY_8,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 9:
    CheckMenuItem(hMenu, ID_STARTDELAY_9,  MF_BYCOMMAND | MF_CHECKED);
    break;
  case 10:
    CheckMenuItem(hMenu, ID_STARTDELAY_10, MF_BYCOMMAND | MF_CHECKED);
    break;
  case 20:
    CheckMenuItem(hMenu, ID_STARTDELAY_20, MF_BYCOMMAND | MF_CHECKED);
    break;
  }

  CheckMenuItem(hMenu, ID_OPERATION_HLINE,            MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_OPERATION_VLINE,            MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_OPERATION_LOGO,             MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_OPERATION_LOGOMOVING,       MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_OPERATION_LOGOSMALL,        MF_BYCOMMAND | MF_UNCHECKED);
  CheckMenuItem(hMenu, ID_OPERATION_LOGOSMALLMOVING,  MF_BYCOMMAND | MF_UNCHECKED);
  switch(handle->setup.operation) {
  case LOGO_OPERATION_HLINE:
    CheckMenuItem(hMenu, ID_OPERATION_HLINE, MF_BYCOMMAND | MF_CHECKED);
    break;
  case LOGO_OPERATION_VLINE:
    CheckMenuItem(hMenu, ID_OPERATION_VLINE, MF_BYCOMMAND | MF_CHECKED);
    break;
  case LOGO_OPERATION_LOGO:
    CheckMenuItem(hMenu, ID_OPERATION_LOGO, MF_BYCOMMAND | MF_CHECKED);
    break;
  case LOGO_OPERATION_LOGO_MOVING:
    CheckMenuItem(hMenu, ID_OPERATION_LOGOMOVING, MF_BYCOMMAND | MF_CHECKED);
    break;
  case LOGO_OPERATION_LOGOSMALL:
    CheckMenuItem(hMenu, ID_OPERATION_LOGOSMALL, MF_BYCOMMAND | MF_CHECKED);
    break;
  case LOGO_OPERATION_LOGOSMALL_MOVING:
    CheckMenuItem(hMenu, ID_OPERATION_LOGOSMALLMOVING, MF_BYCOMMAND | MF_CHECKED);
    break;
  }

  if(handle->setup.input.flush) {
    CheckMenuItem(hMenu, ID_INPUT_FLUSH, MF_BYCOMMAND | MF_CHECKED);
  } else {
    CheckMenuItem(hMenu, ID_INPUT_FLUSH, MF_BYCOMMAND | MF_UNCHECKED);
  }

  if(handle->setup.input.vsyncwait) {
    CheckMenuItem(hMenu, ID_INPUT_VSYNCWAIT, MF_BYCOMMAND | MF_CHECKED);
  } else {
    CheckMenuItem(hMenu, ID_INPUT_VSYNCWAIT, MF_BYCOMMAND | MF_UNCHECKED);
  }


  if(handle->setup.output.flush) {
    CheckMenuItem(hMenu, ID_OUTPUT_FLUSH, MF_BYCOMMAND | MF_CHECKED);
  } else {
    CheckMenuItem(hMenu, ID_OUTPUT_FLUSH, MF_BYCOMMAND | MF_UNCHECKED);
  }

  if(handle->setup.output.vsyncwait) {
    CheckMenuItem(hMenu, ID_OUTPUT_VSYNCWAIT, MF_BYCOMMAND | MF_CHECKED);
  } else {
    CheckMenuItem(hMenu, ID_OUTPUT_VSYNCWAIT, MF_BYCOMMAND | MF_UNCHECKED);
  }
}
#endif


#ifdef WIN32
DWORD APIENTRY logo_dlgproc(HWND hWnd, UINT message, WPARAM wparam, LPARAM lparam) 
{ 
  static logo_handle * handle = NULL;
  int command;

  switch(message) {
  case WM_TIMER:
    SetDlgItemInt(hWnd, IDC_INPUT_DROPPED,  handle->dropped.input, FALSE);
    SetDlgItemInt(hWnd, IDC_OUTPUT_DROPPED, handle->dropped.output, FALSE);    
    SetDlgItemInt(hWnd, IDC_OUTPUT_DROPPED_GETBUFFER, handle->dropped.output_getbuffer, FALSE);    
    break;

  case WM_INITDIALOG:
    handle = (logo_handle *) lparam;
    handle->hWnd = hWnd;

    logo_init(handle);
    logo_update(handle, hWnd);
    SetTimer(hWnd, 0x54321, 100, NULL);
    return FALSE;

  case WM_COMMAND:
    command = GET_WM_COMMAND_ID(wparam, lparam);

    switch(command) {
    case ID_START:
      if(handle->running) {
        logo_exit(handle);
      }
      logo_init(handle);
      logo_update(handle, hWnd);
      break;

    case ID_STOP:
      logo_exit(handle);
      break;

    case ID_OPERATION_HLINE:
      handle->setup.operation = LOGO_OPERATION_HLINE;
      break;
    case ID_OPERATION_VLINE:
      handle->setup.operation = LOGO_OPERATION_VLINE;
      break;
    case ID_OPERATION_LOGO:
      handle->setup.operation = LOGO_OPERATION_LOGO;
      break;
    case ID_OPERATION_LOGOSMALL:
      handle->setup.operation = LOGO_OPERATION_LOGOSMALL;
      break;
    case ID_OPERATION_LOGOMOVING:
      handle->setup.operation = LOGO_OPERATION_LOGO_MOVING;
      break;
    case ID_OPERATION_LOGOSMALLMOVING:
      handle->setup.operation = LOGO_OPERATION_LOGOSMALL_MOVING;
      break;

    case ID_DELAY_X:
      handle->setup.delay = 0;
      break;
    case ID_DELAY_1:
      handle->setup.delay = 1;
      break;
    case ID_DELAY_2:
      handle->setup.delay = 2;
      break;
    case ID_DELAY_3:
      handle->setup.delay = 3;
      break;
    case ID_DELAY_4:
      handle->setup.delay = 4;
      break;
    case ID_DELAY_5:
      handle->setup.delay = 5;
      break;
    case ID_DELAY_6:
      handle->setup.delay = 6;
      break;
    case ID_DELAY_7:
      handle->setup.delay = 7;
      break;
    case ID_DELAY_8:
      handle->setup.delay = 8;
      break;
    case ID_DELAY_9:
      handle->setup.delay = 9;
      break;
    case ID_DELAY_10:
      handle->setup.delay = 10;
      break;
    case ID_DELAY_20:
      handle->setup.delay = 20;
      break;

    case ID_STARTDELAY_0:
      handle->setup.startdelay = 0;
      break;
    case ID_STARTDELAY_1:
      handle->setup.startdelay = 1;
      break;
    case ID_STARTDELAY_2:
      handle->setup.startdelay = 2;
      break;
    case ID_STARTDELAY_3:
      handle->setup.startdelay = 3;
      break;
    case ID_STARTDELAY_4:
      handle->setup.startdelay = 4;
      break;
    case ID_STARTDELAY_5:
      handle->setup.startdelay = 5;
      break;
    case ID_STARTDELAY_6:
      handle->setup.startdelay = 6;
      break;
    case ID_STARTDELAY_7:
      handle->setup.startdelay = 7;
      break;
    case ID_STARTDELAY_8:
      handle->setup.startdelay = 8;
      break;
    case ID_STARTDELAY_9:
      handle->setup.startdelay = 8;
      break;
    case ID_STARTDELAY_10:
      handle->setup.startdelay = 10;
      break;
    case ID_STARTDELAY_20:
      handle->setup.startdelay = 20;
      break;

    case ID_INPUT_FLUSH:
      handle->setup.input.flush       = !handle->setup.input.flush;
      break;
    case ID_INPUT_VSYNCWAIT:
      handle->setup.input.vsyncwait   = !handle->setup.input.vsyncwait;
      break;

    case ID_OUTPUT_FLUSH:
      handle->setup.output.flush      = !handle->setup.output.flush;
      break;
    case ID_OUTPUT_VSYNCWAIT:
      handle->setup.output.vsyncwait  = !handle->setup.output.vsyncwait;
      break;

    case ID_FILE_EXIT:
    case IDOK:
    case IDCANCEL:
      if(handle->running) {
        logo_exit(handle);
      }
      EndDialog(hWnd, TRUE);
      break;
    default:
      return FALSE;
    }

    logo_update(handle, hWnd);

    return TRUE;
  default: 
    return FALSE;
  }

  return FALSE;
}
#endif


#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
  int res = TRUE;
  logo_handle * handle;
    
  memset(&global_preview, 0, sizeof(logo_handle));

  handle = &global_preview;

  logo_inithandle(handle);    

  handle->hInstance = hInstance;

  res = DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_LOGO), NULL, (DLGPROC)logo_dlgproc, (LPARAM)handle);
  if(res == -1) {
    Error("WinMain: DialogBox failed = %d", GetLastError());
  }
  
  return 0;
}
#endif

#ifndef WIN32
void signal_handler(signum)
{
  printf("User terminated program.\n");

  global_preview.running = FALSE;
}

void usage(char *name)
{
  printf("%s [operation [delay[us] [startdelay]]]\n", name);
  printf(" operation:\n");
  printf("  vline  - vertical moving line\n");
  printf("  hline  - horizontal moving line\n");
  printf("  logo   - small logo\n");
  printf("  LOGO   - big logo\n");
  printf("  mlogo  - small logo moving\n");
  printf("  MLOGO  - big logo moving (default)\n");
  printf(" delay:\n");
  printf("  0-20   - do a timed operation (default is 3 when interlace=2\n");
  printf("           or 4 when interlace!=2)\n");
  printf("      us - clocked operation - delay is in microseconds\n");
  printf(" startdelay:\n");
  printf("  0-20   - number of frames between input and output (default is 0)\n"
);
}

int main(int argc, char **argv)
{
  logo_handle *handle;

  memset(&global_preview, 0, sizeof(logo_handle));

  handle = &global_preview;

  logo_inithandle(handle);

  if (argc == 1) {
    handle->setup.operation = LOGO_OPERATION_LOGO_MOVING;
  } else {
    if (!strcmp(argv[1], "vline")) {
      handle->setup.operation = LOGO_OPERATION_VLINE;
    } else if (!strcmp(argv[1], "hline")) {
      handle->setup.operation = LOGO_OPERATION_HLINE;
    } else if (!strcmp(argv[1], "logo")) {
      handle->setup.operation = LOGO_OPERATION_LOGOSMALL;
    } else if (!strcmp(argv[1], "LOGO")) {
      handle->setup.operation = LOGO_OPERATION_LOGO;
    } else if (!strcmp(argv[1], "mlogo")) {
      handle->setup.operation = LOGO_OPERATION_LOGOSMALL_MOVING;
    } else if (!strcmp(argv[1], "MLOGO")) {
      handle->setup.operation = LOGO_OPERATION_LOGO_MOVING;
    } else {
      usage(argv[0]);
      return 1;
    }

    if (argc == 3) {
      if (argv[2][strlen(argv[2])-1] == 's' && argv[2][strlen(argv[2])-2] == 'u') {
        argv[2][strlen(argv[2])-2] = '\0';
        handle->setup.delay_clocked = atoi(argv[2]);
      } else {
        handle->setup.delay = atoi(argv[2]);
      }
    }
    handle->setup.delay = (handle->setup.delay > 20) ? 20 :
      ((handle->setup.delay < 0) ? 0 : handle->setup.delay);

    if (argc == 4) {
      handle->setup.startdelay = atoi(argv[3]);
      handle->setup.startdelay = (handle->setup.startdelay > 20) ? 20 :
        ((handle->setup.startdelay < 0) ? 0 : handle->setup.startdelay);
    }
  }

  signal(SIGTERM, signal_handler);
  signal(SIGKILL, signal_handler);
  signal(SIGINT, signal_handler);

  logo_init(handle);
  logo_paint_thread(handle);
  logo_exit(handle);

  return 0;
}
#endif
