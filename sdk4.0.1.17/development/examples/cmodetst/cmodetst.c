/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    cmodetst: Demo for displaying with changing colormodes / bitdepths / size for each image.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#if defined(_WIN32)
#	include <windows.h>
#	include <direct.h>
#	include <io.h>
#	include <sys/stat.h>
#else
#	include <unistd.h>
#	include <netinet/in.h>
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"

#ifdef _WIN32
# pragma warning (disable : 4244)
#endif
#define XMINVALUE 32
#define YMINVALUE 8
#define ProgramError() printf("ProgramError @ %s/%d\n", __FILE__, __LINE__)
#define STEP_XSIZE 1
#define STEP_YSIZE 1

typedef struct {
  sv_handle *        sv;      //!< Handle to the sv structure
  sv_fifo *          pfifo;   //!< FIFO init structure
  sv_fifo_configinfo config;  //!< FIFO info structure
  sv_storageinfo     storage; //!< FIFO storageinfo structure
  int running;                //!< Running variable for signal handler
  int bchange;                //!< Parameter to change the bitmodes
  int cchange;                //!< Parameter to change the colormodes
  int xchange;                //!< Parameter to change the xsize
  int ychange;                //!< Parameter to change the ysixe
  int dataoffset;             //!< Parameter for the dataoffset
  int halfsize;               //!< Parameter to enable halfsize
  int sleepy;                 //!< Parameter to enable sleepmode
  char * vbuffer_org;         //!< Original videobuffer pointer needed for free
  char * vbuffer;             //!< Aligned videobuffer pointer needed for work
  int xsize;                  //!< Current dvs board storage xsize
  int ysize;                  //!< Current dvs board storage ysize
  int position;               //!< Enables the ability to use own x and y offsets          
  int xoffset;                //!< X offset of the picture (needs position parameter)
  int yoffset;                //!< Y offset of the picture (needs position parameter)
} loop_handle;


loop_handle lhd;              //!< Global struct pointer

int storagemodes[] = {
  SV_MODE_COLOR_YUV422,
  SV_MODE_COLOR_YUV422A,
  SV_MODE_COLOR_YUV444,
  SV_MODE_COLOR_YUV444A,
  SV_MODE_COLOR_RGB_BGR,
  SV_MODE_COLOR_BGRA,
  SV_MODE_COLOR_RGB_RGB,
  SV_MODE_COLOR_RGBA,
}; //!< Global storagemode array

int nbitsmodes[] = {
  SV_MODE_NBIT_8B,
  SV_MODE_NBIT_10B,
  SV_MODE_NBIT_10BDVS,
}; //!< Global nbitsmode array


///Calculate the correct num and denom from the current storage mode.
/**
* It is only a helper function, so it is not important for the example.
*
\param storagemode Value of the current colormode and bitmode.
\param *num        Pointer to the num variable.
\param *denom      Pointer to the denom variable.
*/
void getDenomAndDivisor( int storagemode, int *denominator, int *divisor)
{
  switch(storagemode & SV_MODE_COLOR_MASK)
  {
  case SV_MODE_COLOR_LUMA:
  case SV_MODE_COLOR_CHROMA:
    *denominator = 1;
    break;
  case SV_MODE_COLOR_YUV422:
  case SV_MODE_COLOR_YUV422_YUYV:
    *denominator = 2;
    break;
  case SV_MODE_COLOR_RGB_BGR:
  case SV_MODE_COLOR_RGB_RGB:
  case SV_MODE_COLOR_YUV422A:
  case SV_MODE_COLOR_YUV444:
    *denominator = 3;
    break;
  case SV_MODE_COLOR_YUV444A:
  case SV_MODE_COLOR_ABGR:
  case SV_MODE_COLOR_ARGB:
  case SV_MODE_COLOR_BGRA:
  case SV_MODE_COLOR_RGBA:
    *denominator = 4;
    break;
  default:
    *denominator = -1;
  }

  switch(storagemode & SV_MODE_NBIT_MASK) {
  case SV_MODE_NBIT_8B:
    *denominator = (*denominator) * 1;
    *divisor     = 1;
    break;
  case SV_MODE_NBIT_10B:
  case SV_MODE_NBIT_10BDVS:
  case SV_MODE_NBIT_10BDPX:
    *denominator = (*denominator) * 4;
    *divisor     = 3;
    break;
  case SV_MODE_NBIT_12B:
    *denominator = (*denominator) * 3;
    *divisor     = 2;
    break;
  case SV_MODE_NBIT_16BBE:
  case SV_MODE_NBIT_16BLE:
    *denominator = (*denominator) * 2;
    *divisor     = 1;
    break;
  default:
    *denominator = -1;
    *divisor     = -1;
  }
}


///Create a colorbar in a videobuffer.
/**
* It is only a helper function, so it is not important for the example.
*
\param *vbuffer    Pointer to the videobuffer.
\param sdtv        SDTV, needed for matrix.
\param storagemode Storagemode, needed for bit type and colormode.
\param xsize       XSize of vbuffer.
\param ysize       YSize of vbuffer.
\param *pframesize Framesize from the colorbar.
*/
int colorbar(unsigned char * vbuffer, int sdtv, int storagemode, int xsize, int ysize, int * pframesize)
{
  unsigned char * p;
  unsigned char * q;
  int x,y,k,l;
  int sizeof3lines;
  int framesize;
  int denominator;
  int divisor;
  static unsigned char lum601[8] = { 0xeb, 0xd2, 0xaa, 0x91, 0x6a, 0x51, 0x29, 0x10 };
  static unsigned char bmy601[8] = { 0x80, 0x10, 0xa6, 0x36, 0xca, 0x5a, 0xf0, 0x80 };
  static unsigned char rmy601[8] = { 0x80, 0x92, 0x10, 0x22, 0xde, 0xf0, 0x6e, 0x80 };
  static unsigned char lum709[8] = { 0xeb, 0xdb, 0xbc, 0xad, 0x4e, 0x3f, 0x20, 0x10 };
  static unsigned char bmy709[8] = { 0x80, 0x10, 0x9a, 0x2a, 0xd6, 0x66, 0xf0, 0x80 };
  static unsigned char rmy709[8] = { 0x80, 0x8a, 0x10, 0x1a, 0xe6, 0xf0, 0x76, 0x80 };  
  static unsigned char grn[8] = { 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00 };
  static unsigned char red[8] = { 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00 };
  static unsigned char blu[8] = { 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00 };
  static unsigned char key[8] = { 0xff, 0xe2, 0xb3, 0x96, 0x69, 0x4c, 0x1d, 0x00 };
  unsigned char * lum;
  unsigned char * bmy;
  unsigned char * rmy;

  getDenomAndDivisor( storagemode , &denominator, &divisor );
  framesize = ( xsize * ysize * denominator ) / divisor;
  sizeof3lines = (3 * framesize) / ysize;

  if((storagemode & SV_MODE_NBIT_MASK) == SV_MODE_NBIT_8B) {
    p = &vbuffer[0];
  } else {
    p = &vbuffer[sizeof3lines];
  }

  if(sdtv) {
    lum = lum601;
    rmy = rmy601;
    bmy = bmy601;
  } else {
    lum = lum709;
    rmy = rmy709;
    bmy = bmy709;
  }

  
  for(l = 0; l < 3; l++) {
    for(k = 0; k < 8; k++) {
      switch(storagemode & SV_MODE_COLOR_MASK) {
      case SV_MODE_COLOR_LUMA:
        for(x = 2*k; x < xsize; x += 16) { *p++ = lum[k]; *p++ = lum[k]; }
        break;
      case SV_MODE_COLOR_CHROMA:
        for(x = 2*k; x < xsize; x += 16) { *p++ = bmy[k]; *p++ = rmy[k]; }
        break;
      case SV_MODE_COLOR_YUV422:
        for(x = 2*k; x < xsize; x += 16) { *p++ = bmy[k]; *p++ = lum[k]; *p++ = rmy[k]; *p++ = lum[k]; }
        break;
      case SV_MODE_COLOR_YUV422_YUYV:
        for(x = 2*k; x < xsize; x += 16) { *p++ = lum[k]; *p++ = bmy[k]; *p++ = lum[k]; *p++ = rmy[k]; }
        break;
      case SV_MODE_COLOR_YUV422A:
        for(x = 2*k; x < xsize; x += 16) { *p++ = bmy[k]; *p++ = lum[k]; *p++ = lum[k]; *p++ = rmy[k]; *p++ = lum[k]; *p++ = lum[k]; }
        break;
      case SV_MODE_COLOR_YUV444:
        for(x = k; x < xsize; x += 8) { *p++ = bmy[k]; *p++ = lum[k]; *p++ = rmy[k]; }
        break;
      case SV_MODE_COLOR_YUV444A:
        for(x = k; x < xsize; x += 8) { *p++ = bmy[k]; *p++ = lum[k]; *p++ = rmy[k]; *p++ = lum[k];}
        break;
      case SV_MODE_COLOR_ABGR:
        for(x = k; x < xsize; x += 8) { *p++ = key[k]; *p++ = blu[k]; *p++ = grn[k]; *p++ = red[k]; }
        break;
      case SV_MODE_COLOR_ARGB:
        for(x = k; x < xsize; x += 8) { *p++ = key[k]; *p++ = red[k]; *p++ = grn[k]; *p++ = blu[k]; }
        break;
      case SV_MODE_COLOR_RGB_BGR:
        for(x = k; x < xsize; x += 8) { *p++ = blu[k]; *p++ = grn[k]; *p++ = red[k]; }
        break;
      case SV_MODE_COLOR_BGRA:
        for(x = k; x < xsize; x += 8) { *p++ = blu[k]; *p++ = grn[k]; *p++ = red[k]; *p++ = key[k]; }
        break;
      case SV_MODE_COLOR_RGB_RGB:
        for(x = k; x < xsize; x += 8) { *p++ = red[k]; *p++ = grn[k]; *p++ = blu[k]; }
        break;
      case SV_MODE_COLOR_RGBA:
        for(x = k; x < xsize; x += 8) { *p++ = red[k]; *p++ = grn[k]; *p++ = blu[k]; *p++ = key[k]; }
        break;
      default:
        ProgramError();
        return FALSE;
      }
    }    
  }

  if((storagemode & SV_MODE_NBIT_MASK) != SV_MODE_NBIT_8B) {
    p = &vbuffer[0];
    q = &vbuffer[sizeof3lines];
    switch(storagemode & SV_MODE_NBIT_MASK) {
    case SV_MODE_NBIT_10BDVS:
      for(x = 0; x < sizeof3lines; x+=4) {
        p[0] = q[0];
        p[1] = q[1];
        p[2] = q[2];
        p[3] = 0;
        p+=4;
        q+=3;
      }
      break;
    case SV_MODE_NBIT_10BDPX:
      for(x = 0; x < sizeof3lines; x+=4) {
        p[0] =  q[0];
        p[1] = (q[1]>>2);
        p[2] = (q[1]<<6) | (q[2]>>4);
        p[3] = (q[2]<<4);
        p+=4;
        q+=3;
      }
      break;
    case SV_MODE_NBIT_10B:
      for(x = 0; x < sizeof3lines; x+=4) {
        p[0] =             (q[0]<<2);
        p[1] = (q[0]>>6) | (q[1]<<4);
        p[2] = (q[1]>>4) | (q[2]<<6);
        p[3] = (q[2]>>2);
        p+=4;
        q+=3;
      }
      break;
    case SV_MODE_NBIT_16BBE:
      for(x = 0; x < sizeof3lines; x+=6) {
        p[0] = q[0];
        p[1] = 0;
        p[2] = q[1];
        p[3] = 0;
        p[4] = q[2];
        p[5] = 0;
        p+=6;
        q+=3;
      }
      break;
    case SV_MODE_NBIT_16BLE:
      for(x = 0; x < sizeof3lines; x+=6) {
        p[0] = 0;
        p[1] = q[0];
        p[2] = 0;
        p[3] = q[1];
        p[4] = 0;
        p[5] = q[2];
        p+=6;
        q+=3;
      }
      break;
    default:
      ProgramError();
      return FALSE;
    }
  }

  /*
  // Duplicate all lines
  */
  for(y = 3; y < ysize-2; y+=3) {
    memcpy(&vbuffer[y*sizeof3lines/3], &vbuffer[0], sizeof3lines);
  }
  l = framesize - y*sizeof3lines/3;
  if(l > 0) {
    memcpy(&vbuffer[y*sizeof3lines/3], &vbuffer[0], l);
  }

  if(pframesize) {
    *pframesize = framesize;
  }

  printf("%d %d %d\n", xsize, ysize, framesize);

  return TRUE;
}


///Wait for terminate signals.
/**
* It is only a helper function, so it is not important for the example.
*
\param signum    Current signal.
*/
void signal_handler(int signum)
{
  if(signum); // Disable compiler warning.

  lhd.running = FALSE;
}


///This function closes all dvs handles
/**
\param *hd Pointer to the global loop_handle handle.
*/
int loop_exit(loop_handle * hd)
{
  hd->running = FALSE;

  if(hd->sv != NULL) {
    if(hd->pfifo != NULL) {
      //Cleanup the fifo.
      sv_fifo_free(hd->sv, hd->pfifo);
      hd->pfifo = NULL;
    }
    //Close the handle to the dvsdevice.
    sv_close(hd->sv); 
    hd->sv = NULL;
  }

  if(hd->vbuffer_org) {
    //Free the original pointer and also set the aligned pointer to zero.
    free(hd->vbuffer_org);
    hd->vbuffer_org = NULL;
    hd->vbuffer = NULL;
  }

  return 0;
}


///This function opens all dvs handles
/**
\param *hd Pointer to the global loop_handle handle.
*/
int loop_init(loop_handle * hd)
{
  int res           = SV_OK;
  int currentRaster = 0;

  //Open handle to the dvs video device
  if(res == SV_OK) {
    res = sv_openex(&hd->sv, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_DEFAULT, 0, 0);
    if(res != SV_OK) {
      printf("loop: sv_openex(\"\") failed %d '%s'", res, sv_geterrortext(res));
    }
  } 

  //Get current raster
  if(res == SV_OK) {
  res = sv_option_get( hd->sv, SV_OPTION_VIDEOMODE, &currentRaster);
    if(res != SV_OK) {
      printf("loop: sv_option_get(SV_OPTION_VIDEOMODE) failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  //Set RGBA video mode
  if(res == SV_OK) {
    res = sv_videomode(hd->sv, (currentRaster & SV_MODE_MASK)| SV_MODE_COLOR_RGBA | SV_MODE_NBIT_10B | SV_MODE_STORAGE_FRAME);
    if(res != SV_OK)  {
      printf("sv_videomode() failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  //Get current dvs board configuration
  if(res == SV_OK) {
    res = sv_storage_status(hd->sv, 0, NULL, &hd->storage, sizeof(hd->storage), 0);
    if(res != SV_OK)  {
      printf("sv_storage_status() failed %d '%s'", res, sv_geterrortext(res));
    } else {
      //Save xsize and ysize from the dvs board
      if(!hd->xsize) {
        hd->xsize = hd->storage.storagexsize;
      }
      if(!hd->ysize) {
        hd->ysize = hd->storage.storageysize;
      }
      if(hd->halfsize) {
        hd->ysize /= 2;
        hd->xsize /= 2;
      }
    }
  }

  //Initialize the output fifo.
  if(res == SV_OK) {
    res = sv_fifo_init(hd->sv, &hd->pfifo, FALSE, TRUE, TRUE, FALSE, 0);
    if(res != SV_OK)  {
      printf("sv_fifo_init() failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  //Get dma alignment of fifo
  if(res == SV_OK) {
    res = sv_fifo_configstatus(hd->sv, hd->pfifo, &hd->config);
    if(res != SV_OK)  {
      printf("sv_fifo_configstatus() failed %d '%s'", res, sv_geterrortext(res));
    } else {
      //Allocate video buffer
      hd->vbuffer_org = malloc(8 * hd->storage.xsize * hd->storage.ysize + hd->config.dmaalignment - 1);
      hd->vbuffer = (char *)((uintptr)(hd->vbuffer_org + (hd->config.dmaalignment-1)) & ~(uintptr)(hd->config.dmaalignment-1));
      if(!hd->vbuffer_org) {
        printf("malloc(%d) vbuffer failed", 8 * hd->storage.xsize * hd->storage.ysize);
        res = SV_ERROR_MALLOC;
      }
    }
  }

  //Start the fifo
  if(res ==SV_OK) {
    res = sv_fifo_start(hd->sv, hd->pfifo);
    if(res != SV_OK)  {
      printf("sv_fifo_start() failed %d '%s'", res, sv_geterrortext(res));
    }
  }

  //Cleanup if something was wrong
  if( res != SV_OK ) {
    loop_exit( hd );
  }

  return res;
}


///This is the function which does the major work.
/**
\param *hd Pointer to the global loop_handle handle.
*/
int loop(loop_handle * hd) 
{
  sv_fifo_buffer *pbuffer = 0;

  int res         = SV_OK;
  int storagemode = SV_MODE_COLOR_YUV422;
  int nbitmode    = SV_MODE_NBIT_8B;
  
  int loop          = 0;
  int xsize         = 0;
  int ysize         = 0;
  int xoffset       = 100;
  int yoffset       = -100;
  int dataoffset    = 0;
  int framesize     = 0;
  int divisor       = 1;
  int denominator   = 1;
  int bxsmaller     = TRUE;
  int bdxsmaller    = TRUE;
  int bysmaller     = TRUE;
  int bdysmaller    = TRUE;
  int getbufferflag = 0;
  sv_storageinfo info_in;
  sv_storageinfo info_out;

  //Init dvs device
  res = loop_init( hd );
  if(res != SV_OK)  {
    return res;
  }

  //Create getbuffer flags
  getbufferflag = SV_FIFO_FLAG_STORAGEMODE | SV_FIFO_FLAG_VIDEOONLY;
  if( hd->position ) {
    getbufferflag |= SV_FIFO_FLAG_STORAGENOAUTOCENTER;
  }

  //Set the start xsize and ysize 
  xsize = hd->xsize;
  ysize = hd->ysize;
  bxsmaller  = TRUE;
  bysmaller  = TRUE;
  bdxsmaller = TRUE;
  bdysmaller = TRUE;
  xoffset    = 100;  //-hd->xsize;
  yoffset    = -100; //-hd->ysize;
  
  //The main loop
  while (hd->running)
  {
    //Get fifo buffer
    res = sv_fifo_getbuffer(hd->sv, hd->pfifo, &pbuffer, NULL, getbufferflag);
    if (res != SV_OK)  {
      printf("sv_fifo_getbuffer(dst) failed %d '%s'", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }  

    //Recalculate the next xsize and ysize
    if(hd->xchange) {
      if(bxsmaller) {
        xsize -= STEP_XSIZE;
      } else {
        xsize += STEP_XSIZE;
      }
      if(xsize > hd->xsize) {
        bxsmaller = TRUE;
        xsize = hd->xsize;
      } else if(xsize < STEP_XSIZE) {
        bxsmaller = FALSE;
        xsize     = STEP_XSIZE;
      }
    }

    if(hd->ychange) {
      if(bysmaller) {
        ysize -= STEP_YSIZE;
      } else {
        ysize += STEP_YSIZE;
      }
      if(ysize > hd->ysize) {
        bysmaller = TRUE;
        ysize = hd->ysize;
      } else if(ysize < STEP_YSIZE) {
        bysmaller = FALSE;
        ysize     = STEP_YSIZE;
      }
    }

    //Change storage mode if cchange is activated
    if(hd->cchange) {
      storagemode  = storagemodes[loop%(sizeof(storagemodes)/sizeof(storagemodes[0]))];
    }

    //Change bit mode if bchange is activated
    if(hd->bchange) {
      nbitmode = nbitsmodes[(loop/(sizeof(storagemodes)/sizeof(storagemodes[0])))%(sizeof(nbitsmodes)/sizeof(nbitsmodes[0]))];
    }

    // Get pixelgoup
    res = sv_storage_status( hd->sv, storagemode | nbitmode, &info_in, &info_out, sizeof(info_out), SV_STORAGEINFO_COOKIEISMODE );
    if(res != SV_OK)  {
      printf("sv_storage_status() failed %d '%s'", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    //Adjust the xsize
    getDenomAndDivisor( storagemode | nbitmode, &denominator, &divisor );
    if(hd->xchange) {
      while(xsize % ((info_out.pixelgroup * denominator) / divisor)) {
        if(bxsmaller) {
          xsize -= STEP_XSIZE;
        } else {
          xsize += STEP_XSIZE;
        }
      }
    }

    //Add dataoffset if activated
    if(hd->dataoffset) {
      dataoffset += hd->dataoffset;
      if(dataoffset >= 0x800) {
        dataoffset = 0;
      }
    }

    //Caluclate new position if actived
    if(hd->position) {
      if(bdxsmaller) {
        xoffset-=2;
      } else {
        xoffset+=2;
      }
      if(xoffset < -xsize) {
        bdxsmaller = FALSE;
      } else if(xoffset > xsize) {
        bdxsmaller = TRUE;
      }
      if(bdysmaller) {
        yoffset--;
      } else {
        yoffset++;
      }
      if(yoffset < -ysize) {
        bdysmaller = FALSE;
      } else if(yoffset > ysize) {
        bdysmaller = TRUE;
      }
    } 


    if(hd->sleepy) {
      sv_usleep(hd->sv, 100000);
    }

    //Create new colorbar in buffer for the current bit and colormode
    colorbar((unsigned char *)hd->vbuffer + dataoffset, (hd->storage.xsize<=960), storagemode | nbitmode, xsize, ysize, &framesize);

    //Log
    printf("loop %3d storagemode %08x %dx%d - %dx%d - %d\n", loop, storagemode | nbitmode, xsize, ysize, xoffset, yoffset, framesize);

    //Write dma addresses
    pbuffer->dma.addr = hd->vbuffer;
    pbuffer->dma.size = framesize + 0x800;

    //Enable special parameters for colormode changing
    pbuffer->storage.storagemode  = storagemode | nbitmode | SV_MODE_STORAGE_FRAME;
    pbuffer->storage.xsize        = xsize;
    pbuffer->storage.ysize        = ysize;
    pbuffer->storage.lineoffset   = 0;
    pbuffer->storage.dataoffset   = dataoffset;
    pbuffer->storage.xoffset      = xoffset;
    pbuffer->storage.yoffset      = yoffset;

    //Put the fifobuffer into the fifo
    res = sv_fifo_putbuffer(hd->sv, hd->pfifo, pbuffer, NULL);
    if(res != SV_OK)  {
      printf("sv_fifo_putbuffer() failed %d '%s'", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    loop++;
  }

  //Close dvs device
  loop_exit(hd);
  
  return TRUE;
}


///Dump the help text to the console.
/**
* It is only a helper function, so it is not important for the example.
*
*/
void usage(void)
{
  fprintf(stderr, "SYNTAX:   cmodetst [opts]\n");
  fprintf(stderr, "FUNCTION: Colormode/Size dynamic changing with fifoapi\n");
  fprintf(stderr, "OPTIONS:  -a     Test all.\n");
  fprintf(stderr, "OPTIONS:  -h     Initial size half.\n");
  fprintf(stderr, "OPTIONS:  -o     Test dataoffset changing.\n");
  fprintf(stderr, "OPTIONS:  -s     Test Colormode changing.\n");
  fprintf(stderr, "OPTIONS:  -b     Test Bitmode changing.\n");
  fprintf(stderr, "OPTIONS:  -x     Test X size changing.\n");
  fprintf(stderr, "OPTIONS:  -y     Test Y size changing.\n");
  fprintf(stderr, "OPTIONS:  -z     Test X/Y size changing.\n");
  fprintf(stderr, "OPTIONS:  -w     Sleep 5 seconds between change.\n");
  exit(0);
}


///Parse the command parameters and call main loop.
/**
*/
int main(int argc, char* argv[])
{
  int res;
  int c;
  int value;

  memset(&lhd,0,sizeof(loop_handle));
  lhd.running   = TRUE;

  /*----------------------------------------------------------------------*/
  /* scan command line arguments                                          */
  /*----------------------------------------------------------------------*/
  
  while (--argc) {                                /* scan command line arguments	        */
    argv++;                                       /* skip progamname				*/
    if (**argv == '-') {
      c = toupper(*++*argv);                      /* get option character			*/
      ++*argv;
      if (**argv == '=') {
        ++*argv;
        value = TRUE;
      } else {
        value = FALSE;
      }

      if (**argv == ' ') ++*argv;                 /* skip space					*/
      switch(c) {
      case 'A':
        lhd.bchange = 1;
        lhd.cchange = 1;
        break;
      case 'B':
        lhd.bchange = 1;
        break;
      case 'S':
        lhd.cchange = 1;
        break;
      case 'H':
        lhd.halfsize = 1;
        break;
      case 'W':
        lhd.sleepy = 1;
        break;
      case 'P':
        lhd.position = 1;
        break;
      case 'O':
        if(value) {
          lhd.dataoffset = atoi(*argv);
        } else {
          lhd.dataoffset = 2;
        }
        break;
      case 'X':
        if(value) {
          lhd.xsize = atoi(*argv);
        } else {
          lhd.xchange = 1;
        }
        break;
      case 'Y':
        if(value) {
          lhd.ysize = atoi(*argv);
        } else {
          lhd.ychange = 1;
        }
        break;
      case 'Z':
        lhd.xchange = 1;
        lhd.ychange = 1;
        break;
      default:
        usage();
      }
    } else {
      usage();
    }
  }

  signal(SIGTERM, signal_handler);
  signal(SIGINT,  signal_handler);

  res = loop(&lhd);

  return res;
}
