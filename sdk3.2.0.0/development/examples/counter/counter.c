/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    counter - display a frame counter via FIFO API
//
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <fcntl.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"


#define USE_INTERNALFONT
#define COUNTER_MAXBUFFERS 4
#define VFONT_MAGIC     0x011E
#define swap_short(s) ( (s) = ((((s)&0xFF)<<8)|(((s)>>8)&0xFF)) )

#ifdef USE_INTERNALFONT
#include "cspfnt.h"
#endif

typedef struct vfont {
    short           magic;
    unsigned short  size;
    short           maxx;
    short           maxy;
    short           xtend;
    struct dispatch {
        unsigned short  offs;
        short           nbyte;
        char            up,down,left,right;
        short           width;
    } disp[256];
    char            data[10000];
} VFONT;

typedef struct {
  int   verbose;            //!< Parameter verbose.
  int   xsize;              //!< Current dvs board storage xsize.
  int   ysize;              //!< Current dvs board storage ysize.
  int   psize_num;          //!< Pixel size num.
  int   psize_denom;        //!< Pixel size denom.
  int   lineoffset;         //!< Dvs Board lineoffset.
  int   minsize;            //!< Minimum alignment of the outputraster.
  unsigned char white[16];  //!< White pixel array.
  unsigned char black[16];  //!< Black pixel array.
  unsigned char greyfill[32]; //!< Grey pixel array.
  int   topfield;           //!< Change the field order.
  int   running;            //!< Counter running variable.
  int   height;		        //!< Font height needed for blit.
  int   width;		        //!< Font width need for blit.
  struct {
    int   delay;            //!< Parameter delay.
    int   fieldbased;       //!< Parameter fieldbased mode.
    int   operation;        //!< Parameter operation (count... ).
    int   loops;            //!< Parameter loops.
    int   card;
    int   iochannel;
  } setup;
  struct {
    int   count;
  } hline;
  struct {
    int   count;
  } vline;
  struct {
    int   loaded;           //!< Is true if font file was successfully loaded.
    int   size;             //!< Size of the complete font file.
    int   xzoom;            //!< A value to adjust the font size to the raster xsize.
    int   yzoom;            //!< A value to adjsut the font size to the raster ysize.
    VFONT *vfont;           //!< Pointer to the VFONT struct.
  } font;
  struct {
    int nbuffer;            //!< Current counter buffer id.
    int framesize;          //!< Size of one frame from the outputraster.
    struct {
      unsigned char * buffer;     //!< Pointer to an aligned video buffer.
      unsigned char * buffer_org; //!< Pointer to the original video pointer, need for free.
    } table[COUNTER_MAXBUFFERS];  //!< Ringbuffer array.
  } buffers;
  sv_storageinfo storage;  //!< Dvs structure about the storage settings.
  unsigned char *buffer;   //!< Pointer to a videobuffer with the size of one frame.
  int loop;                //!< Current loop.
} count_handle;  //!< Global count handle.


count_handle global_counter; //!< Global counter handle
#define OPERATION_NONE   0
#define OPERATION_RENDER 1
#define OPERATION_BLIT   2
#define OPERATION_COUNT  3



int hex2tc(int value) 
{
  return  0x10000000 * ((value / 1000000) % 10) |
          0x01000000 * ((value / 100000) % 10) |
          0x00100000 * ((value / 10000) % 10) |
          0x00001000 * ((value / 1000) % 10) |
          0x00000100 * ((value / 100) % 10) |
          0x00000010 * ((value / 10) % 10) |
          0x00000001 * ((value % 10));
}

///Blit the counter into the picture.
/**
* It is only a helper function, so it is not important for the example.
*
\param *handle  Pointer to the global counter handle.
\param *pbuffer Pointer the the fifo buffer struct.
*/
int operation_blit(count_handle * handle, sv_fifo_buffer * pbuffer)
{
    int y;

#if 0
    /*
    ** "double scan": both fields contain the _same_ image data,
    ** thus effectively cutting vertical resolution in half!
    ** (this has been left in just as an example)
    */
    memcpy(pbuffer->video[0].buffer,
                handle->buffer,
                handle->ysize/2*handle->xsize*handle->psize_num/handle->psize_denom);

    memcpy(pbuffer->video[1].buffer,
                handle->buffer,
                handle->ysize/2*handle->xsize*handle->psize_num/handle->psize_denom);
    return TRUE;
#endif

    /*
    ** Note that source buffer is progressive while destination
    ** buffer (onboard CSP memory) is split into fields!
    */

    if(handle->storage.interlace == 1) {
      for(y = 0; y < handle->height; y++) {
         memcpy((char *)pbuffer->dma.addr + handle->storage.fieldoffset[0] + y*handle->storage.lineoffset[0],
                handle->buffer + y*handle->xsize*handle->psize_num/handle->psize_denom,
                handle->width*handle->psize_num/handle->psize_denom);
      }
    } else {
      for(y = 0; y < handle->height; y++) {
         memcpy((char *)pbuffer->dma.addr + handle->storage.fieldoffset[(y^handle->topfield)&1] + y/2*handle->storage.lineoffset[0],
                handle->buffer + y*handle->xsize*handle->psize_num/handle->psize_denom,
                handle->width*handle->psize_num/handle->psize_denom);
      }
    }
    return TRUE;
}

///Put the fonts into the real videobuffer
/**
* 
*
\param *handle  Pointer to the global counter handle.
\param *pbuffer Pointer the the fifo buffer struct.
*/
int operation_render(count_handle * handle, sv_fifo_buffer * pbuffer)
{
    int           i, len, offs, width, height, xz, yz, h, w;
    short         *addr = (short*)&handle->buffer[0];
    unsigned char *caddr = &handle->buffer[0];
    char           string[128];
    VFONT         *fnt = handle->font.vfont;

    sprintf((char*)string,"%07d",handle->loop);
    string[7] = '\0';  /* limit length */
    len = (int)strlen(string);

    for( i=0; i<len; ++i ) {
        if( fnt->disp[string[i]&0xFF].nbyte==0 ) {  /* unix vfont format    */
            continue;                          /* char not defined     */
        }

        offs   = fnt->disp[string[i]&0xFF].offs;
        width  = fnt->disp[string[i]&0xFF].width;
        height = fnt->disp[string[i]&0xFF].nbyte/2;

        handle->width = width * handle->font.xzoom * len * handle->minsize * handle->psize_denom / handle->psize_num   ;
        handle->height = height * handle->font.yzoom;

        if( handle->minsize == 2 ) {
            int blackdata = (handle->black[1] << 8) | handle->black[0];
            int whitedata = (handle->white[1] << 8) | handle->white[0];
            for( h=0; h<height; h++ ) {  /* font height loop     */
                int  bitmap = ((unsigned short*)&fnt->data[offs])[h];
                int  bitmsk = 1<<(width-1);
                for( w=0; w<width; bitmsk>>=1, ++w ) {
                    short pixel = (bitmap&bitmsk) ? whitedata : blackdata;
                    for( yz=0; yz<handle->font.yzoom; ++yz ) {
                        for( xz=0; xz<handle->font.xzoom; ++xz ) {
                            addr[w*handle->font.xzoom + yz*handle->xsize + xz] = pixel;
                        }
                    }
                }
                addr += handle->font.yzoom * handle->xsize;
            }
            addr -= height*handle->font.yzoom * handle->xsize
                    -width*handle->font.xzoom;
        } else {
            for( h=0; h<height; h++ ) {  /* font height loop     */
                int  bitmap = ((unsigned short*)&fnt->data[offs])[h];
                int  bitmsk = 1<<(width-1);
                for( w=0; w<width; bitmsk>>=1, ++w ) {
                    unsigned char *ppixel = ((bitmap&bitmsk) ? &handle->white[0] : &handle->black[0]);
                    for( yz=0; yz<handle->font.yzoom; ++yz ) {
                        for( xz=0; xz<handle->font.xzoom; ++xz ) {
                            int cz;
                            for( cz=0; cz<handle->minsize; ++cz ) {
                                caddr[yz*handle->lineoffset + (w*handle->font.xzoom + xz)*handle->minsize + cz] = ppixel[cz];
                            }
                        }
                    }
                }
                caddr += handle->font.yzoom * handle->lineoffset;
            }
            caddr -= height * handle->font.yzoom * handle->lineoffset;
            caddr += (width * handle->font.xzoom) * handle->minsize;
        }
    }
    return TRUE;
}


///Call first render and then blit.
/**
* It is only a helper function, so it is not important for the example.
*
\param *handle  Pointer to the global counter handle.
\param *pbuffer Pointer the the fifo buffer struct.
*/
int operation_count(count_handle * handle, sv_fifo_buffer * pbuffer)
{
    return operation_render(handle,pbuffer)
         ? operation_blit(handle,pbuffer)
         : FALSE;
}


///Blit the counter into the picture.
/**
*
*
\param *handle  Pointer to the global counter handle.
\param *pbuffer Pointer the the fifo buffer struct.
*/
int count_paint_thread(void * voidhandle)
{
    count_handle *    handle = voidhandle;
    sv_fifo_buffer *  pbuffer;
    sv_fifo *         poutput;
    sv_fifo_info      finfo;
    sv_handle *       sv;
    int               res;
    int               running = TRUE;
    int               dropped_count = 0;
    int               fifoflags     = SV_FIFO_FLAG_FLUSH;
    char card[128]    = "";

    if(handle->setup.fieldbased) {
      fifoflags |= SV_FIFO_FLAG_FIELD;
    }

    if(handle->setup.card != -1) {
      if(handle->setup.iochannel != -1) {
        sprintf(card, "PCI,card=%d,channel=%d", handle->setup.card, handle->setup.iochannel);
      } else {
        sprintf(card, "PCI,card=%d", handle->setup.card);
      }
    } else {
      card[0] = '\0';
    }

    /**
    **    Open the device, for the direct access dll the syntax is
    **    "PCI,card:0" for the first card (this is the default) and 
    **    "PCI,card:n" for the next card in the system.
    */  
    res = sv_openex(&sv, card, SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_OUTPUT, 0, 0);
    if(res != SV_OK) {
      printf("paint_thread: sv_openex(\"\") failed %d '%s'", res, sv_geterrortext(res));
      return FALSE;
    } 

    res = sv_storage_status(sv, 0, NULL, &handle->storage, sizeof(handle->storage), SV_STORAGEINFO_COOKIEISJACK);
    if(res != SV_OK) {
      printf("paint_thread: sv_storage_status(sv, ...) failed = %d '%s'\n", res, sv_geterrortext(res));
      return FALSE;
    }
    handle->buffers.framesize = handle->storage.fieldsize[0] + handle->storage.fieldsize[1] + 0x2000;




    /*
    **  Initialize the output fifo.
    */
    res = sv_fifo_init(sv, &poutput, FALSE, FALSE, TRUE, FALSE, 0);

    if(res != SV_OK)  {
      printf("paint_thread: sv_fifo_init() failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    } else {
      char * p;
      int i,j,k;
      for(k = 0; k < COUNTER_MAXBUFFERS; k++) {
        if((handle->buffers.table[k].buffer_org = malloc(handle->buffers.framesize + 0x40)) == NULL) {
          running = FALSE;
        } else {
          handle->buffers.table[k].buffer = (void*)(((uintptr)handle->buffers.table[k].buffer_org + 0x3f) & ~0x3f);
        }

        for(p = (char*)handle->buffers.table[k].buffer, i = 0; i < handle->buffers.framesize; i+=handle->minsize) {
          for(j = 0; j < handle->minsize; j++) { 
            *p++ = handle->greyfill[j];
          }
        }
      }

      /*
      **  Reset the position of the output fifo.
      */
      res = sv_fifo_start(sv, poutput);
      if(res != SV_OK)  {
        printf("paint_thread: sv_fifo_start() failed = %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
      }
    }

    while(running && handle->running) {

        /**
        //  Fetch the buffer to the next frame to be filled in.
        //
        */
        res = sv_fifo_getbuffer(sv, poutput, &pbuffer, NULL, fifoflags); fifoflags &= ~SV_FIFO_FLAG_FLUSH;
        if(res != SV_OK)  {
            printf("paint_thread: sv_fifo_getbuffer() failed = %d '%s'\n", res, sv_geterrortext(res));
            break;
        }

        pbuffer->dma.addr = (char*)handle->buffers.table[handle->buffers.nbuffer].buffer;
        pbuffer->dma.size = handle->buffers.framesize;

        /*
        **  Here the operation on the displayed video is done, 
        **  Replace this with your own code.
        */
        switch(handle->setup.operation) {
        case OPERATION_NONE:
            ;  /* do nothing */
            break;
        case OPERATION_RENDER:
            running = operation_render(handle, pbuffer);
            break;
        case OPERATION_BLIT:
            running = operation_blit(handle, pbuffer);
            break;
        case OPERATION_COUNT:
            running = operation_count(handle, pbuffer);
            break;
        default:
            printf("paint_thread: operation == %d not implemented\n", handle->setup.operation);
            running = FALSE;
        }

        if( !running ) {
            break;
        }

        pbuffer->timecode.ltc_tc         = hex2tc(handle->loop);
        pbuffer->timecode.ltc_ub         = handle->loop;
        pbuffer->timecode.vitc_tc        = hex2tc(handle->loop);
        pbuffer->timecode.vitc_ub        = handle->loop;
        pbuffer->timecode.vitc_tc2       = handle->loop;
        pbuffer->timecode.vitc_ub2       = handle->loop;
        pbuffer->anctimecode.dltc_tc     = hex2tc(handle->loop);
        pbuffer->anctimecode.dltc_ub     = handle->loop;
        pbuffer->anctimecode.dvitc_tc[0] = hex2tc(handle->loop);
        pbuffer->anctimecode.dvitc_ub[0] = handle->loop;
        pbuffer->anctimecode.dvitc_tc[1] = hex2tc(handle->loop);
        pbuffer->anctimecode.dvitc_ub[1] = handle->loop;

        /*
        **  Release the buffer to be queued for display
        */
        res = sv_fifo_putbuffer(sv, poutput, pbuffer, NULL);
        if(res != SV_OK)  {
            printf("paint_thread: sv_fifo_putbuffer() failed = %d '%s'\n", res, sv_geterrortext(res));
            break;
        }

        if(++handle->buffers.nbuffer >= COUNTER_MAXBUFFERS) {
          handle->buffers.nbuffer = 0;
        }

        res = sv_fifo_status(sv, poutput, &finfo);
        if(res != SV_OK) {
            printf("paint_thread: sv_fifo_status() res=%d '%s'\n", res, sv_geterrortext(res));
            running = FALSE; 
        }
        if( handle->verbose || finfo.dropped>dropped_count ) {
            printf("o: avail %02d drop %02d  nbuf %02d tick %06d\n", finfo.availbuffers, finfo.dropped, finfo.nbuffers, finfo.tick);	/**/
            putchar('.');
            fflush(stdout);
            dropped_count = finfo.dropped;
        }

        ++handle->loop;
        if((handle->setup.loops > 0) && (handle->loop > handle->setup.loops)) {
            if( running && (handle->verbose || finfo.dropped>0) ) {
                printf("o: avail %02d drop %02d tick %06d\n", finfo.availbuffers, finfo.dropped, finfo.tick);	/**/
            }
            break;
        }
        
    }

    if(handle->verbose) {
      printf("\n\n");
    }

    for(handle->loop = 1; handle->loop < handle->verbose; handle->loop++) {
        res = sv_fifo_status(sv, poutput, &finfo);
        if(res != SV_OK) {
            printf("paint_thread: sv_fifo_status() res=%d '%s'\n", res, sv_geterrortext(res));
            running = FALSE; 
        }
        if(handle->verbose || finfo.dropped>dropped_count ) {
            printf("o: avail %02d drop %02d  nbuf %02d tick %06d\n", finfo.availbuffers, finfo.dropped, finfo.nbuffers, finfo.tick);	/**/
            putchar('.');
            fflush(stdout);
            dropped_count = finfo.dropped;
        }
        res = sv_fifo_vsyncwait(sv, poutput);
        if(res != SV_OK) {
            printf("paint_thread: sv_fifo_vsyncwait() res=%d '%s'\n", res, sv_geterrortext(res));
            running = FALSE; 
        }
    }

    /*
    **  Close the output fifo
    */
    if(poutput) {
	res = sv_fifo_wait(sv, poutput);
        if(res != SV_OK)  {
            printf("paint_thread: sv_fifo_wait() failed = %d '%s'\n", res, sv_geterrortext(res));
        }

        res = sv_fifo_free(sv, poutput);
        if(res != SV_OK)  {
            printf("paint_thread: sv_fifo_free() failed = %d '%s'\n", res, sv_geterrortext(res));
        }
    }

    if(sv) {
      int i;
      for(i = 0; i < COUNTER_MAXBUFFERS; i++) {
        if(handle->buffers.table[i].buffer_org) {
          free(handle->buffers.table[i].buffer_org);
        }
      }

      /*
      **  Close the sv handle
      */
      res = sv_close(sv);
      if(res != SV_OK) {
          printf("paint_thread: sv_close() failed = %d '%s'\n", res, sv_geterrortext(res));
      }
    }

    handle->running = FALSE;

    return 0;
}


/// Cleanup the video and the font buffer
/**
* It is only a helper function, so it is not important for the example.
*
\param *handle  Pointer to the global counter handle.
*/
void count_exit(count_handle * handle)
{  

    handle->running = FALSE;
    if (handle->buffer) {
      free(handle->buffer);
      handle->buffer = NULL;
    }
    if( handle->font.vfont ) {
        free(handle->font.vfont);
        handle->font.vfont = NULL;
    }
    handle->font.size = 0;
}


/// Convert the videobuffer to another bit format
/**
* It is only a helper function, so it is not important for the example.
*
\param *pstorage   Pointer to the sv_storageinfo struct.
\param *q          Pointer to source video buffer.
\param inputsize   Size of the input video buffer.
\param *p          Pointer to target video buffer.
\param *outputsize Size of the output video buffer.
*/
void convertbuffer(sv_storageinfo * pstorage, unsigned char * q, int inputsize, unsigned char * p, int * outputsize)
{
#ifdef macintosh
  unsigned char * porg = p;
  int tmp;
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
      printf("Error unknown 10 bit mode\n");
      exit(-1);
    }
    inputsize = 4 * inputsize / 3;
  } else {
    memcpy(p, q, inputsize);
  }

#if defined(macintosh)
  for(p = porg, x = inputsize; x < 4; x++) {
    p[x] = p[x-inputsize];
  }
  for(p = porg, x = 0; x < inputsize; x+=4,p+=4) {
    tmp = p[0]; p[0] = p[3]; p[3] = tmp;
    tmp = p[1]; p[1] = p[2]; p[2] = tmp;
  }
#endif

  if(outputsize) {
    *outputsize = inputsize;
  }
}


/// Init the video and the font buffer
/**
* It is only a helper function, so it is not important for the example.
*
\param *handle  Pointer to the global counter handle.
*/
int count_init(count_handle * handle) 
{
    sv_handle *sv;
    sv_storageinfo storage;
    int       res;
    int	      psize;
    int       j;  
    int       i;
    int       xzoom;
    unsigned char black[16];
    unsigned char white[16];
    unsigned char grey[16];
    unsigned char * p;
    char card[128] = "";

    if(handle->setup.card != -1) {
      if(handle->setup.iochannel != -1) {
        sprintf(card, "PCI,card=%d,channel=%d", handle->setup.card, handle->setup.iochannel);
      } else {
        sprintf(card, "PCI,card=%d", handle->setup.card);
      }
    } else {
      card[0] = '\0';
    }

    res = sv_openex(&sv, card, SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_OUTPUT, 0, 0);
    if(res != SV_OK) {
      printf("init: sv_openex(\"%s\") failed %d '%s'", card, res, sv_geterrortext(res));
      return FALSE;
    } 
   

    res = sv_storage_status(sv, 0, NULL, &storage, sizeof(storage), SV_STORAGEINFO_COOKIEISJACK);
    if(res != SV_OK) {
      printf("count_init: sv_storage_status() failed = %d '%s'\n", res, sv_geterrortext(res));
      sv_close(sv);
      return FALSE;
    }

    sv_close(sv);

    handle->xsize = storage.storagexsize;
    handle->ysize = storage.storageysize;
    if(!handle->xsize || !handle->ysize) {
      printf("There is no valid storage configuration available.\n");
      printf("Please check \"svram memsetup help\".\n");
      return FALSE;
    }
    if(handle->setup.fieldbased) {
      handle->ysize /= 2;
    }
    printf("%dx%d bits:%d\n", handle->xsize, handle->ysize, storage.nbits);
    switch(storage.colormode) {
    case SV_COLORMODE_MONO:
    case SV_COLORMODE_CHROMA:
        psize = storage.nbits==8 ? 1 : 3;
        memset(&white, 0xEB, sizeof(white));
        memset(&black, 0x10, sizeof(black));
        memset(&grey,  0x80, sizeof(grey));
        xzoom = (storage.nbits==8)?5:2;
        break;
    case SV_COLORMODE_YUV2QT:
        psize = storage.nbits==8 ? 2 : 6;
        for(i = 0; i < sizeof(grey); i+=2) {
          white[i  ] = 0xeb;
          white[i+1] = 0x00;
          black[i  ] = 0x10;
          black[i+1] = 0x00;
          grey [i  ] = 0x80;
          grey [i+1] = 0x00;
        }
        xzoom = (storage.nbits==8)?4:2;
        break;
    case SV_COLORMODE_YUV422:
        psize = storage.nbits==8 ? 2 : 6;
        for(i = 0; i < sizeof(grey); i+=2) {
          white[i  ] = 0x80;
          white[i+1] = 0xEB;
          black[i  ] = 0x80;
          black[i+1] = 0x10;
          grey [i  ] = 0x80;
          grey [i+1] = 0x80;
        }
        xzoom = (storage.nbits==8)?4:2;
        break;
    case SV_COLORMODE_RGB_BGR:
    case SV_COLORMODE_RGB_RGB:
        psize = 3;
        memset(&white, 0xEB, sizeof(white));
        memset(&black, 0x10, sizeof(black));
        memset(&grey,  0x80, sizeof(grey));
        xzoom = (storage.nbits==8)?5:4;
        break;
    case SV_COLORMODE_YUV422A:
    case SV_COLORMODE_YUV444:
        psize = 3;
        for(i = 0; i < sizeof(grey)-2; i+=3) {
          white[i  ] = 0x80;
          white[i+1] = 0xEB;
          white[i+2] = 0x80;
          black[i  ] = 0x80;
          black[i+1] = 0x10;
          black[i+2] = 0x80;
        }
        memset(&grey,  0x80, sizeof(grey));
        xzoom = (storage.nbits==8)?5:4;
        break;
    case SV_COLORMODE_ABGR:
    case SV_COLORMODE_ARGB:
    case SV_COLORMODE_BGRA:
    case SV_COLORMODE_RGBA:
        psize = storage.nbits==8 ? 4 : 12; // 4 * 4 / 3
        memset(&white, 0xEB, sizeof(white));
        memset(&black, 0x10, sizeof(black));
        memset(&grey,  0x80, sizeof(grey));
        xzoom = (storage.nbits==8)?5:2;
        break;
    case SV_COLORMODE_YUV444A:
        psize = storage.nbits==8 ? 4 : 12; // 4 * 4 / 3
        for(i = 0; i < sizeof(grey)-3; i+=4) {
          white[i  ] = 0x80;
          white[i+1] = 0xEB;
          white[i+2] = 0x80;
          white[i+3] = 0xEB;
          black[i  ] = 0x80;
          black[i+1] = 0x10;
          black[i+2] = 0x80;
          black[i+3] = 0x10;
        }
        memset(&grey,  0x80, sizeof(grey));
        xzoom = (storage.nbits==8)?5:2;
        break;
    default:
        printf("Colormode mode %d not supported\n", storage.colormode);
        return FALSE;
    }

    

    convertbuffer(&storage, &grey[0],  psize, &handle->greyfill[0], NULL);
    convertbuffer(&storage, &white[0], psize, &handle->white[0], NULL);
    convertbuffer(&storage, &black[0], psize, &handle->black[0], &handle->minsize);

    handle->psize_num   = storage.pixelsize_num;
    handle->psize_denom = storage.pixelsize_denom;
    handle->lineoffset  = storage.lineoffset[0];
    if(handle->lineoffset < 0) {
      handle->lineoffset  = -handle->lineoffset ;
    }

    handle->buffer = malloc(storage.buffersize);
    if (!handle->buffer) {
      perror("malloc()");
      printf("sorry, can't allocate buffer\n");
      return FALSE;
    }
    for(p = handle->buffer, i = 0; i < storage.buffersize; i+=handle->minsize) {
      for(j = 0; j < handle->minsize; j++) { 
        *p++ = handle->greyfill[j];
      }
    }
    handle->running = TRUE;
    handle->topfield = storage.dominance21;
      
    if(!handle->font.loaded) {

#ifndef USE_INTERNALFONT
      int hfont = open("csp.fnt",O_RDONLY|O_BINARY);


        /*
        **  Hard coded to image size 120x64 and the offset in the bmp file,
        **  the file was created with pbrush in 8 bit mode.
        */
        if( hfont<0 ) {
            perror("open()");
            printf("sorry, can't open font file \"csp.fnt\"!\n");
            return FALSE;
        }
        handle->font.size = lseek(hfont,0,SEEK_END);
        if( handle->font.size<=0 ) {
            perror("lseek()");
            printf("sorry, font file seems to be empty!\n");
            close(hfont);
            return FALSE;
        }
        handle->font.vfont = malloc(handle->font.size);
        if( !handle->font.vfont ) {
            perror("malloc()");
            printf("sorry, can't allocate buffer for font file!\n");
            close(hfont);
            return FALSE;
        }
        lseek(hfont,0,SEEK_SET);
        res = read(hfont,handle->font.vfont,handle->font.size);
        if( res!=handle->font.size ) {
            perror("read()");
            printf("sorry, can't read font file!\n");
            close(hfont);
            return FALSE;
        }
        close(hfont);
#else
        handle->font.size = sizeof(cspfnt_rec);
        handle->font.vfont = malloc(handle->font.size);
        if( !handle->font.vfont ) {
            perror("malloc()");
            printf("sorry, can't allocate buffer for font file!\n");
            return FALSE;
        }
        memcpy(handle->font.vfont, cspfnt_rec, sizeof(cspfnt_rec));
#endif
        if( handle->font.vfont->magic==0x1E01 ) {
            int k;

            swap_short(handle->font.vfont->magic);
            swap_short(handle->font.vfont->size);
            swap_short(handle->font.vfont->maxx);
            swap_short(handle->font.vfont->maxy);
            swap_short(handle->font.vfont->xtend);
            for( k=0; k<256; ++k  ) {
                 swap_short(handle->font.vfont->disp[k].offs);
                 swap_short(handle->font.vfont->disp[k].nbyte);
                 swap_short(handle->font.vfont->disp[k].width);
            }
            for( k=0; k<(int)(handle->font.size-offsetof(VFONT,data))/2; ++k ) {
                 swap_short(((short*)&handle->font.vfont->data)[k]);
            }
        }
        if( handle->font.vfont->magic!=VFONT_MAGIC ) {
            printf("sorry, wrong magic number in font file!\n");
            return FALSE;
        }

        /* we expect at most one line of six characters */
        handle->font.xzoom = handle->xsize/(7*handle->font.vfont->maxx) * 2 / psize;
        handle->font.yzoom = handle->ysize/(1*handle->font.vfont->maxy);
        handle->font.xzoom = xzoom;
        handle->font.yzoom = 5;

        handle->font.loaded = TRUE;
    }

    return TRUE;
}


/// Main function
/**
* The main function will  parse the parameter list first and then start the thread.
*
\param argc   Parameter count
\param **argv Pointer to the parameter list
*/
int main(int argc, char ** argv)
{ 
  count_handle * handle = NULL;
  int error = FALSE;

  handle = &global_counter;
  memset(handle, 0, sizeof(count_handle));

  handle->setup.delay = 3;
  handle->setup.card  = -1;
  handle->setup.iochannel = -1;

  handle->setup.operation = OPERATION_COUNT;

  while((argc > 1) && (argv[1][0] == '-')) {
    switch(argv[1][1]) {
    case 'd':
      if(argv[1][2] == '=') {
        handle->setup.delay = atoi(&argv[1][3]);
      } else {
        error = TRUE;
      }
      break;
    case 'f':
      handle->setup.fieldbased = TRUE;
      break;
    case 'l':
      if(argv[1][2] == '=') {
        handle->setup.loops = atoi(&argv[1][3]);
      } else {
        error = TRUE;
      }
      break;
    case 'c':
      if(argv[1][2] == '=') {
        handle->setup.card = atoi(&argv[1][3]);
      } else {
        error = TRUE;
      }
      break;
    case 'i':
      if(argv[1][2] == '=') {
        handle->setup.iochannel = atoi(&argv[1][3]);
      } else {
        error = TRUE;
      }
      break;
    case 'v':
      handle->verbose = TRUE;
      break;
    default:
      error = TRUE;
    }
    argv++; argc--;
  }

  if((handle->setup.iochannel != -1) && (handle->setup.card == -1)) {
    handle->setup.card = 0;
  }

  if(error) {
    printf("usage: %s <operation>\n",argv[0]);
    printf("<operation>: 0..3 (none|render|blit|count)\n");
    printf("\t-d=#\tDelay\n");
    printf("\t-f\tField-based\n");
    printf("\t-v\t\tVerbose\n");
    printf("\t-l=#\tNumber of loops\n");
    printf("\t-c=#\tCard\n");
    printf("\t-i=#\tIoChannel\n");
    exit(TRUE);
  }

  if(argc > 1) {
    handle->setup.operation = atoi(argv[1]);
  }

  if(count_init(handle)) {
    count_paint_thread(handle);
  }
  count_exit(handle);

  return FALSE;
}

