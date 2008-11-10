/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the fifoapi to do display and record of images
//            directly to a file.
//
*/


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>

#ifdef WIN32
#  include <windows.h>
#  include <io.h>
#  pragma warning ( disable : 4100 )
#  pragma warning ( disable : 4244 )
#  pragma warning ( disable : 4127 )
#  pragma warning ( disable : 4702 )
#else
#  ifdef linux
#    define __USE_LARGEFILE64
#  endif
#  include <unistd.h>
#  define MAX_PATH	1024
#endif

#if !defined(WIN32)
#  define ERROR_IO_PENDING              EAGAIN
#  define QueryPerformanceCounter(x)
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"

#ifndef O_BINARY 
#  define O_BINARY 0
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

/*
// dpxio Mode 
*/

#define DPXIO_MAX_TRACE      1000

#define MAX_PATHNAME    1024

#define DPXIO_MAXBUFFER         2
#define DPXIO_BUFFERVIDEOSIZE   (2048 * 1556 * 4)
#define DPXIO_BUFFERAUDIOSIZE   (2 * 4 * 16 * 48000 / 24 ) // 2 x for security


#define AUDIO_FORMAT_INVALID    0
#define AUDIO_FORMAT_AIFF       1
#define AUDIO_FORMAT_WAVE       2


typedef struct {
  FILE *  fp;
  int     binput;
  int     format;

  int     frequency;
  int     nbits;
  int     nchannels;
  int     bsigned;
  int     blittleendian;

  int     nframes;      // Number of frames
  int     nsample;      // Number of Samples
  int     framesize;    // Frame Size
  int     hwchannels;

  int     position;

  int     filebuffersize;
  void *  filebuffer;
} audio_handle;


/**************************** DEFINES ***************************/




#if defined(WIN32) || defined(linux)
#define fromle(x)                    (x)
#else
#define fromle(x)                    ((((x)&0xff000000)>>24)|(((x)&0x00ff0000)>>8)|(((x)&0x0000ff00)<<8)|(((x)&0x000000ff)<<24))
#endif


/****************************** AUDIO *****************************/

#define DPXIO_AUDIO_HEADER_SIZE  512   //Should be sector aligned



/*************************** GENERIC ****************************/


typedef struct {
  sv_handle * sv;

  int         verbose;
  int         baudio;
  int         binput;
  int         bvideo;

  struct {
    int       audiochannels;
    int       devtype;
    int       dmaalignment;
    int       validtimecode;
  } hw;
  
  struct {
    int       dryrun;
  } io;


  struct {
    char   filename[MAX_PATH];
    int    framenr;
    char * allocpointer[DPXIO_MAXBUFFER];
    char * pointer[DPXIO_MAXBUFFER];
    void * handle;
    uint64 position;
  } videoio;

  struct {
    char    filename[MAX_PATH];
    int     position;
    int     format;
  } audioio;

  
  struct {
    int nopreroll;      //  Do not preroll the buffers for display
    int loopmode;       //  Enable loopmode
    int lut;            //  Enable lut toggle
    int repeatfactor;   //  Set frame repeat factor
    int rp215;          //  Use the user anc embedder to transmit rp215 filmcode
    int rp215a;         //  Use the user anc embedder to transmit rp215a videocode
    int rp215alength;
    int verbosetc;      //  Display verbose timecode information
    int verbosetiming;  //  Display verbose timing information
    int vtrcontrol;     //  Enable vtrcontrol
    int verifytimecode;
    int ccverbose;
    int cctext;
    int tracelog;
    int fieldbased;
    int setat;
    int ancdump;
    int ancfill;
    int anccount;
    int ancstream;
  } config;

  int pulldown;
  int pulldownphase;
  int audiochannel;

  audio_handle  audio;

  struct {
    int   tc;
    int   nframes;
  } vtr;

  struct {
    int           tick;
    int           nbuffers;
    int           dropped;
    int           framesize;
#ifdef WIN32
    LARGE_INTEGER start;
    LARGE_INTEGER getbuffer;
    LARGE_INTEGER dpxio;
    LARGE_INTEGER putbuffer;
    LARGE_INTEGER finished;
#endif
  } trace[DPXIO_MAX_TRACE];
  
} dpxio_handle;

#define ProgramError()      printf("ERROR: program error %s/%d\n", __FILE__, __LINE__); running = FALSE;

#include "tags.h"
#include "../common/fileops.h"
