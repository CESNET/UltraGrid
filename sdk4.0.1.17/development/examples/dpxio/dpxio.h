/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the fifoapi to do display and record of images
//            directly to a file.
//
*/


#if defined(WIN32) || defined(macintosh)
#  define O_LARGEFILE 0
#else
#  define _LARGEFILE64_SOURCE
#endif

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
#define DPXIO_BUFFERVIDEOSIZE   (4096 * 3112 * 4)
#define DPXIO_BUFFERAUDIOSIZE   (4 * 16 * 96000 / 24)


#define AUDIO_FORMAT_INVALID    0
#define AUDIO_FORMAT_AIFF       1
#define AUDIO_FORMAT_WAVE       2
#define AUDIO_FORMAT_MXF        3


#define VIDEO_FORMAT_INVALID    0
#define VIDEO_FORMAT_DPX        1
#define VIDEO_FORMAT_JP2        2
#define VIDEO_FORMAT_MXF        3




/**************************** DEFINES ***************************/




#if defined(WIN32) || defined(linux)
#define fromle(x)                    (x)
#else
#define fromle(x)                    ((((x)&0xff000000)>>24)|(((x)&0x00ff0000)>>8)|(((x)&0x0000ff00)<<8)|(((x)&0x000000ff)<<24))
#endif

/****************************** AUDIO *****************************/

#define DPXIO_AUDIO_HEADER_SIZE  62 // exact size of header data



/*************************** GENERIC ****************************/


typedef struct {
  sv_handle * sv;

  int         verbose;
  int         baudio;
  int         binput;
  int         bvideo;

  int         videoformat;

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
    int    xsize;
    int    ysize;
    int    decrypt;
  } videoio;

  struct {
    char   filename[MAX_PATH];
    void * handle;
    uint64 position;
    int    format;
    int    decrypt;
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
    int key_base;
    int matrixtype;
    struct {
      int decrypt;
      char key[512];
    } keys[2];
  } config;

  int pulldown;
  int pulldownphase;
  int audiochannel;

  struct {
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

    int     filebuffersize;
    void *  filebuffer;

    void *  tmpbuffer;
    void *  tmpbuffer_org;
    int     tmpoffset;
    int     tmpremain;
  } audio;

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
