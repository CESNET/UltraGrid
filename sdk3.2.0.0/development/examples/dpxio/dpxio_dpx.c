/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the fifoapi to do display and record 
//            of images directly to a file.
//
*/

#include "dpxio.h"

#include "../common/dpxformat.h"

int dpxio_getstoragemode(dpxio_handle * dpxio, int dpxtype, int nbits)
{
  return dpxformat_getstoragemode(dpxtype, nbits);
}

int dpxio_framesize(dpxio_handle * dpxio, int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding)
{
  return dpxformat_framesize(xsize, ysize, dpxmode, nbits, poffset, ppadding);
}

int dpxio_sequence_verify(dpxio_handle * dpxio, char * filename, int firstframe, int lastframe)
{
  return TRUE;
}

int dpxio_video_readframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits)
{
  char filename[MAX_PATHNAME];

  sprintf(filename, dpxio->videoio.filename, dpxio->videoio.framenr + framenr);

  if(dpxio->io.dryrun) {
    if(dpxio->verbose) {
      printf("read video frame '%s'\n", filename);
    }
    return *offset; /* Simulate dummy dma transfer */
  }

  return dpxformat_readframe(filename, buffer, buffersize, offset, xsize, ysize, dpxtype, nbits);
}


int dpxio_video_writeframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int xsize, int ysize, int dpxtype, int nbits, int offset, int padding, int timecode)
{
  char filename[MAX_PATHNAME];

  sprintf(filename, dpxio->videoio.filename, dpxio->videoio.framenr + framenr);

  if(dpxio->io.dryrun) {
    if(dpxio->verbose) {
      printf("write video frame '%s'\n", filename);
    }
    return 0x1000; /* Simulate dummy dma transfer */
  }

  return dpxformat_writeframe(filename, buffer, buffersize, offset, xsize, ysize, dpxtype, nbits, padding, timecode);
}

