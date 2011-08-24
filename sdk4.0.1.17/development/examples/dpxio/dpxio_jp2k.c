/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the fifoapi to do display and record 
//            of images directly to a file.
//
*/

#include "dpxio.h"


int dpxio_jp2_getstoragemode(dpxio_handle * dpxio, int dpxtype, int nbits)
{
  return SV_MODE_STORAGE_FRAME;
}

int dpxio_jp2_framesize(dpxio_handle * dpxio, int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding)
{
  if(poffset) {
    *poffset = 0;
  }
  if(ppadding) {
    *ppadding = 0;
  }
  return xsize * ysize;
}

int dpxio_jp2_opensequence(dpxio_handle * dpxio, char * filename, int firstframe, int lastframe)
{
  return TRUE;
}

int dpxio_jp2_closesequence(dpxio_handle * dpxio)
{
  return TRUE;
}

int dpxio_jp2_readframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits)
{
  char filename[MAX_PATHNAME];
  int filesize;
  void * fp;

  sprintf(filename, dpxio->videoio.filename, dpxio->videoio.framenr + framenr);

  if(xsize) {
    *xsize = 2048;
  }
  if(ysize) {
    *ysize = 1080;
  }

  if(dpxio->io.dryrun) {
    if(dpxio->verbose) {
      printf("read video frame '%s'\n", filename);
    }
  } else {
    fp = file_open(filename, O_RDONLY|O_BINARY, 0666, TRUE);
    if(fp) {
      filesize = file_read(fp, buffer, buffersize);
      file_close(fp);
    } else {
      printf("ERROR: Could not open file '%s'\n", filename);
      return 0;
    }
    pbuffer->storage.compression = SV_COMPRESSION_JPEG2K;
    if(dpxio->config.keys[0].decrypt) {
      pbuffer->encryption.code = SV_ENCRYPTION_AES;
      pbuffer->encryption.keyid = dpxio->config.key_base;
      pbuffer->encryption.payload = filesize;
      pbuffer->encryption.plaintext = 0;
      pbuffer->encryption.sourcelength = filesize;
    }

    *offset = 0;

    return (filesize + 255) & ~255;
  }

  return *offset; /* Simulate dummy dma transfer */
}


int dpxio_jp2_writeframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int xsize, int ysize, int dpxtype, int nbits, int offset, int padding, int timecode)
{
#if 0
  char filename[MAX_PATHNAME];
  void * fp;

  sprintf(filename, dpxio->videoio.filename, dpxio->videoio.framenr + framenr);

  if(dpxio->io.dryrun) {
    if(dpxio->verbose) {
      printf("write video frame '%s'\n", filename);
    }
  } else {
    dpxio_filldpxheader(dpxio, buffer, offset, xsize, ysize, dpxtype, nbits, padding, timecode);

    fp = file_open(dpxio, filename, O_WRONLY|O_BINARY|O_CREAT, 0666, FALSE);
    if(fp) {
      file_write(dpxio, fp, buffer, buffersize);
      file_close(dpxio, fp);
    } else {
      printf("ERROR: Could not create file '%s'\n", filename);
      return FALSE;
    }
  }
#endif

  return 0x1000; /* Simulate dummy dma transfer */
}

