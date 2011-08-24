/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - File i/o routines for mxf files.
//
*/

#include "dpxio.h"


int dpxio_verifyformat(dpxio_handle * dpxio, char * filename, int firstframenr)
{
  char filepath[MAX_PATH];
  int videoformat = VIDEO_FORMAT_INVALID;
  unsigned char buffer[16];
  int fd;
  int count;

  if(dpxio->binput) {
    videoformat = VIDEO_FORMAT_DPX; // currently only DPX file format supported
  } else if(dpxio->io.dryrun) {
    videoformat = VIDEO_FORMAT_DPX;
  } else {
    sprintf(filepath, filename, firstframenr);

    fd = open(filepath, O_RDONLY|O_BINARY|O_LARGEFILE);
    if(fd > 0) {
      count = read(fd, buffer, sizeof(buffer));

      if(count == sizeof(buffer)) {
        switch(((uint32*)buffer)[0]) {
        case 0x53445058:
        case 0x58504453:
          videoformat = VIDEO_FORMAT_DPX;
          break;
        case 0x342b0e06:
          videoformat = VIDEO_FORMAT_MXF;
          break;
        default:
          if((buffer[0] == 0x00) && (buffer[1] == 0x00) && (buffer[2] == 0x00) && (buffer[3] == 0x0c) && (buffer[4] == 'j') && (buffer[5] == 'P') && (buffer[6] == ' ') && (buffer[7] == ' ') && (buffer[8] == 0x0d) && (buffer[9] == 0x0a) && (buffer[10] == 0x87) && (buffer[11] == 0x0a)) {
            videoformat = VIDEO_FORMAT_JP2;
          }
          if((buffer[0] == 0xff) && (buffer[1] == 0x4f) && (buffer[2] == 0xff) && (buffer[3] == 0x51)) {
            // plain jp2k codestream
            videoformat = VIDEO_FORMAT_JP2;
          }
          if(dpxio->config.keys[0].decrypt) {
            // encrypted jp2k codestream
            videoformat = VIDEO_FORMAT_JP2;
          }
        }
      }

      close(fd);
    }
  }

  dpxio->videoformat = videoformat;

  return (videoformat != VIDEO_FORMAT_INVALID);
}



int dpxio_getstoragemode(dpxio_handle * dpxio, int dpxtype, int nbits)
{
  switch(dpxio->videoformat) {
  case VIDEO_FORMAT_DPX:
    return dpxio_dpx_getstoragemode(dpxio, dpxtype, nbits);
  case VIDEO_FORMAT_JP2:
    return dpxio_jp2_getstoragemode(dpxio, dpxtype, nbits);
  case VIDEO_FORMAT_MXF:
    return dpxio_mxf_getstoragemode(dpxio, dpxtype, nbits);
  }

  return -1;
}

int dpxio_framesize(dpxio_handle * dpxio, int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding)
{
  switch(dpxio->videoformat) {
  case VIDEO_FORMAT_DPX:
    return dpxio_dpx_framesize(dpxio, xsize, ysize, dpxmode, nbits, poffset, ppadding);
  case VIDEO_FORMAT_JP2:
    return dpxio_jp2_framesize(dpxio, xsize, ysize, dpxmode, nbits, poffset, ppadding);
  case VIDEO_FORMAT_MXF:
    return dpxio_mxf_framesize(dpxio, xsize, ysize, dpxmode, nbits, poffset, ppadding);
  }

  return 0;
}


int dpxio_video_opensequence(dpxio_handle * dpxio, char * filename, int firstframe, int lastframe)
{
  switch(dpxio->videoformat) {
  case VIDEO_FORMAT_DPX:
    return dpxio_dpx_opensequence(dpxio, filename, firstframe, lastframe);
  case VIDEO_FORMAT_JP2:
    return dpxio_jp2_opensequence(dpxio, filename, firstframe, lastframe);
  case VIDEO_FORMAT_MXF:
    return dpxio_mxf_opensequence(dpxio, FALSE, filename, firstframe, lastframe);
  }

  return FALSE;
}


int dpxio_video_closesequence(dpxio_handle * dpxio)
{
  switch(dpxio->videoformat) {
  case VIDEO_FORMAT_DPX:
    return dpxio_dpx_closesequence(dpxio);
  case VIDEO_FORMAT_JP2:
    return dpxio_jp2_closesequence(dpxio);
  case VIDEO_FORMAT_MXF:
    return dpxio_mxf_closesequence(dpxio, FALSE);
  }

  return FALSE;
}


int dpxio_video_readframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset)
{
  switch(dpxio->videoformat) {
  case VIDEO_FORMAT_DPX:
    return dpxio_dpx_readframe(dpxio, framenr, pbuffer, buffer, buffersize, offset, xsize, ysize, dpxtype, nbits, lineoffset);
  case VIDEO_FORMAT_JP2:
    *lineoffset = 0;
    return dpxio_jp2_readframe(dpxio, framenr, pbuffer, buffer, buffersize, offset, xsize, ysize, dpxtype, nbits);
  case VIDEO_FORMAT_MXF:
    *lineoffset = 0;
    return dpxio_mxf_readframe(dpxio, framenr, pbuffer, buffer, buffersize, offset, xsize, ysize, dpxtype, nbits);
  }

  return 0;
}


int dpxio_video_writeframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int xsize, int ysize, int dpxtype, int nbits, int offset, int padding, int timecode)
{
  switch(dpxio->videoformat) {
  case VIDEO_FORMAT_DPX:
    return dpxio_dpx_writeframe(dpxio, framenr, pbuffer, buffer, buffersize, xsize, ysize, dpxtype, nbits, offset, padding, timecode);
  case VIDEO_FORMAT_JP2:
    return dpxio_jp2_writeframe(dpxio, framenr, pbuffer, buffer, buffersize, xsize, ysize, dpxtype, nbits, offset, padding, timecode);
  case VIDEO_FORMAT_MXF:
    return dpxio_mxf_writeframe(dpxio, framenr, pbuffer, buffer, buffersize, xsize, ysize, dpxtype, nbits, offset, padding, timecode);
  }

  return 0;
}

