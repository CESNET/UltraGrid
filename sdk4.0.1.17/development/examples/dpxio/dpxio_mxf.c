/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - File i/o routines for mxf files.
//
*/

#include "dpxio.h"

static unsigned char headerid[]     = { 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01, 0x02, 0x04 };
static unsigned char imageid[]      = { 0x06, 0x0e, 0x2b, 0x34, 0x01, 0x02, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x15, 0x01, 0x08 };
static unsigned char audioid[]      = { 0x06, 0x0e, 0x2b, 0x34, 0x01, 0x02, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x16, 0x01, 0x01 };
static unsigned char waveheaderid[] = { 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x48 };
static unsigned char encryptedid[]  = { 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x04, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x02, 0x7e, 0x01 };
static unsigned char encryptedid2[] = { 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x04, 0x01, 0x07, 0x0d, 0x01, 0x03, 0x01, 0x02, 0x7e, 0x01 };

int dpxio_mxf_getstoragemode(dpxio_handle * dpxio, int dpxtype, int nbits)
{
  return SV_MODE_STORAGE_FRAME;
}

int dpxio_mxf_framesize(dpxio_handle * dpxio, int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding)
{
  if(poffset) {
    *poffset = 0;
  }
  if(ppadding) {
    *ppadding = 0;
  }
  return xsize * ysize;
}

static int mxfuint16(unsigned char * p)
{
  return p[0] * 0x100 + p[1];
}

static int mxfuint32(unsigned char * p)
{
  return p[0] * 0x1000000 + p[1] * 0x10000 + p[2] * 0x100 + p[3];
}

int klv_key_compare(char * buffer, unsigned char * compare, int size)
{
  int found = TRUE;
  int i;

  for(i = 0; i < size; i++) {
    if(compare[i] != (unsigned char) buffer[i]) {
      found = FALSE;
    }
  }

  return found;
}

int klv_get_length(char * buffer, int * length)
{
  int nbytes = 1;
  int size = 0;
  int pos = 0;
  int i;

  if(buffer[pos] & 0x80) {
    nbytes = buffer[pos++] & 0x0f;
  }
  for(i = 0; i < nbytes; i++) {
    size = (size << 8) | ((unsigned char*)buffer)[pos++];
  }

  *length = size;

  return pos;
}


int dpxio_mxf_opensequence(dpxio_handle * dpxio, int baudio, char * filename, int firstframe, int lastframe)
{
  unsigned char buffer[1024];
  uint64   offset = 0;
  uint64   tableoffset = 0;
  int framenr = 0;
  int result = 0;
  int size;
  int found;
  void * handle;
  int pos = 0;

  handle = file_open(filename, O_RDONLY|O_BINARY|O_LARGEFILE, 0666, TRUE);
  
  if(handle) {
    result = file_read(handle, (char*)&buffer[pos], sizeof(buffer));

    if(result != sizeof(buffer)) {
      result = 0;
    }

    if(result) {
      result = klv_key_compare((char*)&buffer[pos], headerid, 15);
    }

    do {
      if(result) {
        pos += 16;
        pos += klv_get_length((char*)&buffer[pos], &size);
        offset += pos + size;
        pos = 0;

        result = file_lseek(handle, offset, SEEK_SET);
      }
      if(result) {
        result = file_read(handle, (char*)&buffer[pos], sizeof(buffer));
      }
      if(result) {
        tableoffset = offset;

        if(baudio) {
          found = klv_key_compare((char*)&buffer[pos], waveheaderid, 15);

          if(found) {
            int tag;
            int len;
            int tmp = pos + 16;

            tmp += klv_get_length((char*)&buffer[tmp], &size);

            do {
              tag = mxfuint16(&buffer[tmp]);
              len = mxfuint16(&buffer[tmp+2]);
              
              switch(tag) {
              case 0x3d03: // Audio Sample rate
                dpxio->audio.frequency = mxfuint32(&buffer[tmp+4]);
                if(mxfuint32(&buffer[tmp+8]) != 1) {
                  result = 0;
                }
                break;
              case 0x3d07: // Channel count
                dpxio->audio.nchannels = mxfuint32(&buffer[tmp+4]);
                break;
              case 0x3d01: // Bits per Audio Sample
                dpxio->audio.nbits = mxfuint32(&buffer[tmp+4]);
                break;
#if 0
              case 0x3002: // Length
                mxfuint32(&buffer[tmp+4]);
                break;
#endif
              }

              if(dpxio->verbose) {
                printf("mxfaudio %04x %d\n", tag, len);
              }
              tmp += len + 4;
            } while(tmp < size);
          }

          found = klv_key_compare((char*)&buffer[pos], audioid, 15);

          if(!found) {
            found = klv_key_compare((char*)&buffer[pos], encryptedid, 15);

            if(found) {
              dpxio->audioio.decrypt = TRUE;
            }
          }

          if(!found) {
            found = klv_key_compare((char*)&buffer[pos], encryptedid2, 15);

            if(found) {
              dpxio->audioio.decrypt = TRUE;
            }
          }

          if(!found) {
            tableoffset = 0;
          }
        } else {
          found = klv_key_compare((char*)&buffer[pos], imageid, 15);

          if(!found) {
            found = klv_key_compare((char*)&buffer[pos], encryptedid, 15);

            if(found) {
              dpxio->videoio.decrypt = TRUE;
            }
          }

          if(!found) {
            found = klv_key_compare((char*)&buffer[pos], encryptedid2, 15);

            if(found) {
              dpxio->videoio.decrypt = TRUE;
            }
          }

          if(!found) {
            tableoffset = 0;
          }
        }

        if(tableoffset) {
          if(framenr++ < dpxio->videoio.framenr) {
            tableoffset = 0;
          }
        }
      }
    } while(!tableoffset && result);
  }

  if(baudio) {
    if(!dpxio->audio.nbits) {
      printf("mxfaudio: Audio bits not found\n");
      result = 0;
    }
    if(!dpxio->audio.nchannels) {
      printf("mxfaudio: Audio nchannels not found\n");
      result = 0;
    }
    if(!dpxio->audio.frequency) {
      printf("mxfaudio: Audio frequency not found\n");
      result = 0;
    }
    dpxio->audioio.handle      = handle;
    dpxio->audioio.position    = tableoffset;
    dpxio->audio.binput        = FALSE;
    dpxio->audio.bsigned       = TRUE;
    dpxio->audio.blittleendian = FALSE;
  } else {
    dpxio->videoio.handle   = handle;
    dpxio->videoio.position = tableoffset;
    dpxio->videoio.xsize  = 2048;
    dpxio->videoio.ysize  = 1080;
  }

  return result;
}

int dpxio_mxf_closesequence(dpxio_handle * dpxio, int baudio)
{
  if(baudio) {
    file_close(dpxio->audioio.handle); dpxio->audioio.handle = NULL;
  } else {
    file_close(dpxio->videoio.handle); dpxio->videoio.handle = NULL;
  }

  return TRUE;
}


int dpxio_mxf_readframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits)
{
  uint64 result;
  int size;
  int i;
  int preread = dpxio->videoio.decrypt ? 128 : 20;
  int pos = 0;
  int found;
  int totalsize;
  int dummy;
  uint32 plaintext = 0;
  uint32 sourcelength = 0;
  int payload = 0;

  if(xsize) {
    *xsize = dpxio->videoio.xsize;
  }
  if(ysize) {
    *ysize = dpxio->videoio.ysize;
  }

  if(dpxio->io.dryrun) {
    if(dpxio->verbose) {
      printf("read video frame %d\n", framenr);
    }
  } else {
    result = file_lseek(dpxio->videoio.handle, dpxio->videoio.position, SEEK_SET);
    if(result) {
      if(file_read(dpxio->videoio.handle, buffer, preread) != preread) {
        return 0;
      }

      if(dpxio->videoio.decrypt) {
        found = klv_key_compare(&buffer[pos], encryptedid, 15);
        if(!found) {
          found = klv_key_compare(&buffer[pos], encryptedid2, 15);
        }
      } else {
        found = klv_key_compare(&buffer[pos], imageid, 15);
      }
      pos += 16;

      if(!found) {
        printf("ERROR: Data container not found at 0x%08x%08x (framenr:%d).\n", (uint32)(dpxio->videoio.position>>32), (uint32)dpxio->videoio.position, framenr);
        return 0;
      }

      pos += klv_get_length(&buffer[pos], &size);
      totalsize = pos + size;

      if(dpxio->videoio.decrypt) {
        // skip cryptographic context link
        pos += klv_get_length(&buffer[pos], &dummy);
        pos += dummy;

        // obtain plaintext offset
        pos += klv_get_length(&buffer[pos], &dummy);
        for(i = 0; i < dummy; i++) {
          plaintext = (plaintext << 8) | ((unsigned char*)buffer)[pos++];
        }

        // skip source key
        pos += klv_get_length(&buffer[pos], &dummy);
        pos += dummy;

        // obtain source length
        pos += klv_get_length(&buffer[pos], &dummy);
        for(i = 0; i < dummy; i++) {
          sourcelength = (sourcelength << 8) | ((unsigned char*)buffer)[pos++];
        }

        // obtain payload length
        pos += klv_get_length(&buffer[pos], &payload);

        result = file_lseek(dpxio->videoio.handle, dpxio->videoio.position + pos, SEEK_SET);
      } else {
        payload = size;
      }

      if(dpxio->verbose) {
#ifdef WIN32
        printf("read frame(%dx%d) %d @ %16I64x %d/%08x\n", *xsize, *ysize, framenr, dpxio->videoio.position, payload, payload);
#else
        printf("read frame(%dx%d) %d @ %llx %d/%08x\n", *xsize, *ysize, framenr, dpxio->videoio.position, payload, payload);
#endif
      }
      file_read(dpxio->videoio.handle, buffer, payload);
    } else {
      return 0;
    }

    pbuffer->storage.compression = SV_COMPRESSION_JPEG2K;
    if(dpxio->videoio.decrypt) {
      pbuffer->encryption.code = SV_ENCRYPTION_AES;
      pbuffer->encryption.keyid = dpxio->config.key_base;
      pbuffer->encryption.payload = payload;
      pbuffer->encryption.plaintext = plaintext;
      pbuffer->encryption.sourcelength = sourcelength;
    }

    *offset = 0;

    dpxio->videoio.position += totalsize;

    return (payload + 255) & ~255;
  }

  return 0x1000; /* Simulate dummy dma transfer */
}


int dpxio_mxf_writeframe(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int xsize, int ysize, int dpxtype, int nbits, int offset, int padding, int timecode)
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

    fp = file_open(dpxio, filename, O_WRONLY|O_BINARY|O_CREAT|O_LARGEFILE, 0666, FALSE);
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


int dpxio_mxf_readaudio(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, char * buffer, int buffersize, int * datasize)
{
  uint64 result;
  int size;
  int i;
  int pos = 0;
  int found;
  int dummy;
  uint32 plaintext = 0;
  uint32 sourcelength = 0;
  int payload = 0;

  if(dpxio->io.dryrun) {
    if(dpxio->verbose) {
      printf("read video frame %d\n", framenr);
    }
  } else {
    result = file_lseek(dpxio->audioio.handle, dpxio->audioio.position, SEEK_SET);
    if(result) {
      result = file_read(dpxio->audioio.handle, buffer, 20);
      if(result == 20) {
        if(dpxio->audioio.decrypt) {
          found = klv_key_compare(&buffer[0], encryptedid, 15);
          if(!found) {
            found = klv_key_compare(&buffer[0], encryptedid2, 15);
          }
        } else {
          found = TRUE;
        }
        klv_get_length(&buffer[16], &size);

        if(size > buffersize) {
          printf("audio buffersize to small %08x needed %08x\n", buffersize, size);
          return 0;
        }

        if(!found) {
          printf("ERROR: Audio data container not found at 0x%08x%08x (framenr:%d).\n", (uint32)(dpxio->audioio.position>>32), (uint32)dpxio->audioio.position, framenr);
          return 0;
        }

        if(dpxio->audioio.decrypt) {
          if(file_read(dpxio->audioio.handle, buffer, 128) != 128) {
            return 0;
          }
          pos = 0;

          // skip cryptographic context link
          pos += klv_get_length(&buffer[pos], &dummy);
          pos += dummy;

          // obtain plaintext offset
          pos += klv_get_length(&buffer[pos], &dummy);
          for(i = 0; i < dummy; i++) {
            plaintext = (plaintext << 8) | ((unsigned char*)buffer)[pos++];
          }

          // skip source key
          pos += klv_get_length(&buffer[pos], &dummy);
          pos += dummy;

          // obtain source length
          pos += klv_get_length(&buffer[pos], &dummy);
          for(i = 0; i < dummy; i++) {
            sourcelength = (sourcelength << 8) | ((unsigned char*)buffer)[pos++];
          }

          // obtain payload length
          pos += klv_get_length(&buffer[pos], &payload);

          result = file_lseek(dpxio->audioio.handle, dpxio->audioio.position + 20 + pos, SEEK_SET);
        } else {
          payload = size;
        }

        if(file_read(dpxio->audioio.handle, buffer, size) != size) {
          return 0;
        }
      } else {
        return 0;
      }
    } else {
      return 0;
    }

    if(dpxio->audioio.decrypt) {
      pbuffer->encryption_audio.code = SV_ENCRYPTION_AES;
      pbuffer->encryption_audio.keyid = dpxio->config.key_base + (dpxio->videoio.decrypt ? 1 : 0);
      pbuffer->encryption_audio.payload = payload;
      pbuffer->encryption_audio.plaintext = plaintext;
      pbuffer->encryption_audio.sourcelength = sourcelength;

      pbuffer->encryption_audio.bits = dpxio->audio.nbits;
      pbuffer->encryption_audio.channels = dpxio->audio.nchannels;
      pbuffer->encryption_audio.frequency = dpxio->audio.frequency;
      pbuffer->encryption_audio.littleendian = dpxio->audio.blittleendian;
      pbuffer->encryption_audio.bsigned = dpxio->audio.bsigned;
    }

    dpxio->audioio.position += size + 20;

    if(datasize) {
      *datasize = size;
    }

    return 1;
  }

  return 0;
}
