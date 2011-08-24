/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the FIFO API to do display and record 
//            of images directly from/to a file.
//
*/

#include "dpxio.h"

/************************** Resample ***************************/
#ifdef __sun
# define __inline
#endif

static __inline int swap32(int x)
{
  int y;
  char * p = (char*)&x;
  char * q = (char*)&y;
  
  q[0] = p[3];
  q[1] = p[2];
  q[2] = p[1];
  q[3] = p[0];

  return y;
}

static __inline unsigned short swap16(unsigned short x)
{
  return ((x & 0xff00) >> 8) | ((x & 0x00ff) << 8);
}

#define AUDIORESAMPLE                                     \
  int i;                                                  \
  int ch;                                                 \
                                                          \
  if(inputchannels == outputchannels) {                   \
    for(i = 0; i < samplecount*outputchannels; i++) {     \
      *outputbuffer++ = CONVERT(*inputbuffer);            \
      inputbuffer++;                                      \
    }                                                     \
  } else if(inputchannels < outputchannels) {             \
    for(i = 0; i < samplecount; i++) {                    \
      for(ch = 0; ch < inputchannels; ch++) {             \
        *outputbuffer++ = CONVERT(*inputbuffer);          \
        inputbuffer++;                                    \
      }                                                   \
      for(; ch < outputchannels; ch++) {                  \
        *outputbuffer++ = 0;                              \
      }                                                   \
    }                                                     \
  } else {                                                \
    for(i = 0; i < samplecount; i++) {                    \
      for(ch = 0; ch < outputchannels; ch++) {            \
        *outputbuffer++ = CONVERT(*inputbuffer);          \
        inputbuffer++;                                    \
      }                                                   \
      for(; ch < inputchannels; ch++) {                   \
        inputbuffer++;                                    \
      }                                                   \
    }                                                     \
  }

#define AUDIORESAMPLE24     \
  int i;                                                  \
  int ch;                                                 \
                                                          \
  if(inputchannels == outputchannels) {                   \
    for(i = 0; i < samplecount*outputchannels; i++) {     \
      *outputbuffer++ = CONVERT(inputbuffer);             \
      inputbuffer+=3;                                     \
    }                                                     \
  } else if(inputchannels < outputchannels) {             \
    for(i = 0; i < samplecount; i++) {                    \
      for(ch = 0; ch < inputchannels; ch++) {             \
        *outputbuffer++ = CONVERT(inputbuffer);           \
        inputbuffer+=3;                                   \
      }                                                   \
      for(; ch < outputchannels; ch++) {                  \
        *outputbuffer++ = 0;                              \
      }                                                   \
    }                                                     \
  } else {                                                \
    for(i = 0; i < samplecount; i++) {                    \
      for(ch = 0; ch < inputchannels; ch++) {             \
        *outputbuffer++ = CONVERT(inputbuffer);           \
        inputbuffer+=3;                                   \
      }                                                   \
      for(; ch < inputchannels; ch++) {                   \
        inputbuffer+=3;                                   \
      }                                                   \
    }                                                     \
  }

void dpxio_audio_resample_int8_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((int)(x))<<24)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_uint8_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((int)(x)-0x80)<<24)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int16le_to_int32le(signed short * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((int)(x))<<16)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_uint16le_to_int32le(unsigned short * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((((int)(x))-0x8000)<<16)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int16be_to_int32le(signed short * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((int)swap16(x))<<16)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_uint16be_to_int32le(unsigned short * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((((int)swap16(x))-0x8000)<<16)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int24be_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((((int)x[0] << 16) | ((int)x[1] << 8) | (int)x[2]) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}
    
void dpxio_audio_resample_uint24be_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((((int)x[0] << 16) | ((int)x[1] << 8) | (int)x[2]) - 0x800000) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}

void dpxio_audio_resample_int24le_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((((int)x[2] << 16) | ((int)x[1] << 8) | (int)x[0]) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}

    
void dpxio_audio_resample_uint24le_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((((int)x[2] << 16) | ((int)x[1] << 8) | (int)x[0]) - 0x800000) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_int32le(int * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (x)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_uint32le_to_int32le(unsigned int * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((int)((x)-0x80000000))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32be_to_int32le(int * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (swap32(x))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_uint32be_to_int32le(unsigned int * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((int)(swap32(x)-0x80000000))
  AUDIORESAMPLE
#undef CONVERT
}

int dpxio_audio_resample_output(dpxio_handle * dpxio, void * inputbuffer, void * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
  switch(dpxio->audio.nbits) {
  case 8:
    if(dpxio->audio.bsigned) {
      dpxio_audio_resample_int8_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    } else {
      dpxio_audio_resample_uint8_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    }
    break;
  case 16:
    if(dpxio->audio.blittleendian) {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int16le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint16le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int16be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint16be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  case 24:
    if(dpxio->audio.blittleendian) {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int24le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint24le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int24be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint24be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  case 32:
    if(dpxio->audio.blittleendian) {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int32le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint32le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int32be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint32be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  default:
    printf("ERROR: dpxio_audio_resample_output: Unknown bitdepth %d\n", dpxio->audio.nbits);
    return FALSE;
  }

  return TRUE;
}



void dpxio_audio_resample_int32le_to_int8(int * inputbuffer, signed char * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((signed char)((x)>>24))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_uint8(int * inputbuffer, unsigned char * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((unsigned char)((x)>>24)+0x80))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_int16le(int * inputbuffer, signed short * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((short)((x)>>16))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_uint16le(int * inputbuffer, unsigned short * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((unsigned short)((x)>>16)+0x8000))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_int16be(int * inputbuffer, signed short * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((short)(swap16((uint16)((x)>>16))))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_uint16be(int * inputbuffer, unsigned short * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((unsigned short)((swap16((uint16)((x)>>16)))+0x8000))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_uint32le(int * inputbuffer, unsigned int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((int)(x))+0x80000000)
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_int32be(int * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (swap32(x))
  AUDIORESAMPLE
#undef CONVERT
}

void dpxio_audio_resample_int32le_to_uint32be(unsigned int * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((int)(swap32((x)+0x80000000)))
  AUDIORESAMPLE
#undef CONVERT
}

int dpxio_audio_resample_input(dpxio_handle * dpxio, void * inputbuffer, void * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
  switch(dpxio->audio.nbits) {
  case 8:
    if(dpxio->audio.bsigned) {
      dpxio_audio_resample_int32le_to_int8(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    } else {
      dpxio_audio_resample_int32le_to_uint8(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    }
    break;
  case 16:
    if(dpxio->audio.blittleendian) {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int32le_to_int16le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint16le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int32le_to_int16be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint16be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  case 32:
    if(dpxio->audio.blittleendian) {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int32le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(dpxio->audio.bsigned) {
        dpxio_audio_resample_int32le_to_int32be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint32be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  default:
    printf("ERROR: dpxio_audio_resample_input: Unknown bitdepth %d\n", dpxio->audio.nbits);
    return FALSE;
  }

  return TRUE;
}

/************************** MXF FILE ***************************/
static int dpxio_mxf_open(dpxio_handle * dpxio, char * filename)
{
  if(!dpxio_mxf_opensequence(dpxio, TRUE, filename, 0, 0)) {
    return AUDIO_FORMAT_INVALID;
  }

 return AUDIO_FORMAT_MXF;
}
/************************** Wave FILE ***************************/

static unsigned int wave_uint16(void * vp)
{ 
  unsigned char * p = vp;

  return (((int)p[1])<<8)|((int)(p[0]));
}

static int wave_uint32(void * vp)
{
  unsigned char * p = vp;

  return (((int)p[3])<<24)|(((int)p[2])<<16)|(((int)p[1])<<8)|((int)(p[0]));
}


static int dpxio_wave_create(dpxio_handle * dpxio, char * filename, int channels, int frequency, int nbits)
{
  return FALSE;
}

static int dpxio_wave_open(dpxio_handle * dpxio, char * filename)
{
  void *  handle;
  char    buf[256];
  int     offset;
  int     count = 0;
  int     length;

  handle = file_open(filename, O_RDONLY|O_BINARY, 0666, TRUE);

  if(handle == NULL) {
    return AUDIO_FORMAT_INVALID;
  }

  offset = 12;

  while(count++ < 10000) {
    file_lseek(handle, offset, SEEK_SET);
    if(file_read(handle, &buf[0], sizeof(buf)) != sizeof(buf)) {
      file_close(handle);
      return AUDIO_FORMAT_INVALID;
    }

    length = wave_uint32(&buf[4]);

    if(strncmp(buf, "data", 4) == 0) {
      dpxio->audioio.handle      = handle;
      dpxio->audioio.position    = offset + 8;
      dpxio->audio.binput        = FALSE;
      dpxio->audio.bsigned       = TRUE;
      dpxio->audio.blittleendian = TRUE;
      return AUDIO_FORMAT_WAVE;
    }

    offset += ( 8 + length + 1 ) & ~1;  /* align to even byte boundary */
  }        

  file_close(handle);

  return AUDIO_FORMAT_INVALID;
}

/************************** AIFF FILE ***************************/

static int aiff_readint(uint8 * p)
{
  return (((int)p[0])<<24)|(((int)p[1])<<16)|(((int)p[2])<<8)|((int)(p[3]));
}

static void aiff_writeint(uint8 * p, int value)
{
  p[0] = (uint8)((value >> 24) & 0xff);
  p[1] = (uint8)((value >> 16) & 0xff);
  p[2] = (uint8)((value >>  8) & 0xff);
  p[3] = (uint8)((value      ) & 0xff);
}

static int aiff_readshort(uint8 * p)
{
  return (short)(((int)p[0])<<8)|((int)(p[1]));
}

static void aiff_writeshort(uint8 * p, int value)
{
  p[0] = (uint8)((value >> 8) & 0xff);
  p[1] = (uint8)((value     ) & 0xff);
}


static int dpxio_aiff_open(dpxio_handle * dpxio, char * filename)
{
  void *  handle;
  char    buf[256];
  int     offset;
  int     count = 0;
  int     size;

  handle = file_open(filename, O_RDONLY|O_BINARY, 0666, TRUE);

  if(handle == NULL) {
    return AUDIO_FORMAT_INVALID;
  }

  dpxio->audio.blittleendian = FALSE;

  if(file_read(handle, &buf[0], 12) != 12) {
    file_close(handle);
    return AUDIO_FORMAT_INVALID;
  }

  if(strncmp(buf, "FORM", 4) != 0) {
    file_close(handle);
    return AUDIO_FORMAT_INVALID;
  }

  while(count++ < 100) {
    if(file_read(handle, &buf[0], 8) != 8) {
      file_close(handle);
      return AUDIO_FORMAT_INVALID;
    }  
 
    if(strncmp(buf, "sowt", 4) == 0) {
      dpxio->audio.blittleendian = TRUE;
    } else if(strncmp(buf, "SSND", 4) == 0) {
      if(file_read(handle, buf, 8) != 8) {
        file_close(handle);
        return AUDIO_FORMAT_INVALID;
      }

      offset = aiff_readint((uint8*)&buf[0]);

      dpxio->audioio.handle      = handle;
      dpxio->audioio.position    = file_lseek(handle, 0, SEEK_CUR) + offset;
      dpxio->audio.binput        = FALSE;
      dpxio->audio.bsigned       = TRUE;

      file_lseek(handle, dpxio->audioio.position, SEEK_SET);

      return AUDIO_FORMAT_AIFF;
    } 
    size = (int)aiff_readint((uint8*)&buf[4]);
    while(size > sizeof(buf)) {
      if(file_read(handle, buf, sizeof(buf)) != sizeof(buf)) {
        file_close(handle);
        return AUDIO_FORMAT_INVALID;
      }
      size -= sizeof(buf);
    }
    if(size > 0) {
      if(file_read(handle, buf, size) != size) {
        file_close(handle);
        return AUDIO_FORMAT_INVALID;
      }
    }
  }        


  file_close(handle);

  return AUDIO_FORMAT_INVALID;
}


static int dpxio_aiff_create(dpxio_handle * dpxio, char * filename, int frequency, int channels, int nbits)
{
  uint8  buf[DPXIO_AUDIO_HEADER_SIZE];
  void * handle;

  memset(buf, 0, sizeof(buf));

  handle = file_open(filename, O_CREAT|O_RDWR|O_BINARY, 0666, TRUE);
  if(handle == NULL) {
    printf("ERROR: audio_aiff_create: fopen(%s) failed\n", filename);
    return AUDIO_FORMAT_INVALID;
  }

  memset(buf, 0, sizeof(buf));
  strncpy((char*)&buf[0], "FORM", 4);
  buf[4]  = 0;
  aiff_writeint(&buf[4], 0);
  strncpy((char*)&buf[8], "AIFF(c) ", 8);
  strncpy((char*)&buf[20], "COMM", 4);  
  aiff_writeint(&buf[24], 18);          
  aiff_writeshort(&buf[28], channels);
  aiff_writeint(&buf[30], 0);
  aiff_writeshort(&buf[34], nbits);
  buf[36] = 0x40;
  buf[37] = 0x0e;
  aiff_writeshort(&buf[38], frequency);
  buf[40] = 0;
  buf[41] = 0;
  buf[42] = 0;
  buf[43] = 0;
  buf[44] = 0;
  buf[45] = 0;
  strncpy((char*)&buf[46], "SSND", 4);
  aiff_writeint(&buf[50], 0); // size
  aiff_writeint(&buf[54], 0); // offset
  aiff_writeint(&buf[58], 0); // blocksize
 
  if(file_write(handle, (char*)&buf[0], DPXIO_AUDIO_HEADER_SIZE) != DPXIO_AUDIO_HEADER_SIZE) {
    printf("ERROR: dpxio_aiff_create: fwrite(%d) failed, errno:%d\n", DPXIO_AUDIO_HEADER_SIZE, errno);
    file_close(handle);
    return AUDIO_FORMAT_INVALID;
  }

  dpxio->audioio.handle      = handle;
  dpxio->audioio.position    = sizeof(buf);
  dpxio->audio.format        = AUDIO_FORMAT_AIFF;
  dpxio->audio.binput        = TRUE;
  dpxio->audio.nbits         = nbits;
  dpxio->audio.nchannels     = channels;
  dpxio->audio.frequency     = frequency;
  dpxio->audio.bsigned       = TRUE;
  dpxio->audio.blittleendian = FALSE;

  return AUDIO_FORMAT_AIFF;
}


static void dpxio_aiff_record_setsize(dpxio_handle * dpxio, int fileposition)
{
  uint8 buffer[32];
  size_t res;

  aiff_writeint(buffer, fileposition - 8);
  (void)file_lseek(dpxio->audioio.handle, 0x04, SEEK_SET);
  res = file_write(dpxio->audioio.handle, (char*)buffer, sizeof(uint32));
  if(res != sizeof(uint32)) {
    printf("ERROR: dpxio_aiff_record_setsize: fwrite(%d,%d) failed = %d errno:%d\n", 0x04, (int)sizeof(uint32), (int)res, errno);
  }

  aiff_writeint(buffer, (fileposition - DPXIO_AUDIO_HEADER_SIZE) / 4);
  (void)file_lseek(dpxio->audioio.handle, 0x1e, SEEK_SET);
  res = file_write(dpxio->audioio.handle, (char*)buffer, sizeof(uint32));
  if(res != sizeof(uint32)) {
    printf("ERROR: dpxio_aiff_record_setsize: fwrite(%d,%d) failed = %d errno:%d\n", 0x1e, (int)sizeof(uint32), (int)res, errno);
  }

  aiff_writeint(buffer, fileposition - 54);
  (void)file_lseek(dpxio->audioio.handle, 0x32, SEEK_SET);
  res = file_write(dpxio->audioio.handle, (char*)buffer, sizeof(uint32));
  if(res != sizeof(uint32)) {
    printf("ERROR: dpxio_aiff_record_setsize: fwrite(%d,%d) failed = %d errno:%d\n", fileposition - 54, (int)sizeof(uint32), (int)res, errno);
  }
}


int dpxio_audio_formatcheck(dpxio_handle * dpxio, char * filename)
{
  int     format = AUDIO_FORMAT_INVALID;
  FILE *  fp;
  uint8   header[4096];
  size_t  res;
  int     i;

  fp = fopen(filename, "rb");

  if(fp == NULL) {
    return AUDIO_FORMAT_INVALID;
  }

  res = fread(&header, 1, sizeof(header), fp);
  if(res != sizeof(header)) {
    fclose(fp);
    return AUDIO_FORMAT_INVALID;
  } 

  if((strncmp((char*)header, "FORM", 4) == 0) && (strncmp((char*)(header + 8), "AIFF", 4) == 0)) {
    for(i = 12; (format == AUDIO_FORMAT_INVALID) && (i < sizeof(header)-30); i++) {
      if(strncmp((char*)(header+i), "COMM", 4) == 0) {
        format            = AUDIO_FORMAT_AIFF;
        dpxio->audio.nchannels = aiff_readshort(&header[i+8]);
        dpxio->audio.nsample   = aiff_readint(&header[i+10]);
        dpxio->audio.nbits     = aiff_readshort(&header[i+14]);
        dpxio->audio.frequency = (unsigned int) ((unsigned int) aiff_readshort(&header[i+18]) & 0x0000ffff);
      }
    }
  } else if(!strncmp((char*)header, "RIFF", 4) && !strncmp((char*)header + 8, "WAVE", 4)) {
    for(i = 12; (format == AUDIO_FORMAT_INVALID) && (i < sizeof(header)-30); i++) {
      if(strncmp((char*)(header+i), "fmt ", 4) == 0) {
        format            = AUDIO_FORMAT_WAVE;
        dpxio->audio.nsample   = 5000000;
        dpxio->audio.nchannels = wave_uint16(&header[10+i]);
        dpxio->audio.frequency = wave_uint32(&header[12+i]);
        dpxio->audio.nbits     = wave_uint16(&header[22+i]);
      }
    }
  } else if(((uint32*)header)[0] == 0x342b0e06) {
    format = AUDIO_FORMAT_MXF;
  }

  dpxio->audio.format = format;

  fclose(fp);

  if(format != AUDIO_FORMAT_MXF) {
    switch(dpxio->audio.nbits) {
    case 8:
    case 16:
    case 24:
    case 32:
      break;
    default:
      return AUDIO_FORMAT_INVALID;
    }
  }

  return dpxio->audio.format;
}


int dpxio_audio_create(dpxio_handle * dpxio, char * filename, int format, int frequency, int channels, int nbits, int hwchannels)
{
  switch(format) {
  case AUDIO_FORMAT_AIFF:
    format = dpxio_aiff_create(dpxio, filename, frequency, channels, nbits);
    break;
  case AUDIO_FORMAT_WAVE:
    format = dpxio_wave_create(dpxio, filename, frequency, channels, nbits);
    break;
  case AUDIO_FORMAT_MXF:
    printf("ERROR: Can not create mxf audio files\n");
    break;
  default:
    printf("ERROR: dpxio_audio_create() format:%d invalid\n", format);
    format = AUDIO_FORMAT_INVALID;
  }

  if(format != AUDIO_FORMAT_INVALID) {
    // allocate enough space for one 96kHz frame in 24fps
    dpxio->audio.filebuffersize = 0x1000 * dpxio->audio.nchannels * dpxio->audio.nbits / 8;
    dpxio->audio.filebuffer     = malloc(dpxio->audio.filebuffersize);
    dpxio->audio.hwchannels     = hwchannels;
    if(!dpxio->audio.filebuffer) {
      printf("ERROR: dpxio_audio_create() malloc(%d) failed\n", dpxio->audio.filebuffersize);
      format = AUDIO_FORMAT_INVALID;
    } 
  } 

  return format;
}


int dpxio_audio_open(dpxio_handle * dpxio, char * filename, int hwchannels, int * pchannels, int * pfrequency)
{
  int format;

  format = dpxio_audio_formatcheck(dpxio, filename);

  switch(format) {
  case AUDIO_FORMAT_AIFF:
    format = dpxio_aiff_open(dpxio, filename);
    break;
  case AUDIO_FORMAT_WAVE:
    format = dpxio_wave_open(dpxio, filename);
    break;
  case AUDIO_FORMAT_MXF:
    format = dpxio_mxf_open(dpxio, filename);
    break;
  default:
    format = AUDIO_FORMAT_INVALID;
  }

  if(pchannels) {
    *pchannels = dpxio->audio.nchannels;
  }
  if(pfrequency) {
    *pfrequency = dpxio->audio.frequency;
  }

  if(format != AUDIO_FORMAT_INVALID) {
    // allocate enough space for one 96kHz frame in 24fps
    dpxio->audio.filebuffersize = 0x1000 * dpxio->audio.nchannels * dpxio->audio.nbits / 8;
    dpxio->audio.filebuffer     = malloc(dpxio->audio.filebuffersize);
    dpxio->audio.hwchannels     = hwchannels;
    if(!dpxio->audio.filebuffer) {
      printf("ERROR: dpxio_audio_create() malloc(%d) failed\n", dpxio->audio.filebuffersize);
      format = AUDIO_FORMAT_INVALID;
    } 
  } 

  return format;
}

static int dpxio_audio_do_read(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, void * buffer, int * size)
{
  int nsamples;
  int datasize;
  int res;

  res = dpxio_mxf_readaudio(dpxio, framenr, pbuffer, dpxio->audio.filebuffer, dpxio->audio.filebuffersize, &datasize);
  nsamples = datasize / dpxio->audio.nchannels / (dpxio->audio.nbits / 8);

  dpxio_audio_resample_output(dpxio, dpxio->audio.filebuffer, buffer, dpxio->audio.nchannels, dpxio->audio.hwchannels, nsamples);

  if(size) {
    *size = nsamples * dpxio->audio.hwchannels * 4 /*bytes*/;
  }

  return res;
}

int dpxio_audio_read(dpxio_handle * dpxio, int framenr, sv_fifo_buffer * pbuffer, int startchannel, void * buffer, int nsamples)
{
  int datacount;
  int res = TRUE;

  if(dpxio->audio.format == AUDIO_FORMAT_MXF) {
    if(dpxio->audioio.decrypt) {
      res = dpxio_mxf_readaudio(dpxio, framenr, pbuffer, buffer, dpxio->audio.filebuffersize, NULL);
    } else {
      unsigned char * dst = buffer;
      unsigned char * src = dpxio->audio.tmpbuffer;
      int size = 0;

      // copy remaining samples
      if(dpxio->audio.tmpremain) {
        memcpy(dst, src + dpxio->audio.tmpoffset, dpxio->audio.tmpremain);
        dst += dpxio->audio.tmpremain;

        nsamples -= dpxio->audio.tmpremain / 64;
        dpxio->audio.tmpremain = 0;
      }

      while(res && nsamples > 0) {
        res = dpxio_audio_do_read(dpxio, framenr, pbuffer, src, &size);

        datacount = (size < nsamples * 64) ? size : nsamples * 64;

        if(res) {
          memcpy(dst, src, datacount);
          dst += datacount;

          nsamples -= size / 64;
        }
      }

      if(nsamples < 0) {
        // remember offset/size of remaining samples
        dpxio->audio.tmpoffset = size + nsamples * 64;
        dpxio->audio.tmpremain = -nsamples * 64;
      }
    }

    return res;
  } else {
    int size = nsamples * dpxio->audio.nchannels * dpxio->audio.nbits / 8;
    int res = file_read(dpxio->audioio.handle, dpxio->audio.filebuffer, size);

    dpxio->audioio.position += (int)res;

    if(res == size) {
      buffer = (void *)((uintptr) buffer + (startchannel * 4 /*byte per sample*/ * 2 /*stereo*/));
      dpxio_audio_resample_output(dpxio, dpxio->audio.filebuffer, buffer, dpxio->audio.nchannels, dpxio->audio.hwchannels, nsamples);

      return TRUE;
    }

    return FALSE;
  }
}


int dpxio_audio_write(dpxio_handle * dpxio, int framenr, int startchannel, void * buffer, int nsamples)
{
  int size = nsamples * dpxio->audio.nchannels * dpxio->audio.nbits / 8;
  int res;

  buffer = (void *)((uintptr) buffer + (startchannel * 4 /*byte per sample*/ * 2 /*stereo*/));

  res  = dpxio_audio_resample_input(dpxio, buffer, dpxio->audio.filebuffer, dpxio->audio.hwchannels, dpxio->audio.nchannels, nsamples);
  
  if(res) {
    res = file_write(dpxio->audioio.handle, dpxio->audio.filebuffer, size);

    dpxio->audioio.position += (int)res;

    return (res == size);
  }

  return FALSE;
}


void dpxio_audio_close(dpxio_handle * dpxio)
{
  if(dpxio->audioio.handle) {
    if(dpxio->audio.binput && (dpxio->audio.format == AUDIO_FORMAT_AIFF)) {
      dpxio_aiff_record_setsize(dpxio, dpxio->audioio.position);
    }
    file_close(dpxio->audioio.handle);
    dpxio->audioio.handle = NULL;
  }

  if(dpxio->audio.filebuffer) {
    free(dpxio->audio.filebuffer);
    dpxio->audio.filebuffer = NULL;
  }
}

