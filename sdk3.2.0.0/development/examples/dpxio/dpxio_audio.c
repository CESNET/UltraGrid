/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the fifoapi to do display and record 
//            of images directly to a file.
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
        inputbuffer++;                                    \
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

void dpxio_audio_resample_int24le_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((((int)x[0] << 16) | ((int)x[1] << 8) | (int)x[2]) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}
    
void dpxio_audio_resample_uint24le_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          (((((int)x[0] << 16) | ((int)x[1] << 8) | (int)x[2]) - 0x800000) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}

void dpxio_audio_resample_int24be_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
#define CONVERT(x)          ((((int)x[2] << 16) | ((int)x[1] << 8) | (int)x[0]) << 8)
  AUDIORESAMPLE24
#undef CONVERT
}

    
void dpxio_audio_resample_uint24be_to_int32le(unsigned char * inputbuffer, int * outputbuffer, int inputchannels, int outputchannels, int samplecount)
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

int dpxio_audio_resample_output(audio_handle * paudio, void * inputbuffer, void * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
  switch(paudio->nbits) {
  case 8:
    if(paudio->bsigned) {
      dpxio_audio_resample_int8_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    } else {
      dpxio_audio_resample_uint8_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    }
    break;
  case 16:
    if(paudio->blittleendian) {
      if(paudio->bsigned) {
        dpxio_audio_resample_int16le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint16le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(paudio->bsigned) {
        dpxio_audio_resample_int16be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint16be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  case 24:
    if(paudio->blittleendian) {
      if(paudio->bsigned) {
        dpxio_audio_resample_int24le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint24le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(paudio->bsigned) {
        dpxio_audio_resample_int24be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint24be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  case 32:
    if(paudio->blittleendian) {
      if(paudio->bsigned) {
        dpxio_audio_resample_int32le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint32le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(paudio->bsigned) {
        dpxio_audio_resample_int32be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_uint32be_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  default:
    printf("ERROR: dpxio_audio_resample_output: Unknown bitdepth %d\n", paudio->nbits);
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

int dpxio_audio_resample_input(audio_handle * paudio, void * inputbuffer, void * outputbuffer, int inputchannels, int outputchannels, int samplecount)
{
  switch(paudio->nbits) {
  case 8:
    if(paudio->bsigned) {
      dpxio_audio_resample_int32le_to_int8(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    } else {
      dpxio_audio_resample_int32le_to_uint8(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
    }
    break;
  case 16:
    if(paudio->blittleendian) {
      if(paudio->bsigned) {
        dpxio_audio_resample_int32le_to_int16le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint16le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(paudio->bsigned) {
        dpxio_audio_resample_int32le_to_int16be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint16be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  case 32:
    if(paudio->blittleendian) {
      if(paudio->bsigned) {
        dpxio_audio_resample_int32le_to_int32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint32le(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    } else {
      if(paudio->bsigned) {
        dpxio_audio_resample_int32le_to_int32be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      } else {
        dpxio_audio_resample_int32le_to_uint32be(inputbuffer, outputbuffer, inputchannels, outputchannels, samplecount);
      }
    }
    break;
  default:
    printf("ERROR: dpxio_audio_resample_input: Unknown bitdepth %d\n", paudio->nbits);
    return FALSE;
  }

  return TRUE;
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


static int dpxio_wave_create(audio_handle * paudio, char * filename, int channels, int frequency, int nbits)
{
  return FALSE;
}

static int dpxio_wave_open(audio_handle * paudio, char * filename)
{
  FILE *  fp;
  char    buf[256];
  int     offset;
  int     count = 0;
  int     length;

  fp = fopen(filename, "rb");

  if(fp == NULL) {
    return AUDIO_FORMAT_INVALID;
  }

  offset = 12;

  while(count++ < 10000) {
    (void)fseek(fp, offset, SEEK_SET);
    if(fread(&buf[0], 1, sizeof(buf), fp) != sizeof(buf)) {
      fclose(fp);
      return AUDIO_FORMAT_INVALID;
    }

    length = wave_uint32(&buf[4]);

    if(strncmp(buf, "data", 4) == 0) {
      paudio->fp            = fp;
      paudio->binput        = FALSE;
      paudio->position      = offset + 8;
      paudio->bsigned       = TRUE;
      paudio->blittleendian = TRUE;
      return AUDIO_FORMAT_WAVE;
    }

    offset += ( 8 + length + 1 ) & ~1;  /* align to even byte boundary */
  }        

  fclose(fp);

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


static int dpxio_aiff_open(audio_handle * paudio, char * filename)
{
  FILE *  fp;
  char    buf[256];
  int     offset;
  int     count = 0;

  fp = fopen(filename, "rb");

  if(fp == NULL) {
    return AUDIO_FORMAT_INVALID;
  }

  while(count++ < 10000) {
    if(fread(&buf[0], 1, 2, fp) != 2) {
      fclose(fp);
      return AUDIO_FORMAT_INVALID;
    }  
 
    if (strncmp(buf, "COMM", 4) == 0) {
    } else if (strncmp(buf, "SS", 2) == 0) {
      if(fread(buf, 1, 2, fp) != 2) {
        fclose(fp);
        return AUDIO_FORMAT_INVALID;
      }  
      if(strncmp(buf, "ND", 2) == 0) {
        if(fread(buf, 1, 12, fp) != 12) {
          fclose(paudio->fp);
          paudio->fp = NULL;
          return AUDIO_FORMAT_INVALID;
        }

        offset = aiff_readint((uint8*)&buf[4]);

        paudio->fp            = fp;
        paudio->binput        = FALSE;
        paudio->position      = ftell(fp) + offset;
        paudio->bsigned       = TRUE;
        paudio->blittleendian = FALSE;

        (void)fseek(fp, paudio->position, SEEK_SET);

        return AUDIO_FORMAT_AIFF;
      }
    }
  }        

  return AUDIO_FORMAT_INVALID;
}


static int dpxio_aiff_create(audio_handle * paudio, char * filename, int frequency, int channels, int nbits)
{
  uint8  buf[DPXIO_AUDIO_HEADER_SIZE];
  FILE * fp;

  memset(buf, 0, sizeof(buf));

  fp = fopen(filename, "wb");
  if(fp == NULL) {
    printf("ERROR: audio_aiff_create: fopen(%s) failed\n", filename);
    return FALSE;
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
  aiff_writeint(&buf[50], 0);
  aiff_writeint(&buf[54], DPXIO_AUDIO_HEADER_SIZE - 62);
  aiff_writeint(&buf[58], 0);
 
  if(fwrite((char*)buf, 1, DPXIO_AUDIO_HEADER_SIZE, fp) != DPXIO_AUDIO_HEADER_SIZE) {
    printf("ERROR: dpxio_aiff_create: fwrite(%d) failed, errno:%d\n", DPXIO_AUDIO_HEADER_SIZE, errno);
    fclose(fp);
    return AUDIO_FORMAT_INVALID;
  }

  paudio->fp            = fp;
  paudio->format        = AUDIO_FORMAT_AIFF;
  paudio->binput        = TRUE;
  paudio->position      = sizeof(buf);
  paudio->nbits         = nbits;
  paudio->nchannels     = channels;
  paudio->frequency     = frequency;
  paudio->bsigned       = TRUE;
  paudio->blittleendian = FALSE;

  return AUDIO_FORMAT_AIFF;
}


static void dpxio_aiff_record_setsize(audio_handle * paudio, int fileposition)
{
  uint8 buffer[32];
  size_t res;

  aiff_writeint(buffer, fileposition - 8);
  (void)fseek(paudio->fp, 0x04, SEEK_SET);
  res = fwrite((char*)buffer, 1, sizeof(uint32), paudio->fp);
  if(res != sizeof(uint32)) {
    printf("ERROR: dpxio_aiff_record_setsize: fwrite(%d,%d) failed = %d errno:%d\n", 0x04, sizeof(uint32), res, errno);
  }

  aiff_writeint(buffer, (fileposition - DPXIO_AUDIO_HEADER_SIZE) / 4);
  (void)fseek(paudio->fp, 0x1e, SEEK_SET);
  res = fwrite((char*)buffer, 1, sizeof(uint32), paudio->fp);
  if(res != sizeof(uint32)) {
    printf("ERROR: dpxio_aiff_record_setsize: fwrite(%d,%d) failed = %d errno:%d\n", 0x1e, sizeof(uint32), res, errno);
  }

  aiff_writeint(buffer, fileposition - 54);
  (void)fseek(paudio->fp, 0x32, SEEK_SET);
  res = fwrite((char*)buffer, 1, sizeof(uint32), paudio->fp);
  if(res != sizeof(uint32)) {
    printf("ERROR: dpxio_aiff_record_setsize: fwrite(%d,%d) failed = %d errno:%d\n", fileposition - 54, sizeof(uint32), res, errno);
  }
}


int dpxio_audio_formatcheck(audio_handle * paudio, char * filename)
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
        paudio->nchannels = aiff_readshort(&header[i+8]);
        paudio->nsample   = aiff_readint(&header[i+10]);
        paudio->nbits     = aiff_readshort(&header[i+14]);
        paudio->frequency = (unsigned int) ((unsigned int) aiff_readshort(&header[i+18]) & 0x0000ffff);
      }
    }
  } else if(!strncmp((char*)header, "RIFF", 4) && !strncmp((char*)header + 8, "WAVE", 4)) {
    for(i = 12; (format == AUDIO_FORMAT_INVALID) && (i < sizeof(header)-30); i++) {
      if(strncmp((char*)(header+i), "fmt ", 4) == 0) {
        format            = AUDIO_FORMAT_WAVE;
        paudio->nsample   = 5000000;
        paudio->nchannels = wave_uint16(&header[10+i]);
        paudio->frequency = wave_uint32(&header[12+i]);
        paudio->nbits     = wave_uint16(&header[22+i]);
      }
    }
  }

  paudio->format = format;

  fclose(fp);

  switch(paudio->nbits) {
  case 8:
  case 16:
  case 24:
  case 32:
    break;
  default:
    return AUDIO_FORMAT_INVALID;
  }

  return paudio->format;
}


int dpxio_audio_create(audio_handle * paudio, char * filename, int format, int frequency, int channels, int nbits, int hwchannels)
{
  memset(paudio, 0, sizeof(paudio));

  switch(format) {
  case AUDIO_FORMAT_AIFF:
    format = dpxio_aiff_create(paudio, filename, frequency, channels, nbits);
    break;
  case AUDIO_FORMAT_WAVE:
    format = dpxio_wave_create(paudio, filename, frequency, channels, nbits);
    break;
  default:
    printf("ERROR: dpxio_audio_create() format:%d invalid\n", format);
    format = AUDIO_FORMAT_INVALID;
  }

  if(format != AUDIO_FORMAT_INVALID) {
    paudio->filebuffersize = 0x800 * paudio->nchannels * paudio->nbits / 8;
    paudio->filebuffer     = malloc(paudio->filebuffersize);
    paudio->hwchannels     = hwchannels;
    if(!paudio->filebuffer) {
      printf("ERROR: dpxio_audio_create() malloc(%d) failed\n", paudio->filebuffersize);
      format = AUDIO_FORMAT_INVALID;
    } 
  } 

  return format;
}


int dpxio_audio_open(audio_handle * paudio, char * filename, int hwchannels, int * pchannels, int * pfrequency)
{
  int format;

  memset(paudio, 0, sizeof(paudio));

  format = dpxio_audio_formatcheck(paudio, filename);

  switch(format) {
  case AUDIO_FORMAT_AIFF:
    format = dpxio_aiff_open(paudio, filename);
    break;
  case AUDIO_FORMAT_WAVE:
    format = dpxio_wave_open(paudio, filename);
    break;
  default:
    format = AUDIO_FORMAT_INVALID;
  }

  if(pchannels) {
    *pchannels = paudio->nchannels;
  }
  if(pfrequency) {
    *pfrequency = paudio->frequency;
  }

  if(format != AUDIO_FORMAT_INVALID) {
    paudio->filebuffersize = 0x800 * paudio->nchannels * paudio->nbits / 8;
    paudio->filebuffer     = malloc(paudio->filebuffersize);
    paudio->hwchannels     = hwchannels;
    if(!paudio->filebuffer) {
      printf("ERROR: dpxio_audio_create() malloc(%d) failed\n", paudio->filebuffersize);
      format = AUDIO_FORMAT_INVALID;
    } 
  } 

  return format;
}


int dpxio_audio_read(audio_handle * paudio, int framenr, int startchannel, void * buffer, int nsamples)
{
  size_t size = nsamples * paudio->nchannels * paudio->nbits / 8;
  size_t res  = fread(paudio->filebuffer, 1, size, paudio->fp);

  paudio->position += (int)res;

  if(res == size) {
    buffer = (void *)((uintptr) buffer + (startchannel * 4 /*byte per sample*/ * 2 /*stereo*/));
    dpxio_audio_resample_output(paudio, paudio->filebuffer, buffer, paudio->nchannels, paudio->hwchannels, nsamples);

    return TRUE;
  }

  return FALSE;
}


int dpxio_audio_write(audio_handle * paudio, int framenr, int startchannel, void * buffer, int nsamples)
{
  size_t size = nsamples * paudio->nchannels * paudio->nbits / 8;
  size_t res;

  buffer = (void *)((uintptr) buffer + (startchannel * 4 /*byte per sample*/ * 2 /*stereo*/));

  res  = dpxio_audio_resample_input(paudio, buffer, paudio->filebuffer, paudio->hwchannels, paudio->nchannels, nsamples);
  
  if(res) {
    res = fwrite(paudio->filebuffer, 1, size, paudio->fp);

    paudio->position += (int)res;

    return (res == size);
  }

  return FALSE;
}


void dpxio_audio_close(audio_handle * paudio)
{
  if(paudio->fp) {
    if(paudio->binput && (paudio->format == AUDIO_FORMAT_AIFF)) {
      dpxio_aiff_record_setsize(paudio, paudio->position);
    }
    fclose(paudio->fp);
    paudio->fp = NULL;
  }

  if(paudio->filebuffer) {
    free(paudio->filebuffer);
    paudio->filebuffer = NULL;
  }
}

