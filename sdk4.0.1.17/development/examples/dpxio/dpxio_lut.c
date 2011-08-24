// Lut helper functions
// Creating some example luts for testing.

#include <string.h>

#define limit(value) ((unsigned short)(((value)>0xffff)?0xffff:(value)))

void dpxio_3dlut_invert(unsigned char * lut)
{
  int ramp = 0;
  int r,g,b;

  for(g = 16; g >= 0; g--) {
    for(b = 16; b >= 0; b--) {
      for(r = 16; r >= 0; r--) {
        *((unsigned short *)lut + (ramp++)) = limit(0x10000 * g / 16);
        *((unsigned short *)lut + (ramp++)) = limit(0x10000 * b / 16);
        *((unsigned short *)lut + (ramp++)) = limit(0x10000 * r / 16);
      }
    }
  }
}
void dpxio_3dlut_identity(unsigned char * lut)
{
  int ramp = 0;
  int r,g,b;

  for(g = 0; g < 17; g++) {
    for(b = 0; b < 17; b++) {
      for(r = 0; r < 17; r++) {
        *((unsigned short *)lut + (ramp++)) = limit(0x10000 * g / 16);
        *((unsigned short *)lut + (ramp++)) = limit(0x10000 * b / 16);
        *((unsigned short *)lut + (ramp++)) = limit(0x10000 * r / 16);
      }
    }
  }
}

void dpxio_1dlut_invert(unsigned char * lut)
{
  unsigned int * lastnode;
  int ramp;

  for(ramp = 0; ramp < 1024; ramp++) {
    *((unsigned int *)lut + ramp) = limit(0x10000 * (1023 - ramp) / 1024);
  }

  memcpy(lut + 0x1000, lut, 0x1000);
  memcpy(lut + 0x2000, lut, 0x1000);
  memcpy(lut + 0x3000, lut, 0x1000);

  lastnode = (unsigned int *)(lut + 0x4000);
  lastnode[0] = lastnode[1] = lastnode[2] = lastnode[3] = 0;
}

void dpxio_1dlut_identity(unsigned char * lut)
{
  unsigned int * lastnode;
  int ramp;

  for(ramp = 0; ramp < 1024; ramp++) {
    *((unsigned int *)lut + ramp) = limit(0x10000 * ramp / 1024);
  }

  memcpy(lut + 0x1000, lut, 0x1000);
  memcpy(lut + 0x2000, lut, 0x1000);
  memcpy(lut + 0x3000, lut, 0x1000);

  lastnode = (unsigned int *)(lut + 0x4000);
  lastnode[0] = lastnode[1] = lastnode[2] = lastnode[3] = 0xffff;
}

void dpxio_1dlutsmall_invert(unsigned char * lut)
{
  int ramp;

  for(ramp = 0; ramp < 1024; ramp++) {
    *((unsigned short *)lut + ramp) = (unsigned short)((1023 - ramp) << 6);
  }

  memcpy(lut + 0x800, lut, 0x800);
  memcpy(lut + 0x1000, lut, 0x800);
}

void dpxio_1dlutsmall_identity(unsigned char * lut)
{
  int ramp;

  for(ramp = 0; ramp < 1024; ramp++) {
    *((unsigned short *)lut + ramp) = (unsigned short)(ramp << 6);
  }

  memcpy(lut + 0x800, lut, 0x800);
  memcpy(lut + 0x1000, lut, 0x800);
}

