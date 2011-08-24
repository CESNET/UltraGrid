/*
// LUT helper functions which are creating some example LUTs.
*/

#include <string.h>

#define limit(value) ((unsigned short)(((value)>0xffff)?0xffff:(value)))


void generate3DLutInverse(void * pLut)
{
  int ramp = 0;
  int r,g,b;

  for(g = 16; g >= 0; g--) {
    for(b = 16; b >= 0; b--) {
      for(r = 16; r >= 0; r--) {
        *((unsigned short *)pLut + (ramp++)) = limit(0x10000 * g / 16);
        *((unsigned short *)pLut + (ramp++)) = limit(0x10000 * b / 16);
        *((unsigned short *)pLut + (ramp++)) = limit(0x10000 * r / 16);
      }
    }
  }
}


void generate3DLutIdentity(void * pLut)
{
  int ramp = 0;
  int r,g,b;

  for(g = 0; g < 17; g++) {
    for(b = 0; b < 17; b++) {
      for(r = 0; r < 17; r++) {
        *((unsigned short *)pLut + (ramp++)) = limit(0x10000 * g / 16);
        *((unsigned short *)pLut + (ramp++)) = limit(0x10000 * b / 16);
        *((unsigned short *)pLut + (ramp++)) = limit(0x10000 * r / 16);
      }
    }
  }
}


void generate1DLutInverse(void * pLut, int lutElements, int components)
{
  int component = 0;
  int element   = 0;
  int ramp      = 0;

  // Fill inverse lut
  for(component = 0; component < components; component++) {
    for(element = 0; element < lutElements; element++) {
      if(component < 3) { // R,G,B
        *((unsigned int *)pLut + (ramp++)) = limit(0x10000 * ((lutElements - 1) - element) / lutElements);
      } else {            // Alpha
        *((unsigned int *)pLut + (ramp++)) = 0;
      }
    }
  }
}


void generate1DLutIdentity(void * pLut, int lutElements, int components)
{
  int component = 0;
  int element   = 0;
  int ramp      = 0;

  // Fill identity lut
  for(component = 0; component < components; component++) {
    for(element = 0; element < lutElements; element++) {
      if(component < 3) { // R,G,B
        *((unsigned int *)pLut + (ramp++)) = limit(0x10000 * element / lutElements);
      } else {            // Alpha
        *((unsigned int *)pLut + (ramp++)) = 0;
      }
    }
  }
}
