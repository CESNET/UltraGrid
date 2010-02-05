/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK
//
//    convert2bgra - converts a memory buffer to bgra 8 bit
*/

#ifndef DVSSDK_CONVERT2BGRA_H
#define DVSSDK_CONVERT2BGRA_H

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"

int convert_tobgra(unsigned char * psource, unsigned char * pdest, int xsize, int ysize, int interlace, int lineoffset, int storagemode, int bottom2top, int bviewalpha, int fields);

#endif
