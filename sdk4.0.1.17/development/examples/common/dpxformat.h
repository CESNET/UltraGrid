#ifndef DVSSDK_DPXFORMAT_H
#define DVSSDK_DPXFORMAT_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#ifndef WIN32
# include <fcntl.h>
#endif
#include <string.h>

#include "../../header/dvs_setup.h"
#include "../../header/dvs_version.h"

int dpxformat_getstoragemode(int dpxformat, int nbits);
int dpxformat_fillheader(void * vpheader, int offset, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode);
int dpxformat_readheader(void * vpheader, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset);
int dpxformat_framesize(int xsize, int ysize, int dpxmode, int nbits, int * poffset, int * ppadding);
int dpxformat_framesize_noheader(int xsize, int ysize, int dpxmode, int nbits, int * ppadding);
int dpxformat_dpxtype(int storagemode, int * pdpxtype, int * pnbits, int * pdittokey);
int dpxformat_readframe(char * filename, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset);
int dpxformat_readframe_noheader(char * filename, char * buffer, int buffersize, int * offset, int * xsize, int * ysize, int * dpxtype, int * nbits, int * lineoffset);
int dpxformat_writeframe(char * filename, char * buffer, int buffersize, int offset, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode);
int dpxformat_writeframe_noheader(char * filename, char * buffer, int buffersize, int xsize, int ysize, int dpxtype, int nbits, int padding, int timecode);
int dpxformat_getoffset(void);
int dpxformat_file_exists(char * filename);

#endif

