/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This applications test the hosttransfer performance for DDR and the DMA
//    performance for the OEM products.
//
*/



#include <stdio.h>
#ifndef WIN32
#  include <sys/time.h>
#  include <unistd.h>
#else
#  include <windows.h>
#endif
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"

int  verbose = 0;
int  running = 1;

#define ALIGNMENT 0x10000


int sv_transfertest(sv_handle * sv, int savemode, char * buffer, int size, int
start, int nframes, int xsize, int ysize, int nbits)
{
  int         res = SV_OK;
  int         count;

  for(count = 0; (count < nframes) && running; count++) {
    int frame = start + count; 

    if(savemode) {
      res = sv_sv2host(sv, buffer, size, xsize, ysize, frame, 1, 0);
    } else {
      res = sv_host2sv(sv, buffer, size, xsize, ysize, frame, 1, 0);
    }

    if(res != SV_OK) {
      fprintf(stderr, "sv_sv2host failed: ");
      sv_errorprint(sv, res);
      return res;
    }
  }

  return res;
}



void signal_handler(int signum)
{
  running = 0;
}

void usage(void)
{
  printf("usage: dmaspeed {load,save,read,write}\n");
  printf("       dmaspeed load #nframes -> Transfers from CPU memory to Device memory\n");
  printf("       dmaspeed save #nframes -> Transfers from Device memory to CPU memory\n");
  printf("       dmaspeed read  #blocksize -> DMA Read from System Memory, Transfers from CPU memory to Device memory\n");
  printf("       dmaspeed write #blocksize -> DMA Write to System Memory, Transfers from Device memory to CPU memory\n");
}

int resultus[32] ={0};
int alignment[] = {
  16384,
  8192,
  4096,
  2048,
  1024,
  512,
  256,
  128,
  64,
  32,
  16,
  8,
  0,
};

int main(int argc, char ** argv)
{
  sv_handle * sv    = NULL;
  int         res   = SV_OK;
  int         bsavemode;
  int         bhosttransfer;
  int         bsetbuffersize = 0;
  int         nframes;
  sv_storageinfo     storage;
  char *      buffer;
  char *      buffer_org;
  int         buffersize;
  int         us;
  int         i, j;
#ifdef WIN32
  LARGE_INTEGER frequency;
  LARGE_INTEGER timestart;
  LARGE_INTEGER timeend;
  QueryPerformanceFrequency(&frequency);
#else
  struct timeval timestart;
  struct timeval timeend;
  struct timezone tz;
#endif


  if(argc < 2) {
    usage();
    return -1;
  }

  if(!strcmp(argv[1], "load")) {
    bsavemode     = 0;
    bhosttransfer = 1;
  } else if(!strcmp(argv[1], "save")) {
    bsavemode     = 1;
    bhosttransfer = 1;
  } else if(!strcmp(argv[1], "read")) {
    bsavemode     = 0;
    bhosttransfer = 0;
  } else if(!strcmp(argv[1], "write")) {
    bsavemode     = 1;
    bhosttransfer = 0;
  } else {
    usage();
    return -1;
  }

  bhosttransfer = 1;

#ifndef WIN32
  signal(SIGINT,  signal_handler);
  signal(SIGKILL, signal_handler);
#endif

  res = sv_openex(&sv, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_DEFAULT, 0, 0);
  if(res != SV_OK) {
    printf("dmaspeed: Error '%s' opening video device", sv_geterrortext(res));
    return -1;
  } 

  res = sv_storage_status(sv, 0, NULL, &storage, sizeof(storage), 0);

  if(bhosttransfer) {
    buffersize = storage.fieldsize[0] + storage.fieldsize[1];
    if(argc >= 3) {
      nframes = atoi(argv[2]);
    } else {
      nframes = 10;
    }
  } else {
    if(argc >= 3) {
      buffersize = atoi(argv[2]);
      if(buffersize == 0) {
        buffersize = strtol(argv[2], NULL, 16);
      }
      bsetbuffersize = 1;
    } else {
      buffersize = 0x200000;
    }
    nframes    = 1;
  }

  buffer_org = malloc(buffersize + ALIGNMENT + alignment[0]);
  buffer = (void*)(((uintptr)buffer_org + ALIGNMENT - 1) & ~(ALIGNMENT-1));

  if(buffer == NULL) {
    fprintf(stderr, "dmaspeed: Failure to allocate video buffer\n");
    res = SV_ERROR_MALLOC;
  }

  if(res == SV_OK) {
    if(bhosttransfer) {
      if(bsavemode) {
        printf("dmaspeed: frames     %d frames, Device -> CPUMemory\n", nframes);
      } else {
        printf("dmaspeed: frames     %d frames, CPUMemory -> Device\n", nframes);
      } 
      printf("dmaspeed: framesize  %d\n", buffersize);
      printf("dmaspeed: totalsize  %d bytes total\n", nframes * buffersize);
  
#ifdef WIN32
      QueryPerformanceCounter(&timestart);
#else
      gettimeofday(&timestart, &tz);
#endif

      res = sv_transfertest(sv, bsavemode, buffer, buffersize, 0, nframes, storage.storagexsize, storage.storageysize, storage.nbits);

#ifdef WIN32
      QueryPerformanceCounter(&timeend);
#else
      gettimeofday(&timeend, &tz);
#endif

      if(res == SV_ERROR_PARAMETER) {
        printf("dmaspeed: returned SV_ERROR_PARAMTER, did you specify more frames than in memory ?\n");
      } else if(res == SV_OK) {
#ifdef WIN32
#ifdef VERBOSE
        printf("dmaspeed: frequency  %d\n", (int)frequency.QuadPart);
        printf("dmaspeed: time       %d\n", (int)(timeend.QuadPart - timestart.QuadPart));
#endif

        if(frequency.QuadPart) {
          us = (int)(((timeend.QuadPart - timestart.QuadPart) * 1000000) / frequency.QuadPart);
        } else {
          us = 0;
        }
#else
        us = (int)((timeend.tv_sec * 1000000 + timeend.tv_usec) - (timestart.tv_sec * 1000000 + timestart.tv_usec));

#ifdef VERBOSE
        printf("dmaspeed: frequency  %d\n", 1000000);
        printf("dmaspeed: time       %d\n", us);
#endif
#endif

        if(us) {
          printf("dmaspeed: %d us %d B/s %d MB/s\n", us, (int)(((int64)nframes * buffersize * 1000000) / us), (int)(((int64)nframes * buffersize) / us));
        } else {
          printf("dmaspeed: Error 0 us execution time ?\n");
        }
      }
    } else {
      sv_memory_dma(sv, bsavemode, (void*)((uintptr)buffer), 0x100000, buffersize, NULL);

      for(i = 0; (res == SV_OK) && (i < arraysize(alignment)); i++) {
#ifdef WIN32
        QueryPerformanceCounter(&timestart);
#else
        gettimeofday(&timestart, &tz);
#endif
        if(bsetbuffersize) {
          res = sv_memory_dma(sv, bsavemode, (void*)((uintptr)buffer+alignment[i]), 0x100000, buffersize, NULL);
        } else {
          res = sv_memory_dma(sv, bsavemode, (void*)((uintptr)buffer), 0x100000, buffersize, NULL);
        }

        if((res != SV_OK) && (res != SV_ERROR_BUFFER_NOTALIGNED)) {
          fprintf(stderr, "sv_memory_dma failed: "); sv_errorprint(sv, res);
        }
#ifdef WIN32
        QueryPerformanceCounter(&timeend);
        if(frequency.QuadPart) {
          us = (int)(((timeend.QuadPart - timestart.QuadPart) * 1000000) / frequency.QuadPart);
        } else {
          us = 0;
        }
#else
        gettimeofday(&timeend, &tz);

        us = (int)((timeend.tv_sec * 1000000 + timeend.tv_usec) - (timestart.tv_sec * 1000000 + timestart.tv_usec));
#endif
        resultus[i] = us;
      }

      j = i - 1;
      if(res == SV_ERROR_BUFFER_NOTALIGNED) {
        res = SV_OK;
      }

      if(bsetbuffersize) {
        for(i = 0; (i < arraysize(alignment)) && (i < j); i++) {
          if(resultus[i]) {
            printf("dmaspeed %3d e6/s  alignment:%08x/%6d time %d us\n", buffersize / resultus[i], alignment[i], alignment[i], resultus[i]);
          } else {
            printf("dmaspeed %3d e6/s  alignment:%08x/%6d time %d us ****\n", 0, alignment[i], alignment[i], 0);
          }
        } 
      } else {
        for(us = 0, i = 0; (i < arraysize(alignment)) && (i < j); i++) {
          us += resultus[i];
        }
        printf("Transferspeed: %3d e6/s %s\n", (int)(buffersize / (us / arraysize(alignment))), bsavemode?"CPUMemory->Device":"Device->CPUMemory");
      } 
    }
  } else {
    printf("dmaspeed: returned %s\n", sv_geterrortext(res));
  }
  sv_close(sv);

  free(buffer_org);

  if(res != SV_OK) {
    sv_errorprint(sv, res);
  }

  if(res == SV_OK) {
    return 0;  
  } else {
    return -1;
  }
}
