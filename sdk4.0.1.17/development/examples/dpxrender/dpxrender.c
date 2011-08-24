/*
//    Part of the DVS SDK (http://www.dvs.de)
//
//    dpxrender - Shows the usage of the Render API.
//
*/
#include "dpxrender.h"

// Globals
typedef struct {
  int xsize;
  int ysize;
  int nbits;
  int dpxType;
  int offset;
} frameInfo;

typedef struct {
  struct {
    int     enable;
    double  xsize;
    double  ysize;
    double  xoffset;
    double  yoffset;
    double  sharpness;
    int     scalerType;
  } scale;

  struct {
    int    enable;
  } crop;

  struct {
    int    enable;
    double matrix[10];
    double offset[4];
  } matrix;
} renderConfig;

typedef struct {
  char * pBuffer;
  char * pAligned;
  unsigned int bufferSize;
} videoBuffer_t;

typedef struct {
  int id;
  int result;

  dvs_thread handle;
  dvs_cond exit;
  dvs_cond running;

  sv_render_context * pSvRc;
  renderHandle * pRh;
} renderThreadInfo;

#include "tags.h"

#define LUT_ELEMENTS_12_BIT   (1 << 12)
#define COMPONENTS_RGBA       4

dvs_mutex gFrameLock;
int gFrameNr = 0;
int gRunning = TRUE;



/*
// Signal handler
*/
void signalHandler(int signal) 
{
  printf("closing application...\n");
  gRunning = FALSE;
}


/*
// Helper function for making sure that all frames will be rendered.
*/
void GetNextFrame(int * pFrameNr)
{
  dvs_mutex_enter(&gFrameLock);
  *pFrameNr = gFrameNr++;
  dvs_mutex_leave(&gFrameLock);
}


/*
// Helper function to allocate and align buffers.
*/
int allocateAlignedBuffer(videoBuffer_t * pVideoBuffer, unsigned int size, unsigned int alignment)
{
  int result = TRUE;

  if(result) {    
    pVideoBuffer->bufferSize = (size + (alignment - 1)) & ~(alignment - 1);
    pVideoBuffer->pBuffer    = (char *) malloc(((pVideoBuffer->bufferSize + alignment) + (alignment - 1)) & ~(uintptr)(alignment - 1));
    pVideoBuffer->pAligned   = (char *)(((uintptr)pVideoBuffer->pBuffer + (alignment - 1)) & ~(uintptr)(alignment - 1));
    if(!pVideoBuffer->pAligned) {
      result = FALSE;
      printf("allocateAlignedBuffer(): Unable to allocate video buffer!\n");
    }
  }

  if(result) {
    memset(pVideoBuffer->pAligned, 0x00, pVideoBuffer->bufferSize);
  }

  return result;
}


/*
// In this helper function the scaler parameter interpolation is done.
*/
double renderInterpolateDouble(int frame, int frameCount, double start, double end)
{
  return start + ((end - start) * frame) / frameCount;
}

/*
// This helper function is called to set scaler parameters.
// If specified by the user there will be done an interpolation of the scaler values.
*/
int renderInterpolate(renderHandle * pRender, renderConfig * pConfig, int frame, int frameCount)
{
  double factor;

  if(!frameCount) {
    return FALSE;
  }

  /*
  //  Scaler parameters
  */
  pConfig->scale.enable     = pRender->scale.enable;
  pConfig->scale.scalerType = pRender->scale.scalerType;
  pConfig->scale.xsize      = renderInterpolateDouble(frame, frameCount, pRender->scale.xsize.start,     pRender->scale.xsize.end);
  pConfig->scale.ysize      = renderInterpolateDouble(frame, frameCount, pRender->scale.ysize.start,     pRender->scale.ysize.end);
  pConfig->scale.xoffset    = renderInterpolateDouble(frame, frameCount, pRender->scale.xoffset.start,   pRender->scale.xoffset.end);
  pConfig->scale.yoffset    = renderInterpolateDouble(frame, frameCount, pRender->scale.yoffset.start,   pRender->scale.yoffset.end);
  pConfig->scale.sharpness  = renderInterpolateDouble(frame, frameCount, pRender->scale.sharpness.start, pRender->scale.sharpness.end);

  pConfig->matrix.enable    = pRender->matrix.enable;

  if(pRender->matrix.enable) {
    // If there was no user value set, specify an identity matrix.
    if(!pRender->matrix.valueSet) {
      pConfig->matrix.matrix[0] = 1.0;
      pConfig->matrix.matrix[1] = 0;
      pConfig->matrix.matrix[2] = 0;
      pConfig->matrix.matrix[3] = 0;
      pConfig->matrix.matrix[4] = 1.0;
      pConfig->matrix.matrix[5] = 0;
      pConfig->matrix.matrix[6] = 0;
      pConfig->matrix.matrix[7] = 0;
      pConfig->matrix.matrix[8] = 1.0;
      pConfig->matrix.matrix[9] = 1.0;
      pConfig->matrix.offset[0] = 0;
      pConfig->matrix.offset[1] = 0;
      pConfig->matrix.offset[2] = 0;
      pConfig->matrix.offset[3] = 0;
    }

    if(pRender->matrix.factorRed.enable) {
      factor = renderInterpolateDouble(frame, frameCount, pRender->matrix.factorRed.start, pRender->matrix.factorRed.end);
  
      pConfig->matrix.matrix[0] = pConfig->matrix.matrix[0] * factor;
      pConfig->matrix.matrix[3] = pConfig->matrix.matrix[3] * factor;
      pConfig->matrix.matrix[6] = pConfig->matrix.matrix[6] * factor;
    }
    if(pRender->matrix.factorGreen.enable) {
      factor = renderInterpolateDouble(frame, frameCount, pRender->matrix.factorGreen.start, pRender->matrix.factorGreen.end);
  
      pConfig->matrix.matrix[1] = pConfig->matrix.matrix[1] * factor;
      pConfig->matrix.matrix[4] = pConfig->matrix.matrix[4] * factor;
      pConfig->matrix.matrix[7] = pConfig->matrix.matrix[7] * factor;
    }
    if(pRender->matrix.factorBlue.enable) {
      factor = renderInterpolateDouble(frame, frameCount, pRender->matrix.factorBlue.start, pRender->matrix.factorBlue.end);
  
      pConfig->matrix.matrix[2] = pConfig->matrix.matrix[2] * factor;
      pConfig->matrix.matrix[5] = pConfig->matrix.matrix[5] * factor;
      pConfig->matrix.matrix[8] = pConfig->matrix.matrix[8] * factor;
    }
  }

  return TRUE;
}


/*
// This is the render execution function of the example program used by each thread.
*/
int renderExecute(renderThreadInfo * pThreadInfo, videoBuffer_t * pSourceBuffer, videoBuffer_t * pTargetBuffer, int frameNr)
{
  char sourceFile[MAX_PATHNAME];
  char destFile[MAX_PATHNAME];

  int sourceImageSize = 0;
  int destImageSize   = 0;
  int frameSize       = 0;
  int dmaOffset       = 0;
  int dmaSize         = 0;
  int timeout         = 0;
  int res             = SV_OK;
  int running         = TRUE;
  int retryCount      = 0;

  renderHandle * pRender   = pThreadInfo->pRh;

  sv_render_handle  * pSvRh = pRender->pSvRh;
  sv_render_context * pSvRc = pThreadInfo->pSvRc;

  renderConfig config;
  renderConfig * pConfig = &config;
  
  struct {
    frameInfo               info;
    sv_render_image *       pRenderImage;
    sv_render_bufferlayout  bufferLayout;
  } source,dest;

  if(pRender->verbose) {
    printf(" renderExecute(%d): Start rendering frame %d\n", pThreadInfo->id, frameNr);
  }

  if(pRender->bRetry) {
    retryCount = 3;
  }

  if(gRunning) {
    sprintf(sourceFile, pRender->source.fileName, pRender->startFrame + frameNr);
    sprintf(destFile, pRender->destination.fileName, frameNr);

    // Read needed properties (e.g. xsize, ysize, dpxtype, nbits, ...) from source dpx file.
    frameSize = dpxformat_readframe_noheader(sourceFile, pSourceBuffer->pAligned, pSourceBuffer->bufferSize, &source.info.offset, &source.info.xsize, &source.info.ysize, &source.info.dpxType, &source.info.nbits, NULL);

    if(frameSize) {
      // Just do a 1:1 copy from source to target frameinfo struct.
      memcpy(&dest.info, &source.info, sizeof(frameInfo));
     
      /*
      // By setting a smaller destination x-size (parameter '-dx=#') and/or y-size (parameter '-dy=#')
      // than the source x-size/y-size and if there is no explicit scaler setting pushed (sv_render_push_scaler())
      // there will be performed an automatic downscaling.
      */
      if(pRender->output.xsize) {
        dest.info.xsize = pRender->output.xsize;
      }      
      if(pRender->output.ysize) {
        dest.info.ysize = pRender->output.ysize;
      } 

      if(!renderInterpolate(pRender, pConfig, frameNr, pRender->frameCount)) {
        printf(" renderExecute(%d): renderInterpolate() failed!\n", pThreadInfo->id);
        running = FALSE;
      }

      if(running) {
        // Set the properties of the source buffer's layout.
        memset(&source.bufferLayout, 0, sizeof(sv_render_bufferlayout));
        source.bufferLayout.v1.xsize       = source.info.xsize;
        source.bufferLayout.v1.ysize       = source.info.ysize;
        source.bufferLayout.v1.storagemode = dpxformat_getstoragemode(source.info.dpxType, source.info.nbits);
        source.bufferLayout.v1.lineoffset  = 0;
        source.bufferLayout.v1.dataoffset  = source.info.offset;
        sourceImageSize                    = dpxformat_framesize_noheader(source.info.xsize, source.info.ysize, source.info.dpxType, source.info.nbits, NULL);

        /*
        // Try to allocate space on the DVS video board for the source buffer.
        // This may fail due to memory fragmentation, so retry up to 100 times only.
        */
        timeout = 100;
        do {
          res = sv_render_malloc(pRender->pSv, pSvRh, &source.pRenderImage, 1, sizeof(sv_render_bufferlayout), &source.bufferLayout);
          if(res == SV_ERROR_MALLOC) {
            sv_usleep(pRender->pSv, 2000); // Wait 2 milliseconds until the next retry.
          }
        } while((res == SV_ERROR_MALLOC) && (--timeout > 0));
 
        if((res != SV_OK) || (timeout <= 0)) {
          printf(" renderExecute(%d): sv_render_malloc(src) failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
          running = FALSE;
        }
      }

      if(running) {
        // Set the properties of the destination buffer's layout.
        memset(&dest.bufferLayout, 0, sizeof(sv_render_bufferlayout));
        dest.bufferLayout.v1.xsize       = dest.info.xsize;
        dest.bufferLayout.v1.ysize       = dest.info.ysize;
        dest.bufferLayout.v1.storagemode = dpxformat_getstoragemode(dest.info.dpxType, dest.info.nbits);
        dest.bufferLayout.v1.lineoffset  = 0;
        dest.bufferLayout.v1.dataoffset  = dest.info.offset;
        destImageSize                    = dpxformat_framesize_noheader(dest.info.xsize, dest.info.ysize, dest.info.dpxType, dest.info.nbits, NULL);

        /*
        // Try to allocate space on the DVS video board for the destination buffer.
        // This may fail due to memory fragmentation, so retry up to 100 times only.
        */
        timeout = 100;
        do {
          res = sv_render_malloc(pRender->pSv, pSvRh, &dest.pRenderImage, 1, sizeof(sv_render_bufferlayout), &dest.bufferLayout);
          if(res == SV_ERROR_MALLOC) {
            sv_usleep(pRender->pSv, 2000); // Wait 2 milliseconds until the next retry.
          }
        } while((res == SV_ERROR_MALLOC) && (--timeout > 0));

        if((res != SV_OK) || (timeout <= 0)) {
          printf(" renderExecute(%d): sv_render_malloc(dst) failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
          running = FALSE;
        }
      }

      // Transfer the source buffer from the system RAM to the DVS video board RAM.
      // This is done in maximum chunk sizes of 4MB.
      dmaOffset = 0;
      while(running && (dmaOffset < sourceImageSize)) {
        dmaSize = sourceImageSize - dmaOffset;
        if(dmaSize > 0x400000) {
          dmaSize = 0x400000;
        }

        res = sv_render_dma(pRender->pSv, pSvRh, TRUE, source.pRenderImage, &pSourceBuffer->pAligned[dmaOffset], dmaOffset, dmaSize, NULL);
        if(res != SV_OK) {
          printf(" renderExecute(%d): sv_render_dma() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
          running = FALSE;
        }

        dmaOffset += dmaSize;
      }


      if(running) {
        do
        {
          /*
          // The following sv_render_push_XXX functions show how the render stack can be prepared.
          */

          /*
          // Push the source image data on the stack.
          // note: At the moment we can push only one image on the stack.
          //       This might change in future SDK versions.
          */
          if (running) {
            res = sv_render_push_image(pRender->pSv, pSvRh, pSvRc, source.pRenderImage);
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_push_image() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          // If a 1D LUT is selected by the user push a 1dlut setting for the preceding image on the render stack.
          if(running && pRender->lut1d.enable) {
            int lutElements = LUT_ELEMENTS_12_BIT;
            int components  = COMPONENTS_RGBA;
            unsigned int lutData[COMPONENTS_RGBA * LUT_ELEMENTS_12_BIT];
            sv_render_1dlut lut;

            pRender->lut1d.pLut = lutData;
            pRender->lut1d.size = sizeof(unsigned int) * components * lutElements;

            memset(&lut, 0, sizeof(sv_render_1dlut));
            memset(pRender->lut1d.pLut, 0, pRender->lut1d.size);

            // Create identity 1D LUT.
            generate1DLutIdentity(pRender->lut1d.pLut, lutElements, components);

            lut.v1.lutid = 0; // First one in the pipeline
            lut.v1.plut  = pRender->lut1d.pLut;
            lut.v1.size  = pRender->lut1d.size;
            lut.v1.flags = 0;

            res =  sv_render_push_1dlut(pRender->pSv, pSvRh, pSvRc, 1, sizeof(sv_render_1dlut), &lut );
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_push_1dlut() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          // If a 3D LUT is selected by the user push a 3dlut setting for the preceding image on the render stack.
          if(running && pRender->lut3d.enable) {
            unsigned short * pOutBuffer = NULL;
            sv_render_3dlut lut;
            memset(&lut, 0, sizeof(sv_render_3dlut));

            // Allocate LUT buffer.
            pOutBuffer = (unsigned short *) malloc(0x8000); // 32768 bytes as expected by the driver
            memset(pOutBuffer, 0, 0x8000);

            pRender->lut3d.size = 0x8000;
            pRender->lut3d.pLut = pOutBuffer;

            // Fill buffer with 3D identity LUT.
            generate3DLutIdentity(pRender->lut3d.pLut);

            lut.v1.lutid = 0; // First one in the pipeline
            lut.v1.plut  = pRender->lut3d.pLut;
            lut.v1.size  = pRender->lut3d.size;
            lut.v1.flags = 0;

            res = sv_render_push_3dlut(pRender->pSv, pSvRh, pSvRc, 1, sizeof(sv_render_3dlut), &lut);
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_push_3dlut() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          // If scaling is selected by the user push a scaler setting for the preceding image on the render stack.
          // Please look into the DVS Render API Extension supplement reference guide for possible limitations.
          if(running && pRender->scale.enable) {
            sv_render_scaler scaler;
            memset(&scaler, 0, sizeof(scaler));

            scaler.v1.xsize   = pConfig->scale.xsize;
            scaler.v1.ysize   = pConfig->scale.ysize;
            scaler.v1.xoffset = pConfig->scale.xoffset; // xoffset > 0 is currently not implemented
            scaler.v1.yoffset = pConfig->scale.yoffset; // yoffset > 0 is currently not implemented

            res = sv_render_push_scaler(pRender->pSv, pSvRh, pSvRc, source.pRenderImage, 1, sizeof(sv_render_scaler), &scaler);
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_push_scaler() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          // If matrix is selected by the user push a matrix setting for the preceding image on the render stack.
          if(running && pConfig->matrix.enable) {
            int i;
            sv_render_matrix matrix;

            for(i = 0; i < 10; i++) {
              matrix.v1.matrix[i] = pConfig->matrix.matrix[i];
            }
            for(i = 0; i < 4; i++) {
              matrix.v1.offset[i] = pConfig->matrix.offset[i];
            }
            res = sv_render_push_matrix(pRender->pSv, pSvRh, pSvRc, 1, sizeof(sv_render_matrix), &matrix);
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_push_matrix() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          // Push a render operator on the stack for the preceding image and the preceding setting(s).
          if (running) {
            res = sv_render_push_render(pRender->pSv, pSvRh, pSvRc, dest.pRenderImage);
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_push_render() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          /*
          // Start the render operation.
          // note: This example does not use parameter 'poverlapped' which means
          //       that sv_render_issue() is blocking until rendering is finished.
          //       So waiting with sv_render_ready() is not required.
          */
          if (running) {
            res = sv_render_issue(pRender->pSv, pSvRh, pSvRc, NULL);
            if(res != SV_OK) {
              printf(" renderExecute(%d): sv_render_issue() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
              running = FALSE;
            }
          }

          // Reset the render context for the next frame that shall be rendered.
          // Do this even if an error happenend because then the render context is in a valid state again.
          res = sv_render_reuse(pRender->pSv, pSvRh, pSvRc);
          if(res != SV_OK) {
            printf(" renderExecute(%d): sv_render_reuse() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
            running = FALSE;
          }

          if(!running && gRunning) {
            if(retryCount-- > 0) {
              running = TRUE;
              if(pRender->verbose) {
                printf(" renderExecute(%d): retry rendering of frame %d...\n", pThreadInfo->id, frameNr);
              }
            }
          } else {
            retryCount = 0;
          }
        } while(gRunning && (retryCount > 0));
      }

      // Transfer the target buffer back to the system memory.
      // This is done in maximum chunk sizes of 4MB.
      dmaOffset = 0;
      while(running && (dmaOffset < destImageSize)) {
        dmaSize = destImageSize - dmaOffset;
        if(dmaSize > 0x400000) {
          dmaSize = 0x400000;
        }

        res = sv_render_dma(pRender->pSv, pSvRh, FALSE, dest.pRenderImage, &pTargetBuffer->pAligned[dmaOffset], dmaOffset, dmaSize, NULL);
        if(res != SV_OK) {
          printf(" renderExecute(%d): sv_render_dma() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
          running = FALSE;
        }

        dmaOffset += dmaSize;
      }

      // Free the allocated source buffer on the DVS video device to make it available again.
      if (running) {
        res = sv_render_free(pRender->pSv, pSvRh, source.pRenderImage);
        if(res != SV_OK) {
          printf(" renderExecute(%d): sv_render_free() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
          running = FALSE;
        }
      }

      // Free the allocated destination buffer on the DVS video device to make it available again.
      if (running) {
        res = sv_render_free(pRender->pSv, pSvRh, dest.pRenderImage);
        if(res != SV_OK) {
          printf(" renderExecute(%d): sv_render_free() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
          running = FALSE;
        }
      }

      // Write the target buffer into a dpx file.
      if(running) {
        frameSize = dpxformat_writeframe_noheader(destFile, pTargetBuffer->pAligned, pTargetBuffer->bufferSize, dest.info.xsize, dest.info.ysize, dest.info.dpxType, dest.info.nbits, 0, 0);
        if(!frameSize) {
          printf(" renderExecute(%d): dpxformat_writeframe() failed\n", pThreadInfo->id);
          running = FALSE;
        }
      }
    }
  }

  if(pRender->verbose) {
    printf(" renderExecute(%d): Finished rendering frame %d\n", pThreadInfo->id, frameNr);
  }

  return running;
}


/*
// This is the render thread.
*/
void * renderThread(void * p)
{
  int res       = SV_OK;
  int noError   = TRUE;
  int alignment = 0;
  int frameNr   = 0;

  renderThreadInfo * pThreadInfo = (renderThreadInfo *) p;
  sv_handle        * pSv   = pThreadInfo->pRh->pSv;
  sv_render_handle * pSvRh = pThreadInfo->pRh->pSvRh;

  videoBuffer_t source, target;

  memset(&source, 0, sizeof(videoBuffer_t));
  memset(&target, 0, sizeof(videoBuffer_t));

  if(pThreadInfo->pRh->verbose) {
    printf("renderThread(%d): Starting\n", pThreadInfo->id);
  }

  // Query the DMA alignment of the DVS video card.
  res = sv_query(pSv, SV_QUERY_DMAALIGNMENT, 0, &alignment);
  if(res != SV_OK) {
    printf("renderThread(%d): sv_query(SV_QUERY_DMAALIGNMENT) failed = %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
    noError = FALSE;
  }

  // Allocate source and target buffers
  if(noError) {
    noError = allocateAlignedBuffer(&source, MAX_BUFFERSIZE, alignment);
  }
  if(noError) {
    noError = allocateAlignedBuffer(&target, MAX_BUFFERSIZE, alignment);
  }

  if(noError) {
    // Create a render context.
    res = sv_render_begin(pSv, pSvRh, &pThreadInfo->pSvRc);
    if(!pThreadInfo->pSvRc || (res != SV_OK)) {
      printf("renderThread(%d): sv_render_begin() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
      pThreadInfo->pSvRc = NULL;
      noError = FALSE;
    }
  }

  if(noError) {
    // Send message that thread is running.
    dvs_cond_broadcast(&pThreadInfo->running, NULL, FALSE);

    do {
      // Get the next frame number for this thread.
      GetNextFrame(&frameNr);

      // Do the rendering.
      if(frameNr < pThreadInfo->pRh->frameCount) {
        noError = renderExecute(pThreadInfo, &source, &target, frameNr);
      }
    } while(gRunning && noError && (frameNr < pThreadInfo->pRh->frameCount));
  }

  // Clean-up
  if(source.pBuffer) {
    free(source.pBuffer);
  }
  if(target.pBuffer) {
    free(target.pBuffer);
  }

  // Close the render context again.
  if(pThreadInfo->pSvRc) {
    res = sv_render_end(pSv, pSvRh, pThreadInfo->pSvRc);
    if(res != SV_OK) {
      printf("renderThread(%d): sv_render_end() failed! res: %d '%s'\n", pThreadInfo->id, res, sv_geterrortext(res));
    }
  }

  if(pThreadInfo->pRh->verbose) {
    printf("renderThread(%d): Finished\n", pThreadInfo->id);
  }

  dvs_thread_exit(&pThreadInfo->result, &pThreadInfo->exit);
  return 0;
}


/*
// Here are the render threads created.
*/
int startRendering(renderHandle * pRender)
{
  int res   = SV_OK;
  int error = FALSE;
  int i;
  renderThreadInfo * pThreadInfo;

  // Allocate some space for each thread's renderThreadInfo struct.
  pThreadInfo = (renderThreadInfo *) malloc(pRender->threadCount * sizeof(renderThreadInfo));
  memset(pThreadInfo, 0, pRender->threadCount * sizeof(renderThreadInfo));

  // Open a handle to the DVS video device.
  res = sv_openex(&pRender->pSv, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_DEFAULT, 0, 0);
  if(!pRender->pSv || (res != SV_OK)) {
    printf("start_rendering: sv_openex() failed! res: %d '%s'\n", res, sv_geterrortext(res));
    error = TRUE;
  }

  if(!error) {
    // Get the one and only needed/allowed sv_render_handle pointer.
    res = sv_render_open(pRender->pSv, &pRender->pSvRh);
    if(!pRender->pSvRh || (res != SV_OK)) {
      printf("start_rendering: sv_render_open() failed! res: %d '%s'\n", res, sv_geterrortext(res));
      error = TRUE;
    }
  }

  if(!error) {
    // init frame counter mutex
    dvs_mutex_init(&gFrameLock);

    // Create the render threads.
    for(i = 0; i < pRender->threadCount; i++) {
      pThreadInfo[i].id     = i;
      pThreadInfo[i].pRh    = pRender;
      pThreadInfo[i].result = 0;

      dvs_cond_init(&pThreadInfo[i].running);
      dvs_thread_create(&pThreadInfo[i].handle, renderThread, (void*) &pThreadInfo[i], &pThreadInfo[i].exit);
    }

    // Wait until every thread is running.
    for(i = 0; i < pRender->threadCount; i++) {
      dvs_cond_wait(&pThreadInfo[i].running, NULL, FALSE);
    }

    // Wait until thread termination and free thread handles.
    for(i = 0; i < pRender->threadCount; i++) {
      dvs_thread_exitcode(&pThreadInfo[i].handle, &pThreadInfo[i].exit);
      dvs_cond_free(&pThreadInfo[i].running);
    }
  }

  /*
  // Clean-up.
  */
  dvs_mutex_free(&gFrameLock);

  if(pThreadInfo) {
    free(pThreadInfo);
  }
  if(pRender->pSvRh) {
    sv_render_close(pRender->pSv, pRender->pSvRh);
  }
  if(pRender->pSv) {
    sv_close(pRender->pSv);
  }
  
  return error;
}


/*
// Main entry function.
*/
int main(int argc, char ** argv)
{
  int error   = FALSE;
  int noPrint = FALSE;
  int i;
  char * p;

  renderHandle   gRender = { 0 };
  renderHandle * pRender = &gRender;

  memset(pRender, 0, sizeof(renderHandle));

  // default thread count
  pRender->threadCount = 10;

  signal(SIGTERM, signalHandler);
  signal(SIGINT,  signalHandler);

  while((argc > 2) && (argv[1][0] == '-')) {
    switch(argv[1][1]) {
    case '1':
      switch(argv[1][2]) {
        case 'd':
          pRender->lut1d.enable = TRUE;
          break;
        default:
          error = TRUE;
      }
      break;
    case '3':
      switch(argv[1][2]) {
        case 'd':
          pRender->lut3d.enable = TRUE;
          break;
        default:
          error = TRUE;
      }
      break;
    case 'd':
      switch(argv[1][2]) {
      case 'x':
        if(argv[1][3] == '=') {
          pRender->output.xsize = atoi(&argv[1][4]);
        } else {
          error = TRUE;
        }
        break;
      case 'y':
        if(argv[1][3] == '=') {
          pRender->output.ysize = atoi(&argv[1][4]);
        } else {
          error = TRUE;
        }
        break;
      default:
        error = TRUE;
      }
      break;
    case 'm':
      switch(argv[1][2]) {
      case 'r':
        if(argv[1][3] == '=') {
          pRender->matrix.enable = TRUE;
          pRender->matrix.factorRed.enable = TRUE;
          pRender->matrix.factorRed.start = pRender->matrix.factorRed.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->matrix.factorRed.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      case 'g':
        if(argv[1][3] == '=') {
          pRender->matrix.enable = TRUE;
          pRender->matrix.factorGreen.enable = TRUE;
          pRender->matrix.factorGreen.start = pRender->matrix.factorGreen.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->matrix.factorGreen.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      case 'b':
        if(argv[1][3] == '=') {
          pRender->matrix.enable = TRUE;
          pRender->matrix.factorBlue.enable = TRUE;
          pRender->matrix.factorBlue.start = pRender->matrix.factorBlue.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->matrix.factorBlue.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      default:
        error = TRUE;
      }
      break;
#if 0 // not usable at the moment
    case 'p':
      switch(argv[1][2]) {
      case 'x':
        if(argv[1][3] == '=') {
          pRender->scale.enable = TRUE;
          pRender->scale.xoffset.start = pRender->scale.xoffset.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->scale.xoffset.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      case 'y':
        if(argv[1][3] == '=') {
          pRender->scale.enable = TRUE;
          pRender->scale.yoffset.start = pRender->scale.yoffset.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->scale.yoffset.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      default:
        error = TRUE;
      }
      break;
#endif
    case 'r':
      pRender->bRetry = TRUE;
      break;
    case 's':
      switch(argv[1][2]) {
      case 'x':
        if(argv[1][3] == '=') {
          pRender->scale.enable = TRUE;
          pRender->scale.xsize.start = pRender->scale.xsize.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->scale.xsize.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      case 'y':
        if(argv[1][3] == '=') {
          pRender->scale.enable = TRUE;
          pRender->scale.ysize.start = pRender->scale.ysize.end = atof(&argv[1][4]);
          p = strchr(&argv[1][4], ':');
          if(p) {
            pRender->scale.ysize.end = atof(&p[1]);
          }
        } else {
          error = TRUE;
        }
        break;
      default:
        error = TRUE;
      }
      break;
    case 't':
      if(argv[1][2] == '=') {
        pRender->threadCount = atoi(&argv[1][3]);
      } else {
        error = TRUE;
      }
      break;
    case 'v':
      pRender->verbose = TRUE;
      break;
    case 'l':
      pRender->loopMode = TRUE;
      break;
    default:
      error = TRUE;
    }
    argv++; argc--;
  }

  if(pRender->verbose) {
    for(i = 0; i < argc; i++) {
      printf("%s ", argv[i]);
    }
    printf("\n");
  }

  if(argc >= 5) {
    strcpy(pRender->source.fileName, argv[1]);
    argv++; argc--;

    strcpy(pRender->destination.fileName, argv[1]);
    argv++; argc--;

    pRender->startFrame = atoi(argv[1]);
    pRender->frameCount = atoi(argv[2]);

    do {
      if(!error) {
        // Reset globals to default values
        gFrameNr = 0;

        // Start the rendering.
        error = startRendering(pRender);
      }
    } while(pRender->loopMode && gRunning && !error);
  } else {
    error = TRUE;
  }

  if(error && !noPrint) { 
    printf("dpxrender [options] dpx_source dpx_dest #start #nframes\n");
    printf("                     dpx_source/dpx_dest syntax is printf syntax\n");
    printf(" options:\n");
    printf("         -l          Loop forever\n");
    printf("         -r          Enable retry for rendering if something went wrong.\n");
    printf("         -v          Verbose\n");
    printf("\n");
    printf("         -1d         Set 1D LUT\n");
    printf("         -3d         Set 3D LUT (only available on Atomix)\n");
    printf("\n");
    printf("         -t=#        Number of render threads (default: 10)\n");
    printf("\n");
    printf("         -dx=#       Set destination X size\n");
    printf("         -dy=#       Set destination Y size\n");
    printf("\n");
    printf("         -mr=#[:#]   Matrix factor for red\n");
    printf("         -mg=#[:#]   Matrix factor for green\n");
    printf("         -mb=#[:#]   Matrix factor for blue\n");
#if 0 // not usable at the moment
    printf("\n");
    printf("         -px=#[:#]   Pan X\n");
    printf("         -py=#[:#]   Pan Y\n");
#endif
    printf("\n");
    printf("         -sx=#[:#]   Scale to X size\n");
    printf("         -sy=#[:#]   Scale to Y size\n");
    printf("\n");
    printf("         note: When a second value is specified by [:#] there\n");
    printf("               will be done a value interpolation from\n");
    printf("               first value to second value over the rendered sequence.\n");
  }

  return error;
}
