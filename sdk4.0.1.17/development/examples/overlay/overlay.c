#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"

#define USE_STATIC_OVERLAY 0
#define USE_10BIT_DPX 0

int running = TRUE;

void signal_handler(int signum)
{
  (void)signum; // disable compiler warning

  running = FALSE;
}

int generate_image(char * buffer, sv_storageinfo * info);

int generate_lut1d(char * lut)
{
  int ramp;

  for(ramp = 0; ramp < 1024; ramp++) {
    // do an invert LUT
    *((unsigned short *)lut + ramp) = (1023 - ramp) << 6;
  }

  memcpy(lut + 0x800, lut, 0x800);
  memcpy(lut + 0x1000, lut, 0x800);

  return SV_OK;
}

int main(int argc, char * argv[])
{
  sv_handle * sv = NULL;
  sv_fifo * pfifo_in = NULL;
  sv_fifo * pfifo_out = NULL;
  sv_fifo_buffer * pbuffer_out = NULL;
  sv_storageinfo storageinfo = { 0 };
  sv_info info = { 0 };
  sv_fifo_info fifo_info = { 0 };
  int res = SV_OK;
  int alignment = 1;
  int mode = 0;
  int newmode = 0;
  int dropped = -1;
  char * image_org = NULL;
  char * image = NULL;
  char lut[3 * 1024 * 2];
  int fifo_running = FALSE;

  signal(SIGTERM, signal_handler);
  signal(SIGINT,  signal_handler);

  // Open device.
  res = sv_openex(&sv, "", SV_OPENPROGRAM_DEMOPROGRAM, SV_OPENTYPE_MASK_DEFAULT, 0, 0);
  if(res != SV_OK) {
    fprintf(stderr, "Opening card failed\n");
  }

  if(res == SV_OK) {
    // Query DMA alignment for proper buffer allocation.
    res = sv_query(sv, SV_QUERY_DMAALIGNMENT, 0, &alignment);
    if(res != SV_OK) {
      fprintf(stderr, "sv_query(SV_QUERY_DMAALIGNMENT): %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Query current video raster xsize
    res = sv_status(sv, &info);
    if(res != SV_OK) {
      fprintf(stderr, "sv_status(): %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    /*
    // Enable ANC pass-through mode. This has to be set before sv_videomode().
    // Further, it is not available in SDTV rasters.
    */
    if(info.xsize > 960) {
      res = sv_option_set(sv, SV_OPTION_ANCCOMPLETE, SV_ANCCOMPLETE_STREAMER);
      if(res == SV_ERROR_WRONG_HARDWARE) {
        res = SV_OK;
      } else if(res != SV_OK) {
        fprintf(stderr, "sv_option_set(SV_OPTION_ANCCOMPLETE): %s\n", sv_geterrortext(res));
      }
    }
  }

  if(res == SV_OK) {
    // Query current video raster.
    res = sv_query(sv, SV_QUERY_MODE_CURRENT, 0, &mode);
    if(res != SV_OK) {
      fprintf(stderr, "sv_query(SV_QUERY_MODE_CURRENT): %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Enable YUV422A/8B/FRAME video storage and also activate audio.
    newmode = (mode & ~(SV_MODE_COLOR_MASK | SV_MODE_NBIT_MASK | SV_MODE_AUDIO_MASK)) | SV_MODE_COLOR_YUV422A | SV_MODE_STORAGE_FRAME | SV_MODE_AUDIO_8CHANNEL;
    newmode |= (USE_10BIT_DPX) ? SV_MODE_NBIT_10BLABE :  SV_MODE_NBIT_8B;
    if(mode != newmode) {
      res = sv_videomode(sv, newmode);
      if(res != SV_OK) {
        fprintf(stderr, "sv_videomode: %s\n", sv_geterrortext(res));
      }
    }
  }

  if(res == SV_OK) {
    // set genlock to SDI input, reuired for this example program.
    res = sv_sync(sv, SV_SYNC_EXTERNAL);
    if(res != SV_OK) {
      fprintf(stderr, "sv_sync: %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Query various parameters for buffer allocation and generation.
    res = sv_storage_status(sv, 0, NULL, &storageinfo, sizeof(storageinfo), 0);
    if(res != SV_OK) {
      fprintf(stderr, "sv_storage_status: %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Allocate video buffer according to alignment requirement.
    image_org = malloc(storageinfo.buffersize + (alignment - 1));
    image = (char *)(((uintptr)image_org + (alignment - 1)) & ~(alignment - 1));
    if(!image) {
      fprintf(stderr, "memory allocation failed\n");
      res = SV_ERROR_MALLOC;
    }
  }

  if(res == SV_OK) {
    // Generate 1D LUT.
    res = generate_lut1d(lut);
    if(res != SV_OK) {
      fprintf(stderr, "generate_lut: %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Initialize input FIFO.
    res = sv_fifo_init(sv, &pfifo_in, 1, FALSE, TRUE, 0, 0);
    if(res != SV_OK) {
      fprintf(stderr, "sv_fifo_init(in): %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Initialize output FIFO.
    res = sv_fifo_init(sv, &pfifo_out, 0, FALSE, TRUE, SV_FIFO_FLAG_VIDEO_B, 0);
    if(res != SV_OK) {
      fprintf(stderr, "sv_fifo_init(out): %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    // Start input FIFO immediately.
    res = sv_fifo_start(sv, pfifo_in);
    if(res != SV_OK) {
      fprintf(stderr, "sv_fifo_start(in): %s\n", sv_geterrortext(res));
    }
  }

  if(res == SV_OK) {
    res = sv_fifo_sanitylevel(sv, pfifo_out, SV_FIFO_SANITY_LEVEL_FATAL, SV_FIFO_SANITY_VERSION_1);
    if(res != SV_OK)  {
      fprintf(stderr, "sv_fifo_sanitylevel(out) failed %d '%s'\n", res, sv_geterrortext(res));
    }
  }

  while(running && (res == SV_OK))
  {
    // check for sync is detected an if genlocked when sync is on the refin.
    while((sv_fifo_sanitycheck(sv, pfifo_out) == SV_ERROR_SYNC_MISSING) && running)
    {
      if(fifo_running) {
        fifo_running = FALSE;
        printf("WARNING: lost genlock! Output fifo stopped.\n");
        res = sv_fifo_stop(sv, pfifo_out, SV_FIFO_FLAG_FLUSH);
        if(res != SV_OK) {
          printf("sv_fifo_stop(out) failed %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
        res = sv_fifo_reset(sv, pfifo_out);
        if(res != SV_OK) {
          printf("sv_fifo_reset(out) failed %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
 	      dropped = -1;
      }

      sv_fifo_vsyncwait(sv, pfifo_out);
    }

    if(!fifo_running) {
      printf("====> Restarting output fifo. <=====\n");
      res = sv_fifo_start(sv, pfifo_out);
      if(res != SV_OK)  {
        printf("sv_fifo_start(out) failed %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
      }
      fifo_running = TRUE;
    }

    // Generate new overlay plane.
    res = generate_image(image, &storageinfo);
    if(res != SV_OK) {
      fprintf(stderr, "generate_overlay: %s\n", sv_geterrortext(res));
    }

    // Get FIFO buffer for output.
    res = sv_fifo_getbuffer(sv, pfifo_out, &pbuffer_out, NULL, SV_FIFO_FLAG_NODMAADDR | SV_FIFO_FLAG_VIDEO_B);
    if(res == SV_ERROR_SYNC_MISSING) {
      res = SV_OK;
      continue;
    } else if(res != SV_OK) {
      fprintf(stderr, "sv_fifo_getbuffer(out): %s\n", sv_geterrortext(res));
    }

    if(res == SV_OK) {
      res = sv_fifo_status(sv, pfifo_out, &fifo_info);
      if(res != SV_OK) {
        fprintf(stderr, "sv_fifo_status(out): %s\n", sv_geterrortext(res));
      }
      if (dropped != fifo_info.dropped) {
		    if (dropped > 0) {
		      fprintf (stderr, "%08d  dropped %d  avail %d\n", fifo_info.displaytick, fifo_info.dropped - dropped, fifo_info.availbuffers);
		    }
        dropped = fifo_info.dropped;
	    }
    }

    if(res == SV_OK) {
      // Configure bypass for video channel A and audio channels.
      res = sv_fifo_bypass(sv, pfifo_out, pbuffer_out, 0x01, 0xffff);
      if(res != SV_OK) {
        fprintf(stderr, "sv_fifo_bypass(out): %s\n", sv_geterrortext(res));
      }
    }

    if(res == SV_OK) {
      // Configure 1D LUT on output.
      res = sv_fifo_lut(sv, pfifo_out, pbuffer_out, (unsigned char *)lut, sizeof(lut), 0, SV_FIFO_LUT_TYPE_1D_RGB);
      if(res != SV_OK) {
        fprintf(stderr, "sv_fifo_lut(out): %s\n", sv_geterrortext(res));
      }
    }

    if(res == SV_OK) {
      if(pbuffer_out) {
        // Transfer no video for channel A - it is controlled by sv_fifo_bypass().
        pbuffer_out->video[0].addr = 0;
        pbuffer_out->video[0].size = 0;
        pbuffer_out->video[1].addr = 0;
        pbuffer_out->video[1].size = 0;

        // Transfer alpha image to channel B.
        pbuffer_out->video_b[0].addr = image;
        pbuffer_out->video_b[0].size = storageinfo.buffersize;

        pbuffer_out->video_b[1].addr = 0;
        pbuffer_out->video_b[1].size = 0;

        // Transfer no audio - it is controlled by sv_fifo_bypass().
        pbuffer_out->audio[0].size = 0;
        pbuffer_out->audio[1].size = 0;
      }
    }

    if(res == SV_OK) {
      // Release FIFO buffer for transfer to output.
      res = sv_fifo_putbuffer(sv, pfifo_out, pbuffer_out, NULL);
      if(res != SV_OK) {
        fprintf(stderr, "sv_fifo_putbuffer(out): %s\n", sv_geterrortext(res));
      }
    }

#   if USE_STATIC_OVERLAY
    // block sv_fifo_getbuffer/sv_fifo_putbuffer loop here
    while(running && (res == SV_OK)) {
      res = sv_fifo_vsyncwait(sv, pfifo_out);
      if(res != SV_OK) {
        fprintf(stderr, "sv_fifo_vsyncwait(out): %s\n", sv_geterrortext(res));
      }
    }
#   endif

  } // end while loop

  // Free video buffer.
  if(image) {
    free(image_org);
    image_org = image = NULL;
  }

  if(sv) {
    int res2;

    // Stop and free output FIFO.
    if(pfifo_out) {
      res2 = sv_fifo_stop(sv, pfifo_out, 0);
      if(res2 != SV_OK) {
        fprintf(stderr, "sv_fifo_stop(out): %s\n", sv_geterrortext(res2));
      }

      res2 = sv_fifo_free(sv, pfifo_out);
      if(res2 != SV_OK) {
        fprintf(stderr, "sv_fifo_free(out): %s\n", sv_geterrortext(res2));
      }
    }

    // Stop and free input FIFO.
    if(pfifo_in) {
      res2 = sv_fifo_stop(sv, pfifo_in, 0);
      if(res2 != SV_OK) {
        fprintf(stderr, "sv_fifo_stop(input): %s\n", sv_geterrortext(res2));
      }

      res2 = sv_fifo_free(sv, pfifo_in);
      if(res2 != SV_OK) {
        fprintf(stderr, "sv_fifo_free(input): %s\n", sv_geterrortext(res2));
      }
    }

    // Disable ANC pass-through mode.
    res2 = sv_option_set(sv, SV_OPTION_ANCCOMPLETE, SV_ANCCOMPLETE_OFF);
    if(res2 == SV_ERROR_WRONG_HARDWARE) {
      res2 = SV_OK;
    } else if(res2 != SV_OK) {
      fprintf(stderr, "sv_option_set(SV_OPTION_ANCCOMPLETE): %s\n", sv_geterrortext(res2));
    }

    // Close device.
    res2 = sv_close(sv);
    if(res2 != SV_OK) {
      fprintf(stderr, "sv_close: %s\n", sv_geterrortext(res2));
    }
    sv = NULL;
  }

  signal(SIGTERM, NULL);
  signal(SIGINT, NULL);

  return res;
}
