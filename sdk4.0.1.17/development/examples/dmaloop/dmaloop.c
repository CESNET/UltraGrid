/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dmaloop: Demo for simultaneous recording of video and audio data.
//             This example can either operate on two separate boards or one
//             single board. Additionally an input to output delay can be
//             specified.
*/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

#if defined(WIN32)
# include <windows.h>
#else
# include <ctype.h>
# include <string.h>
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"
#include "../../header/dvs_thread.h"

#define MAX_ID 20

typedef struct anc_element{
  struct anc_element * prev;
  struct anc_element * next;
  int valid;
  sv_fifo_ancbuffer ancbuffer;
} anc_element_t;

typedef struct {
  sv_handle *        svsrc;
  sv_fifo *          fifosrc;
  sv_fifo_buffer *   bufsrc;
  sv_handle *        svdst;
  sv_fifo *          fifodst;
  sv_fifo_buffer *   bufdst;

  char * livebuffer[MAX_ID];
  char * livebuffer_org[MAX_ID];
  char * ancbuffer[MAX_ID];
  char * ancbuffer_org[MAX_ID];
  anc_element_t anclist[MAX_ID];
  char * blackbuffer;
  char * blackbuffer_org;
  char * nobuffer;
  char * nobuffer_org;
  int videosize[2];
  int audiosize[MAX_ID][2];
  sv_fifo_buffer tmp;
  sv_storageinfo storage;

  struct {
    int id;
    uint32 inputtick[MAX_ID];
    dvs_cond ready;
    dvs_mutex lock;
  } common;

  dvs_cond finish;
  int exitcode;
  int running;
  int vinterlace;

  int ndelay;
  int buse2cards;
  int bvideoonly;
  int baudioonly;
  int bfieldbased;
  int bverbose;
  int bplayback;
  int banc;
  int bancstreamer;
  int indelay_info;
  int outdelay_info;
  int inavail_info;
  int bdualsdi;
  int bottom2top;
} loop_handle;

void delete_element( int id, anc_element_t * anc_element );


int get_free_bufferid(loop_handle * hd)
{
  int id = hd->common.id;

  if(++hd->common.id >= MAX_ID) {
    hd->common.id = 0;
  }

  return id;
}

int get_bufferid(loop_handle * hd, uint32 displaytick, uint32 delay )
{
  int id;
  unsigned int recordTick = displaytick - delay;

  for(id = MAX_ID - 1; id >= 0; id--) {
    if(hd->common.inputtick[id] == recordTick) {
      break;
    }
  }

  return id;
}

int is_bufferid_available(loop_handle * hd)
{
  int id;

  for(id = MAX_ID - 1; id >= 0; id--) {
    if(hd->common.inputtick[id]) {
      return FALSE;
    }
  }

  return TRUE;
}


loop_handle global_handle;


void signal_handler(int signum)
{
  (void)signum; // disable compiler warning

  global_handle.running = FALSE;
}


int loop_exit(loop_handle *hd);

int loop_init(loop_handle * hd)
{
  sv_fifo_configinfo config;
  int modesrc  = 0;
  int modedst  = 0;
  int syncmode = 0;
  int res;
  int i;
  char anclayout[1024];

  // Open first card.
  hd->svsrc = sv_open("");
  if (hd->svsrc == NULL) {
    printf("loop_init: sv_open(\"\") failed\n");
    loop_exit(hd);
    return FALSE;
  }

  if(hd->buse2cards) {
    // Open second card.
    hd->svdst = sv_open("PCI,card:1");
    if(hd->svdst == NULL) {
      printf("loop_init: sv_open(\"PCI.card:1\") failed\n");
      loop_exit(hd);
      return FALSE;
    }
  } else {
    // Use first card for output as well.
    hd->svdst = hd->svsrc;
  }

  // Get current videomode from first card.
  res = sv_option_get(hd->svsrc, SV_OPTION_VIDEOMODE, &modesrc);
  if(res != SV_OK) {
    printf("loop_init: sv_option_get() failed\n");
    loop_exit(hd);
    return FALSE;
  }

  // Get current videomode from second card.
  res = sv_option_get(hd->svdst, SV_OPTION_VIDEOMODE, &modedst);
  if(res != SV_OK) {
    printf("loop_init: sv_option_get() failed\n");
    loop_exit(hd);
    return FALSE;
  }

  // Compare if both videomodes do match.
  if((modesrc & SV_MODE_MASK) != (modedst & SV_MODE_MASK)) {
    printf("loop_init: Raster of source and destination board do not match.\n");
    loop_exit(hd);
    return FALSE;
  }

  // Check sync mode
  res = sv_query(hd->svsrc, SV_QUERY_SYNCMODE, 0, &syncmode);
  if(res != SV_OK) {
    printf("loop_init: sv_query(SV_QUERY_SYNCMODE) failed\n");
    loop_exit(hd);
    return FALSE;
  }

  if(syncmode == SV_SYNC_INTERNAL) {
    printf("Error:\tPlease configure another sync mode,\n\tit is not possible to have a stable in to out delay with SV_SYNC_INTERNAL.\n");
    loop_exit(hd);
    return FALSE;
  }

  switch(modesrc & SV_MODE_MASK) {
  case SV_MODE_SMPTE274_47P:
  case SV_MODE_SMPTE274_48P:
  case SV_MODE_SMPTE274_50P:
  case SV_MODE_SMPTE274_59P:
  case SV_MODE_SMPTE274_60P:
  case SV_MODE_SMPTE274_71P:
  case SV_MODE_SMPTE274_72P:
    hd->bdualsdi = TRUE;
    break;
  }

  // Get information about current raster.
  res = sv_storage_status(hd->svsrc, 0, NULL, &hd->storage, sizeof(sv_storageinfo), 0);
  if(res != SV_OK) {
    printf("loop_init: sv_storage_status() failed = %d '%s'\n", res, sv_geterrortext(res));
  }

  // How many ticks does a frame last?
  hd->vinterlace = hd->storage.vinterlace;

  //frame to field correction
  hd->ndelay = hd->ndelay * hd->storage.vinterlace;

  //fieldbased correction
  if(hd->bfieldbased) {
    hd->vinterlace = 1;
  }

  if(hd->banc) {
    // Disable the fifo ancgenerator because else you will get double packets in loopback
    res = sv_option_set(hd->svdst, SV_OPTION_ANCGENERATOR, SV_ANCDATA_DISABLE );
    if(res != SV_OK) {
      printf("loop_init: sv_option_set() failed\n");
      loop_exit(hd);
      return FALSE;
    }
  }

  // Init input FIFO.
  res = sv_fifo_init(hd->svsrc, &hd->fifosrc,
    TRUE,  // input FIFO
    FALSE,
    TRUE,  // enable DMA mode
    (hd->bancstreamer ? SV_FIFO_FLAG_ANC : 0) |
    (hd->bfieldbased ? SV_FIFO_FLAG_FIELD : 0),  // enable field-based mode
    0      // use maximum available buffers
  );
  if(res != SV_OK)  {
    printf("sv_fifo_init(src) failed %d '%s'\n", res, sv_geterrortext(res));
    loop_exit(hd);
    return FALSE;
  }

  res = sv_fifo_sanitylevel(hd->svsrc, hd->fifosrc, SV_FIFO_SANITY_LEVEL_FATAL, SV_FIFO_SANITY_VERSION_1);
  if(res != SV_OK)  {
    printf("sv_fifo_sanitylevel(dst) failed %d '%s'\n", res, sv_geterrortext(res));
  }

  // Init output FIFO.
  res = sv_fifo_init(hd->svdst, &hd->fifodst,
    FALSE,  // output FIFO
    FALSE,
    TRUE,   // enable DMA mode
    (hd->bancstreamer ? SV_FIFO_FLAG_ANC : 0),
    0       // use maximum available buffers
  );
  if(res != SV_OK)  {
    printf("sv_fifo_init(dst) failed %d '%s'\n", res, sv_geterrortext(res));
    loop_exit(hd);
    return FALSE;
  }

  res = sv_fifo_sanitylevel(hd->svdst, hd->fifodst, SV_FIFO_SANITY_LEVEL_FATAL, SV_FIFO_SANITY_VERSION_1);
    if(res != SV_OK)  {
    printf("sv_fifo_sanitylevel(dst) failed %d '%s'\n", res, sv_geterrortext(res));
  }

  if(hd->bverbose && hd->bancstreamer) {
    int required = 0;

    res = sv_fifo_anclayout(hd->svsrc, hd->fifosrc, anclayout, sizeof(anclayout), &required);
    if(res == SV_ERROR_BUFFERSIZE)  {
      printf("sv_fifo_anclayout(src) buffer too small (required %d)\n", required);
    } else if(res != SV_OK)  {
      printf("sv_fifo_anclayout(src) failed %d '%s'\n", res, sv_geterrortext(res));
    } else {
      printf("ANC layout (input):\n%s\n", anclayout);
    }

    res = sv_fifo_anclayout(hd->svdst, hd->fifodst, anclayout, sizeof(anclayout), &required);
    if(res == SV_ERROR_BUFFERSIZE)  {
      printf("sv_fifo_anclayout(dst) buffer too small (required %d)\n", required);
    } else if(res != SV_OK)  {
      printf("sv_fifo_anclayout(dst) failed %d '%s'\n", res, sv_geterrortext(res));
    } else {
      printf("ANC layout (output):\n%s\n", anclayout);
    }
  }

  // Fetch some information about FIFO buffer sizes.
  res = sv_fifo_configstatus(hd->svsrc, hd->fifosrc, &config);
  if(res != SV_OK)  {
    printf("sv_fifo_configstatus(src) failed %d '%s'\n", res, sv_geterrortext(res));
    loop_exit(hd);
    return FALSE;
  }

  // Allocate sufficient memory for video and audio data.
  for(i = 0; i < MAX_ID; i++) {
    memset( &hd->anclist[i], 0, sizeof(hd->anclist[i]) );
    hd->livebuffer_org[i] = malloc(config.vbuffersize + config.abuffersize + (config.dmaalignment-1));
    hd->livebuffer[i] = (char *)((uintptr)(hd->livebuffer_org[i] + (config.dmaalignment-1)) & ~(uintptr)(config.dmaalignment-1));
    if(!hd->livebuffer_org[i]) {
      printf("malloc(%d) livebuffer %d failed\n", config.vbuffersize + config.abuffersize + (config.dmaalignment-1), i);
      loop_exit(hd);
      return FALSE;
    }

    if(config.ancbuffersize && hd->bancstreamer) {
      hd->ancbuffer_org[i] = malloc(config.ancbuffersize + (config.dmaalignment-1));
      hd->ancbuffer[i] = (char *)((uintptr)(hd->ancbuffer_org[i] + (config.dmaalignment-1)) & ~(uintptr)(config.dmaalignment-1));
      if(!hd->ancbuffer_org[i]) {
        printf("malloc(%d) ancbuffer %d failed\n", config.ancbuffersize + (config.dmaalignment-1), i);
        loop_exit(hd);
        return FALSE;
      }
    }
  }

  // Allocate sufficient memory for video and audio data.
  hd->blackbuffer_org = malloc(config.vbuffersize + config.abuffersize + (config.dmaalignment-1));
  hd->blackbuffer = (char *)((uintptr)(hd->blackbuffer_org + (config.dmaalignment-1)) & ~(uintptr)(config.dmaalignment-1));
  if(!hd->blackbuffer_org) {
    printf("malloc(%d) blackbuffer failed\n", config.vbuffersize + config.abuffersize + (config.dmaalignment-1));
    loop_exit(hd);
    return FALSE;
  }
  memset(hd->blackbuffer_org, 0, config.vbuffersize + config.abuffersize + (config.dmaalignment-1));

  // Allocate sufficient memory for video and audio data.
  hd->nobuffer_org = malloc(config.vbuffersize + config.abuffersize + (config.dmaalignment-1));
  hd->nobuffer = (char *)((uintptr)(hd->nobuffer_org + (config.dmaalignment-1)) & ~(uintptr)(config.dmaalignment-1));
  if(!hd->nobuffer_org) {
    printf("malloc(%d) nobuffer failed\n", config.vbuffersize + config.abuffersize + (config.dmaalignment-1));
    loop_exit(hd);
    return FALSE;
  }
  memset(hd->nobuffer_org, 0xff, config.vbuffersize + config.abuffersize + (config.dmaalignment-1));

  dvs_mutex_init(&hd->common.lock);
  dvs_cond_init(&hd->common.ready);

  return TRUE;
}


int loop_exit(loop_handle *hd)
{
  int i;

  hd->running = FALSE;

  if(hd->svdst != NULL) {
    if (hd->fifodst != NULL) {
      sv_fifo_free(hd->svdst, hd->fifodst);
      hd->fifodst = NULL;
    }
    if(hd->svdst != hd->svsrc) {
      sv_close(hd->svdst);
      hd->svdst = NULL;
    }
  }

  if (hd->svsrc != NULL) {
    if (hd->fifosrc != NULL) {
      sv_fifo_free(hd->svsrc, hd->fifosrc);
      hd->fifosrc = NULL;
    }
    sv_close(hd->svsrc);
    hd->svsrc = NULL;
  }

  if(hd->nobuffer_org) {
    free(hd->nobuffer_org);
  }

  if(hd->blackbuffer_org) {
    free(hd->blackbuffer_org);
  }

  for(i = 0; i < MAX_ID; i++) {
    if(hd->ancbuffer_org[i]) {
      free(hd->ancbuffer_org[i]);
    }
    if(hd->livebuffer_org[i]) {
      free(hd->livebuffer_org[i]);
    }
    if(hd->banc) {
      //If there are valid packets delete them
      if( hd->anclist[i].next ) {
        //Delete all elements in list
        delete_element( i, hd->anclist[i].next );
      }
    }
  }

  dvs_mutex_free(&hd->common.lock);
  dvs_cond_free(&hd->common.ready);

  printf("exiting dma loop through\n");

  return 0;
}


void fill_buffer(loop_handle * hd, int binput, sv_fifo_buffer * pbuffer, int bufferid)
{
  uintptr buffer;
  uintptr ancbuffer;

  /*
  // Decide if live or black buffer is to be played out
  */
  if(bufferid < 0) {
    if(binput) {
      // do not record anything
      pbuffer->video[0].size = 0;
      pbuffer->video[1].size = 0;
      pbuffer->audio[0].size = 0;
      pbuffer->audio[1].size = 0;
      return;
    } else {
      buffer = (uintptr) hd->nobuffer;
      ancbuffer = (uintptr) 0;
    }
  } else {
    buffer    = (uintptr) hd->livebuffer[bufferid];
    ancbuffer = (uintptr) hd->ancbuffer[bufferid];
  }

  if(!binput) {
    if(hd->bplayback) {
      buffer = (uintptr) hd->blackbuffer;
    }
  }

  /*
  // Enter DMA addresses as ordered by SV_FIFO_FLAG_NODMAADDR.
  // All the addr fields below now have a zero-based offset, so we can
  // simply add a local base address.
  */
  pbuffer->video[0].addr     += buffer;
  pbuffer->video[1].addr     += buffer;
  pbuffer->audio[0].addr[0]  += buffer;
  pbuffer->audio[0].addr[1]  += buffer;
  pbuffer->audio[0].addr[2]  += buffer;
  pbuffer->audio[0].addr[3]  += buffer;
  pbuffer->audio[1].addr[0]  += buffer;
  pbuffer->audio[1].addr[1]  += buffer;
  pbuffer->audio[1].addr[2]  += buffer;
  pbuffer->audio[1].addr[3]  += buffer;
  pbuffer->audio2[0].addr[0] += buffer;
  pbuffer->audio2[0].addr[1] += buffer;
  pbuffer->audio2[0].addr[2] += buffer;
  pbuffer->audio2[0].addr[3] += buffer;
  pbuffer->audio2[1].addr[0] += buffer;
  pbuffer->audio2[1].addr[1] += buffer;
  pbuffer->audio2[1].addr[2] += buffer;
  pbuffer->audio2[1].addr[3] += buffer;

  if(hd->bancstreamer) {
    if(ancbuffer) {
      pbuffer->anc[0].addr     += ancbuffer;
      pbuffer->anc[1].addr     += ancbuffer;
    } else {
      pbuffer->anc[0].addr      = 0;
      pbuffer->anc[1].addr      = 0;
      pbuffer->anc[0].size      = 0;
      pbuffer->anc[1].size      = 0;
    }
  }

  /*
  // Adjust the DMA buffer sizes on the output FIFO.
  // Together with SV_FIFO_FLAG_SETAUDIOSIZE, this allows to compensate
  // the fluctuating audio sample distribution which occurs in dropframe
  // rasters.
  */
  if(bufferid >= 0) {
    if(binput) {
      hd->videosize[0] = hd->bufsrc->video[0].size;
      hd->videosize[1] = hd->bufsrc->video[1].size;

      hd->audiosize[bufferid][0] = hd->bufsrc->audio[0].size;
      hd->audiosize[bufferid][1] = hd->bufsrc->audio[1].size;
    } else {
      hd->bufdst->video[0].size = hd->videosize[0];
      hd->bufdst->video[1].size = hd->videosize[1];

      hd->bufdst->audio[0].size = hd->audiosize[bufferid][0];
      hd->bufdst->audio[1].size = hd->audiosize[bufferid][1];
    }
  }

  /*
  // Transfer incoming timecode data from input to output FIFO.
  */
  if(binput) {
    hd->tmp.timecode.ltc_tc             = hd->bufsrc->timecode.ltc_tc;
    hd->tmp.timecode.ltc_ub             = hd->bufsrc->timecode.ltc_ub;
    hd->tmp.timecode.vitc_tc            = hd->bufsrc->timecode.vitc_tc;
    hd->tmp.timecode.vitc_ub            = hd->bufsrc->timecode.vitc_ub;
    hd->tmp.timecode.vitc_tc2           = hd->bufsrc->timecode.vitc_tc2;
    hd->tmp.timecode.vitc_ub2           = hd->bufsrc->timecode.vitc_ub2;
    hd->tmp.timecode.vtr_tc             = hd->bufsrc->timecode.vtr_tc;
    hd->tmp.timecode.vtr_ub             = hd->bufsrc->timecode.vtr_ub;
    hd->tmp.timecode.vtr_info           = hd->bufsrc->timecode.vtr_info;
    hd->tmp.timecode.vtr_info2          = hd->bufsrc->timecode.vtr_info2;
    hd->tmp.anctimecode.dvitc_tc[0]     = hd->bufsrc->anctimecode.dvitc_tc[0];
    hd->tmp.anctimecode.dvitc_tc[1]     = hd->bufsrc->anctimecode.dvitc_tc[1];
    hd->tmp.anctimecode.dvitc_ub[0]     = hd->bufsrc->anctimecode.dvitc_ub[0];
    hd->tmp.anctimecode.dvitc_ub[1]     = hd->bufsrc->anctimecode.dvitc_ub[1];
    hd->tmp.anctimecode.film_tc[0]      = hd->bufsrc->anctimecode.film_tc[0];
    hd->tmp.anctimecode.film_tc[1]      = hd->bufsrc->anctimecode.film_tc[1];
    hd->tmp.anctimecode.film_ub[0]      = hd->bufsrc->anctimecode.film_ub[0];
    hd->tmp.anctimecode.film_ub[1]      = hd->bufsrc->anctimecode.film_ub[1];
    hd->tmp.anctimecode.prod_tc[0]      = hd->bufsrc->anctimecode.prod_tc[0];
    hd->tmp.anctimecode.prod_tc[1]      = hd->bufsrc->anctimecode.prod_tc[1];
    hd->tmp.anctimecode.prod_ub[0]      = hd->bufsrc->anctimecode.prod_ub[0];
    hd->tmp.anctimecode.prod_ub[1]      = hd->bufsrc->anctimecode.prod_ub[1];
    hd->tmp.anctimecode.dltc_tc         = hd->bufsrc->anctimecode.dltc_tc;
    hd->tmp.anctimecode.dltc_ub         = hd->bufsrc->anctimecode.dltc_ub;
    hd->tmp.anctimecode.afilm_tc[0]     = hd->bufsrc->anctimecode.afilm_tc[0];
    hd->tmp.anctimecode.afilm_tc[1]     = hd->bufsrc->anctimecode.afilm_tc[1];
    hd->tmp.anctimecode.afilm_ub[0]     = hd->bufsrc->anctimecode.afilm_ub[0];
    hd->tmp.anctimecode.afilm_ub[1]     = hd->bufsrc->anctimecode.afilm_ub[1];
    hd->tmp.anctimecode.aprod_tc[0]     = hd->bufsrc->anctimecode.aprod_tc[0];
    hd->tmp.anctimecode.aprod_tc[1]     = hd->bufsrc->anctimecode.aprod_tc[1];
    hd->tmp.anctimecode.aprod_ub[0]     = hd->bufsrc->anctimecode.aprod_ub[0];
    hd->tmp.anctimecode.aprod_ub[1]     = hd->bufsrc->anctimecode.aprod_ub[1];
  } else {
    hd->bufdst->timecode.ltc_tc         = hd->tmp.timecode.ltc_tc;
    hd->bufdst->timecode.ltc_ub         = hd->tmp.timecode.ltc_ub;
    hd->bufdst->timecode.vitc_tc        = hd->tmp.timecode.vitc_tc;
    hd->bufdst->timecode.vitc_ub        = hd->tmp.timecode.vitc_ub;
    hd->bufdst->timecode.vitc_tc2       = hd->tmp.timecode.vitc_tc2;
    hd->bufdst->timecode.vitc_ub2       = hd->tmp.timecode.vitc_ub2;
    hd->bufdst->timecode.vtr_tc         = hd->tmp.timecode.vtr_tc;
    hd->bufdst->timecode.vtr_ub         = hd->tmp.timecode.vtr_ub;
    hd->bufdst->timecode.vtr_info       = hd->tmp.timecode.vtr_info;
    hd->bufdst->timecode.vtr_info2      = hd->tmp.timecode.vtr_info2;
    hd->bufdst->anctimecode.dvitc_tc[0] = hd->tmp.anctimecode.dvitc_tc[0];
    hd->bufdst->anctimecode.dvitc_tc[1] = hd->tmp.anctimecode.dvitc_tc[1];
    hd->bufdst->anctimecode.dvitc_ub[0] = hd->tmp.anctimecode.dvitc_ub[0];
    hd->bufdst->anctimecode.dvitc_ub[1] = hd->tmp.anctimecode.dvitc_ub[1];
    hd->bufdst->anctimecode.film_tc[0]  = hd->tmp.anctimecode.film_tc[0];
    hd->bufdst->anctimecode.film_tc[1]  = hd->tmp.anctimecode.film_tc[1];
    hd->bufdst->anctimecode.film_ub[0]  = hd->tmp.anctimecode.film_ub[0];
    hd->bufdst->anctimecode.film_ub[1]  = hd->tmp.anctimecode.film_ub[1];
    hd->bufdst->anctimecode.prod_tc[0]  = hd->tmp.anctimecode.prod_tc[0];
    hd->bufdst->anctimecode.prod_tc[1]  = hd->tmp.anctimecode.prod_tc[1];
    hd->bufdst->anctimecode.prod_ub[0]  = hd->tmp.anctimecode.prod_ub[0];
    hd->bufdst->anctimecode.prod_ub[1]  = hd->tmp.anctimecode.prod_ub[1];
    hd->bufdst->anctimecode.dltc_tc     = hd->tmp.anctimecode.dltc_tc;
    hd->bufdst->anctimecode.dltc_ub     = hd->tmp.anctimecode.dltc_ub;
    hd->bufdst->anctimecode.afilm_tc[0] = hd->tmp.anctimecode.afilm_tc[0];
    hd->bufdst->anctimecode.afilm_tc[1] = hd->tmp.anctimecode.afilm_tc[1];
    hd->bufdst->anctimecode.afilm_ub[0] = hd->tmp.anctimecode.afilm_ub[0];
    hd->bufdst->anctimecode.afilm_ub[1] = hd->tmp.anctimecode.afilm_ub[1];
    hd->bufdst->anctimecode.aprod_tc[0] = hd->tmp.anctimecode.aprod_tc[0];
    hd->bufdst->anctimecode.aprod_tc[1] = hd->tmp.anctimecode.aprod_tc[1];
    hd->bufdst->anctimecode.aprod_ub[0] = hd->tmp.anctimecode.aprod_ub[0];
    hd->bufdst->anctimecode.aprod_ub[1] = hd->tmp.anctimecode.aprod_ub[1];
  }
}


void * loop_in(void * arg) 
{
  loop_handle * hd = (loop_handle *) arg;
  sv_fifo_bufferinfo infosrc;
  sv_fifo_info status_src;
  unsigned int lastdrop_src = (unsigned int) ~0;
  int bufferid;
  int res = SV_OK;

  /*
  // Start the input FIFO.
  */
  res = sv_fifo_start(hd->svsrc, hd->fifosrc);
  if(res != SV_OK)  {
    printf("sv_fifo_start(src) failed %d '%s'\n", res, sv_geterrortext(res));
    hd->running = FALSE;
  }

  while (hd->running) {
    /*
    // >>> Handle input FIFO <<<
    */
    do {
      res = sv_fifo_getbuffer(hd->svsrc, hd->fifosrc, &hd->bufsrc, NULL,
        SV_FIFO_FLAG_NODMAADDR |
        (hd->bancstreamer ? SV_FIFO_FLAG_ANC : 0) |
        (hd->bvideoonly ? SV_FIFO_FLAG_VIDEOONLY : 0) |
        (hd->baudioonly ? SV_FIFO_FLAG_AUDIOONLY : 0) |
        (hd->bplayback ? SV_FIFO_FLAG_FLUSH : 0)
      );

      switch(res) {
      case SV_ERROR_NOCARRIER:
      case SV_ERROR_INPUT_VIDEO_NOSIGNAL:
      case SV_ERROR_INPUT_VIDEO_RASTER:
      case SV_ERROR_INPUT_VIDEO_DETECTING:
      case SV_ERROR_INPUT_KEY_NOSIGNAL:
      case SV_ERROR_INPUT_KEY_RASTER:
        printf("v");	// Inform user about missing video
        fflush(stdout);
        res = SV_ACTIVE;
        break;
      case SV_ERROR_INPUT_AUDIO_NOAIV:
      case SV_ERROR_INPUT_AUDIO_NOAESEBU:
        // In these two cases, getbuffer returned a valid buffer.
        // So, we can continue without audio.
        printf("a");	// Inform user about missing audio
        fflush(stdout);
        res = SV_OK;
        break;
      case SV_ERROR_VSYNCPASSED:
        printf("NOTE: vsyncpassed possibly caused by signal disconnect\n");
        res = SV_ACTIVE;
        break;
      default:;
      }

      if(res == SV_ACTIVE) {
        // wait for next vsync and try again
        sv_fifo_vsyncwait(hd->svsrc, hd->fifosrc);
      }
    } while(res == SV_ACTIVE && hd->running);

    if (res != SV_OK)  {
      printf("sv_fifo_getbuffer(src) failed %d '%s'\n", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    dvs_mutex_enter(&hd->common.lock);
    bufferid = get_free_bufferid(hd);
    dvs_mutex_leave(&hd->common.lock);

    fill_buffer(hd, TRUE, hd->bufsrc, bufferid);

    if(hd->banc && (bufferid >= 0)) {
      //Get list pointer
      anc_element_t * current_element = &hd->anclist[bufferid];

      //If there are valid packets delete them
      if( current_element->next ) {
        //Delete all elements in list
        delete_element( bufferid, current_element->next );
        //Set first element to zero
        memset( current_element, 0, sizeof(anc_element_t) );
      }
      
      //Try to get anc packets
      while( res==SV_OK ) {
        res = sv_fifo_anc( hd->svsrc, hd->fifosrc, hd->bufsrc, &current_element->ancbuffer );
        if( res == SV_OK ) {
          current_element->valid = 1;

          //Log
          if( hd->bverbose ) {
            printf("Recorded anc_element. Buffer:%d, DID:0x%x, SDID:0x%x\n", bufferid, current_element->ancbuffer.did, current_element->ancbuffer.sdid );
          }

          //I need to allocate one more packet buffer
          current_element->next = malloc( sizeof( anc_element_t ));
          if( current_element->next != 0 )
          {
            //Set all values to zero
            memset( current_element->next, 0, sizeof( anc_element_t ));
            
            //Build the list
            current_element->next->prev = current_element;
            current_element = current_element->next;
          } else {
            printf("malloc failed %d '%s'\n", res, sv_geterrortext(res));
            hd->running = FALSE;
          }
        }
      }
      // This errorcode shows that no more packets are in the frame
      if( res == SV_ERROR_NOTAVAILABLE ) {
        res = SV_OK;
      }
    }

    res = sv_fifo_putbuffer(hd->svsrc, hd->fifosrc, hd->bufsrc, &infosrc);
    if(res != SV_OK)  {
      printf("sv_fifo_putbuffer(src) failed %d '%s'\n", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    res = sv_fifo_status(hd->svsrc, hd->fifosrc, &status_src);
    if(res != SV_OK)  {
      printf("sv_fifo_status(src) failed %d '%s'\n", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    // Save input delay
    hd->indelay_info  = status_src.recordtick - infosrc.when;
    hd->inavail_info  = status_src.availbuffers; 
    
    if(hd->bverbose || ((unsigned int)status_src.dropped > lastdrop_src)) {
      printf("ID %2d - in  %06d %d %d %d (%u)\n", bufferid, infosrc.when, status_src.nbuffers, status_src.availbuffers, status_src.dropped - lastdrop_src, infosrc.clock_low);
    }

    // Report dropped frames in input FIFO.
    if((unsigned int)status_src.dropped > lastdrop_src) {
      printf("input fifo dropped - delay might have changed\n");
    }
    lastdrop_src = status_src.dropped;

    dvs_mutex_enter(&hd->common.lock);
    if(bufferid >= 0) {
      hd->common.inputtick[bufferid] = infosrc.when;
    }
    dvs_mutex_leave(&hd->common.lock);

    /*
    // Signal that a new input buffer is ready
    */
    dvs_cond_broadcast(&hd->common.ready, &hd->common.lock, FALSE);
  }

  res = sv_fifo_stop(hd->svsrc, hd->fifosrc, 0);
  if(res != SV_OK)  {
    printf("sv_fifo_stop(src) failed %d '%s'\n", res, sv_geterrortext(res));
  }

  hd->exitcode = TRUE;
  dvs_thread_exit(&hd->exitcode, &hd->finish);

  return NULL;
}



void * loop_out(void * arg) 
{
  loop_handle * hd = (loop_handle *) arg;
  sv_fifo_bufferinfo infodst;
  sv_fifo_info status_dst;
  int lastdrop_dst = 0;
  int displaytick = 0;
  int res = SV_OK;
  int bufferid;
  int fifo_running = FALSE;
  int flip = 0;

  dvs_cond_wait(&hd->common.ready, &hd->common.lock, FALSE);

  while (hd->running) {
    // check for sync is detected and if genlocked when sync is on the refin
    while((res == SV_ERROR_SYNC_MISSING) || sv_fifo_sanitycheck(hd->svdst, hd->fifodst) == SV_ERROR_SYNC_MISSING &&
               hd->running)
    {
      if(fifo_running) {
        fifo_running = FALSE;
        printf("WARNING: lost genlock! Output fifo stopped.\n");
        res = sv_fifo_stop(hd->svdst, hd->fifodst, SV_FIFO_FLAG_FLUSH);
        if(res != SV_OK)  {
          printf("sv_fifo_stop(dst) failed %d '%s'\n", res, sv_geterrortext(res));
          hd->running = FALSE;
        }
        res = sv_fifo_reset(hd->svdst, hd->fifodst);
        if(res != SV_OK)  {
          printf("sv_fifo_reset(dst) failed %d '%s'\n", res, sv_geterrortext(res));
          hd->running = FALSE;
        }
      }

      sv_fifo_vsyncwait(hd->svdst, hd->fifodst);
    }

    if(!fifo_running) {
      printf("====> Restarting output fifo. <=====\n");
#if 0
      res = sv_fifo_free(hd->svdst, hd->fifodst);
      if(res != SV_OK)  {
        printf("sv_fifo_reset(dst) failed %d '%s'\n", res, sv_geterrortext(res));
        hd->running = FALSE;
      }

      // Init output FIFO.
      res = sv_fifo_init(hd->svdst, &hd->fifodst, FALSE, FALSE, TRUE, (hd->bancstreamer ? SV_FIFO_FLAG_ANC : 0), 0 );
      if(res != SV_OK)  {
        printf("sv_fifo_init(dst) failed %d '%s'\n", res, sv_geterrortext(res));
        hd->running = FALSE;
        return;
      }

      res = sv_fifo_sanitylevel(hd->svdst, hd->fifodst, SV_FIFO_SANITY_LEVEL_FATAL, SV_FIFO_SANITY_VERSION_1);
      if(res != SV_OK)  {
                printf("sv_fifo_sanitylevel(dst) failed %d '%s'\n", res, sv_geterrortext(res));
      }
#endif

      res = sv_fifo_startex(hd->svdst, hd->fifodst, &displaytick, NULL, NULL, NULL);
      if(res != SV_OK)  {
        printf("sv_fifo_start(dst) failed %d '%s'\n", res, sv_geterrortext(res));
        hd->running = FALSE;
      }
      fifo_running = TRUE;
    }

    res = sv_fifo_getbuffer(hd->svdst, hd->fifodst, &hd->bufdst, NULL,
      SV_FIFO_FLAG_NODMAADDR |
      (hd->bancstreamer ? SV_FIFO_FLAG_ANC : 0) |
      (hd->bvideoonly ? SV_FIFO_FLAG_VIDEOONLY : 0) |
      (hd->baudioonly ? SV_FIFO_FLAG_AUDIOONLY : 0) |
      (hd->bfieldbased ? SV_FIFO_FLAG_FIELD : 0) |  // enable field-based mode
      (hd->bvideoonly ? 0 : SV_FIFO_FLAG_SETAUDIOSIZE) |
      (hd->bottom2top ? SV_FIFO_FLAG_STORAGEMODE : 0)
    );
    switch(res) {
      case SV_ERROR_SYNC_MISSING:
        continue;
        break;
      default:
        if(res != SV_OK)  {
          printf("sv_fifo_getbuffer(dst) failed %d '%s'\n", res, sv_geterrortext(res));
          hd->running = FALSE;
        }
        break;
    }

    if(hd->bottom2top) {
      hd->bufdst->storage.storagemode = hd->storage.videomode & (SV_MODE_COLOR_MASK|SV_MODE_NBIT_MASK|SV_MODE_STORAGE_FRAME|SV_MODE_STORAGE_BOTTOM2TOP);
      hd->bufdst->storage.xsize       = hd->storage.storagexsize;
      hd->bufdst->storage.ysize       = hd->storage.storageysize;

      if(flip) {
        hd->bufdst->storage.storagemode |= SV_MODE_STORAGE_BOTTOM2TOP;
      } else {
        hd->bufdst->storage.storagemode &= ~SV_MODE_STORAGE_BOTTOM2TOP;
      }

      flip = (displaytick / 100) & 1;
    }

    dvs_mutex_enter(&hd->common.lock);
    bufferid = get_bufferid(hd, displaytick, hd->ndelay );
    dvs_mutex_leave(&hd->common.lock);

    fill_buffer(hd, FALSE, hd->bufdst, bufferid);

    if(hd->banc && (bufferid >= 0)) {
      //Get list
      anc_element_t * current_element = &hd->anclist[bufferid];
       
      //Try to get anc packets
      while( current_element->valid ) {
        current_element->valid = 0;
        res = sv_fifo_anc( hd->svsrc, hd->fifodst, hd->bufdst, &current_element->ancbuffer );
        if( res == SV_OK ) {
          //Log
          if( hd->bverbose ) {
            printf("Displayed anc_element. Buffer:%d, DID:0x%x, SDID:0x%x\n", bufferid, current_element->ancbuffer.did, current_element->ancbuffer.sdid );
          }
          if( current_element->next ) {
            current_element = current_element->next;
          }
        }
      }
    }

    res = sv_fifo_putbuffer(hd->svdst, hd->fifodst, hd->bufdst, &infodst);
    if(res != SV_OK)  {
      printf("sv_fifo_putbuffer(dst) failed %d '%s'\n", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    // Calculate next output tick.
    displaytick = infodst.when + hd->vinterlace;

    res = sv_fifo_status(hd->svdst, hd->fifodst, &status_dst);
    if(res != SV_OK)  {
      printf("sv_fifo_status(dst) failed %d '%s'\n", res, sv_geterrortext(res));
      hd->running = FALSE;
      break;
    }

    // Save output delay
    hd->outdelay_info = infodst.when - status_dst.displaytick;
    
    if(!hd->bverbose) {
      printf("\rDmaloop: %s  in-delay:%2d/%d ticks  out-delay:%2d ticks              ", (bufferid!=-1) ? "running" : "no buffer", hd->indelay_info, hd->inavail_info, hd->outdelay_info);
    }

    if(hd->bverbose || (status_dst.dropped > lastdrop_dst)) {
      printf("ID %2d - out %06d %d %d %d (%u)\n", bufferid, infodst.when, status_dst.nbuffers, status_dst.availbuffers, status_dst.dropped - lastdrop_dst, infodst.clock_low);
    }

    // Report dropped frames in output FIFO.
    if(status_dst.dropped > lastdrop_dst) {
      printf("output fifo dropped - delay might have changed\n");
    }
    lastdrop_dst = status_dst.dropped;

    /*
     * Wait until I need a new buffer
     */
    if(hd->bfieldbased || hd->bdualsdi) {
      while(hd->running && ((status_dst.nbuffers - status_dst.availbuffers) > 3/*Fields*/)) {
        // Wait for next vsync
        res = sv_fifo_vsyncwait(hd->svdst, hd->fifodst);
        if(res != SV_OK)  {
          printf("sv_fifo_vsyncwait(dst) failed %d '%s'\n", res, sv_geterrortext(res));
	        hd->running = FALSE;  
        }
      
        // Check buffer count
        res = sv_fifo_status(hd->svdst, hd->fifodst, &status_dst);
        if(res != SV_OK)  {
          printf("sv_fifo_status(dst) failed %d '%s'\n", res, sv_geterrortext(res));
          hd->running = FALSE;  
        }
      }
    } else {
      while(hd->running && ((status_dst.nbuffers - status_dst.availbuffers) > 2 /*Frames*/)) {
        // Wait for next vsync
        res = sv_fifo_vsyncwait(hd->svdst, hd->fifodst);
        if(res != SV_OK) {
          printf("sv_fifo_vsyncwait(dst) failed %d '%s'\n", res, sv_geterrortext(res));
          hd->running = FALSE;
        }
            
        // Check buffer count
        res = sv_fifo_status(hd->svdst, hd->fifodst, &status_dst);
        if(res != SV_OK)  {
          printf("sv_fifo_status(dst) failed %d '%s'\n", res, sv_geterrortext(res));
          hd->running = FALSE;  
        }
      }
    }
  }

  res = sv_fifo_stop(hd->svdst, hd->fifodst, SV_FIFO_FLAG_FLUSH);
  if(res != SV_OK)  {
    printf("sv_fifo_stop(dst) failed %d '%s'\n", res, sv_geterrortext(res));
  }

  hd->exitcode = TRUE;
  dvs_thread_exit(&hd->exitcode, &hd->finish);

  return NULL;
}


int usage(void)
{
  printf("SYNTAX:   dmaloop [opts]\n");
  printf("FUNCTION: Video loop through between one or two boards via DMA\n");
  printf("OPTIONS:  -2     use 2 boards\n");
  printf("OPTIONS:  -a     use audio only\n");
  printf("OPTIONS:  -d[=]n loop delay in fifo buffers (default 4)\n");
  printf("OPTIONS:  -f     use field-based mode\n");
  printf("OPTIONS:  -v     use video only\n");
  printf("OPTIONS:  -p     enable anc loop\n");
  printf("OPTIONS:  -q     enable anc streamer loop\n");

  return FALSE;
}


int main(int argc, char* argv[])
{
  loop_handle * hd = &global_handle;
  dvs_thread thread_in;
  dvs_thread thread_out;
  int res = TRUE;
  int c;

  memset(hd, 0, sizeof(loop_handle));
  hd->running = TRUE;
  hd->ndelay = 4;

  while(--argc) {
    argv++;                                       // skip progamname
    if(**argv == '-') {
      c = tolower(*++*argv);                      // get option character
      ++*argv;
      if(**argv == '=') ++*argv;                  // skip equalsign
      if(**argv == ' ') ++*argv;                  // skip space
      switch(c) {
      case '2':
        hd->buse2cards = TRUE;
        break;
      case 'a':
        hd->baudioonly = TRUE;
        break;
      case 'b':
        hd->bottom2top = TRUE;  // flipping by using bottom2top
        break;
      case 'd':	
        hd->ndelay = atoi(*argv);	
        break;
      case 'f':	
        hd->bfieldbased = TRUE;	
        break;
      case 'v':
        hd->bvideoonly = TRUE;
        break;
      case 'x':
        hd->bverbose = TRUE;
        break;
      case 'p':
        hd->banc = TRUE;
        break;
      case 'q':
        hd->bancstreamer = TRUE;
        break;
      default:
        res = usage();
      }
    } else {
      res = usage();
    }
  }

  if(hd->banc && hd->bancstreamer) {
    printf("Parameter for anc overrides anc streamer.\n");
    hd->bancstreamer = FALSE;
  }

  signal(SIGTERM, signal_handler);
  signal(SIGINT,  signal_handler);

  if(res) {
    res = loop_init(hd);
  }

  if(res) {
    if(!dvs_thread_create(&thread_in, loop_in, hd, &hd->finish)) {
      printf("Creating input thread failed.\n");
      res = FALSE;
    }
  }

  if(res) {
    if(!dvs_thread_create(&thread_out, loop_out, hd, &hd->finish)) {
      printf("Creating output thread failed.\n");
      res = FALSE;
    }
  }

  if(res) {
    do {
      if(hd->bplayback) {
        printf("playback mode\n");
      } else {
        printf("loop-through mode\n");
      }

      c = getc(stdin);
      hd->bplayback = !hd->bplayback;
    } while (hd->running);
  }

  dvs_cond_broadcast(&hd->common.ready, &hd->common.lock, FALSE);
  dvs_thread_exitcode(&thread_in, &hd->finish);
  dvs_thread_exitcode(&thread_out, &hd->finish);

  loop_exit(hd);

  signal(SIGTERM, NULL);
  signal(SIGINT, NULL);

  return (res == TRUE) ? 0 : 1;
}


void delete_element( int id, anc_element_t * anc_element )
{
  if( anc_element ) {
    if( anc_element->next ) {
      delete_element( id, anc_element->next );
    }
    free( anc_element );
  }
}
