/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/


#include "svprogram.h"

#include <stdarg.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void jpeg_errorprint(sv_handle * sv, int res)
{
  if(res != SV_OK) {
    if(sv) {
      if(sv->vgui) {
        printf("<error>\n");
      }
    }
    printf("%s\n", sv_geterrortext(res));
  }
}


void jpeg_errorcode(sv_handle * sv, int res, int argc, char ** argv)
{
  if(res != SV_OK) {
    if(sv) {
      if(sv->vgui) {
        printf("<error>\n");
      }
    }
    printf("%s %s : %s\n", argv[-1], argv[0], sv_geterrortext(res));
  }
}


void jpeg_errorprintf(sv_handle * sv, char * string, ...)
{
  va_list va;
  char    ach[1024];
  
  va_start(va, string);

  vsprintf(ach, string, va);

  va_end(va);

  if(sv) {
    if(sv->vgui) {
      printf("<error>\n");
    }
  }
  printf("%s\n", ach);
}




void jpeg_record(sv_handle * sv, char * type, char * filename, int start, int nframes, char * tc, char * speed, char * loopmode)
{
  int      timecode = 0;
  int      res      = SV_OK;
  int      size     = 0;
  ubyte   *buffer   = NULL;
  sv_info  info;
  int      sv_xsize = 0;
  int      sv_ysize = 0;
  double speed_double = 1.0;
  int    speed_int = 0x10000;
  int    loop;

  if(tc) {
    res = sv_asc2tc(sv, tc, &timecode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }
  }

  if(strcmp("once",     loopmode) == 0) {
    loop = SV_LOOPMODE_ONCE;
  } else if(strcmp("infinite", loopmode) == 0) {
    loop = SV_LOOPMODE_INFINITE;
  } else {
    jpeg_errorprintf(sv, "sv speed: Unknown loopmode: %s\n", loopmode);
    return;
  }
  
  res = sv_status(sv, &info);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  res = sv_option(sv, SV_OPTION_RECORD_SPEED, speed_int);
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
   

  res = sv_option(sv, SV_OPTION_RECORD_LOOPMODE, loop);
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }

  speed_double = atof(speed);
  speed_int    = (int) (speed_double * info.video.speedbase);

  if(filename != NULL) { 
  
    fm_initialize();
  
    size   = 2 * info.setup.storagexsize * info.setup.storageysize;
    buffer = (ubyte *) malloc(size);

    if(buffer == NULL) {
      jpeg_errorprintf(sv, "sv record: Malloc(%d) failed for image buffer\n", size);
      return; 
    }
  } else {
    buffer = 0;
    size   = 0;
  }

  if(res == SV_OK) {
    res = sv_record(sv, (char *) buffer, size, &sv_xsize, &sv_ysize, start, nframes, timecode); 
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    free(buffer);
    return;
  }

  if(filename != NULL) {

    res = jpeg_convertimage(sv, FALSE, filename, type, 0,
          sv_xsize, sv_ysize, buffer, size, 1, info.colormode, 0, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, NULL);

    free(buffer);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }

    fm_deinitialize();
  }
}



void jpeg_display(sv_handle * sv, char * type, char * filename, int start, int nframes, char * tc)
{
  int      timecode = 0;
  int      res      = SV_OK;
  int      size     = 0;
  ubyte   *buffer   = NULL;
  sv_info  info;
  
  if(tc) {
    res = sv_asc2tc(sv, tc, &timecode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }
  }

  res = sv_status(sv, &info);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  if(filename != NULL) {
    fm_initialize();
  
    size   = 4 * info.setup.storagexsize * info.setup.storageysize;
    buffer = (ubyte *) malloc(size);

    if(buffer == NULL) {
      jpeg_errorprintf(sv, "sv display: Malloc(%d) failed for image buffer\n", size);
      return; 
    }
  
    res = jpeg_convertimage(sv, TRUE, filename, type, 0, 
          info.setup.storagexsize, info.setup.storageysize, buffer, size, 1, info.colormode, 0, info.nbit, (info.config&SV_MODE_STORAGE_FRAME?1:2), (info.config&SV_MODE_STORAGE_BOTTOM2TOP?1:0), info.xsize, NULL); 
  } else {
    buffer = 0;
    size   = 0;
  }
  
  if(res == SV_OK) {
    res = sv_display(sv, (char *) buffer, size, info.xsize, info.ysize, start, nframes, timecode);      
  }
  
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  if(filename != NULL) {
    free(buffer);
  
    fm_deinitialize();
  }
}


void jpeg_vtredit(sv_handle * sv, char * tc, int nframes)
{
  int      timecode = 0;
  int      res      = SV_OK;
 
  if(tc) {
    res = sv_asc2tc(sv, tc, &timecode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }
  }

  res = sv_vtredit(sv, timecode, nframes); 

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }
}


    

void jpeg_black(sv_handle * sv)
{
  int res = sv_black(sv);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_colorbar(sv_handle * sv)
{
  int res = sv_colorbar(sv);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_slave(sv_handle * sv, char * what)
{
  int mode = 0;
  int res  = SV_OK;

  if((strcmp(what, "on") == 0)) {
    mode = SV_SLAVE_ENABLED;
  } else if((strcmp(what, "off") == 0)) {
    mode = SV_SLAVE_DISABLED;
  } else if((strcmp(what, "always") == 0)) {
    mode = SV_SLAVE_ALWAYS;
  } else if((strcmp(what, "slaveinfo") == 0)) {
    mode = SV_SLAVE_SLAVEINFO;
  } else if((strcmp(what, "driver") == 0)) {
    mode = SV_SLAVE_DRIVER;
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res == SV_OK) {
    res = sv_slave(sv, mode);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }  
}    
 

void jpeg_live(sv_handle * sv, char * showmode)
{
  int res = SV_OK;
  int mode = -1;

  if(showmode) {
    if(strcmp(showmode, "bypass") == 0) {
      mode = SV_SHOWINPUT_BYPASS;
    } else if(strcmp(showmode, "frame") == 0) {
      mode = SV_SHOWINPUT_FRAMEBUFFERED;
    } else if(strcmp(showmode, "field") == 0) {
      mode = SV_SHOWINPUT_FIELDBUFFERED;
    } else if(strcmp(showmode, "default") == 0) {
      mode = SV_SHOWINPUT_DEFAULT;
    } else if((strcmp(showmode, "?") == 0) || (strcmp(showmode, "help") == 0)) {
      printf("sv live help\n");
      printf("\tdefault\n");
      printf("\tbypass\n");
      printf("\tframe\n");
      printf("\tfield\n");
    } else {
      jpeg_errorprintf(sv, "sv live : Unknown live mode: %s\n", showmode);
    }
  } else {
    mode = SV_SHOWINPUT_DEFAULT;
  } 

  if(mode != -1) {
    res = sv_showinput(sv, mode, 1);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



sv_handle * jpeg_open(int hideerror, int infodevice)
{
  sv_handle * sv;
  int res = sv_openex(&sv, "", SV_OPENPROGRAM_SVPROGRAM, infodevice ? SV_OPENTYPE_QUERY : SV_OPENTYPE_DEFAULT, 0, 0);

  if(sv == NULL) {
    if(!hideerror) {
      switch(res) {
      case SV_ERROR_DEVICEINUSE:
        jpeg_errorprintf(sv, "Failure to connect to Video Device, device in use\n");
        break;
      case SV_ERROR_DEVICENOTFOUND:
        jpeg_errorprintf(sv, "Failure to connect to Video Device, device not found\n");
        break;
      case SV_ERROR_USERNOTALLOWED:
        jpeg_errorprintf(sv, "That system dosn't had permission to work with the Video Server software\n");
        break;
      case SV_ERROR_DRIVER_MISMATCH:
        jpeg_errorprintf(sv, "Driver and library version do not match\n");
        break;
      case SV_ERROR_WRONGMODE:
        res = sv_openex(&sv, "", SV_OPENPROGRAM_SVPROGRAM, SV_OPENTYPE_QUERY, 0, 0);
        if(sv != NULL) {
          int multichannel = 0;

          res = sv_option_get(sv, SV_OPTION_MULTICHANNEL, &multichannel);
          if(res == SV_OK) {
            if(!multichannel) {
              jpeg_errorprintf(sv, "Failure to connect to Video Device as multichannel mode is off\n");
            } else {
              jpeg_errorprintf(sv, "Failure to connect to Video Device (multichannel is on)\n");
            }
          }

          sv_close(sv);
          sv = NULL;
        } else {
          jpeg_errorprintf(sv, "Failure to connect to Video Device\n");
        }
        break;
      default:
        jpeg_errorprintf(sv, "Failure to connect to Video Device\n");
      }
      if(getenv("VGUI")) {
        printf("<exit>\n");
      }
    }
  }

  return sv;
}



void jpeg_close(sv_handle * sv)
{
  int res = sv_close(sv);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


int jpeg_support_string2videomode(sv_handle * sv, char * videomode)
{
  int video = sv_support_string2videomode(videomode, 0);
  sv_rasterheader raster;
  int res;
  int feature;
  int i,j;
  int nrasters;

  if(video == -1) {
    res = sv_query(sv, SV_QUERY_FEATURE, 0, &feature);
    if((res == SV_OK) && (feature & SV_FEATURE_RASTERLIST)) {
      for(nrasters = 1, i = 0; (i < nrasters) && (video == -1); i++) {
        res = sv_raster_status(sv, i, &raster, sizeof(raster), &nrasters, 0);
        if(res == SV_OK) {
          for(j = 0; (j < sizeof(raster.name)-1) && (raster.name[j] != ' '); j++);
          if(!strncmp(videomode, raster.name, j)) {
            video = sv_support_string2videomode(videomode, j) | SV_MODE_FLAG_RASTERINDEX;
            if (video != -1) {
              video |= i;
            }
          }
        }
      }
    }
  }

  return video;
}


void jpeg_mode(sv_handle * sv, char * videomode)
{
  int res   = SV_OK;
  int video = -1;
  int rasterindex = FALSE;

  if((strcmp(videomode, "?") == 0) || (strcmp(videomode, "help") == 0)) {
    printf("sv mode help\n");
    printf("\tUse the mode description shown below to enter a video mode.\n");
    printf("\tFx: sv mode SMPTE274/30I or sv mode PAL.\n");
    printf("\t    Additional qualifiers are /10B for 10 bit video modes.\n");
    printf("List of supported video modes:\n");
    jpeg_guiinfo(sv, "mode");
    return;    
  }

  if((videomode[0] >= '0') && (videomode[0] <= '9')) {
    rasterindex = TRUE;
    video       = atoi(videomode);
    if(video < 1000) {
      video |=  SV_MODE_FLAG_RASTERINDEX;
    } else {
      rasterindex = FALSE;
    }
  }

  if(!rasterindex) {
    video = jpeg_support_string2videomode(sv, videomode);
  }
   
  if(video == -1) {
    jpeg_errorprintf(sv, "sv mode : Unknown videomode: %s\n", videomode);	
    return;
  }

  res = sv_videomode(sv, video);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


static char *str_colormode[] = {  "MONO", "BGR", "YUV422", "YUV411",
                                  "YUV422A", "RGBA", "YUV422STEREO", "YUV444", 
                                  "YUV444A", "YUV420", "RGBVIDEO", "RGBAVIDEO", 
                                  "YUV2QT", "RGB", "BGRA", "YUV422_YUYV", 
                                  "CHROMA", "ARGB", "ABGR" };


static char *str_inputfilter[]  = { "default", "nofilter", "5taps", "9taps", "13taps", "17taps",  };

static char *str_outputfilter[] = { "default", "nofilter", "5taps", "9taps", "13taps", "17taps",  };

static char *str_yuvmatrix[] = { "CCIR601", "CCIR709", "RGB", "RGB/CGR" };

static char *str_fstype[]    = { "FLAT", "PVFS", "CSFS", "OSFS" };

static char *str_slowmotion[]= { "frame", "field", "field1", "field2" };
 
static char *str_repeatmode[]= { "frame", "field1", "field2", "current" };

static char *str_audioinput[] = { "aiv", "aesebu" };

static char *str_audiomode[] = { "always", "on_speed1", "on_motion" };

static char *str_loopmode[]  = { "loop", "reverse", "shuttle", "once",
                                 "loop" };

static char *str_wordclock[] = { "off", "on" };

static char *str_ancdata[] = { "default", "disabled", "userdef", "rp188", "rp201", "rp196", 
                               "rp196ltc", "rp215" };


static char *str_fastmode[]  = { "default", "field1", "field2", "best_match",
                                 "aba", "abab" };


static char *str_master_tolerance[] = { "none", "normal", "large", "rough" };

static char *str_master_timecodetype[] = { "VITC", "LTC", "auto", "timer1", "timer2" };

static char * str_ltcsource[] = { "default", "intern", "playlist", "master", "freerunning", "ltcoffset", "proxy" };

static char * str_ltcsource_rec[] = { "intern", "EE"};

static char * str_tcout_aux[] = { "Off", "Timecode", "Frame", "Timecode & Frame"};

static char *str_multidevice[] = { "off", "master", "slave", "master delayline" };

static char *str_recordmode[] = { "normal", "gpi", "variframe" };

static char *str_proxyvideomode[]  = { "PAL", "NTSC" };
static char *str_proxysyncmode[]  = { "auto", "internal", "genlock"  };
static char *str_proxyoutput[]  = { "underscan", "letterbox", "cropped", "anamorph" };
static char *str_analogout[]  = { "rgbfull", "rgbhead", "yuvfull", "yuvhead" };

static struct {
  int    mode;
  char * name;
  int    frequency;
  int    xsize;
  int    ysize;
} table_rasters[] = {
  { SV_MODE_PAL,          "PAL/25I       625/25.00Hz 2:1",  27000000,  720,  576 },
  { SV_MODE_PALHR,        "PALHR/25I     625/25.00Hz 2:1",  36000000,  960,  576 },
  { SV_MODE_NTSC,         "NTSC/29I      525/29.97Hz 2:1",  27000000,  720,  486 },
  { SV_MODE_NTSCHR,       "NTSCHR/29I    525/29.97Hz 2:1",  36000000,  960,  486 },
  { SV_MODE_HD360,        "HD360         HDTV Compress  ",  36000000,  960,  504 },

  
  { SV_MODE_SMPTE240_29I, "SMPTE240/29I  1125/29.97Hz 2:1", 74175824, 1920, 1035 },
  { SV_MODE_SMPTE240_30I, "SMPTE240/30I  1125/30.00Hz 2:1", 74250000, 1920, 1035 },
   
  { SV_MODE_SMPTE274_23I, "SMPTE274/23I  1125/23.98Hz 2:1", 74175824, 1920, 1080 },
  { SV_MODE_SMPTE274_24I, "SMPTE274/24I  1125/24.00Hz 2:1", 74250000, 1920, 1080 },
  { SV_MODE_SMPTE274_25I, "SMPTE274/25I  1125/25.00Hz 2:1", 74250000, 1920, 1080 },
  { SV_MODE_SMPTE274_29I, "SMPTE274/29I  1125/29.97Hz 2:1", 74175824, 1920, 1080 },
  { SV_MODE_SMPTE274_30I, "SMPTE274/30I  1125/30.00Hz 2:1", 74250000, 1920, 1080 },

  { SV_MODE_SMPTE274_23sF, "SMPTE274/23sF  1125/23.98Hz sF", 74175824, 1920, 1080 },
  { SV_MODE_SMPTE274_24sF, "SMPTE274/24sF  1125/24.00Hz sF", 74250000, 1920, 1080 },
  { SV_MODE_SMPTE274_25sF, "SMPTE274/25sF  1125/25.00Hz sF", 74250000, 1920, 1080 },
  { SV_MODE_SMPTE274_29sF, "SMPTE274/29sF  1125/29.97Hz sF", 74175824, 1920, 1080 },
  { SV_MODE_SMPTE274_30sF, "SMPTE274/30sF  1125/30.00Hz sF", 74250000, 1920, 1080 },

  { SV_MODE_SMPTE274_23P, "SMPTE274/23P  1125/23.98Hz 1:1",  74175824, 1920, 1080 },
  { SV_MODE_SMPTE274_24P, "SMPTE274/24P  1125/24.00Hz 1:1",  74250000, 1920, 1080 },
  { SV_MODE_SMPTE274_25P, "SMPTE274/25P  1125/25.00Hz 1:1",  74250000, 1920, 1080 },
  { SV_MODE_SMPTE274_29P, "SMPTE274/29P  1125/29.97Hz 1:1",  74175824, 1920, 1080 },
  { SV_MODE_SMPTE274_30P, "SMPTE274/30P  1125/30.00Hz 1:1",  74250000, 1920, 1080 },

  { SV_MODE_SMPTE274_47P, "SMPTE274/47P  1125/47.96Hz 1:1", 148351648, 1920, 1080 },
  { SV_MODE_SMPTE274_48P, "SMPTE274/48P  1125/48.00Hz 1:1", 148500000, 1920, 1080 },
  { SV_MODE_SMPTE274_59P, "SMPTE274/59P  1125/59.94Hz 1:1", 148351648, 1920, 1080 },
  { SV_MODE_SMPTE274_60P, "SMPTE274/60P  1125/60.00Hz 1:1", 148500000, 1920, 1080 },
  { SV_MODE_SMPTE274_71P, "SMPTE274/71P  1125/71.93Hz 1:1", 148351648, 1920, 1080 },
  { SV_MODE_SMPTE274_72P, "SMPTE274/72P  1125/72.00Hz 1:1", 148500000, 1920, 1080 },

  { SV_MODE_SMPTE295_25I, "SMPTE295/25I  1250/25.00Hz 2:1",  74250000, 1920, 1080 },

  { SV_MODE_SMPTE296_59P, "SMPTE296/59P  750/59.94Hz 1:1",   74175824, 1280,  720 },
  { SV_MODE_SMPTE296_60P, "SMPTE296/60P  750/60.00Hz 1:1",   74250000, 1280,  720 },
  { SV_MODE_SMPTE296_71P, "SMPTE296/71P  750/71.93Hz 1:1",   74175824, 1280,  720 },
  { SV_MODE_SMPTE296_72P, "SMPTE296/72P  750/72.00Hz 1:1",   74250000, 1280,  720 },

  { SV_MODE_SMPTE296_71P_89MHZ, "SMPTE296/71P/89MHZ  750/71.93Hz 1:1", 89010989, 1280, 720},
  { SV_MODE_SMPTE296_72P_89MHZ, "SMPTE296/72P/89MHZ  750/72.00Hz 1:1", 89100000, 1280, 720},

  { SV_MODE_EUREKA,       "EUREKA/25I   1250/25.00Hz 2:1",   72000000, 1920, 1152 },

  { SV_MODE_SMPTE274_2560_23P,"SMPTE_2560/23P 1125/23.98Hz 1:1", 74175824, 2560, 1080 },
  { SV_MODE_SMPTE274_2560_24P,"SMPTE_2560/24P 1125/24.00Hz 1:1", 74250000, 2560, 1080 },

  { SV_MODE_VESA_800x600_71P,  "VESA_800x600/71P   71.93Hz 1:1",  74175824,  800,  600 },
  { SV_MODE_VESA_800x600_72P,  "VESA_800x600/72P   72.00Hz 1:1",  74250000,  800,  600 },
  { SV_MODE_VESA_1024x768_71P, "VESA_1024x768/71P  71.93Hz 1:1",  74175824, 1024,  768 },
  { SV_MODE_VESA_1024x768_72P, "VESA_1024x768/72P  72.00Hz 1:1",  74250000, 1024,  768 },
  { SV_MODE_VESA_1280x1024_71P,"VESA_1280x1024/71P 71.93Hz 1:1", 129470530, 1280, 1024 },
  { SV_MODE_VESA_1280x1024_72P,"VESA_1280x1024/72P 72.00Hz 1:1", 129600000, 1280, 1024 },

  { SV_MODE_FILM2K_1536_24P,   "FILM2K_1536/24P    24.00Hz 1:1",  96000000, 2048, 1536 },
  { SV_MODE_FILM2K_1536_24sF,  "FILM2K_1536/24sF   24.00Hz sF ",  96000000, 2048, 1536 },
  { SV_MODE_FILM2K_1536_48P,   "FILM2K_1536/48P    48.00Hz 1:1", 192000000, 2048, 1536 },
  { SV_MODE_FILM2K_1556_24P,   "FILM2K_1556/24P    24.00Hz 1:1",  96000000, 2048, 1556 },
  { SV_MODE_FILM2K_1556_24sF,  "FILM2K_1556/24sF   24.00Hz sF ",  96000000, 2048, 1556 },
  { SV_MODE_FILM2K_1556_48P,   "FILM2K_1556/48P    48.00Hz 1:1", 192000000, 2048, 1556 },

  { SV_MODE_TEST,         "TEST                         ",           1,   1,     1},

};


void jpeg_info_clock(sv_handle * sv)
{
  int res = SV_OK;
  sv_clock_info clock;

  res = sv_realtimeclock(sv, 0, &clock, 0);
  if(res == SV_OK) {
    printf("%04d/%02d/%02d %02d:%02d:%02d-%06d\n", clock.year, clock.month, clock.day, clock.hours, clock.minutes, clock.seconds, clock.microseconds);
  } else {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_info_publickey(sv_handle * sv)
{
  int res = SV_OK;
  uint8 publickey[512];
  int j;

  res = sv_publickey(sv, (char*)publickey, 512);
  if(res == SV_OK) {
    if(publickey[0] > 128) {
      printf("No public key found\n");
    } else {
      for(j = 0; publickey[j] && (publickey[j] < 128) && (j < 512); j++) {
        printf("%c", publickey[j]);
      }
    }
  } else {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_info_random(sv_handle * sv)
{
  int res = SV_OK;
  uint8 random[128];
  int i,j;

  res = sv_random(sv, (char*)random, 128);
  if(res == SV_OK) {
    for(j = 0; j < 128; j+=16) {
      for(i = 0; i < 16; i++) {
        printf("%02x ", random[i+j]);
      }
      printf("\n");
    }
  } else {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_info_card(sv_handle * sv)
{
  int res;
  int temp;

  res = sv_query(sv, SV_QUERY_HW_MEMORYSIZE, 0, &temp);
  if(res == SV_OK) {
    printf("Memorysize:   %d MB\n", temp/0x100000);
  }
  res = sv_query(sv, SV_QUERY_HW_MAPPEDSIZE, 0, &temp);
  if(res == SV_OK) {
    printf("Mappedsize:   %d MB\n", temp/0x100000);
  }
  res = sv_query(sv, SV_QUERY_HW_FLASHSIZE, 0, &temp);
  if(res == SV_OK) {
    printf("Flashsize:    %d MB\n", temp/0x100000);
  }
  res = sv_query(sv, SV_QUERY_HW_FLASHVERSION, 0, &temp);
  if(res == SV_OK) {
    printf("Flashversion: %d\n", temp);
  }
  res = sv_query(sv, SV_QUERY_HW_EPLDTYPE, 0, &temp);
  if(res == SV_OK) {
    printf("EpldType:     %x\n", temp);
  }
  res = sv_query(sv, SV_QUERY_HW_PCISPEED, 0, &temp);
  if(res == SV_OK) {
    printf("PCISpeed:     %d MHz\n", temp);
  }
  res = sv_query(sv, SV_QUERY_HW_PCIWIDTH, 0, &temp);
  if(res == SV_OK) {
    printf("PCIWidth:     %d bit\n", temp);
  }
  res = sv_query(sv, SV_QUERY_HW_PCIELANES, 0, &temp);
  if(res == SV_OK) {
    printf("PCI-E lanes:  %d\n", temp);
  }
  res = sv_query(sv, SV_QUERY_HW_EPLDVERSION, 0, &temp);
  if(res == SV_OK) {
    printf("Firmware:     %d.%d.%d.%d\n", (temp>>16) & 0xff, (temp>>8) & 0xff, (temp) & 0xff, (temp>>24) & 0xff);
  }
  res = sv_query(sv, SV_QUERY_HW_EPLDOPTIONS, 0, &temp);
  if(res == SV_OK) {
    printf("Options:      %08x\n", temp);
  }
  res = sv_query(sv, SV_QUERY_HW_CARDVERSION, 0, &temp);
  if(res == SV_OK) {
    printf("Cardversion:  %d.%d.%d.%d\n", (temp>>16) & 0xff, (temp>>8) & 0xff, (temp) & 0xff, (temp>>24) & 0xff);
  }
  res = sv_query(sv, SV_QUERY_HW_CARDOPTIONS, 0, &temp);
  if(res == SV_OK) {
    printf("Cardoptions:  %08x\n", temp);
  }
}

void jpeg_info_hardware(sv_handle * sv)
{
  int res;
  int temp;
  int max = 1;
  int i;

  res = sv_query(sv, SV_QUERY_DEVTYPE, 0, &temp);
  if(res == SV_OK) {
    if(temp > SV_DEVTYPE_CENTAURUS) {
      max = 5;
    }
  }

  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_TEMPERATURE, i, &temp) == SV_OK); i++) {
    printf("Temperature     : %1.1f C\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V0, i, &temp) == SV_OK); i++) {
    printf("Voltage (1.0V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V2, i, &temp) == SV_OK); i++) {
    printf("Voltage (1.2V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V5, i, &temp) == SV_OK); i++) {
    printf("Voltage (1.5V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V8, i, &temp) == SV_OK); i++) {
    printf("Voltage (1.8V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_2V3, i, &temp) == SV_OK); i++) {
    printf("Voltage (2.3V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_2V5, i, &temp) == SV_OK); i++) {
    printf("Voltage (2.5V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_3V3, i, &temp) == SV_OK); i++) {
    printf("Voltage (3.3V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_5V0, i, &temp) == SV_OK); i++) {
    printf("Voltage (5.0V)  : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_12V0, i, &temp) == SV_OK); i++) {
    printf("Voltage (12.0V) : %1.2f V\n", ((double)temp)/0x10000);
  }
  for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_FANSPEED, i, &temp) == SV_OK); i++) {
    printf("Fanspeed        : %d rpm\n", temp);
  }
}

void jpeg_info_input(sv_handle * sv)
{
  int res;
  int temp;
  int tmp2;
  int tmp3;
  int iospeed;

  res = sv_query(sv, SV_QUERY_INPUTRASTER, 0, &temp);
  if(res == SV_OK) {
    printf("Inputraster         : %d '%s'\n", temp, sv_support_videomode2string(temp));
  }
  res = sv_query(sv, SV_QUERY_INPUTPORT, 0, &temp);
  if(res == SV_OK) {
    printf("Inputport           : '%s'\n", sv_query_value2string(sv, SV_QUERY_INPUTPORT, temp));
  }
  res = sv_query(sv, SV_QUERY_SMPTE352, 0, &temp);
  if(res == SV_OK) {
    printf("Smpte352            : %02x %02x %02x %02x\n", temp & 0xff, (temp >> 8) & 0xff, (temp >> 16) & 0xff, (temp >> 24) & 0xff);
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_SDIA, 0, &temp);
  if(sv_query(sv, SV_QUERY_IOSPEED_SDIA, 0, &iospeed) != SV_OK) {
    iospeed = SV_IOSPEED_UNKNOWN;
  }
  if(res == SV_OK) {
    printf("Inputraster SDI A   : %d '%s' %s\n", temp, sv_support_videomode2string(temp), (iospeed != SV_IOSPEED_UNKNOWN)?sv_option_value2string(sv, SV_OPTION_IOSPEED, iospeed):"");
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_SDIB, 0, &temp);
  if(sv_query(sv, SV_QUERY_IOSPEED_SDIB, 0, &iospeed) != SV_OK) {
    iospeed = SV_IOSPEED_UNKNOWN;
  }
  if(res == SV_OK) {
    printf("Inputraster SDI B   : %d '%s' %s\n", temp, sv_support_videomode2string(temp), (iospeed != SV_IOSPEED_UNKNOWN)?sv_option_value2string(sv, SV_OPTION_IOSPEED, iospeed):"");
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_SDIC, 0, &temp);
  if(sv_query(sv, SV_QUERY_IOSPEED_SDIC, 0, &iospeed) != SV_OK) {
    iospeed = SV_IOSPEED_UNKNOWN;
  }
  if(res == SV_OK) {
    printf("Inputraster SDI C   : %d '%s' %s\n", temp, sv_support_videomode2string(temp), (iospeed != SV_IOSPEED_UNKNOWN)?sv_option_value2string(sv, SV_OPTION_IOSPEED, iospeed):"");
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_SDID, 0, &temp);
  if(sv_query(sv, SV_QUERY_IOSPEED_SDID, 0, &iospeed) != SV_OK) {
    iospeed = SV_IOSPEED_UNKNOWN;
  }
  if(res == SV_OK) {
    printf("Inputraster SDI D   : %d '%s' %s\n", temp, sv_support_videomode2string(temp), (iospeed != SV_IOSPEED_UNKNOWN)?sv_option_value2string(sv, SV_OPTION_IOSPEED, iospeed):"");
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_DVI, 0, &temp);
  if(res == SV_OK) {
    printf("Inputraster DVI     : %d '%s'\n", temp, sv_support_videomode2string(temp));
  }
  res = sv_query(sv, SV_QUERY_GENLOCK, 0, &tmp2);
  if(res != SV_OK) {
    tmp2 = FALSE;
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_GENLOCK_TYPE, 0, &tmp3);
  if(res != SV_OK) {
    tmp3 = FALSE;
  }
  res = sv_query(sv, SV_QUERY_INPUTRASTER_GENLOCK, 0, &temp);
  if(res == SV_OK) {
    printf("Inputraster Genlock : %d '%s' %s %s\n", temp, sv_support_videomode2string(temp), tmp2?"Locked":"", tmp3==SV_SYNC_BILEVEL?"Bilevel":(tmp3==SV_SYNC_TRILEVEL?"Trilevel":""));
  }
  res = sv_query(sv, SV_QUERY_AUDIOINERROR, 0, &temp);
  if(res == SV_OK) {
    printf("AudioInError        : %s\n", sv_geterrortext(temp));
  }
  res = sv_query(sv, SV_QUERY_SDILINKDELAY, 0, &temp);
  if(res == SV_OK) {
    printf("LinkDelay A->B      : %d\n", temp);
  }
  res = sv_query(sv, SV_QUERY_VIDEOINERROR, 0, &temp);
  if(res == SV_OK) {
    printf("VideoInError        : %s\n", sv_geterrortext(temp));
  }
  res = sv_query(sv, SV_QUERY_IOMODEINERROR, 0, &temp);
  if(res == SV_OK) {
    printf("IOModeInError       : %s\n", sv_geterrortext(temp));
  }
  res = sv_query(sv, SV_QUERY_AUDIO_AIVCHANNELS, 0, &temp);
  if(res == SV_OK) {
    printf("AiVChannels         : 0x%08x\n", temp);
  }
  res = sv_query(sv, SV_QUERY_AUDIO_AESCHANNELS, 0, &temp);
  if(res == SV_OK) {
    printf("AESChannels         : 0x%08x\n", temp);
  }
  res = sv_query(sv, SV_QUERY_VALIDTIMECODE, 0, &temp);
  if(res == SV_OK) {
    printf("Timecodes           : %s%s%s%s%s%s%s%s%s%s%s%s%s%s\n", 
      (temp & SV_VALIDTIMECODE_VTR)?"VTR ":"",
      (temp & SV_VALIDTIMECODE_DLTC)?"DLTC ":"",
      (temp & SV_VALIDTIMECODE_LTC)?"LTC ":"",
      (temp & SV_VALIDTIMECODE_RP215)?"RP215 ":"",
      (temp & SV_VALIDTIMECODE_VITC_F1)?"VITC/F1 ":"",
      (temp & SV_VALIDTIMECODE_DVITC_F1)?"DVITC/F1 ":"",
      (temp & SV_VALIDTIMECODE_RP201_F1)?"RP201/F1 ":"",
      (temp & SV_VALIDTIMECODE_CC_F1)?"CC/F1 ":"",
      (temp & SV_VALIDTIMECODE_ARP201_F1)?"ARP201/F1 ":"",
      (temp & SV_VALIDTIMECODE_VITC_F2)?"VITC/F2 ":"",
      (temp & SV_VALIDTIMECODE_DVITC_F2)?"DVITC/F2 ":"",
      (temp & SV_VALIDTIMECODE_RP201_F2)?"RP201/F2 ":"",
      (temp & SV_VALIDTIMECODE_CC_F2)?"CC/F2 ":"",
      (temp & SV_VALIDTIMECODE_ARP201_F2)?"ARP201/F2 ":"");
  }
}


int jpeg_support_videomode2string(sv_handle * sv, char * buffer, int mode)
{
  sv_rasterheader raster;
  char * smode = NULL;
  int res;
  int feature;

  res = sv_query(sv, SV_QUERY_FEATURE, 0, &feature);

  if((res == SV_OK) && (feature & SV_FEATURE_RASTERLIST)) {
    if(mode & SV_MODE_FLAG_RASTERINDEX) {
      res = sv_raster_status(sv, mode & SV_MODE_MASK, &raster, sizeof(raster), NULL, 0);
      if(res == SV_OK) {
        smode = &raster.name[0];
      }
    } else {
      int nrasters = 1;
      int i;
      for(i = 0; (i < nrasters) && !smode; i++) {
        res = sv_raster_status(sv, i, &raster, sizeof(raster), &nrasters, 0);
        if(res == SV_OK) {
          if(raster.svind == (mode & SV_MODE_MASK)) {
            smode = &raster.name[0];
          }
        }
      }
    }
    if(smode) {
      char * p = strchr(&raster.name[0], ' ');
      if(p) {
        *p = 0;
      }
    }
  }
  if(!smode) {
    smode = sv_support_videomode2string(mode);
  }

  strcpy(buffer, smode);

  switch(mode & SV_MODE_COLOR_MASK) {
  case SV_MODE_COLOR_CHROMA:
    strcat(buffer, "/CHROMA");
    break;
  case SV_MODE_COLOR_LUMA:
    strcat(buffer, "/LUMA");
    break;
  case SV_MODE_COLOR_RGB_BGR:
    strcat(buffer, "/BGR");
    break;
  case SV_MODE_COLOR_BGRA:
    strcat(buffer, "/BGRA");
    break;
  case SV_MODE_COLOR_RGB_RGB:
    strcat(buffer, "/RGB");
    break;
  case SV_MODE_COLOR_ABGR:
    strcat(buffer, "/ABGR");
    break;
  case SV_MODE_COLOR_ARGB:
    strcat(buffer, "/ARGB");
    break;
  case SV_MODE_COLOR_RGBA:
    strcat(buffer, "/RGBA");
    break;
  case SV_MODE_COLOR_YUV2QT:
    strcat(buffer, "/YUV2QT");
    break;
  case SV_MODE_COLOR_YUV422:
    strcat(buffer, "/YUV422");
    break;
  case SV_MODE_COLOR_YUV422_YUYV:
    strcat(buffer, "/YUV422_YUYV");
    break;
  case SV_MODE_COLOR_YUV422A:
    strcat(buffer, "/YUV422A");
    break;
  case SV_MODE_COLOR_YUV444:
    strcat(buffer, "/YUV444");
    break;
  case SV_MODE_COLOR_YUV444_VYU:
    strcat(buffer, "/YUV444_VYU");
    break;
  case SV_MODE_COLOR_YUV444A:
    strcat(buffer, "/YUV444A");
    break;
  case SV_MODE_COLOR_ALPHA:
    strcat(buffer, "/ALPHA");
    break;
  case SV_MODE_COLOR_ALPHA_422A:
    strcat(buffer, "/ALPHA_422A");
    break;
  case SV_MODE_COLOR_ALPHA_444A:
    strcat(buffer, "/ALPHA_444A");
    break;
  case SV_MODE_COLOR_ALPHA_A444:
    strcat(buffer, "/ALPHA_A444");
    break;
  case SV_MODE_COLOR_XYZ:
    strcat(buffer, "/XYZ");
    break;
  case SV_MODE_COLOR_YCC:
    strcat(buffer, "/YCC");
    break;
  case SV_MODE_COLOR_YCC422:
    strcat(buffer, "/YCC422");
    break;
  case SV_MODE_COLOR_BAYER_BGGR:
    strcat(buffer, "/BAYER_BGGR");
    break;
  case SV_MODE_COLOR_BAYER_GBRG:
    strcat(buffer, "/BAYER_GBRG");
    break;
  case SV_MODE_COLOR_BAYER_GRBG:
    strcat(buffer, "/BAYER_GRBG");
    break;
  case SV_MODE_COLOR_BAYER_RGGB:
    strcat(buffer, "/BAYER_RGGB");
    break;
  case SV_MODE_COLOR_WHITE:
    strcat(buffer, "/WHITE");
    break;
  case SV_MODE_COLOR_BLACK:
    strcat(buffer, "/BLACK");
    break;
  default:
    strcat(buffer, "/?colormode?");
  }

  switch(mode & SV_MODE_NBIT_MASK) {
  case SV_MODE_NBIT_8B:
    break;
  case SV_MODE_NBIT_10B:
    strcat(buffer, "/10B");
    break;
  case SV_MODE_NBIT_10BDVS:
    strcat(buffer, "/10BDVS");
    break;
  case SV_MODE_NBIT_10BDPX:
    strcat(buffer, "/10BDPX");
    break;
  case SV_MODE_NBIT_10BLALE:
    strcat(buffer, "/10BLALE");
    break;
  case SV_MODE_NBIT_10BRABE:
    strcat(buffer, "/10BRABE");
    break;
  case SV_MODE_NBIT_10BRALEV2:
    strcat(buffer, "/10BRALEV2");
    break;
  case SV_MODE_NBIT_10BLABEV2:
    strcat(buffer, "/10BLABEV2");
    break;
  case SV_MODE_NBIT_10BLALEV2:
    strcat(buffer, "/10BLALEV2");
    break;
  case SV_MODE_NBIT_10BRABEV2:
    strcat(buffer, "/10BRABEV2");
    break;
  case SV_MODE_NBIT_12B:
    strcat(buffer, "/12B");
    break;
  case SV_MODE_NBIT_12BDPX:
    strcat(buffer, "/12BDPX");
    break;
  case SV_MODE_NBIT_12BDPXLE:
    strcat(buffer, "/12BDPXLE");
    break;
  case SV_MODE_NBIT_16BBE:
    strcat(buffer, "/16BBE");
    break;
  case SV_MODE_NBIT_16BLE:
    strcat(buffer, "/16BLE");
    break;
  default:
    strcat(buffer, "/?nbits?");
  }

  if(mode & SV_MODE_STORAGE_FRAME) { 
    strcat(buffer, "/FRAME");
  } 
  if(mode & SV_MODE_FLAG_PACKED) {
    strcat(buffer, "/PACKED");
  } 
  if(mode & SV_MODE_STORAGE_BOTTOM2TOP) {
    strcat(buffer, "/BOTTOM2TOP");
  } 


  switch(mode & SV_MODE_AUDIO_MASK) {
  case 0:
    break;
  case SV_MODE_AUDIO_1CHANNEL:
    strcat(buffer, "/AUDIO1CH");
    break;
  case SV_MODE_AUDIO_2CHANNEL:
    strcat(buffer, "/AUDIO2CH");
    break;
  case SV_MODE_AUDIO_4CHANNEL:
    strcat(buffer, "/AUDIO4CH");
    break;
  case SV_MODE_AUDIO_6CHANNEL:
    strcat(buffer, "/AUDIO6CH");
    break;
  case SV_MODE_AUDIO_8CHANNEL:
    strcat(buffer, "/AUDIO8CH");
    break;
  default:
    strcat(buffer, "/?achannels?");
  }

  if(mode & SV_MODE_AUDIO_MASK) {
    switch(mode & SV_MODE_AUDIOBITS_MASK) {
    case SV_MODE_AUDIOBITS_16:
      strcat(buffer, "/16");
      break;
    case SV_MODE_AUDIOBITS_32:
      strcat(buffer, "/32");
      break;
    default:
      strcat(buffer, "/?abits?");
    }
  }
  
  return res;
}


void jpeg_info(sv_handle * sv)
{
  sv_info info;
  char    buffer[256];
  int     jackinput = -1;
  int     jackoutput = -1;
  int     intid;
  int     mode[2];
  int     res = sv_status(sv, &info);
  int     tmp_timecode;
  int     tmp_userbytes;
  char *  state;
  int     syncstate;
  int     syncselect;
  char    syncselect_string[32];
  char *  b10bit;
  int     value;


  if(sv_jack_find(sv, -1, "output", 0, &jackoutput) != SV_OK) {
    jackoutput = -1;
  }
  if(sv_jack_find(sv, -1, "input", 0, &jackinput) != SV_OK) {
    jackinput = -1;
  }


  if(res == SV_OK) {

    state = sv_support_devmode2string(info.video.state);
    sv_tc2asc(sv, info.video.positiontc, buffer, sizeof(buffer));
    printf("Timecode        : %-12s  %8d  %s\n", buffer, info.video.position, state);

    sv_query(sv, SV_QUERY_LTCTIMECODE, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_LTCUSERBYTES, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    printf("Timecode LTC    : %-12s  %08x\n", buffer, tmp_userbytes);

    res = sv_query(sv, SV_QUERY_VITCTIMECODE, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_VITCUSERBYTES, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode VITC   : %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_DVITC_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_DVITC_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode DVITC  : %-12s  %08x\n", buffer, tmp_userbytes);
    }
    
    res = sv_query(sv, SV_QUERY_FILM_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_FILM_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode FILM   : %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_PROD_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_PROD_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode PROD   : %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_AFILM_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_AFILM_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode AFILM  : %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_APROD_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_APROD_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode APROD  : %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_DLTC_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_DLTC_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("Timecode DLTC   : %-12s  %08x\n", buffer, tmp_userbytes);
    }
    
    printf("\n");
    
    if(sv_query(sv, SV_QUERY_INTERLACEID_STORAGE, 0, &intid) != SV_OK) {
      intid = 0;
    }

    res = sv_query(sv, SV_QUERY_MODE_CURRENT, 0, &value);
    if(res == SV_OK) {
      if((jackinput != -1) && (jackoutput != -1)) {
        res = sv_jack_option_get(sv, jackoutput, SV_OPTION_VIDEOMODE, &mode[0]);
        if(res == SV_OK) {
          res = sv_jack_option_get(sv, jackinput, SV_OPTION_VIDEOMODE, &mode[1]);
        }
        if(res != SV_OK) {
          mode[1] = mode[0] = value;
        }
      } else {
        mode[1] = mode[0] = value;
      }
  
      jpeg_support_videomode2string(sv, buffer, mode[0]);
      printf("Video mode      : %s\n", buffer);
      if(mode[0] != mode[1]) {
        jpeg_support_videomode2string(sv, buffer, mode[1]);
        printf("Input mode      : %s\n", buffer);
      }
    }

    res = sv_option_get(sv, SV_OPTION_DVI_VIDEOMODE, &value);
    if(res == SV_OK) {
      if(value != -1) {
        res = jpeg_support_videomode2string(sv, buffer, value);
        if(res == SV_OK) {
          printf("DVI mode        : %s\n", buffer);
        }
      }
    }

    res = sv_option_get(sv, SV_OPTION_IOMODE, &mode[0]); mode[1] = mode[0];
    if(res == SV_OK) {
      if((jackinput != -1) && (jackoutput != -1)) {
        res = sv_jack_option_get(sv, jackoutput, SV_OPTION_IOMODE, &mode[0]);
        if(res == SV_OK) {
          res = sv_jack_option_get(sv, jackinput, SV_OPTION_IOMODE, &mode[1]);
        }
        if(res != SV_OK) {
          mode[1] = mode[0] = value;
        }
      } else {
        mode[1] = mode[0] = value;
      }
  
      sv_support_iomode2string(mode[0], buffer, sizeof(buffer));
      printf("Video iomode    : %s\n", buffer);
      if(mode[0] != mode[1]) {
        sv_support_iomode2string(mode[1], buffer, sizeof(buffer));
        printf("Input iomode    : %s\n", buffer);
      }
    }

    switch(info.nbit) {
    case 10:
      b10bit = "x10b";
      break;
    case 12:
      b10bit = "x12b";
      break;
    case 16:
      b10bit = "x16b";
      break;
    default:
      b10bit = "";
    }

    if((info.setup.storageysize != info.ysize) || (info.setup.storagexsize != info.xsize)) {
      printf("Video raster    : %dx%d%s (storage:%dx%d)\n", info.xsize, info.ysize, b10bit, info.setup.storagexsize, info.setup.storageysize);
    } else {
      printf("Video raster    : %dx%d%s\n", info.xsize, info.ysize, b10bit);
    }

    res = sv_query(sv, SV_QUERY_SYNCSTATE, -1, &syncstate);
    if(res != SV_OK) {
      syncstate = 0;
    }
    res = sv_option_get(sv, SV_OPTION_SYNCSELECT, &syncselect);
    if(res != SV_OK) {
      syncselect = 0;
    }
    sprintf(syncselect_string, "(Link-%c)", 'A' + syncselect);
    sv_support_syncmode2string(info.sync, buffer, sizeof(buffer));
    printf("Video sync      : %s %s %s\n", buffer, syncstate?"":"(Sync Missing)", syncselect?syncselect_string:"");

    printf("\n");

    res = sv_tc2asc(sv, info.master.timecode, buffer, sizeof(buffer));

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }

    printf("Master timecode : %s\n", buffer);

    sv_support_vtrinfo2string(buffer, sizeof(buffer), info.master.info);

    printf("Master flags    : %s\n", buffer);
    printf("Master error    : %s\n", sv_geterrortext(info.master.error));
  } else {
    jpeg_errorprint(sv, res);
  }
}

static void jpeg_info_timecode_print( sv_handle * sv, char* name, int tc, int ub )
{
  char buffer[256];
  sv_tc2asc(sv, tc, buffer, sizeof(buffer));
  printf("%s\tTC: \t%-12s UB: %8d\n",name, buffer, ub );
}

void jpeg_info_timecode(sv_handle * sv , char* choice)
{
  int res = SV_OK;
  sv_timecode_info timecodes;

  //Input
  if( !strncmp(choice,"input",strlen(choice)) )
  {
   res = sv_timecode_feedback( sv, &timecodes, 0 );
   if( res == SV_OK ) //Print input timecodes
   {
      printf("Input\n");
      printf("\n");
      printf("Tick: \t\t%d\n",timecodes.tick);
      printf("\n");
      printf("Vtr received: %d\n",timecodes.vtr_received );
      jpeg_info_timecode_print( sv, "Vtr",    timecodes.vtr_tc,      timecodes.vtr_ub );
      printf("\n");
      printf("ALtc received: %d\n",timecodes.altc_received );
      jpeg_info_timecode_print( sv, "ALtc",   timecodes.altc_tc,     timecodes.altc_ub );
      jpeg_info_timecode_print( sv, "AVitc",  timecodes.avitc_tc[0], timecodes.avitc_ub[0] );
      jpeg_info_timecode_print( sv, "AVitc*", timecodes.avitc_tc[1], timecodes.avitc_ub[1] );
      jpeg_info_timecode_print( sv, "AFilm",  timecodes.afilm_tc[0], timecodes.afilm_ub[0] );
      jpeg_info_timecode_print( sv, "AFilm*", timecodes.afilm_tc[1], timecodes.afilm_ub[1] );
      jpeg_info_timecode_print( sv, "AProd",  timecodes.aprod_tc[0], timecodes.aprod_ub[0] );
      jpeg_info_timecode_print( sv, "AProd*", timecodes.aprod_tc[1], timecodes.aprod_ub[1] );
      printf("\n");
      jpeg_info_timecode_print( sv, "DLtc",   timecodes.dltc_tc,     timecodes.dltc_ub );
      jpeg_info_timecode_print( sv, "DVitc",  timecodes.dvitc_tc[0], timecodes.dvitc_ub[0] );
      jpeg_info_timecode_print( sv, "DVitc*", timecodes.dvitc_tc[1], timecodes.dvitc_ub[1] );
      jpeg_info_timecode_print( sv, "DFilm",  timecodes.dfilm_tc[0], timecodes.dfilm_ub[0] );
      jpeg_info_timecode_print( sv, "DFilm*", timecodes.dfilm_tc[1], timecodes.dfilm_ub[1] );
      jpeg_info_timecode_print( sv, "DProd",  timecodes.dprod_tc[0], timecodes.dprod_ub[0] );
      jpeg_info_timecode_print( sv, "DProd*", timecodes.dprod_tc[1], timecodes.dprod_ub[1] );
    }
  }
  //Output
  else if( !strncmp(choice,"output",strlen(choice)) )
  {
    res = sv_timecode_feedback( sv, 0, &timecodes );
    if( res == SV_OK ) //Print input timecodes
    {
      printf("Output\n");
      printf("\n");
      printf("Tick: \t\t%d\n",timecodes.tick);
      printf("\n");
      jpeg_info_timecode_print( sv, "Vtr",    timecodes.vtr_tc,      timecodes.vtr_ub );
      printf("\n");
      jpeg_info_timecode_print( sv, "ALtc",   timecodes.altc_tc,     timecodes.altc_ub );
      jpeg_info_timecode_print( sv, "AVitc",  timecodes.avitc_tc[0], timecodes.avitc_ub[0] );
      jpeg_info_timecode_print( sv, "AVitc*", timecodes.avitc_tc[1], timecodes.avitc_ub[1] );
      jpeg_info_timecode_print( sv, "AFilm",  timecodes.afilm_tc[0], timecodes.afilm_ub[0] );
      jpeg_info_timecode_print( sv, "AFilm*", timecodes.afilm_tc[1], timecodes.afilm_ub[1] );
      jpeg_info_timecode_print( sv, "AProd",  timecodes.aprod_tc[0], timecodes.aprod_ub[0] );
      jpeg_info_timecode_print( sv, "AProd*", timecodes.aprod_tc[1], timecodes.aprod_ub[1] );
      printf("\n");
      jpeg_info_timecode_print( sv, "DLtc",   timecodes.dltc_tc,     timecodes.dltc_ub );
      jpeg_info_timecode_print( sv, "DVitc",  timecodes.dvitc_tc[0], timecodes.dvitc_ub[0] );
      jpeg_info_timecode_print( sv, "DVitc*", timecodes.dvitc_tc[1], timecodes.dvitc_ub[1] );
      jpeg_info_timecode_print( sv, "DFilm",  timecodes.dfilm_tc[0], timecodes.dfilm_ub[0] );
      jpeg_info_timecode_print( sv, "DFilm*", timecodes.dfilm_tc[1], timecodes.dfilm_ub[1] );
      jpeg_info_timecode_print( sv, "DProd",  timecodes.dprod_tc[0], timecodes.dprod_ub[0] );
      jpeg_info_timecode_print( sv, "DProd*", timecodes.dprod_tc[1], timecodes.dprod_ub[1] );
    }
  }
  //Help
  else
  {
    printf("Name:\n");
    printf("\tsv info timecode\n\n");
    printf("Synopsis:\n");
    printf("\tsv info timecode #\n");
    printf("\t\tinput\n");
    printf("\t\toutput\n\n");
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_info_closedcaption(sv_handle * sv)
{
  int ch;
  int res;

  do {
    res = sv_query(sv, SV_QUERY_CLOSEDCAPTION, -1, &ch);
    if(res == SV_ERROR_NOTAVAILABLE) {
      res = SV_OK;
      sv_usleep(sv, 10000);
    } else if(res == SV_OK) {
      if(ch & 0x7f) {
        printf("%c", ch & 0x7f); fflush(stdout);
      }
    }
  } while(res == SV_OK);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_stop(sv_handle * sv)
{
  int res = sv_stop(sv);

  if(res != SV_OK)
    jpeg_errorprint(sv, res);
}


void jpeg_memsetup(sv_handle * sv, int argc, char ** argv)
{
  sv_jack_memoryinfo info[16];
  sv_jack_memoryinfo * pinfo[16];
  int res = SV_OK;
  int jack;
  int njacks;
  int free = FALSE;

  memset(info, 0, sizeof(info));
  for(jack = 0; jack < 16; jack++) {
    pinfo[jack] = &info[jack];
  }

  if(argc) {
    if(!strcmp(argv[0], "help")) {
      printf("svram memsetup info\n");
      printf("svram memsetup free\n");
      printf("      This call removes all memory assignments.\n");
      printf("svram memsetup [percent bytes [percent bytes [...]]]\n");
      printf("      Each pair of percent and bytes configures one additional jack.\n");
    } else if(!strcmp(argv[0], "info")) {
      res = sv_jack_memorysetup(sv, TRUE, pinfo, 8, &njacks, 0);
      if(res == SV_OK) {
        for(jack = 0; jack < njacks; jack++) {
          if(info[jack].info.frames > 0) {
            printf("jack:%d - percent:%d bytes:%d (frames:%d videosize:%d audiosize:%d)\n", jack, info[jack].usage.percent, info[jack].usage.bytes, info[jack].info.frames, info[jack].info.videosize, info[jack].info.audiosize);
          }
        }
      }

      if(res != SV_OK) {
        jpeg_errorprint(sv, res);
      }
    } else if(!strcmp(argv[0], "free")) {
      free = TRUE;
    }
  }

  jack = 0;
  while(argc >= 2) {
    info[jack].usage.percent = atoi(argv[0]);
    info[jack].usage.bytes   = atoi(argv[1]);

    argv+=2; argc-=2;
    jack++;
  }

  if(jack || free) {
    res = sv_jack_memorysetup(sv, FALSE, pinfo, jack, NULL, 0);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_goto(sv_handle * sv, int argc, char ** argv)
{
  int res   = SV_OK;
  int mode  = SV_REPEAT_DEFAULT;
  int frame = 0;

  if(argc >= 1) {
    if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv goto help\n");
      printf("\t#framenumber\n");
      return;
    } else {
      frame = atoi(argv[0]);
    }

    if(argc >= 2) {
      if(strcmp(argv[1], "frame") == 0) {
        mode = SV_REPEAT_FRAME;
      } else if(strcmp(argv[1], "field1") == 0) {
        mode = SV_REPEAT_FIELD1;
      } else if(strcmp(argv[1], "field2") == 0) {
        mode = SV_REPEAT_FIELD2;
      } else if(strcmp(argv[1], "default") == 0) {
        mode = SV_REPEAT_DEFAULT;
      } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
        printf("sv goto %d help\n", frame);
        printf("\tfield1\n");
        printf("\tfield2\n");
        printf("\tframe\n");
        printf("\tdefault\n");
        return;
      } else {
        jpeg_errorprintf(sv, "sv goto: Unknown repeat command: %s\n", argv[1]);
        return;
      }
    }
  }

  if(res == SV_OK) {
    res = sv_position(sv, frame, 0, mode, 0);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_version(sv_handle * sv, int argc, char ** argv)
{
  int   res   = SV_OK;
  int   device, devicecount;
  int   module, modulecount;
  sv_version version;
  char tmp[256];
 
  if(argc >= 1) {
    if(!strcmp(argv[0], "check")) {
      res = sv_version_check(sv, DVS_VERSION_MAJOR, DVS_VERSION_MINOR, DVS_VERSION_PATCH, DVS_VERSION_FIX);
    } else if(!strcmp(argv[0], "verify")) {
      res = sv_version_verify(sv, 0, tmp, sizeof(tmp));
      printf("sv_version_verify %s - '%s'\n", sv_geterrortext(res), tmp);
      return;
    } else if(!strcmp(argv[0], "info")) {
      devicecount = 1;
      modulecount = 1;
      
      for(device = 0; (res == SV_OK) && (device < devicecount); device++) {
        for(module = 0; (res == SV_OK) && (module < modulecount); module++) {
          res = sv_version_status(sv, &version, sizeof(version), device, module, 0);
          if(res == SV_OK) {
            if(module == 0) {
              if(device == 0) {
                devicecount = version.devicecount;
              }
              modulecount = version.modulecount;
            } 
            //Driver
            if( strstr( version.module, "driver" ) ) {
              sprintf(tmp, "%d.%d.%d.%d", version.release.v.major, version.release.v.minor, version.release.v.patch, version.release.v.fix);
              printf ("Driver Version:\t\t%s\n", tmp);
            }  
            //Dvsoem
            if( strstr( version.module, "clib-oem" ) ) {
              sprintf(tmp, "%d.%d.%d.%d", version.release.v.major, version.release.v.minor, version.release.v.patch, version.release.v.fix);
              printf ("Dvsoem Version:\t\t%s\n", tmp);
            }
            //Hardware Version
            if( strstr( version.module, "hardware vers." ) ) {
              printf ("Hardware Version:\t%s\n", version.comment);
            }  
            //Firmware Version
            if( strstr( version.module, "firmware vers." ) ) {
              printf ("Firmware Version:\t%s\n", version.comment);
            }  
            //Serial
            if( strstr( version.module, "cardversion" ) ) {
              sprintf( tmp, strstr( version.comment, "serial:" ) );
              sprintf( tmp, strstr( tmp, ":" ) );
              printf ("Serialnumber:\t\t%s\n", tmp+1);
            }
            //PCI
            if( strstr( version.module, "epldversion" ) ) {
              sprintf( tmp, strstr( version.comment, "PCI" ) );
              sprintf( tmp, strstr( tmp, ":" ) );
              printf ("PCI Settings:\t\t%s\n", tmp+1);
            }  
          }
        } 
      }
      return;
    } else if(!strcmp(argv[0], "certify")) {
      int c_sw = 0, c_fw = 0, r_sw = 0, r_fw = 0;
      int result = 0;

      if(res==SV_OK) {
        res = sv_query( sv, SV_QUERY_VERSION_DRIVER, 0, &c_sw );
      }
      if(res==SV_OK) {
        res = sv_query( sv, SV_QUERY_HW_EPLDVERSION, 0, &c_fw );
      }
      if(res==SV_OK) {
        res = sv_version_certify(sv, argv[1], &r_sw, &r_fw, &result, 0 );
        if(res!=SV_OK) {
          printf("sv_version_certify %s\n", sv_geterrortext(res));
        }
      }
      if(res==SV_OK) {
        printf("Current  Versions: SDK %d.%d.%d.%d  FW %d.%d.%d.%d\n", ((c_sw&0x00FF0000)>>16),((c_sw&0x0000FF00)>>8),((c_sw&0x0000FF)>>0), ((c_sw&0xFF000000)>>24), ((c_fw&0x00FF0000)>>16),((c_fw&0x0000FF00)>>8),((c_fw&0x0000FF)>>0), ((c_fw&0xFF000000)>>24) );
        printf("Required Versions: SDK %d.%d.%d.%d  FW %d.%d.%d.%d\n", ((r_sw&0x00FF0000)>>16),((r_sw&0x0000FF00)>>8),((r_sw&0x0000FF)>>0), ((r_sw&0xFF000000)>>24), ((r_fw&0x00FF0000)>>16),((r_fw&0x0000FF00)>>8),((r_fw&0x0000FF)>>0), ((r_fw&0xFF000000)>>24) );
        printf("Certified: %s\n", sv_geterrortext(result) );
      }
      return;
    }

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    return;
  }

  sprintf(tmp, "%d.%d.%d.%d", DVS_VERSION_MAJOR, DVS_VERSION_MINOR, DVS_VERSION_PATCH, DVS_VERSION_FIX);
  printf("%-15s %10s\n", "distribution:", tmp);
  printf("%-15s %10s\n", "sv:", tmp); 

  devicecount = 1;
  modulecount = 1;

  for(device = 0; (res == SV_OK) && (device < devicecount); device++) {
    for(module = 0; (res == SV_OK) && (module < modulecount); module++) {
      res = sv_version_status(sv, &version, sizeof(version), device, module, 0);
      if(res == SV_OK) {
        if(module == 0) {
          if(device == 0) {
            devicecount = version.devicecount;
          }
          modulecount = version.modulecount;
        } 
        sprintf(tmp, "%d.%d.%d.%d", version.release.v.major, version.release.v.minor, version.release.v.patch, version.release.v.fix);
        strcat(version.module, ":");
        printf ("%-15s %10s ", version.module, tmp);

        if (version.flags & SV_VERSION_FLAG_BETA) {
          printf("beta (%d) ",version.rbeta);
        } 
        if(version.date.date || version.time.time) {
          printf("%04x/%02x/%02x %02x:%02x ",
            version.date.d.yyyy, version.date.d.mm, version.date.d.dd,
            version.time.t.hh, version.time.t.mm);
        }
        printf("%s\n", version.comment);
      }
    } 
  }
 
  if((res != SV_OK) && (res != SV_ERROR_PARAMETER)) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_inpoint(sv_handle * sv, int inpoint)
{
  int res = sv_inpoint(sv, inpoint);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_outpoint(sv_handle * sv, int outpoint)
{
  int res = sv_outpoint(sv, outpoint);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_showlicence(sv_handle * sv)
{
  int             res;
  int             i;
  unsigned int    keys[32];
  unsigned int    featurebits[8];
  int             type, version, serial, firmware, ram, disk, flags;
  char            *what;
  int             lictype = 0;
  int             expiry = 0;
  char            text[128];
  int             bitno;

  res = sv_getlicence(sv, &type, &version, &serial, &firmware, &ram, &disk, &flags, 9, keys);
  if(res == SV_ERROR_WRONG_HARDWARE) {
    res = sv_licenceinfo(sv, NULL, &serial, &expiry, (unsigned char *)&featurebits, sizeof(featurebits), (unsigned char*)&keys[0], sizeof(keys));
    lictype = 3;
  }
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }


  printf(" System type     : ");
  switch( type ) {
  case 10:
    printf("ProntoVideo");
    lictype = 0;
    break;
  case 20:
    printf("PCIStudio");
    lictype = 1;
    break;
  case 21:
    printf("ClipBoard");
    lictype = 1;
    break;
  case 22:
    printf("HDBoard");
    lictype = 1;
    break;
  case 23:
    printf("SDBoard");
    lictype = 1;
    break;
  case 24:
    printf("Clipster");
    lictype = 3;
    break;
  case 25:
    printf("Centaurus");
    if(!lictype) {
      lictype = 2;
    }
    break;
  case 26:
    printf("Hydra");
    lictype = 3;
    break;
  case 27:
    printf("Digilab");
    lictype = 3;
    break;
  case 28:
    printf("HydraX");
    lictype = 3;
    break;
  case 30:
    printf("ClipStation");
    lictype = 1;
    break;
  case 31:
    printf("ProntoServer");
    lictype = 1;
    break;
  case 32:
    printf("HDStationPRO");
    lictype = 1;
    break;
  case 33:
    printf("SDStationPRO");
    lictype = 1;
    break;
  case 34:
    printf("HDStationPlus");
    lictype = 2;
    break;
  case 42:
    printf("HDXWay");
    lictype = 1;
    break;
  case 43:
    printf("SDXWay");
    lictype = 1;
    break;
  case 50:
    printf("MovieVideo");
    lictype = 0;
    break;      
  default:
    printf("UNKNOWN (%d)",type);
  }
  printf("\n");
  printf(" Hardware serial : %8d\n",serial);
  if(expiry) {
    printf(" Valid until     : %08d\n", expiry);
  }
  sprintf(text,"%d.%d",(version>>8)&0xFF,version&0xFF);
  printf(" Hardware version: %8s\n",text);
  sprintf(text,"%d.%d.%d",(firmware>>8)&0xFF,firmware&0xFF,(firmware>>16)&0xFFFF);
  printf(" Firmware version: %8s\n",text);
  if(lictype < 3) {
    for(i = 0; i < 3; i++) {
      printf(" Licence key%d    : %08X %08X %08X\n",1+i, (int)keys[i*3+0], (int)keys[i*3+1], (int)keys[i*3+2]);
    }
  } else {
    for(i = 0; i < 2; i++) {
      printf(" Licence key%d    : %08X %08X %08X %08X\n",1+i, (int)keys[i*0x10+0x0], (int)keys[i*0x10+0x1], (int)keys[i*0x10+0x2], (int)keys[i*0x10+0x3]);
      printf("                 : %08X %08X %08X %08X\n",      (int)keys[i*0x10+0x4], (int)keys[i*0x10+0x5], (int)keys[i*0x10+0x6], (int)keys[i*0x10+0x7]);
      printf("                 : %08X %08X %08X %08X\n",      (int)keys[i*0x10+0x8], (int)keys[i*0x10+0x9], (int)keys[i*0x10+0xa], (int)keys[i*0x10+0xb]);
      printf("                 : %08X %08X %08X %08X\n",      (int)keys[i*0x10+0xc], (int)keys[i*0x10+0xd], (int)keys[i*0x10+0xe], (int)keys[i*0x10+0xf]);
    }
  }
  if((ram > 0) && (lictype <= 1)) {
    printf(" Licence option  : %d MB RAM limit\n", ram);
  }
  if((disk > 0) && (lictype <= 2)) {
    printf(" Licence option  : %d MB disk limit\n",disk);
  }
  switch(lictype) {
  case 3:
    for(bitno = 0; bitno < 32*8; bitno++) {
      if(featurebits[bitno / 32] & (1<<(bitno&31))) {
        char * txt = sv_licencebit2string(sv, bitno);
        if(txt) {
          printf("%d %s\n", bitno, txt);
        }
      }
    }
    break;
  case 2:
    if(ram & SV_OPSYS_WINDOWS) {
      printf(" Licenced opsys  : Windows\n");
    }
    if(ram & SV_OPSYS_LINUX) {
      printf(" Licenced opsys  : Linux\n");
    }
    if(ram & SV_OPSYS_SOLARIS) {
      printf(" Licenced opsys  : Solaris\n");
    }
    if(ram & SV_OPSYS_IRIX) {
      printf(" Licenced opsys  : Irix\n");
    }
    if(ram & SV_OPSYS_MACOS) {
      printf(" Licenced opsys  : MacOS\n");
    }

    for(i = 0; i < 32; i++) {
      switch(flags&(1<<i)) {
      case 0:
        what = NULL;
        break;
      case SV_LICENCED_AUDIO:
        what = "Audio";
        break;
      case SV_LICENCED_AUTOCONFORMING:
        what = "Autoconforming";
        break;
      case SV_LICENCED_CLIPSTER:
        what = "Clipster";
        break;
      case SV_LICENCED_COLORCORRECTOR:
        what = "ColorCorrector";
        break;
      case SV_LICENCED_COLORMANAGEMENT:
        what = "ColorManagement";
        break;
      case SV_LICENCED_CUSTOMRASTER:
        what = "CustomRaster";
        break;
      case SV_LICENCED_DISKRECORDER:
        what = "DiskRecorder";
        break;
      case SV_LICENCED_EVALUATION:
        what = "Evaluation";
        break;
      case SV_LICENCED_HDTV:
        what = "HDTV";
        break;
      case SV_LICENCED_HDTVDUALLINK:
        what = "HDTVDualLink";
        break;
      case SV_LICENCED_HDTVKEYCHANNEL:
        what = "HDTVKeyChannel";
        break;
      case SV_LICENCED_HIRES:
        what = "Hires";
        break;
      case SV_LICENCED_OEM:
        what = "OEM";
        break;
      case SV_LICENCED_MULTIDEVICE:
        what = "Multidevice";
        break;
      case SV_LICENCED_SDTV:
        what = "SDTV";
        break;
      case SV_LICENCED_SDTVDUALLINK:
        what = "SDTVDualLink";
        break;
      case SV_LICENCED_SDTVKEYCHANNEL:
        what = "SDTVKeyChannel";
        break;
      case SV_LICENCED_FILM2K:
        what = "FILM2K";
        break;
      case SV_LICENCED_FILM2KPLUS:
        what = "FILM2K+";
        break;
      case SV_LICENCED_FILM4K:
        what = "FILM4K";
        break;
      case SV_LICENCED_HSDL:
        what = "HSDL";
        break;
      case SV_LICENCED_HSDL4K:
        what = "HSDL4K";
        break;
      case SV_LICENCED_MIXER:
        what = "Mixer";
        break;
      case SV_LICENCED_PROCESSING:
        what = "Processing";
        break;
      case SV_LICENCED_ZOOMANDPAN:
        what = "ZoomAndPan";
        break;
      case SV_LICENCED_12BIT:
        what = "12Bit";
        break;
      case SV_LICENCED_SGI:
        what = "SGI";
        break;
      case SV_LICENCED_2K_1080PLAY:
        what = "2K_1080PLAY";
        break;
      case SV_LICENCED_DVI16:
        what = "DVI16";
        break;
      default:
        sprintf(text,"flag #%d (unknown!)",1+i);
        what = text;
      }
      if(what) {
        printf(" Licence option  : %s\n",what);
      }
    }
    break;
  case 1:
    for( i=0; i<32; ++i ) {
      switch(flags&(1<<i) ) {
      case  0:
      case  1:
        what = NULL;
        break;
      case SV_CSPLICENCE_MULTICHANNEL:
        what = "Multichannel";
        break;
      case SV_CSPLICENCE_AUDIO1:
        sprintf(text,"%d Channel Audio", SV_CSPLICENCE_AUDIO_GET(flags));
        what = text;
        break;
      case SV_CSPLICENCE_AUDIO2:
        if(flags & SV_CSPLICENCE_AUDIO1) {
          what = NULL;
        } else {
          sprintf(text,"%d Channel Audio", SV_CSPLICENCE_AUDIO_GET(flags));
          what = text;
        }
        break;
      case SV_CSPLICENCE_BETA:
        what = "Beta";
        break;
      case SV_CSPLICENCE_STREAMER:
        what = "Streamer";
        break;
      case SV_CSPLICENCE_CLIPMANAGEMENT:
        what = "Clipmanagement";
        break;
      case SV_CSPLICENCE_DUALLINK:
        what = "Duallink";
        break;
      case SV_CSPLICENCE_DISKRECORDER:
        what = "Disk Recorder";
        break;
      case SV_CSPLICENCE_FILM2K:
        what = "Film2k";
        break;
      case SV_CSPLICENCE_FILM2KPLUS:
        what = "Film2kplus";
        break;
      case SV_CSPLICENCE_HD360:
        what = "HD360";
        break;
      case SV_CSPLICENCE_HSDL:
        what = "HSDL";
        break;
      case SV_CSPLICENCE_IRIXOEM:
        what = "Irix OEM";
        break;
      case SV_CSPLICENCE_IRIXSGI:
        what = "Irix SGI";
        break;
      case SV_CSPLICENCE_KEYCHANNEL:
        what = "Key Channel";
        break;
      case SV_CSPLICENCE_LOUTHVDCP:
        what = "Louth Protocol";
        break;
      case SV_CSPLICENCE_MIXER:
        what = "Mixer";
        break;
      case SV_CSPLICENCE_MULTIDEVICE:
        what = "MultiDeviceMode";
        break;
      case SV_CSPLICENCE_ODETICS:
        what = "Odetics Protocol";
        break;
      case SV_CSPLICENCE_OSFS:
        what = "OSFS";
        break;
      case SV_CSPLICENCE_PULLDOWN:
        what = "Pulldown";
        break;
      case SV_CSPLICENCE_SDTV:
        what = "SDTV";
        break;
      case SV_CSPLICENCE_RGBSUPPORT:
        what = "RGB Support";
        break;
      case SV_CSPLICENCE_TILEMODE:
        what = "Tilemode";
        break;
      case SV_CSPLICENCE_CINECONTROL:
        what = "CineControl";
        break;
      default:
        sprintf(text,"flag #%d (unknown!)",1+i);
        what = text;
      }
      if(what) {
        printf(" Licence option  : %s\n",what);
      }
    }
    break;
  default:
    for( i=0; i<32; ++i ) {
          switch( flags&(1<<i) ) {
          case  0:
#if     0
                  if( (1<<i)&SV_LICENCE_FLAG_MAIN ) {
                          printf(" NOTE:  firmware licence has expired!\n");
                  }
#endif
                  continue;
          case SV_LICENCE_FLAG_MAIN:
                  continue;
          case SV_LICENCE_FLAG_VTR_MASTER:
                  what = "VTR master control";
                  break;
          case SV_LICENCE_FLAG_VTR_SLAVE:
                  what = "VTR slave mode";
                  break;
          case SV_LICENCE_FLAG_D5:
                  what = "D5 mode";
                  break;
          case SV_LICENCE_FLAG_JPEG:
                  what = "M-JPEG mode";
                  break;
          case SV_LICENCE_FLAG_ETHERNET:
                  what = "ethernet interface";
                  break;
          case SV_LICENCE_FLAG_AUDIO:
                  what = "digital audio interface";
                  break;
          case SV_LICENCE_FLAG_LTC:
                  what = "LTC interface";
                  break;
          case SV_LICENCE_FLAG_KEY:
                  what = "digital key interface";
                  break;
          case SV_LICENCE_FLAG_RASTER:
                  what = "custom video raster";
                  break;
          case SV_LICENCE_FLAG_10BIT:
                  what = "10 bit mode";
                  break;
          case SV_LICENCE_FLAG_ULTRA:
                  what = "Ultra SCSI host interface";
                  break;
          case SV_LICENCE_FLAG_RGB:
                  what = "RGB operation";
                  break;
          case SV_LICENCE_FLAG_SERVER:
                  what = "server operation";
                  break;
          case SV_LICENCE_FLAG_HOTKEY:
                  what = "onscreen menu hotkey";
                  break;
          case SV_LICENCE_FLAG_HOPPER:
                  what = "ethernet emulation";
                  break;
          case SV_LICENCE_FLAG_PULLDOWN:
                  what = "3:2 pulldown";
                  break;
          case SV_LICENCE_FLAG_CLIP:
                  what = "clip management";
                  break;
          default:
                  sprintf(text,"flag #%d (unknown!)",1+i);
                  what = text;
                  break;
          }
          printf(" Licence option  : %s\n",what);
    }
    if( ~flags&SV_LICENCE_FLAG_MAIN ) {
          printf(" NOTE:  firmware licence has expired!\n");
    }
  }
}


void jpeg_licence(sv_handle * sv, int argc, char ** argv)
{
  FILE * fp;
  char buffer[65536];
  int res = SV_OK;
  size_t count;
  
  if(argc > 1) {
    if(!strcmp(argv[1], "help") || !strcmp(argv[1], "?")) {
      printf("sv licence help\n");
      printf("\tkey1 - Set key1 for Centaurus II\n");
      printf("\tkey2 - Set key2 for Centaurus II\n");
      printf("\tkey3 - Set key3 for Centaurus II\n");
      printf("\tkeyfile1 - Set key1 from file for Atomix and Centaurus II\n");
      printf("\tkeyfile2 - Set key2 from file for Atomix and Centaurus II\n");
      return;
    }
  }

  if(argc > 2) {
    if(!strcmp(argv[1], "key1")) {
      res = sv_licence(sv, 1, argv[2]);
    } else if(!strcmp(argv[1], "key2")) {
      res = sv_licence(sv, 2, argv[2]);
    } else if(!strcmp(argv[1], "key3")) {
      res = sv_licence(sv, 3, argv[2]);
    } else if(!strcmp(argv[1], "keyfile1") || !strcmp(argv[1], "keyfile2")) {
      fp = fopen(argv[2], "r");
      if(fp) {
        count = fread(buffer, 1, sizeof(buffer), fp);
        fclose(fp);
        if(count < sizeof(buffer)) {
          if(count > 64) {
            if(!strcmp(argv[1], "keyfile1")) {
              res = sv_licence(sv, 4, buffer);
            } else {
              res = sv_licence(sv, 5, buffer);
            }
          } else {
            res = SV_ERROR_FILEREAD;
          }
        } else {
          printf("Licence file too big.\n");
          res = SV_ERROR_FILEREAD;
        }
      } else {
        printf("Could not open licence file '%s'\n", argv[2]);
        res = SV_ERROR_FILEOPEN;
      }
    } else {
      printf("sv %s %s - Unknown command\n", argv[0], argv[1]);
    }
  } else {
    printf("sv %s %s - Unknown command\n", argv[0], argv[1]);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_fastmode(sv_handle * sv, char * what)
{
  int res  = SV_OK;
  int mode = -1;

  if(strcmp(what, "default") == 0) {
    mode = SV_FASTMODE_DEFAULT; 
  } else if (strcmp(what, "odd_fields") == 0) {
    mode = SV_FASTMODE_ODDFIELDS;
  } else if (strcmp(what, "even_fields") == 0) {
    mode = SV_FASTMODE_EVENFIELDS;
  } else if (strcmp(what, "best_match") == 0) {
    mode = SV_FASTMODE_BESTMATCH;
  } else if (strcmp(what, "frame_rep_aba") == 0) {
    mode = SV_FASTMODE_FRAMEREP_ABA;
  } else if (strcmp(what, "frame_rep_abab") == 0) {
    mode = SV_FASTMODE_FRAMEREP_ABAB;
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_FASTMOTION, 0, &mode);
    if(res == SV_OK) {
      if(mode < arraysize(str_fastmode)) {
        printf("sv fastmode %s\n", str_fastmode[mode]);
      } else {
        printf("sv fastmode %d\n", mode);
      }
    }
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv fastmode help\n");
    printf("\tinfo\n");
    printf("\tdefault\n");
    printf("\todd_fields\n");
    printf("\teven_fields\n");
    printf("\tbest_match\n");
    printf("\tframe_rep_aba\n");
    printf("\tframe_rep_abab\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }
  
  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_FASTMODE, mode);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}
 
void jpeg_iospeed(sv_handle * sv, char * what)
{
  int res  = SV_OK;
  int mode = -1;

  if((strcmp(what, "1gb5") == 0) || (strcmp(what, "1.5GBit") == 0)) {
    mode = SV_IOSPEED_1GB5;
  } else if((strcmp(what, "3gba") == 0) || (strcmp(what, "3GBit/A") == 0)) {
    mode = SV_IOSPEED_3GBA;
  } else if((strcmp(what, "3gbb") == 0) || (strcmp(what, "3GBit/B") == 0)) {
    mode = SV_IOSPEED_3GBB;
  } else if((strcmp(what, "sdtv") == 0) || (strcmp(what, "SDTV") == 0)) {
    mode = SV_IOSPEED_SDTV;
  } else if (strcmp(what, "info") == 0) {
    res = sv_option_get(sv, SV_OPTION_IOSPEED, &mode);
    if(res == SV_OK) {
      printf("sv iospeed %s\n", sv_option_value2string(sv, SV_OPTION_IOSPEED, mode));
      return;
    }
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv iospeed help\n");
    printf("\tinfo\n");
    printf("\t1gb5 / 1.5GBit\n");
    printf("\t3gba / 3GBit/A\n");
    printf("\t3gbb / 3GBit/B\n");
    printf("\tsdtv / SDTV\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }
  
  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_IOSPEED, mode);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_iomode(sv_handle * sv, char * what, char * param2)
{
  char buffer[256];
  int res = SV_OK;
  int mode;
  int tmp;

  if(!strcmp(what, "info")) {
    res = sv_query(sv, SV_QUERY_IOMODE, 0, &mode);
    if(res == SV_OK) {
      sv_support_iomode2string(mode, buffer, sizeof(buffer));
      printf("sv iomode %s\n", buffer);
    }
    return;
  } else if(!strcmp(what, "auto")) {
    sv_option(sv, SV_OPTION_IOMODE_AUTODETECT, TRUE);
    return;
  } else if(!strcmp(what, "?") || !strcmp(what, "help")) {
    printf("sv iomode help\n");
    printf("\tYUV422    Sets IO mode to YUV422 10 bit\n");
    printf("\tYUV422A   Sets IO mode to YUV422A 10 bit\n");
    printf("\tYUV444    Sets IO mode to YUV444 10 bit\n");
    printf("\tYUV444A   Sets IO mode to YUV444A 10 bit\n");
    printf("\tRGB       Sets IO mode to RGB 10 bit\n");
    printf("\tRGBA      Sets IO mode to RGBA 10 bit\n");
    printf("\tRGB/8     Sets IO mode to RGB 8 bit\n");
    printf("\tRGBA/8    Sets IO mode to RGBA 8 bit\n");
    printf("\tYUV422/12 Sets IO mode to YUV422 12 bit\n");
    printf("\tYUV444/12 Sets IO mode to YUV444 12 bit\n");
    printf("\tRGB/12    Sets IO mode to RGB 12 bit\n");
    printf("\tinfo      Shows current settings\n");
    return;
  } else {
    mode = sv_support_string2iomode(what, 0);
    if(param2) {
      tmp = sv_support_string2iomode(param2, 0);
      if(tmp != -1) {
        mode |= SV_IOMODE_MODE2OUTPUT(tmp) | SV_IOMODE_RANGE2OUTPUT(tmp);
      } else {
        mode = -1;
      }
    }
    if(mode != -1) {
      res = sv_option(sv, SV_OPTION_IOMODE, mode);
    } else {
      res = SV_ERROR_PARAMETER;
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}
 
void jpeg_audiomode(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int mode = -1;
  int tmp;

  if(strcmp(what, "always") == 0) {
    mode = SV_AUDIOMODE_ALWAYS;
  } else if (strcmp(what, "on_speed1") == 0) {
    mode = SV_AUDIOMODE_ON_SPEED1;
  } else if (strcmp(what, "on_motion") == 0) {
    mode = SV_AUDIOMODE_ON_MOTION;
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_AUDIOMODE, 0, &tmp);
    if(res == SV_OK) {
      if(tmp < arraysize(str_audiomode)) {
        printf("sv audiomode %s\n", str_audiomode[tmp]);
      } else {
        printf("sv audiomode %d\n", tmp);
      }
    }
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv audiomode help\n");
    printf("\tinfo\t\tShow current settings\n");
    printf("\ton_speed1\tAudio only at speed 1.0\n");
    printf("\ton_motion\tAudio on speed != 0.0\n");
    printf("\talways\t\tAudio is always sent out, also at speed 0.0\n");
  } else {
    res = SV_ERROR_PARAMETER;
  }  

  if(mode != -1) {
    res = sv_option(sv, SV_OPTION_AUDIOMODE, mode);
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}    
 

void jpeg_audioanalogout(sv_handle * sv, int argc, char ** argv)
{
  int res    = SV_OK;
  int values = 0;
  int valuem = 0;
  int value;
  int i,j;

  if(argc > 0) {
    if(!strcmp(argv[0], "info")) {
      res = sv_option_get(sv, SV_OPTION_AUDIOANALOGOUT, &value);
      if(res == SV_OK) {
        printf("sv audioanalogout %x%x %x%x", (value & 0xf), (value & 0xf0)>>4, (value & 0xf00)>>8, (value & 0xf000)>>12);
      }
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv audioanalogout help\n");
      printf("sv audioanalogout 12\tSet analog out to mono channel 12\n");
      printf("sv audioanalogout 2\tSet analog out to stereo pair channel 2\n");
      printf("\tinfo\tShow current setting\n");
      return;   
    } else {
      for(value = j = 0; (res == SV_OK) && (j < 2); j++) {
        if(argc > j) {
          values = -1;
          valuem =  0;
          for(i = 0; i < 2; i++) {
            switch(argv[j][i]) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
              valuem |= (argv[j][i] - '0') << (4*i);
              values  = (argv[j][i] - '0') * 0x22 + 0x10;
              break;
              /* FALLTHROUGH */
            case '8':
            case '9':
              valuem |= (argv[j][i] - '0') << (4*i);
              values  = -1;
              break;
            case 'a':
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'f':
              valuem |= (argv[j][i] - 'a' + 10) << (4*i);
              values  = -1;
              break;
            case 'A':
            case 'B':
            case 'C':
            case 'D':
            case 'E':
            case 'F':
              valuem |= (argv[j][i] - 'A' + 10) << (4*i);
              values  = -1;
              break;
            case 0:
            case ' ':
              valuem = values;
              break;
            default:
              res = SV_ERROR_PARAMETER;
            } 
            if(valuem == -1) {
              res = SV_ERROR_PARAMETER;
            }
          }
        }
        value |= (valuem << (8*j));
      }
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_AUDIOANALOGOUT, value);
      }
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_proxy(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int tmp = 0;
  int i;

  if(argc > 0) {
    if(!strcmp(argv[0], "info")) {
      res = sv_option_get(sv, SV_OPTION_PROXY_VIDEOMODE, &tmp);
      if(res == SV_OK) { 
        if((tmp >= 0) && (tmp < arraysize(str_proxyvideomode))) {
          printf("sv proxy mode %s\n", str_proxyvideomode[tmp]);
        } else {
          printf("sv proxy mode ?\n");
        }
      } 
      res = sv_option_get(sv, SV_OPTION_PROXY_SYNCMODE, &tmp);
      if(res == SV_OK) { 
        if((tmp >= 0) && (tmp < arraysize(str_proxysyncmode))) {
          printf("sv proxy sync %s\n", str_proxysyncmode[tmp]);
        } else {
          printf("sv proxy sync ?\n");
        }
      } 
      res = sv_option_get(sv, SV_OPTION_PROXY_ASPECTRATIO, &tmp);
      if(res == SV_OK) {
        printf("sv proxy aspectratio %d.%d\n", tmp>>16, ((tmp&0xffff)*10000+0x8000)>>16);
      } 
      res = sv_option_get(sv, SV_OPTION_PROXY_TIMECODE, &tmp);
      if(res == SV_OK) {
        printf("sv proxy timecode %d\n", tmp);
      }
      res = sv_option_get(sv, SV_OPTION_PROXY_OUTPUT, &tmp);
      if(res == SV_OK) {
        if((tmp >= 0) && (tmp < arraysize(str_proxyoutput))) {
          printf("sv proxy output %s\n", str_proxyoutput[tmp]);
        } else {
          printf("sv proxy output %d\n", tmp);
        }
      }
      res = sv_option_get(sv, SV_OPTION_PROXY_OPTIONS, &tmp);
      if(res == SV_OK) {
        printf("sv proxy option %s%s%s%s%s\n", 
          (tmp == 0)?"none":"", 
          (tmp & SV_PROXY_OPTION_NTSCJAPAN)?"ntscjapan ":"", 
          (tmp & SV_PROXY_OPTION_SDTVFULL)?"sdtvfull ":"",
          (tmp & SV_PROXY_OPTION_FREEZEFIELD)?"freezefield ":"",
          (tmp & SV_PROXY_OPTION_DESKTOPONLY)?"desktoponly ":"");
      }
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv proxy help\n");
      printf("sv proxy mode {PAL,NTSC}\tSet proxy videmode\n");
      printf("sv proxy sync {auto,internal,genlock}\n");
      printf("sv proxy option [ntscjapan,sdtvfull,none]\n");
      printf("sv proxy output [underscan,letterbox,cropped,anamorph]\n");
      printf("sv proxy aspectratio #.#\n");
      printf("sv proxy info\n");
      return;   
    } else if(!strcmp(argv[0], "aspectratio") && (argc > 1)) {
      tmp = (int) (0x10000 * atof(argv[1]));
      res = sv_option_set(sv, SV_OPTION_PROXY_ASPECTRATIO, tmp);
    } else if(!strcmp(argv[0], "option") ) {
      for(tmp = 0, i = 1; i < argc; i++) {
        if(!strcmp(argv[i], "ntscjapan")) {
          tmp |= SV_PROXY_OPTION_NTSCJAPAN;
        } else if(!strcmp(argv[i], "sdtvfull")) {
          tmp |= SV_PROXY_OPTION_SDTVFULL;
        } else if(!strcmp(argv[i], "freezefield")) {
          tmp |= SV_PROXY_OPTION_FREEZEFIELD;
        } else if(!strcmp(argv[i], "desktoponly")) {
          tmp |= SV_PROXY_OPTION_DESKTOPONLY;
        } else if(strcmp(argv[i], "none")) {
          res = SV_ERROR_PARAMETER;
        }
      }
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_PROXY_OPTIONS, tmp);
      }
    } else if(!strcmp(argv[0], "mode") && (argc > 1)) {
      res = sv_option(sv, SV_OPTION_PROXY_VIDEOMODE, sv_support_string2videomode(argv[1], 0));
    } else if(!strcmp(argv[0], "timecode") && (argc > 1)) {
      res = sv_option(sv, SV_OPTION_PROXY_TIMECODE, atoi(argv[1]));
    } else if(!strcmp(argv[0], "sync") && (argc > 1)) {
      if(!strcmp(argv[1], "auto")) {
        tmp = SV_PROXY_SYNC_AUTO;
      } else if(!strcmp(argv[1], "internal")) {
        tmp = SV_PROXY_SYNC_INTERNAL;
      } else if(!strcmp(argv[1], "genlock")) {
        tmp = SV_PROXY_SYNC_GENLOCKED;
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_PROXY_SYNCMODE, tmp);
      }
    } else if(!strcmp(argv[0], "output") && (argc > 1)) {
      if(!strcmp(argv[1], "underscan")) {
        tmp = SV_PROXY_OUTPUT_UNDERSCAN;
      } else if(!strcmp(argv[1], "letterbox")) {
        tmp = SV_PROXY_OUTPUT_LETTERBOX;
      } else if(!strcmp(argv[1], "cropped")) {
        tmp = SV_PROXY_OUTPUT_CROPPED;
      } else if(!strcmp(argv[1], "anamorph")) {
        tmp = SV_PROXY_OUTPUT_ANAMORPH;
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_PROXY_OUTPUT, tmp);
      }
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}
 

 


void jpeg_audio_speed_compensation(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int value;
  
  if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv audio_speed_compensation #speed\tSets the speed\n");
    printf("                     speed >= 1.0 and speed <= 4.0\n");
    printf("               info\tShow current speed compensation\n");
    return;
  }

  if(!strcmp(what, "info")) {
    res = sv_query(sv, SV_QUERY_AUDIO_SPEED_COMPENSATION, 0, &value);
    if(res == SV_OK) {
      printf("sv audio_speed_compensation %2f\n", ((double)value / (double)0x10000));
    }
  } else {
    value = (int) (0x10000 * atof(what));

    res = sv_option(sv, SV_OPTION_AUDIO_SPEED_COMPENSATION, value);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_audioinput(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int mode;

  if(!strcmp(what, "aes") || !strcmp(what, "aesebu")) {
    res = sv_option(sv, SV_OPTION_AUDIOINPUT, SV_AUDIOINPUT_AESEBU);
  } else if(!strcmp(what, "aiv")) {
    res = sv_option(sv, SV_OPTION_AUDIOINPUT, SV_AUDIOINPUT_AIV);
  } else if(!strcmp(what, "info")) {
    res = sv_query(sv, SV_QUERY_AUDIOINPUT, 0, &mode);
    if(res == SV_OK) {
      if(mode < arraysize(str_audioinput)) {
        printf("sv audioinput %s\n", str_audioinput[mode]);
      } else {
        printf("sv audioinput %d\n", mode);
      }
    }
  } else if(!strcmp(what, "?") || !strcmp(what, "help")) {
    printf("sv audioinput aes\n");
    printf("              aiv\n");
    printf("              info\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_audiomaxaiv(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int count = 0;

  if(!strcmp(what, "0")) {
    res = sv_option_set(sv, SV_OPTION_AUDIOMAXAIV, 0);
  } else if(!strcmp(what, "4")) {
    res = sv_option_set(sv, SV_OPTION_AUDIOMAXAIV, 4);
  } else if(!strcmp(what, "8")) {
    res = sv_option_set(sv, SV_OPTION_AUDIOMAXAIV, 8);
  } else if(!strcmp(what, "12")) {
    res = sv_option_set(sv, SV_OPTION_AUDIOMAXAIV, 12);
  } else if(!strcmp(what, "16")) {
    res = sv_option_set(sv, SV_OPTION_AUDIOMAXAIV, 16);
  } else if(!strcmp(what, "info")) {
    res = sv_option_get(sv, SV_OPTION_AUDIOMAXAIV, &count);
    if(res == SV_OK) {
      printf("sv audiomaxaiv %d\n", count);
    }
  } else if(!strcmp(what, "?") || !strcmp(what, "help")) {
    printf("sv audimaxaiv 0,4,8,12,16\n");
    printf("              info\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_audioaesrouting(sv_handle * sv, int argc, char ** argv)
{
  int channels_a = 0;
  int channels_b = 0;
  int value;
  int res = SV_OK;

  if(argc >= 1) {
    if(!strcmp(argv[0], "info")) {
      res = sv_option_get(sv, SV_OPTION_AUDIOAESROUTING, &value);
      if(res == SV_OK) {
        if(value == SV_AUDIOAESROUTING_DEFAULT) {
          printf("svram audioaesrouting default\n");
        } else if(value == SV_AUDIOAESROUTING_16_0) {
          printf("svram audioaesrouting 16 0\n");
        } else if(value == SV_AUDIOAESROUTING_8_8) {
          printf("svram audioaesrouting 8 8\n");
        } else if(value == SV_AUDIOAESROUTING_4_4) {
          printf("svram audioaesrouting 4 4\n");
        } else {
          printf("svram audioaesrouting (unknown)\n");
        }
      }
    } else if(!strcmp(argv[0], "?") || !strcmp(argv[0], "help")) {
      printf("svram audioaesrouting 16 0\n");
      printf("svram audioaesrouting 8 8\n");
      printf("svram audioaesrouting 4 4\n");
      printf("                      info\n");
    } else {
      channels_a = atoi(argv[0]);

      if(argc >= 2) {
        channels_b = atoi(argv[1]);
      }

      if((channels_a == 16) && (channels_b == 0)) {
        res = sv_option_set(sv, SV_OPTION_AUDIOAESROUTING, SV_AUDIOAESROUTING_16_0);
      } else if((channels_a == 8) && (channels_b == 8)) {
        res = sv_option_set(sv, SV_OPTION_AUDIOAESROUTING, SV_AUDIOAESROUTING_8_8);
      } else if((channels_a == 4) && (channels_b == 4)) {
        res = sv_option_set(sv, SV_OPTION_AUDIOAESROUTING, SV_AUDIOAESROUTING_4_4);
      } else {
        res = sv_option_set(sv, SV_OPTION_AUDIOAESROUTING, SV_AUDIOAESROUTING_DEFAULT);
      }
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_audiomute(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int mode;

  if(!strcmp(what, "off")) {
    res = sv_option(sv, SV_OPTION_AUDIOMUTE, 0);
  } else if(!strcmp(what, "on")) {
    res = sv_option(sv, SV_OPTION_AUDIOMUTE, 1);
  } else if(!strcmp(what, "info")) {
    res = sv_query(sv, SV_QUERY_AUDIOMUTE, 0, &mode);
    if(res == SV_OK) {
      printf("sv audiomute %s\n", mode?"on":"off");
    }
  } else if(!strcmp(what, "?") || !strcmp(what, "help")) {
    printf("sv audiomute on\t\tAudio output is silent\n");
    printf("             off\tAudio output is normal\n");
    printf("             info\tShow current settings\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_audiochannels(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int value;

  if((what[0] >= '0') && (what[0] <= '8')) {
    res = sv_option_set(sv, SV_OPTION_AUDIOCHANNELS, atoi(what));
  } else if(!strcmp(what, "info")) {
    res = sv_option_get(sv, SV_OPTION_AUDIOCHANNELS, &value);
    if(res == SV_OK) {
      printf("svram audiochannels %d\n", value);
    }
  } else if(!strcmp(what, "?") || !strcmp(what, "help")) {
    printf("svram audiochannels 8\t\tSet audio channels to 8 stereo channels\n");
    printf("                  info\tShow current settings\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

 
void jpeg_audiofrequency(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int value;

  if((what[0] >= '0') && (what[0] <= '9')) {
    res = sv_option_set(sv, SV_OPTION_AUDIOFREQ, atoi(what));
  } else if(!strcmp(what, "info")) {
    res = sv_option_get(sv, SV_OPTION_AUDIOFREQ, &value);
    if(res == SV_OK) {
      printf("sv audiofrequency %d\n", value);
    }
  } else if(!strcmp(what, "?") || !strcmp(what, "help")) {
    printf("sv audiofrequency 48000\t\tSet audio frequency to 48000 Hz\n");
    printf("                  96000\t\tSet audio frequency to 96000 Hz\n");
    printf("                  info\tShow current settings\n");
    return;
  } else {
    res = SV_ERROR_PARAMETER;
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_gpi(sv_handle * sv, int argc, char ** argv)
{
  int value = 0;
  int res = SV_OK;

  if(argc > 0) {
    if(!strcmp(argv[0], "set") && (argc > 1)) {
      if(argc > 1) {
        value = atoi(argv[1]);
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_OPTIONGPI);
      }
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_GPI, value);
      }
    } else if(!strcmp(argv[0], "inoutpoint")) {
      res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_INOUTPOINT);
    } else if(!strcmp(argv[0], "pulldown")) {
      res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_PULLDOWN);
    } else if(!strcmp(argv[0], "pulldownphase")) {
      res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_PULLDOWNPHASE);
    } else if(!strcmp(argv[0], "repeated")) {
      res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_REPEATED);
    } else if(!strcmp(argv[0], "spirit")) {
      res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_SPIRIT);
    } else if(!strcmp(argv[0], "info")) {
      res = sv_query(sv, SV_QUERY_GPI, -1, &value);
      if(res == SV_OK) { 
        printf("sv gpi %c%c\n", (value & 2)?'1':'0', (value & 1)?'1':'0');
        return;
      }
    } else if(!strcmp(argv[0], "default")) {
      res = sv_option(sv, SV_OPTION_GPIOUT, SV_GPIOUT_DEFAULT);
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv gpi help\n");
      printf("\tset {00,01,10,11}\tSet gpi output to static value\n");
      printf("\tdefault\tSet default mode, recorded gpi ddr/ram, fifoapi output fifo\n");
      printf("\tpulldown\tSet on pulldown phase A\n");
      printf("\tpulldownphase\tShow current pulldown phase {A-00/B-01/C-10/D-11}\n");
      printf("\tinfo\tShow current incomming gpi\n");
      return;   
    } else {
      res = SV_ERROR_PARAMETER;
    } 
  } else {
    res = SV_ERROR_PARAMETER;
  }
 
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_inputport(sv_handle * sv, char * what)
{
  int res, mode;

  if(strcmp(what, "sdi") == 0) {
    mode = SV_INPUTPORT_SDI;
  } else if (strcmp(what, "sdi2") == 0) {
    mode = SV_INPUTPORT_SDI2;
  } else if (strcmp(what, "sdi3") == 0) {
    mode = SV_INPUTPORT_SDI3;
  } else if (strcmp(what, "dvi") == 0) {
    mode = SV_INPUTPORT_DVI;
  } else if (strcmp(what, "analog") == 0) {
    mode = SV_INPUTPORT_ANALOG;
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_INPUTPORT, 0, &mode);
    if(res == SV_OK) {
      printf("sv inputport %s\n", sv_query_value2string(sv, SV_QUERY_INPUTPORT, mode));
    }
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv inputport help\n");
    printf("\tanalog\n");
    printf("\tdvi\n");
    printf("\tsdi\n");
    printf("\tsdi2\n");
    printf("\tsdi3\n");
    return;
  } else {
    jpeg_errorprint(sv, SV_ERROR_PARAMETER);
    return;
  }
  res = sv_option(sv, SV_OPTION_INPUTPORT, mode);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }  
}
 

void jpeg_outputport(sv_handle * sv, char * what)
{
  int res = SV_OK;
  int mode;

  if(strcmp(what, "default") == 0) {
    mode = SV_OUTPUTPORT_DEFAULT;
  } else if (strcmp(what, "mirror") == 0) {
    mode = SV_OUTPUTPORT_MIRROR;
  } else if (strcmp(what, "swapped") == 0) {
    mode = SV_OUTPUTPORT_SWAPPED;
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_OUTPUTPORT, 0, &mode);
    if(res == SV_OK) {
      printf("sv outputport %s\n", sv_query_value2string(sv, SV_QUERY_OUTPUTPORT, mode));
    }
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv outputport help\n");
    printf("\tdefault\n");
    printf("\tswapped\n");
    printf("\tmirror\n");
    return;
  } else {
    jpeg_errorprint(sv, SV_ERROR_PARAMETER);
    return;
  }

  res = sv_option(sv, SV_OPTION_OUTPUTPORT, mode);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }  
}


void jpeg_mainoutput(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int ok   = TRUE;
  int mode = SV_MAINOUTPUT_SDI;

  if(argc >= 1) {
    if(strcmp(argv[0], "sdi") == 0) {
      mode = SV_MAINOUTPUT_SDI;
    } else if (strcmp(argv[0], "dvi") == 0) {
      mode = SV_MAINOUTPUT_DVI;
    } else if (strcmp(argv[0], "info") == 0) {
      res = sv_option_get(sv, SV_OPTION_MAINOUTPUT, &mode);
      if(res == SV_OK) {
        printf("sv mainoutput %s", sv_option_value2string(sv, SV_OPTION_MAINOUTPUT, mode & SV_MAINOUTPUT_MASK));
        if(mode & SV_MAINOUTPUT_FLAG_MASK) {
          printf(" %d", mode & SV_MAINOUTPUT_FLAG_MASK);
        }
        printf("\n");
      }
      return;
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv mainoutput help\n");
      printf("\tsdi\n");
      printf("\tdvi\n");
      return;
    } else {
      ok = FALSE;
    }

    if(argc >= 2) {
      mode |= atoi(argv[1]) & SV_MAINOUTPUT_FLAG_MASK;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv mainoutput: Unknown command: %s\n", argv[0]);
    return;
  }

  res = sv_option(sv, SV_OPTION_MAINOUTPUT, mode);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }  
}

int jpeg_read_line(int path, char * buffer)
{
 int i,j;
 unsigned char tmp;

 i = 0;
 do {
   j = read(path,&tmp,1);
   if (i < 126) {
     buffer[i] = (char) tmp;
     i += j;
   }
 } while ((j>0)&&(tmp!=0x0a)&&(tmp!=0x8a));

 if(j>0)  {  buffer[i] = 0;  }
 else     {  buffer[i+1] = 0; }

 return i;
}


void jpeg_lut(sv_handle * sv, int argc, char ** argv)
{
  int res    = SV_OK;
  int pos   = 0;
  int tmp;
  int interpolate = FALSE;
  int path = -1;
  int lastvalue;
  int lut[4096];
  int mode = -1;
  int load = TRUE;
  char buffer[256];
  int i;
  int count = 0;
  int value;
  int lutid = 0;

  if(argc >= 1) {
    if(!strcmp(argv[0], "luma")) {
      mode = SV_LUT_LUMA;
    } else if(!strcmp(argv[0], "red")) {
      mode = SV_LUT_RED;
    } else if(!strcmp(argv[0], "green")) {
      mode = SV_LUT_GREEN;
    } else if(!strcmp(argv[0], "blue")) {
      mode = SV_LUT_BLUE;
    } else if(!strcmp(argv[0], "rgba")) {
      mode = SV_LUT_RGBA;
    } else if(!strcmp(argv[0], "disable")) {
      mode = SV_LUT_DISABLE;
      load = FALSE;
    } else if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("sv lut {luma,red,green,blue,rgba} [file.lut [lutid]]\n");
      printf("sv lut disable [lutid]\n");
      printf("\tFile format:\n");
      printf("\t#LUT\n");
      printf("\t0        value\n");
      printf("\t1        value\n");
      printf("\t2        value\n");
      printf("\t...\n");
      printf("\t1023     value\n");
      printf("\n");
      printf("\tFile format (interpolated):\n");
      printf("\t#LUT/I\n");
      printf("\t0        value\n");
      printf("\t1023     value\n");
      printf("\n");
      printf("\tIf /I is not used then the file must contain 1024 values.\n");
      printf("\tWith /I the values inbetween are interpolated.\n");
      return;
    } else {
      printf("sv lut: %s unknown command\n", argv[0]);
      return;
    }
  }
  argv++; argc--;

  if(load) {
    if(argc >= 1) {
      path = open(argv[0], O_RDONLY);
      if(path < 0) {
        res = SV_ERROR_FILEOPEN;
      }
    } else {
      res = SV_ERROR_PARAMETER;
    }


    if(res == SV_OK) {
      jpeg_read_line(path, buffer);
      if(strncmp(buffer, "#LUT", 4)) {
        fprintf(stderr, "lut file '%s' does not start with #LUT\n", argv[0]);
        res = SV_ERROR_PARAMETER;
      }
      for(i = 0; buffer[i]; i++) {
        if(buffer[i] == '/') {
          if(!strncmp(&buffer[i], "/I", 2)) { 
            interpolate = TRUE;
          }
        }
      }
    }

    if(res == SV_OK) {
      do {
        count = jpeg_read_line(path, buffer);
      } while(buffer[0] == '#');
    }

    if(res == SV_OK) {
      for(lastvalue = pos = 0; pos < 1024;) {
        if(count) {
          tmp = atoi(buffer);
          for(i = 0; count && buffer[i] && ((buffer[i] != ' ') && (buffer[i] != '\t')); i++);
          if(buffer[i]) {
            i++;
          }
          value = atoi(&buffer[i]);
          if((value < 0) || (value > 1023)) {
            jpeg_errorprintf(sv, "sv_lut: pos %d value %d out of range (0 .. 1023)\n", pos, value);
            return;
          }
        } else {
          tmp = 1023;
          value = 1023;
        }
        if(!interpolate) {
          if(tmp != pos) {
            jpeg_errorprintf(sv, "sv_lut: value %d expected got %d\n", pos, tmp);
            return;
          }
          lut[pos++] = value;
        } else {
          if(tmp < pos) {
            jpeg_errorprintf(sv, "sv_lut: value %d out of order expected at least %d\n", tmp, pos);
            return;
          }
          for( ; pos < tmp; pos++) {
            lut[pos] = lastvalue + pos * (value - lastvalue) / tmp;
          }
          lut[pos++] = value;
        }
        count = jpeg_read_line(path, buffer);
        lastvalue = value;
      }
    }

    argv++; argc--;
  }

  if(argc >= 1) {
    lutid = atoi(argv[0]);
    argv++; argc--;
  }

  if(res == SV_OK) {
    if(mode == SV_LUT_RGBA) {
      memcpy(&lut[  1024], &lut[0], 1024 * sizeof(int));
      memcpy(&lut[2*1024], &lut[0], 1024 * sizeof(int));
      memcpy(&lut[3*1024], &lut[0], 1024 * sizeof(int));
    }

    res = sv_lut(sv, mode, lut, lutid);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_debug(sv_handle * sv, int value)
{
  int res = sv_option(sv, SV_OPTION_DEBUG, value);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_debugvalue(sv_handle * sv, int value)
{
  int res = sv_option(sv, SV_OPTION_DEBUGVALUE, value);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_trace(sv_handle * sv, char * pvalue)
{
  int res = SV_OK;
  int value;
  
  if((pvalue[0] == '-') || ((pvalue[0] >= '0') && (pvalue[0] <= '9'))) {
    value = atoi(pvalue);
  } else if(!strcmp(pvalue, "info")) {
    res = sv_option_get(sv, SV_OPTION_TRACE, &value);
    if(res == SV_OK) {
      printf("sv trace %s%s%s%s%s%s%s%s\n", 
        (value == 0)?"0 ":"",
        (value & SV_TRACE_ERROR)?"error ":"",
        (value & SV_TRACE_SETUP)?"setup ":"",
        (value & SV_TRACE_STORAGE)?"storage ":"",
        (value & SV_TRACE_FIFOAPI)?"fifoapi ":"",
        (value & SV_TRACE_CAPTURE)?"capture ":"",
        (value & SV_TRACE_VTRCONTROL)?"vtrcontrol ":"",
        (value & SV_TRACE_TCCHECK)?"tccheck ":"");
    }
  } else if(!strcmp(pvalue, "help")) {
    printf("sv trace {#value,[error|setup|storage|fifoapi|capture|vtrcontrol|tccheck]}\n");
    return;
  } else {
    value = 0;
    if(!strstr(pvalue, "error")) {
      value |= SV_TRACE_ERROR; 
    }
    if(!strstr(pvalue, "setup")) {
      value |= SV_TRACE_SETUP; 
    }
    if(!strstr(pvalue, "storage")) {
      value |= SV_TRACE_STORAGE; 
    }
    if(!strstr(pvalue, "fifoapi")) {
      value |= SV_TRACE_FIFOAPI; 
    }
    if(!strstr(pvalue, "capture")) {
      value |= SV_TRACE_CAPTURE; 
    }
    if(!strstr(pvalue, "vtrcontrol")) {
      value |= SV_TRACE_VTRCONTROL; 
    }
    if(!strstr(pvalue, "tccheck")) {
      value |= SV_TRACE_TCCHECK; 
    }
    if(!strstr(pvalue, "all")) {
      value |= SV_TRACE_ALL; 
    }
  }

  if(res == SV_OK) {
    res = sv_option_set(sv, SV_OPTION_TRACE, value);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_syncout(sv_handle * sv, int argc, char ** argv)
{
  char buffer[256];
  int res;
  int mode;
  int tmp = 0;
  int tmp2 = 0;

  mode = sv_support_string2syncout(argc, argv);

  if(mode == -1) {
    if(!strcmp(argv[0], "delay") && (argc > 1)) {
      if(!strcmp(argv[1], "info")) {
        res = sv_query(sv, SV_QUERY_SYNCOUTDELAY, 0, &tmp);
        if(res == SV_OK) {
          res = sv_query(sv, SV_QUERY_SYNCOUTVDELAY, 0, &tmp2);
        }
        printf("sv syncout delay %d %d\n", tmp, tmp2);
      } else {
        tmp = atoi(argv[1]);
        res = sv_option(sv, SV_OPTION_SYNCOUTDELAY, tmp);
        if(argc > 2) {
          tmp = atoi(argv[2]);
          res = sv_option(sv, SV_OPTION_SYNCOUTVDELAY, tmp);
        }
      }
    } else if(!strcmp(argv[0], "info")) {
      res = sv_query(sv, SV_QUERY_SYNCOUT, 0, &mode);

      if(res == SV_OK) {
        sv_support_syncout2string(mode, buffer, sizeof(buffer));
        printf("sv syncout %s\n", buffer);
      }
    } else if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("sv syncout help\n");
      printf("sv syncout delay #delay\n");
      printf("sv syncout {mode=default [options]}\n");
      printf("\tdefault\tUse raster default sync output\n");
      printf("\toff\n");
      printf("\tbilevel\tUse Bilevel sync output\n");
      printf("\ttrilevel\tUse TriLevel sync output\n");
      printf("\thvttl\tUse H/V TTL sync output\n");
      printf("\tuser\n");
      printf("\toptions: ongreen\tSync on green\n");
      printf("\toptions: high\tHigher voltage output (4.0V instead of 0.3V)\n");
    } else {
      jpeg_errorprintf(sv, "sv syncout : Unknown syncout command: %s\n", argv[0]);	
    }
    return;
  }

  res = sv_sync_output(sv, mode);

  if(res != SV_OK) {
    sv_errorprint(sv, res);
  }
}


void jpeg_timecode(sv_handle * sv, int argc, char ** argv)
{
  int value;
  int res = SV_OK;

  if(argc >= 2) {
    if(!strcmp(argv[0], "dropframe")) {
      value = atoi(argv[1]);
      res = sv_option(sv, SV_OPTION_TIMECODE_DROPFRAME, value);
    } else if(!strcmp(argv[0], "offset")) {
      res = sv_asc2tc(sv, argv[1], &value);
      if(res == SV_OK) {
        res = sv_option(sv, SV_OPTION_TIMECODE_OFFSET, value);
      }
    } else if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("sv timecode dropframe 0\tdisable dropframe timecode\n");
      printf("sv timecode dropframe 1\tenable  dropframe timecode\n");
      printf("sv timecode offset #timecode\n");
      return;
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    sv_errorprint(sv, res);
  }
}


void jpeg_sync(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int tmp;
  int syncmode = -1;
  int syncselect = 0;
  int hdelay   = 0;
  int vdelay   = 0;

  syncmode = sv_support_string2syncmode(argv[0], (argc>1)?argv[1]:NULL);

  if(syncmode == -1) {
    if(!strcmp(argv[0], "hdelay") && (argc > 1)) {
      if(!strcmp(argv[1], "info")) {
        res = sv_query(sv, SV_QUERY_HDELAY, 0, &hdelay);
        if(res == SV_OK) {
          printf("sv sync hdelay %d\n", hdelay);
        }
      } else {
        hdelay = atoi(argv[1]);
        res = sv_option(sv, SV_OPTION_HDELAY, hdelay);
      }
    } else if((!strcmp(argv[0], "vdelay")) && (argc > 1)) {
      if(!strcmp(argv[1], "info")) {
        res = sv_query(sv, SV_QUERY_VDELAY, 0, &vdelay);
        if(res == SV_OK) {
          printf("sv sync vdelay %d\n", vdelay);
        }
      } else {
        vdelay = atoi(argv[1]);
        res = sv_option(sv, SV_OPTION_VDELAY, vdelay);
      }
    } else if(!strcmp(argv[0], "delay") && (argc > 1)) {
      if(!strcmp(argv[1], "info")) {
        res = sv_query(sv, SV_QUERY_HDELAY, 0, &hdelay);
        if(res == SV_OK) {
          res = sv_query(sv, SV_QUERY_VDELAY, 0, &vdelay);
        }
        if(res == SV_OK) {
          printf("sv sync delay %d %d\n", hdelay, vdelay);
        }
      } else {
        hdelay = atoi(argv[1]);
        if(argc > 2) {
          vdelay = atoi(argv[2]);
        } else {
          vdelay = 0;
        }
        res = sv_option(sv, SV_OPTION_HDELAY, hdelay);
        tmp = sv_option(sv, SV_OPTION_VDELAY, vdelay);
        if(res == SV_OK) {
          res = tmp;
        }
      }
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv sync help\n");
      printf("\tinternal\tFreerunning\n");
      printf("\texternal #\tSync to incoming SDI signal (on source #)\n");
      printf("\tbilevel  \t[{4.0v,0.3v}] (SD), sync to the reference input signal.\n");
      printf("\ttrilevel\t(HD), sync to the reference input signal.\n");
      printf("\tanalog\t\tEither trilevel (HD) or bilevel (SD)\n");
      printf("\thdelay #\tHorizontal sync delay\n");
      printf("\tvdelay #\tVertical sync delay\n");
      printf("\tdelay  # #\tH and V sync delay\n");
      printf("\tsdtv-bilevel\tSync to analog ntsc/pal sync in HD raster\n");
    } else {
      jpeg_errorprintf(sv, "sv sync : Unknown sync command: %s %s\n", argv[0], argv[1]?argv[1]:"");	
    }
  } else {
    if(argc > 1) {
      syncselect = atoi(argv[1]);
    }
    res = sv_option_set(sv, SV_OPTION_SYNCSELECT, syncselect);
    if(res == SV_ERROR_NOTIMPLEMENTED) {
      res = SV_OK;
    }
    if(res == SV_OK) {
      res = sv_sync(sv, syncmode);
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_croppingmode(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int option = 0;
  int value  = 0;
  

  if(!strcmp(argv[0], "default")) {
    option = SV_OPTION_CROPPINGMODE;
    value  = SV_CROPPINGMODE_DEFAULT;
  } else if(!strcmp(argv[0], "head")) {
    option = SV_OPTION_CROPPINGMODE;
    value  = SV_CROPPINGMODE_HEAD;
  } else if(!strcmp(argv[0], "full")) {
    option = SV_OPTION_CROPPINGMODE;
    value  = SV_CROPPINGMODE_FULL;
  } else if(!strcmp(argv[0], "info")) {
    res = sv_option_get(sv, SV_OPTION_CROPPINGMODE, &value);
    if(res == SV_OK) {
      printf("sv croppingmode %s\n", sv_option_value2string(sv, SV_OPTION_CROPPINGMODE, value));
    }
  } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
    printf("sv croppingmode help\n");
    printf("\tdefault\tSet according to iomode settings\n");
    printf("\thead\tSet to head range\n");
    printf("\tfull\tSet to full range\n");
  } else {
    jpeg_errorprintf(sv, "sv croppingmode : Unknown command: %s\n", argv[0]);	
  }

  if(res == SV_OK) {
    if(option) {
      res = sv_option_set(sv, option, value);
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

char * str_mixer_edge[] = {
  "hard",
  "ramp",
  "soft",
  "steps",
};

char * str_mixer_port[] = {
  "-",
  "black",
  "video",
  "input",
};

char * str_mixer_mode[] = {
  "disable",
  "bottomleft",
  "bottom2top",
  "bottomright",
  "left2right",
  "center",
  "right2left",
  "topleft",
  "top2bottom",
  "topright",
  "blend",
  "alpha",
  "curtainh",
  "curtainv",
  "curtainhopen",
  "curtainvopen",
  "linesh",
  "linesv",
  "fade",
  "stripesh",
  "stripesv",
  "stripeshswap",
  "stripesvswap",
  "timerstop",
  "toblack",
  "zoomandpan",
  "framenumber",
  "cropmarks",
  "rect",
  "overlay",
};

void jpeg_mixer(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int tmp;
  int port[3];
  int mode = SV_MIXER_MODE_DISABLE;
  int start;
  int end;
  int nframes;
  int steps;
  int width;
  int param;
  sv_mixer_info info;
  int nargpos;

  if(argc >= 1) {
    nframes = 0;
    start   = 0;
    end     = 0;
    if(!strcmp(argv[0], "edge") && (argc >= 2)) {
      width = 1;
      steps = 1;
      if(!strcmp(argv[1], "hard")) {
        mode = SV_MIXER_EDGE_HARD;
      } else if(!strcmp(argv[1], "soft") || !strcmp(argv[1], "cos")) {
        mode = SV_MIXER_EDGE_SOFT;
      } else if(!strcmp(argv[1], "ramp")) {
        mode = SV_MIXER_EDGE_RAMP;
      } else if(!strcmp(argv[1], "steps")) {
        mode = SV_MIXER_EDGE_STEPS;
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if((res == SV_OK) && (argc >= 3)) {
        if(mode == SV_MIXER_EDGE_STEPS) {
          steps = atoi(argv[2]);
          if(argc >= 3) {
            width = atoi(argv[3]);
          }
        } else {
          width = atoi(argv[2]);
        }
      }
      if(res == SV_OK) {
        res = sv_mixer_edge(sv, mode, steps, width, 0);
      }
    } else if(!strcmp(argv[0], "input") && (argc >= 3)) {
      memset(port, 0, sizeof(port));
      start = 1;
      while((res == SV_OK) && (argc > start + 1)) {
        if(!strcmp(argv[start], "porta")) {
          tmp = 0;
        } else if(!strcmp(argv[start], "portb")) {
          tmp = 1;
        } else if(!strcmp(argv[start], "portc")) {
          tmp = 2;
        } else {
          tmp = 0; // Compiler warning disable.
          res = SV_ERROR_PARAMETER;
        } 
        if(res == SV_OK) {
          if(       !strcmp(argv[start+1], "video")) {
            port[tmp] = SV_MIXER_INPUT_VIDEO;
          } else if(!strcmp(argv[start+1], "black")) {
            port[tmp] = SV_MIXER_INPUT_BLACK;
          } else if(!strcmp(argv[start+1], "input")) {
            port[tmp] = SV_MIXER_INPUT_INPUT;
          } else {
            res = SV_ERROR_PARAMETER;
          }
        } else {
          res = SV_ERROR_PARAMETER;
        } 
        start += 2;
      }
      if(res == SV_OK) {
        res = sv_mixer_input(sv, port[0], port[1], port[2], 0);
      }
    } else if(!strcmp(argv[0], "mode") && (argc >= 2)) {
      param = 1;
      nargpos = 2;
      if(!strcmp(argv[1], "disable")) {
        mode = SV_MIXER_MODE_DISABLE;
      } else if(!strcmp(argv[1], "blend")) {
        mode = SV_MIXER_MODE_BLEND;
      } else if(!strcmp(argv[1], "alpha")) {
        mode = SV_MIXER_MODE_ALPHA;
      } else if(!strcmp(argv[1], "curtainhopen")) {
        mode = SV_MIXER_MODE_CURTAINHOPEN;
      } else if(!strcmp(argv[1], "curtainh")) {
        mode = SV_MIXER_MODE_CURTAINH;
      } else if(!strcmp(argv[1], "curtainvopen")) {
        mode = SV_MIXER_MODE_CURTAINVOPEN;
      } else if(!strcmp(argv[1], "curtainv")) {
        mode = SV_MIXER_MODE_CURTAINV;
      } else if(!strcmp(argv[1], "1") || !strcmp(argv[1], "bottomleft") ) {
        mode = SV_MIXER_MODE_BOTTOMLEFT;
      } else if(!strcmp(argv[1], "2") || !strcmp(argv[1], "bottom2top")) {
        mode = SV_MIXER_MODE_BOTTOM2TOP;
      } else if(!strcmp(argv[1], "3") || !strcmp(argv[1], "bottomright")) {
        mode = SV_MIXER_MODE_BOTTOMRIGHT;
      } else if(!strcmp(argv[1], "4") || !strcmp(argv[1], "left2right")) {
        mode = SV_MIXER_MODE_LEFT2RIGHT;
      } else if(!strcmp(argv[1], "5") || !strcmp(argv[1], "center")) {
        mode = SV_MIXER_MODE_CENTER;
      } else if(!strcmp(argv[1], "6") || !strcmp(argv[1], "right2left")) {
        mode = SV_MIXER_MODE_RIGHT2LEFT;
      } else if(!strcmp(argv[1], "7") || !strcmp(argv[1], "topleft")) {
        mode = SV_MIXER_MODE_TOPLEFT;
      } else if(!strcmp(argv[1], "8") || !strcmp(argv[1], "top2bottom")) {
        mode = SV_MIXER_MODE_TOP2BOTTOM;
      } else if(!strcmp(argv[1], "9") || !strcmp(argv[1], "topright")) {
        mode = SV_MIXER_MODE_TOPRIGHT;
      } else if(!strcmp(argv[1], "linesh")) {
        mode = SV_MIXER_MODE_LINESH;
      } else if(!strcmp(argv[1], "linesv")) {
        mode = SV_MIXER_MODE_LINESV;
      } else if(!strcmp(argv[1], "fade")) {
        mode = SV_MIXER_MODE_FADE;
      } else if(!strcmp(argv[1], "stripesh")) {
        mode = SV_MIXER_MODE_STRIPESH;
        if(argc >= 3) {
          param = atoi(argv[2]);
          nargpos++;
        }
      } else if(!strcmp(argv[1], "stripesv")) {
        mode = SV_MIXER_MODE_STRIPESV;
        if(argc >= 3) {
          param = atoi(argv[2]);
          nargpos++;
        }
      } else if(!strcmp(argv[1], "stripeshswap")) {
        mode = SV_MIXER_MODE_STRIPESHSWAP;
        if(argc >= 3) {
          param = atoi(argv[2]);
          nargpos++;
        }
      } else if(!strcmp(argv[1], "stripesvswap")) {
        mode = SV_MIXER_MODE_STRIPESVSWAP;
        if(argc >= 3) {
          param = atoi(argv[2]);
          nargpos++;
        }
      } else if(!strcmp(argv[1], "timerstop")) {
        mode = SV_MIXER_MODE_TIMERSTOP;
      } else if(!strcmp(argv[1], "toblack")) {
        mode = SV_MIXER_MODE_TOBLACK;
      } else if(!strcmp(argv[1], "framenumber")) {
        mode = SV_MIXER_MODE_FRAMENUMBER;
        param = 0;
      } else if(!strcmp(argv[1], "timecode")) {
        mode = SV_MIXER_MODE_TIMECODE;
        param = 0;
      } else if(!strcmp(argv[1], "cropmarks")) {
        mode = SV_MIXER_MODE_CROPMARKS;
        param = 0;
        if(argc >= 3) {
          param = atoi(argv[2]);
          nargpos++;
        }
        if(argc >= 4) {
          param |= atoi(argv[3])<<16;
          nargpos++;
        }
      } else if(!strcmp(argv[1], "rect") || !strcmp(argv[1], "overlay")) {
        if(!strcmp(argv[1], "overlay")) {
          mode = SV_MIXER_MODE_OVERLAY;
        } else {
          mode = SV_MIXER_MODE_RECT;
        } 
        if(argc >= 6) {
          param = (atoi(argv[2]) << 16) | atoi(argv[3]);
          start = (atoi(argv[4]) << 16) | atoi(argv[5]);
          nargpos+=4;
        } else {
          res = SV_ERROR_PARAMETER;
        }
      } else {
        res = SV_ERROR_PARAMETER;
      } 
      if(argc > nargpos) {
        if(!strcmp(argv[nargpos], "time")) {
          if(argc > nargpos + 3) {
            start   = (int)(0x10000 * atof(argv[nargpos+1]));
            end     = (int)(0x10000 * atof(argv[nargpos+2]));
            nframes = atoi(argv[nargpos+3]);
          } else {
            res = SV_ERROR_PARAMETER;
          }
        } else {
          start   = (int)(0x10000 * atof(argv[nargpos]));
          end     = 0;
          nframes = 0;
        }
      } 

      if(res == SV_OK) {
        res = sv_mixer_mode(sv, mode, param, start, end, nframes, 0);
      }
    } else if(!strcmp(argv[0], "guiinfo") || !strcmp(argv[0], "info")) {
      info.size = sizeof(info);
      res = sv_mixer_status(sv, &info);
      if(res == SV_OK) {
        if((info.mode >= 0) && (info.mode < arraysize(str_mixer_mode))) {
          printf("MODE %s\n", str_mixer_mode[info.mode]);
        }
        printf("MODEPARAM %d\n", info.modeparam);
        printf("POSITION %f\n", (double)info.position/0x10000);
        printf("START %f\n", (double)info.start/0x10000);
        printf("END %f\n", (double)info.end/0x10000);
        printf("NFRAMES %d\n", info.nframes);
        if((info.porta >= 0) && (info.porta < arraysize(str_mixer_port))) {
          printf("PORTA %s\n", str_mixer_port[info.porta]);
        }
        if((info.portb >= 0) && (info.portb < arraysize(str_mixer_port))) {
          printf("PORTB %s\n", str_mixer_port[info.portb]);
        }
        if((info.portc >= 0) && (info.portc < arraysize(str_mixer_port))) {
          printf("PORTC %s\n", str_mixer_port[info.portc]);
        }
        if((info.edge >= 0) && (info.edge < arraysize(str_mixer_edge))) {
          printf("EDGE %s\n", str_mixer_edge[info.edge]);
        }
        printf("EDGEPARAM %d\n", info.edgeparam);
        printf("EDGEWIDTH %d\n", info.edgewidth);
        printf("XZOOM     %d\n", info.xzoom);
        printf("YZOOM     %d\n", info.yzoom);
        printf("XPANNING  %d\n", info.xpanning);
        printf("YPANNING  %d\n", info.ypanning);
        printf("ZOOMFLAGS %d\n", info.zoomflags);
      }
    } else if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("sv mixer help\n");
      printf("\tedge {hard,ramp,soft, step #nsteps} [#width=1]\n");
      printf("\tinput {{porta,portb,portc} {video,black,input}}*\n");
      printf("\tmode {mode, see below} [{#position=0, time #start #end #nframes}]\n");
      printf("\t  Range for position/start/end is 0.0 -> 1.0\n");
      printf("\tmode {linesh,linesv}\n");
      printf("\tmode {top2bottom,bottom2top,left2right,right2left}\n");
      printf("\tmode {bottomleft,bottomright,center,topleft,topright}\n");
      printf("\tmode {curtainh,curtainv,curtainhopen,curtainvopen}\n");
      printf("\tmode {stripesh,stripeshswap,stripesv,stripesvswap}\n");
      printf("\tmode {hslots,vslots} #nslots\n");
      printf("\tmode {alpha,blend}\n");
      printf("\tmode disable\n");
      printf("\tinfo\n");
      return;
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_zoom(sv_handle * sv, int argc, char ** argv)
{
  int res       = SV_OK;
  int xzoom     = 1;
  int yzoom     = 1;
  int xpanning  = 0;
  int ypanning  = 0;
  int zoomflags = 0;
  int bfloat    = FALSE;
  int i;


  if(argc >= 1) {
    for(i = 0; !bfloat && (i < argc); i++) {
      if(strchr(argv[i], '.')) {
        bfloat = TRUE;
        xzoom  = 0x10000;
        yzoom  = 0x10000;
      }
    }
    if((argv[0][0] >= '0') && (argv[0][0] <= '9')) {
      switch(argc) {
      case 5:
        if(!strcmp(argv[4], "progressive")) {
          zoomflags = SV_ZOOMFLAGS_PROGRESSIVE;
        } else if(!strcmp(argv[4], "interlaced")) {
          zoomflags = SV_ZOOMFLAGS_INTERLACED;
        } else if(!strcmp(argv[4], "default")) {
          zoomflags = 0;
        } else {
          res = SV_ERROR_PARAMETER;
        }
        /* FALLTHROUGH */
      case 4:
        if(bfloat) {
          ypanning = (int)(0x10000 * atof(argv[3]));
        } else {
          ypanning = atoi(argv[3]);
        }
        /* FALLTHROUGH */
      case 3:
        if(bfloat) {
          xpanning = (int)(0x10000 * atof(argv[2]));
        } else {
          xpanning = atoi(argv[2]);
        }
        /* FALLTHROUGH */
      case 2:
        if(bfloat) {
          yzoom = (int)(0x10000 * atof(argv[1]));
        } else {
          yzoom = atoi(argv[1]);
        }
        /* FALLTHROUGH */
      case 1:
        if(bfloat) {
          xzoom = (int)(0x10000 * atof(argv[0]));
        } else {
          xzoom = atoi(argv[0]);
        }
        /* FALLTHROUGH */
      case 0:
        if(bfloat) {
          zoomflags |= SV_ZOOMFLAGS_FIXEDFLOAT;
        }
        res = sv_zoom(sv, xzoom, yzoom, xpanning, ypanning, zoomflags);
        break;
      default:
        res = SV_ERROR_PARAMETER;
      }
    } else if(!strcmp(argv[0], "guiinfo") || !strcmp(argv[0], "info")) {
      sv_info info;
      res = sv_status(sv, &info);
      if (res == SV_OK) {
        printf("XSIZE     %d\n", info.xsize);
        printf("YSIZE     %d\n", info.ysize);
      }
      res = sv_query(sv, SV_QUERY_ZOOMFLAGS, 0, &zoomflags);
      if(zoomflags & SV_ZOOMFLAGS_FIXEDFLOAT) {
        bfloat = TRUE;
      }
      res = sv_query(sv, SV_QUERY_XZOOM, 0, &xzoom);
      if(res == SV_OK) {
        printf("XZOOM     %g\n", bfloat?((double)xzoom / 0x10000):xzoom);
      }
      res = sv_query(sv, SV_QUERY_YZOOM, 0, &yzoom);
      if(res == SV_OK) {
        printf("YZOOM     %g\n", bfloat?((double)yzoom / 0x10000):yzoom);
      }
      res = sv_query(sv, SV_QUERY_XPANNING, 0, &xpanning);
      if(res == SV_OK) {
        printf("XPANNING  %g\n", bfloat?((double)xpanning/ 0x10000):xpanning);
      }
      res = sv_query(sv, SV_QUERY_YPANNING, 0, &ypanning);
      if(res == SV_OK) {
        printf("YPANNING  %g\n", bfloat?((double)ypanning / 0x10000):ypanning);
      }
      res = sv_query(sv, SV_QUERY_ZOOMFLAGS, 0, &zoomflags);
      if(res == SV_OK) {
        printf("ZOOMFLAGS %s%s%s\n", 
          (zoomflags&SV_ZOOMFLAGS_PROGRESSIVE)?"progressive ":"", 
          (zoomflags&SV_ZOOMFLAGS_INTERLACED)?"interlaced ":"",
          (zoomflags&SV_ZOOMFLAGS_FIXEDFLOAT)?"fixedfloat ":"");
      }
    } else if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("sv zoom help\n");
      printf("\tzoom [xzoom=1 [yzoom=1 [xpos=0 [ypos=0 [flags={default,progressive,interlaced]]]]]\n");
      printf("\tinfo\n");
      return;
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}




void jpeg_repeat(sv_handle * sv, char * what)
{
  int mode = -1;
  int res;

  if(strcmp(what, "frame")== 0) {
    mode = SV_REPEAT_FRAME;
  } else if (strcmp(what, "field") == 0) {
    mode = SV_REPEAT_FIELD1;
  } else if (strcmp(what, "field1") == 0) {
    mode = SV_REPEAT_FIELD1;
  } else if (strcmp(what, "field2") == 0) {
    mode = SV_REPEAT_FIELD2;
  } else if (strcmp(what, "current") == 0) {
    mode = SV_REPEAT_CURRENT;
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_REPEATMODE, 0, &mode);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    switch(mode) {
    case SV_REPEAT_FRAME:
    case SV_REPEAT_FIELD1:
    case SV_REPEAT_FIELD2:
    case SV_REPEAT_CURRENT:
      printf("sv repeat %s\n", str_repeatmode[mode]);
      break;
    default:
      printf("sv repeat mode=%d", mode);
    } 
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv repeat help\n");
    printf("\tcurrent\n");
    printf("\tfield1\n");
    printf("\tfield2\n");
    printf("\tframe\n");
    printf("\tinfo\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv repeat: Unknown repeat command: %s\n", what);
    return;
  }

  if (mode != -1) {
    res = sv_option(sv, SV_OPTION_REPEAT, mode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}


void jpeg_dropmode(sv_handle * sv, char * what)
{
  int mode = -1;
  int res;

  if(strcmp(what, "repeat")== 0) {
    mode = SV_DROPMODE_REPEAT;
  } else if (strcmp(what, "black") == 0) {
    mode = SV_DROPMODE_BLACK;
  } else if (strcmp(what, "info") == 0) {
    res = sv_option_get(sv, SV_OPTION_DROPMODE, &mode);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    switch(mode) {
    case SV_DROPMODE_REPEAT:
      printf("sv dropmode repeat\n");
      break;
    case SV_DROPMODE_BLACK:
      printf("sv dropmode black\n");
      break;
    default:
      printf("sv dropmode mode=%d", mode);
    } 
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv dropmode help\n");
    printf("\trepeat\n");
    printf("\tblack\n");
    printf("\tinfo\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv dropmode: Unknown dropmode command: %s\n", what);
    return;
  }

  if (mode != -1) {
    res = sv_option(sv, SV_OPTION_DROPMODE, mode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}


void jpeg_slowmotion(sv_handle * sv, char * what)
{
  int mode = -1;
  int res;

  if(strcmp(what, "frame") == 0) {
    mode = SV_SLOWMOTION_FRAME;
  } else if(strcmp(what, "field1") == 0) {
    mode = SV_SLOWMOTION_FIELD1;
  } else if(strcmp(what, "field2") == 0) {
    mode = SV_SLOWMOTION_FIELD2;
  } else if(strcmp(what, "field") == 0) {
    mode = SV_SLOWMOTION_FIELD;
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_SLOWMOTION, 0, &mode);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    switch(mode) {
    case SV_SLOWMOTION_FRAME:
    case SV_SLOWMOTION_FIELD1:
    case SV_SLOWMOTION_FIELD2:
    case SV_SLOWMOTION_FIELD:
      printf("sv slowmotion %s\n", str_slowmotion[mode]);
      break;
    default:
      printf("sv slowmotion mode=%d", mode);
    } 
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv slowmotion help\n");
    printf("\tfield\n");
    printf("\tframe\n");
    printf("\tfield1\n");
    printf("\tfield2\n");
    printf("\tinfo\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv slowmotion: Unknown mode: %s\n", what);
    return;
  }

  if (mode != -1) {
    res = sv_option(sv, SV_OPTION_SLOWMOTION, mode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}




void jpeg_framesync(sv_handle * sv, char * what)
{
  int mode = -1;
  int res;

  if(strcmp(what, "on") == 0) {
    mode = SV_SLOWMOTION_FRAME;
  } else if(strcmp(what, "off") == 0) {
    mode = SV_SLOWMOTION_FIELD;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv framesync help\n");
    printf("\ton\n");
    printf("\toff\n");
  } else {
    jpeg_errorprintf(sv, "sv framesync: Unknown mode: %s\n", what);
  }


  if (mode != -1) {
    res = sv_option(sv, SV_OPTION_SLOWMOTION, mode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}


void jpeg_preset(sv_handle * sv, char * preset)
{
  char buffer[32];
  int  res = SV_OK;
  int  par = -1;
  int  tmp;
  
  if((!strcmp(preset, "help")) || (!strcmp(preset, "?"))) {
    printf("sv preset help\n");
    printf("\tall\tFull Preset\n");
    printf("\tinfo\tShow Current Preset\n");
    printf("\tVT12345678\tPartial Preset\n");
    printf("\t\tV\tVideo\n");
    printf("\t\tT\tTimecode\n");
    printf("\t\t12\tAudio channel pair 12\n");
    printf("\t\t34\tAudio channel pair 34\n");
    printf("\t\t56\tAudio channel pair 56\n");
    printf("\t\t78\tAudio channel pair 78\n");
    return;
  } else if(!strcmp(preset, "info")) {
    res = sv_query(sv, SV_QUERY_PRESET, 0, &tmp);
    if(res == SV_OK) {
      sv_support_preset2string(tmp, buffer, sizeof(buffer));
      printf("sv preset %s\n", buffer);
      return;
    }
  } else if(!strcmp(preset, "all")) {
    par = SV_PRESET_VIDEO | SV_PRESET_TIMECODE | SV_PRESET_AUDIOMASK;
  } else {
    par = sv_support_string2preset(preset);
    if(par == -1) {
      jpeg_errorprintf(sv, "sv preset: Illegal preset definition\n");
      return;
    }
  }

  if(res == SV_OK) {
    res = sv_preset(sv, par);
  } else {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_vitcline(sv_handle * sv, int argc, char ** argv)
{
  int mode = -1;
  int res;
  char * p;
  char buffer[256];

  if(!argc) {
    return;
  }

  if(strcmp(argv[0], "default") == 0) {
    mode = SV_VITCLINE_DEFAULT;
  } else if(strcmp(argv[0], "test") == 0) {
    mode = SV_VITCLINE_DEFAULT | SV_VITCLINE_VCOUNT;
  } else if(strcmp(argv[0], "disabled") == 0) {
    mode = SV_VITCLINE_DISABLED;
  } else if(isdigit((unsigned char)argv[0][0])) {
    mode = atoi(argv[0]);
    p = strchr(argv[0], '/');
    if(p) {
      mode |= SV_VITCLINE_DUPLICATE(atoi(p+1));
    }
  } else if(strcmp(argv[0], "info") == 0) {
    res = sv_query(sv, SV_QUERY_VITCLINE, 0, &mode);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    memset(buffer, 0, sizeof(buffer));
    strcat(buffer, "");
    if(mode & SV_VITCLINE_VCOUNT) {
      strcat(buffer, " vcount");
    }
    if(mode & SV_VITCLINE_ARP201) {
      strcat(buffer, " rp201");
    }
    if(mode & SV_VITCLINE_DYNAMIC) {
      strcat(buffer, " dynamic");
    }
    switch(mode & SV_VITCLINE_MASK) {
    case SV_VITCLINE_DISABLED:
      printf("sv vitcline disabled%s\n", buffer);
      break;
    case SV_VITCLINE_DEFAULT:
      printf("sv vitcline default%s\n", buffer);
      break;
    default:
      if(SV_VITCLINE_DUPLICATE_GET(mode)) {
        printf("sv vitcline %d/%d%s\n", mode & SV_VITCLINE_MASK, SV_VITCLINE_DUPLICATE_GET(mode), buffer);
      } else {
        printf("sv vitcline %d%s\n", mode & SV_VITCLINE_MASK, buffer);
      }
    } 
    return;
  } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
    printf("sv vitcline help\n");
    printf("\tdisabled\n");
    printf("\tdefault [rp201]\n");
    printf("\tinfo\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv vitcline: Unknown mode: %s\n", argv[0]);
    return;
  }

  if((argc >= 2) && (mode != -1)) {
    if(!strcmp(argv[1], "rp201")) {
      mode |= SV_VITCLINE_ARP201;
    }
    if(!strcmp(argv[1], "dynamic")) {
      mode |= SV_VITCLINE_DYNAMIC;
    }
  }

  if (mode != -1) {
    res = sv_option(sv, SV_OPTION_VITCLINE, mode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}


void jpeg_vitcreaderline(sv_handle * sv, char * what)
{
  int mode = -1;
  int res;

  if(strcmp(what, "default") == 0) {
    mode = SV_VITCLINE_DISABLED;
  } else if(isdigit((unsigned char)what[0])) {
    mode = atoi(what);
  } else if (strcmp(what, "info") == 0) {
    res = sv_query(sv, SV_QUERY_VITCREADERLINE, 0, &mode);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    if(!mode || ((mode & SV_VITCLINE_MASK) == SV_VITCLINE_DEFAULT)) {
      printf("sv vitcreaderline default\n");
    } else {
      printf("sv vitcreaderline %d\n", mode);
    }
    res = sv_query(sv, SV_QUERY_VITCINPUTLINE, 0, &mode);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
    if(!mode || ((mode & SV_VITCLINE_MASK) == SV_VITCLINE_DEFAULT)) {
      printf("   vitcinputline  notdetected\n");
    } else if(SV_VITCLINE_DUPLICATE_GET(mode)) {
      printf("   vitcinputline  %d/%d\n", mode & SV_VITCLINE_MASK, SV_VITCLINE_DUPLICATE_GET(mode));
    } else {
      printf("   vitcinputline  %d\n", mode & SV_VITCLINE_MASK);
    }
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv vitcreaderline help\n");
    printf("\tdefault\n");
    printf("\t#\tlinenumer\n");
    printf("\tinfo\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv vitcreaderline: Unknown mode: %s\n", what);
    return;
  }

  if (mode != -1) {
    res = sv_option(sv, SV_OPTION_VITCREADERLINE, mode);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}


int jpeg_dominance(sv_handle * sv, int argc, char ** argv)
{
  int res   = SV_OK;
  int error = FALSE;
  int value = 0;

  if(argc >= 1) {
    if(!strcmp(argv[0], "help")) {
      printf("sv dominance <value>\n");
      printf("\t0 - Field 1 is dominant (default).\n");
      printf("\t1 - Field 2 is dominant\n");
    } else if(!strcmp(argv[0], "info")) {
      res = sv_option_get(sv, SV_OPTION_FIELD_DOMINANCE, &value);
      if(res == SV_OK) {
        printf("sv dominance %d\n", value);
      }
    } else {
      switch(argv[0][0]) {
      case '0':
        value = 0;
        break; 
      case '1':
        value = 1;
        break;
      default:
        error = TRUE;
      }

      if(!error) {
        res = sv_option_set(sv, SV_OPTION_FIELD_DOMINANCE, value);
      }
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }

  if(error) {
    jpeg_errorprintf(sv, "sv dominance %s: Wrong parameter\n", argv[0]);
  }

  return !error;
}


int jpeg_rs422pinout(sv_handle * sv, int argc, char ** argv)
{
  char * pinout;
  char * task;
  int device;
  int iochannel;
  int error = FALSE;
  int command = SV_OPTION_NOP;
  int value = 0;
  int res = SV_OK;

  if(argc >= 1) {
    if(!strcmp(argv[0], "help")) {
      printf("sv rs422pinout help\n");
      printf("\tporta <pinout> <task> [iochannel]\n");
      printf("\tportb <pinout> <task> [iochannel]\n");
      printf("\tportc <pinout> <task> [iochannel]\n");
      printf("\tportd <pinout> <task> [iochannel]\n");
      printf("\t\t<pinout>     default,normal,swapped,master,slave\n");
      printf("\t\t<task>       default,none,master,slave\n");
      printf("\t\t[iochannel]  Multichannel pipeline (default 0)\n");
    } else if(!strcmp(argv[0], "info")) {
      printf("               %-5s %-7s %-6s %-9s\n", "port", "pinout", "task", "iochannel");
      for(device = 0; device < 4; device++) {
        if(device == 0) {
          res = sv_option_get(sv, SV_OPTION_RS422A, &value);
        } else if(device == 1) {
          res = sv_option_get(sv, SV_OPTION_RS422B, &value);
        } else if(device == 2) {
          res = sv_option_get(sv, SV_OPTION_RS422C, &value);
        } else if(device == 3) {
          res = sv_option_get(sv, SV_OPTION_RS422D, &value);
        }

        if(res == SV_OK) {
          switch(value & SV_RS422_PINOUT_MASK) {
          case SV_RS422_PINOUT_NORMAL:
            pinout = "normal";
            break;
          case SV_RS422_PINOUT_SWAPPED:
            pinout = "swapped";
            break;
          case SV_RS422_PINOUT_MASTER:
            pinout = "master";
            break;
          case SV_RS422_PINOUT_SLAVE:
            pinout = "slave";
            break;
          default:
            pinout = "?";
          }
          switch(value & SV_RS422_TASK_MASK) {
          case SV_RS422_TASK_NONE:
            task = "none";
            break;
          case SV_RS422_TASK_MASTER:
            task = "master";
            break;
          case SV_RS422_TASK_SLAVE:
            task = "slave";
            break;
          case SV_RS422_TASK_VDCPSLAVE:
            task = "vdcpslave";
            break;
          default:
            task = "?";
          }
          iochannel = SV_RS422_IOCHANNEL_GET(value);
          printf("sv rs422pinout port%c %-7s %-6s %-9d\n", device + 'a', pinout, task, iochannel);
        }
      }
    } else if(!strncmp(argv[0], "port", 4)) {
      if(argc >= 3) {
        if(!strcmp(argv[0], "porta")) {
          command = SV_OPTION_RS422A;
        } else if(!strcmp(argv[0], "portb")) {
          command = SV_OPTION_RS422B;
        } else if(!strcmp(argv[0], "portc")) {
          command = SV_OPTION_RS422C;
        } else if(!strcmp(argv[0], "portd")) {
          command = SV_OPTION_RS422D;
        } else {
          error = TRUE;
        }

        if(!strcmp(argv[1], "default")) {
          value |= SV_RS422_PINOUT_DEFAULT;
        } else if(!strcmp(argv[1], "normal")) {
          value |= SV_RS422_PINOUT_NORMAL;
        } else if(!strcmp(argv[1], "swapped")) {
          value |= SV_RS422_PINOUT_SWAPPED;
        } else if(!strcmp(argv[1], "master")) {
          value |= SV_RS422_PINOUT_MASTER;
        } else if(!strcmp(argv[1], "slave")) {
          value |= SV_RS422_PINOUT_SLAVE;
        } else {
          error = TRUE;
        }

        if(!strcmp(argv[2], "default")) {
          value |= SV_RS422_TASK_DEFAULT;
        } else if(!strcmp(argv[2], "none")) {
          value |= SV_RS422_TASK_NONE;
        } else if(!strcmp(argv[2], "master")) {
          value |= SV_RS422_TASK_MASTER;
        } else if(!strcmp(argv[2], "slave")) {
          value |= SV_RS422_TASK_SLAVE;
        } else {
          error = TRUE;
        }

        if(argc >= 4) {
          value |= SV_RS422_IOCHANNEL_SET(atoi(argv[3]));
        }

        if(!error) {
          res = sv_option_set(sv, command, value);
        }
        if(res == SV_ERROR_WRONG_HARDWARE) {
          jpeg_errorprintf(sv, "sv rs422pinout: Function not available on this hardware.\n");
        } else if(res != SV_OK) {
          jpeg_errorprint(sv, res);
        }
      } else {
        error = TRUE;
      }
    } else {
      error = TRUE;
    }
  }

  if(error) {
    jpeg_errorprintf(sv, "sv rs422pinout %s: Wrong parameters\n", argv[0]);
  }

  return !error;
}


int jpeg_master(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int cmd = SV_MASTER_NOP;
  int par = 0;
  char * raw = NULL;
  char * rawreply = NULL;
  int i;
  char rawbuffer[256];

  if(argc == 1) {
    if(strcmp(argv[0], "eject") == 0) {
      cmd = SV_MASTER_EJECT;
    } else if(strcmp(argv[0], "forward") == 0) {
      cmd = SV_MASTER_FORWARD;
    } else if(strcmp(argv[0], "live") == 0) {
      cmd = SV_MASTER_LIVE;
    } else if(strcmp(argv[0], "pause") == 0) {
      cmd = SV_MASTER_PAUSE;
    } else if(strcmp(argv[0], "play") == 0) {
      cmd = SV_MASTER_PLAY;
    } else if(strcmp(argv[0], "record") == 0) {
      cmd = SV_MASTER_RECORD;
    } else if(strcmp(argv[0], "rewind") == 0) {
      cmd = SV_MASTER_REWIND;
    } else if(strcmp(argv[0], "stboff") == 0) {
      cmd = SV_MASTER_STBOFF;
    } else if(strcmp(argv[0], "stop") == 0) {
      cmd = SV_MASTER_STOP;
    } else if((strcmp(argv[0], "help") == 0) || (strcmp(argv[0], "?") == 0)) {
      printf("sv master help\n");
      printf("\t\tdisoffset #frames\n");
      printf("\t\teditlag #frames\n");
      printf("\t\teject\n");
      printf("\t\tflags {autoedit,forcedropframe}\n");
      printf("\t\tforward\n");
      printf("\t\tgoto #timecode\n");
      printf("\t\tjog #speed\n");
      printf("\t\tlive\n");
      printf("\t\tmoveto #timecode\n");
      printf("\t\tpause\n");
      printf("\t\tplay\n");
      printf("\t\tpostroll #timecode\n");
      printf("\t\tpreroll #timecode\n");
      printf("\t\tpreset [VAD]123456789abcdef\n");
      printf("\t\traw #command {#data0 ...}\n");
      printf("\t\trawreply #command {#data0 ...} with reply\n");
      printf("\t\trecord\n");
      printf("\t\trecoffset #frames\n");
      printf("\t\trewind\n");
      printf("\t\tshuttle #speed\n");
      printf("\t\tstandby on/off\n");
      printf("\t\tstep #frames\n");
      printf("\t\tstop\n");
      printf("\t\ttimecode vitc/ltc/auto/timer1/timer2\n");
      printf("\t\ttolerance none/normal/large/rough\n");
      printf("\t\tvar #speed\n");
      return TRUE;
    }
  } else if(argc == 2) {
    if(strcmp(argv[0], "autopark") == 0) {
      if(strcmp(argv[1], "on") == 0) {
        cmd = SV_MASTER_AUTOPARK;
        par = SV_MASTER_AUTOPARK_ON;
      } else if(strcmp(argv[1], "off") == 0) {
        cmd = SV_MASTER_AUTOPARK;
        par = SV_MASTER_AUTOPARK_OFF;
      } else {
        jpeg_errorprintf(sv, "sv master autopark : Parameter Error : %s (valid: on/off)\n", argv[1]);
      }
    } else if(strcmp(argv[0], "disoffset") == 0) {
      cmd = SV_MASTER_DISOFFSET;
      par = atoi(argv[1]);
    } else if(strcmp(argv[0], "editlag") == 0) {
      cmd = SV_MASTER_EDITLAG;
      par = atoi(argv[1]);
    } else if(strcmp(argv[0], "forcedropframe") == 0) {
      if(strcmp(argv[1], "auto") == 0) {
        cmd = SV_MASTER_FORCEDROPFRAME;
        par = SV_MASTER_FORCEDROPFRAME_AUTO;
      } else if(strcmp(argv[1], "on") == 0) {
        cmd = SV_MASTER_FORCEDROPFRAME;
        par = SV_MASTER_FORCEDROPFRAME_ON;
      } else {
        jpeg_errorprintf(sv, "sv master forcedropframe : Parameter Error : %s (valid: auto/on)\n", argv[1]);
      }
    } else if(strcmp(argv[0], "goto") == 0) {
      if(sv_asc2tc(sv, argv[1], &par) != SV_OK) {
        jpeg_errorprintf(sv, "sv master %s : Timecode Error : %s (valid hh:mm:ss:ff)\n",  argv[0], argv[1]);
      } else {
        cmd = SV_MASTER_GOTO;
      }
    } else if(strcmp(argv[0], "jog") == 0) {
      double speed_double = atof(argv[1]);
      cmd = SV_MASTER_JOG;
      par = (int) (speed_double * 65536.0);
    } else if(strcmp(argv[0], "standby") == 0) {
      if(strcmp(argv[1], "on") == 0) {
        cmd = SV_MASTER_STANDBY;
        par = SV_MASTER_STANDBY_ON;
      } else if(strcmp(argv[1], "off") == 0) {
        cmd = SV_MASTER_STANDBY;
        par = SV_MASTER_STANDBY_OFF;
      } else {
        jpeg_errorprintf(sv, "sv master standby : Parameter Error : %s (valid: on/off)\n", argv[1]);
      }
    } else if(strcmp(argv[0], "shuttle") == 0) {
      double speed_double = atof(argv[1]);
      cmd = SV_MASTER_SHUTTLE;
      par = (int) (speed_double * 65536.0);
    } else if(strcmp(argv[0], "var") == 0) {
      double speed_double = atof(argv[1]);
      cmd = SV_MASTER_VAR;
      par = (int) (speed_double * 65536.0);
    } else if(strcmp(argv[0], "step") == 0) {
      cmd = SV_MASTER_STEP;
      par = atoi(argv[1]);
    } else if(strcmp(argv[0], "stepmoveto") == 0) {
      cmd = SV_MASTER_STEP_MOVETO;
      par = atoi(argv[1]);
    } else if(strcmp(argv[0], "stepgoto") == 0) {
      cmd = SV_MASTER_STEP_GOTO;
      par = atoi(argv[1]);
    } else if(strcmp(argv[0], "moveto") == 0) {
      if(sv_asc2tc(sv, argv[1], &par) != SV_OK) {
        jpeg_errorprintf(sv, "sv master %s : Timecode Error : %s (valid hh:mm:ss:ff)\n", argv[0], argv[1]);
      } else {
        cmd = SV_MASTER_MOVETO;
      }
    } else if(strcmp(argv[0], "parktime") == 0) {
      if(sv_asc2tc(sv, argv[1], &par) != SV_OK) {
        jpeg_errorprintf(sv, "sv master %s : Timecode Error : %s (valid hh:mm:ss:ff)\n", argv[0], argv[1]);
      } else {
        cmd = SV_MASTER_PARKTIME;
      }
    } else if(strcmp(argv[0], "postroll") == 0) {
      if(sv_asc2tc(sv, argv[1], &par) != SV_OK) {
        jpeg_errorprintf(sv, "sv master %s : Timecode Error : %s (valid hh:mm:ss:ff)\n", argv[0], argv[1]);
      } else {
        cmd = SV_MASTER_POSTROLL;
      }
    } else if(strcmp(argv[0], "preroll") == 0) {
      if(sv_asc2tc(sv, argv[1], &par) != SV_OK) {
        jpeg_errorprintf(sv, "sv master %s : Timecode Error : %s (valid hh:mm:ss:ff)\n", argv[0], argv[1]);
      } else {
        cmd = SV_MASTER_PREROLL;
      }
    } else if(strcmp(argv[0], "preset") == 0) {
      int digital = FALSE;
      cmd = SV_MASTER_PRESET;
      par = 0;
      for(i = 0; i < (int)strlen(argv[1]); i++) {
        switch (argv[1][i]) {
        case 'V':
        case 'v':
          par |= SV_MASTER_PRESET_VIDEO;
          break;
        case 'Q':
        case 'q':
          par |= SV_MASTER_PRESET_ASSEMBLE;
          break;
        case 'A':
          digital = FALSE;
          break;
        case 'D':
          digital = TRUE;
          break;
        case '1':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO1;
          } else {
            par |= SV_MASTER_PRESET_AUDIO1;
          }
          break;
        case '2':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO2;
          } else {
            par |= SV_MASTER_PRESET_AUDIO2;
          }
          break;
        case '3':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO3;
          } else {
            par |= SV_MASTER_PRESET_AUDIO3;
          }
          break;
        case '4':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO4;
          } else {
            par |= SV_MASTER_PRESET_AUDIO4;
          }
          break;
        case '5':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO5;
          }
          break;
        case '6':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO6;
          }
          break;
        case '7':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO7;
          }
          break;
        case '8':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO8;
          }
          break;
        case '9':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO9;
          }
          break;
        case 'a':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO10;
          }
          break;
        case 'b':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO11;
          }
          break;
        case 'c':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO12;
          }
          break;
        case 'd':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO13;
          }
          break;
        case 'e':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO14;
          }
          break;
        case 'f':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO15;
          }
          break;
        case 'g':
          if(digital) {
            par |= SV_MASTER_PRESET_DIGAUDIO16;
          }
          break;
        default:
          cmd = SV_MASTER_NOP;
        }
      }
      if(cmd == SV_MASTER_NOP) {
        jpeg_errorprintf(sv, "sv master preset : Preset definition illegal : %s\n", argv[1]);
      }
    } else if(strcmp(argv[0], "recoffset") == 0) {
      cmd = SV_MASTER_RECOFFSET;
      par = atoi(argv[1]);
    } else if(strcmp(argv[0], "timecode") == 0) {
      if((strcmp(argv[1], "LTC") == 0) || (strcmp(argv[1], "ltc") == 0)) {
        cmd = SV_MASTER_TIMECODE;
        par = SV_MASTER_TIMECODE_LTC;
      } else if((strcmp(argv[1], "ASTC") == 0) || (strcmp(argv[1], "VITC") == 0) || (strcmp(argv[1], "vitc") == 0)) {
        cmd = SV_MASTER_TIMECODE;
        par = SV_MASTER_TIMECODE_VITC;
      } else if(strcmp(argv[1], "auto") == 0) {
        cmd = SV_MASTER_TIMECODE;
        par = SV_MASTER_TIMECODE_AUTO;
      } else if(strcmp(argv[1], "timer1") == 0) {
        cmd = SV_MASTER_TIMECODE;
        par = SV_MASTER_TIMECODE_TIMER1;
      } else if(strcmp(argv[1], "timer2") == 0) {
        cmd = SV_MASTER_TIMECODE;
        par = SV_MASTER_TIMECODE_TIMER2;
      } else {
        jpeg_errorprintf(sv, "sv master timecode : Parameter Error : %s (valid: ltc/vitc/auto/timer1/timer2)\n", argv[1]);
      }
    } else if(strcmp(argv[0], "tolerance") == 0) {
      if(strcmp(argv[1], "none") == 0) {
        cmd = SV_MASTER_TOLERANCE;
        par = SV_MASTER_TOLERANCE_NONE;
      } else if(strcmp(argv[1], "normal") == 0) {
        cmd = SV_MASTER_TOLERANCE;
        par = SV_MASTER_TOLERANCE_NORMAL;
      } else if(strcmp(argv[1], "large") == 0) {
        cmd = SV_MASTER_TOLERANCE;
        par = SV_MASTER_TOLERANCE_LARGE;
      } else if(strcmp(argv[1], "rough") == 0) {
        cmd = SV_MASTER_TOLERANCE;
        par = SV_MASTER_TOLERANCE_ROUGH;
      } else {
        jpeg_errorprintf(sv, "sv master tolerance: Parameter Error : %s (valid: none/normal/large/rough)\n", argv[1]);
      }
    } else if(strcmp(argv[0], "code") == 0) {
      cmd = SV_MASTER_CODE;
      raw = argv[1];
    }
  }

  if(cmd == SV_MASTER_NOP) {
    if(argc >= 2) {
      if(strcmp(argv[0], "raw") == 0) {
        cmd = SV_MASTER_CODE;
    
        raw = rawbuffer;

        strcpy(rawbuffer, argv[1]);
        for(i = 2; i < argc; i++) {
          strcat(rawbuffer, " ");
          strcat(rawbuffer, argv[i]);
        }
      } else if(strcmp(argv[0], "rawreply") == 0) {
        cmd = SV_MASTER_CODE;
    
        raw       = rawbuffer;
        rawreply  = rawbuffer;

        strcpy(rawbuffer, argv[1]);
        for(i = 2; i < argc; i++) {
          strcat(rawbuffer, " ");
          strcat(rawbuffer, argv[i]);
        }
      } else if(strcmp(argv[0], "flags") == 0) {
        cmd = SV_MASTER_FLAG;
        par = 0;
        for(i = 1; i < argc; i++) {
          if(!strcmp(argv[i], "help") || !strcmp(argv[i], "?")) {
            printf("sv master preset help\n");
            printf("  autoedit,forcedropframe,emulatestepcmd\n");
          } else if(!strcmp(argv[i], "autoedit")) {
            par |= SV_MASTER_FLAG_AUTOEDIT;
          } else if(!strcmp(argv[i], "forcedropframe")) {
            par |= SV_MASTER_FLAG_FORCEDROPFRAME;
          } else if(!strcmp(argv[i], "emulatestepcmd")) {
            par |= SV_MASTER_FLAG_EMULATESTEPCMD;
          } else {
            jpeg_errorprintf(sv, "sv master flags: Parameter Error : %s\n", argv[i]);
          }
        }
      }
    }
  }

  if(cmd == SV_MASTER_CODE) {
    res = sv_vtrmaster_raw(sv, raw, rawreply);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);    
    } else if(rawreply) {
      printf("reply %s\n", rawreply);
    }
  } else if(cmd != SV_MASTER_NOP) {
    res = sv_vtrmaster(sv, cmd, par);
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);    
    }
  } else {
    jpeg_errorprintf(sv, "sv master %s : Unknown master command\n", argv[0]);
  }
  
  return (cmd != SV_MASTER_NOP);
}


void jpeg_storageinfo_show(sv_storageinfo * psi)
{
    int j;

    printf("SIZE          %d\n", psi->size);
    printf("VERSION       %d\n", psi->version);
    printf("COOKIE        %d\n", psi->cookie);
    printf("XSIZE         %d\n", psi->xsize);
    printf("YSIZE         %d\n", psi->ysize);
    printf("STORAGEXSIZE  %d\n", psi->storagexsize);
    printf("STORAGEYSIZE  %d\n", psi->storageysize);
    printf("INTERLACE     %d\n", psi->interlace);
    printf("NBITS         %d\n", psi->nbits);
    printf("FPS           %d\n", psi->fps);
    printf("DROPFRAME     %d\n", psi->dropframe);
    printf("BUFFERSIZE    %d\n", psi->buffersize);
    printf("ALIGNMENT     %d\n", psi->alignment);
    printf("BIGENDIAN     %d\n", psi->bigendian);
    printf("COMPONENTS    %d\n", psi->components);
    printf("DOMINACE21    %d\n", psi->dominance21);
    printf("FULLRANGE     %d\n", psi->fullrange);
    printf("RGBFORMAT     %d\n", psi->rgbformat);
    printf("SUBSAMPLE     %d\n", psi->subsample);
    if((uint32)psi->yuvmatrix  < arraysize(str_yuvmatrix)) {
      printf("YUVMATRIX     %s\n", str_yuvmatrix[psi->yuvmatrix]);
    }
    printf("FIELDSIZE0    %d\n", psi->fieldsize[0]);
    printf("FIELDSIZE1    %d\n", psi->fieldsize[1]);
    printf("FIELDOFFSET0  %d\n", psi->fieldoffset[0]);
    printf("FIELDOFFSET1  %d\n", psi->fieldoffset[1]);
    printf("LINEOFFSET0   %d\n", psi->lineoffset[0]);
    printf("LINEOFFSET1   %d\n", psi->lineoffset[1]);
    printf("LINESIZE      %d\n", psi->linesize);
    printf("PIXELSIZE     %d/%d\n", psi->pixelsize_num, psi->pixelsize_denom);
    for( j = 0; (j < 8) && (j < psi->components); j++ ) {
      printf("DATAOFFSET%d  %d/%d\n", j, psi->dataoffset_num[j], psi->pixelsize_denom);
    }
    for( j = 0; (j < 8) && (j < psi->components); j++ ) {
      printf("PIXELOFFSET%d %d/%d\n", j, psi->pixeloffset_num[j], psi->pixelsize_denom);
    }
}


void jpeg_guiinfo_init(sv_handle * sv, sv_info * info)
{
  int res;
  int feature;
  int nrasters = 1;
  int rasterindex = -1;
  sv_rasterheader raster, current;
  int i;
  int temp;
  
  res = sv_query(sv, SV_QUERY_FEATURE, 0, &feature);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  printf("VMODINIT   %s\n", "BURST");
  printf("NPELINIT   %d\n", info->xsize);
  printf("NLININIT   %d\n", info->ysize);
  printf("XSIZEINIT  %d\n", info->setup.storagexsize);
  printf("YSIZEINIT  %d\n", info->setup.storageysize);
  printf("NFRAMINIT  %d\n", info->setup.nframes);
  printf("COLMINIT   %s\n", str_colormode[info->colormode]);
  printf("NBITINIT   %d\n", info->nbit);
  printf("ABITINIT   %d\n", 0);
  printf("COMPINIT   %d\n", 0);
  printf("VMODLIST   BURST\n");
  printf("COLMLIST   %s\n", info->setup.rgbm?"YUV422,RGB,BGR,RGBA,BGRA":"YUV422");
  printf("NBITLIST   %s\n", "8,10");
  printf("ABITLIST   0\n");
  printf("COMPLIST   0\n");

  if(feature & SV_FEATURE_RASTERLIST) {
    res = sv_raster_status(sv, -1, &current, sizeof(current), &nrasters, 0);
    
    printf("RASTERINIT %d\n", current.index);
    printf("FREQINIT   %d\n", current.dfreq);

    for(i = 0; i < nrasters; i++) {
      res = sv_raster_status(sv, i, &raster, sizeof(raster), NULL, 0);
      if(res == SV_OK) {
        printf("RASTERLIST %3d \"%-32s\" %9d %4d %4d %s\n",
           raster.index,
           raster.name,
           raster.dfreq,
           raster.hlen,
           raster.vlen[0] + raster.vlen[1], 
           raster.disable?"disabled":"enabled");
      }
    }
  } else {
    res = sv_query(sv, SV_QUERY_MODE_CURRENT, 0, &rasterindex);
    if(res == SV_OK) {
      printf("RASTERINIT %d\n", rasterindex & SV_MODE_MASK);
    }
    for(i = 0; (i < arraysize(table_rasters)); i++) {
      if(rasterindex == table_rasters[i].mode) {
        printf("FREQINIT   %d\n", table_rasters[i].frequency);
      }
    }
    for(i = 0; (i < arraysize(table_rasters)); i++) {
      res = sv_query(sv, SV_QUERY_MODE_AVAILABLE, table_rasters[i].mode, &temp);
      if((res == SV_OK) && temp) {
        printf("RASTERLIST %2d \"%-32s\" %8d %4d %4d %s\n",
           table_rasters[i].mode,
           table_rasters[i].name,
           table_rasters[i].frequency,
           table_rasters[i].xsize,
           table_rasters[i].ysize, 
           "enabled");
      }
    }
  }
}


void jpeg_guiinfo_menu(sv_handle * sv)
{
  int res;
  int menu,       menucount;
  int menuitem,   menuitemcount;
  int menulabel,  menulabelcount;
  char buffer[64];
  int option;
  int value;
  int mask;

  res = sv_option_menu(sv, 0, 0, 0, buffer, sizeof(buffer), &menucount, NULL, NULL);
  printf("DEVICE '%s'\n", buffer);
  for(menu = 1; (res == SV_OK) && (menu <= menucount); menu++) {
    res = sv_option_menu(sv, menu, 0, 0, buffer, sizeof(buffer), &menuitemcount, NULL, NULL);
    printf("MENU '%s'\n", buffer);
    for(menuitem = 1; (res == SV_OK) && (menuitem <= menuitemcount); menuitem ++) {
      res = sv_option_menu(sv, menu, menuitem, 0, buffer, sizeof(buffer), &menulabelcount, &option, NULL);
      printf("SUBMENU '%s' %08x\n", buffer, option);
      for(menulabel = 1; (res == SV_OK) && (menulabel <= menulabelcount); menulabel++) {
        res = sv_option_menu(sv, menu, menuitem, menulabel, buffer, sizeof(buffer), &value, &mask, NULL);
        printf("MENULABEL '%s' %08x %08x\n", buffer, value, mask);
      }
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

   

void jpeg_guiinfo(sv_handle * sv, char * what)
{
  int     res;
  sv_info info;
  double  speed;
  char    buffer[256];
  int     i,j;
  char    tmp[20];
  int     type;
  int     intid;
  char *  argv;
  int     temp;
 
  res = sv_status(sv, &info);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  if(strncmp(what, "tc", 5) == 0) {
    res = sv_tc2asc(sv, info.master.timecode, buffer, sizeof(buffer));
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }
    printf("VTRTC %s\n", buffer);
    printf("VTRERROR %s\n", sv_geterrortext(info.master.error));
    printf("STANDBY %s\n", (info.master.info & SV_MASTER_INFO_STANDBY)?"on":"off");
  } else if(strncmp(what, "position", 8) == 0) {
    int tmp_timecode, tmp_userbytes;
    printf("POSITION      %d\n", info.video.position);
    res = sv_tc2asc(sv, info.video.positiontc, buffer, sizeof(buffer));
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }
    printf("POSITIONTC    %s\n", buffer);

    res = sv_query(sv, SV_QUERY_LTCTIMECODE, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_LTCUSERBYTES, -1, &tmp_userbytes);
    if(res == SV_OK) {
      if(sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer)) == SV_OK) {
        printf("LTC           %-12s  %08x\n", buffer, tmp_userbytes);
      }
    }

    res = sv_query(sv, SV_QUERY_DVITC_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_DVITC_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("DVITC         %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_DLTC_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_DLTC_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("DLTC          %-12s  %08x\n", buffer, tmp_userbytes);
    }
    
    res = sv_query(sv, SV_QUERY_FILM_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_FILM_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("FILMTC        %-12s  %08x\n", buffer, tmp_userbytes);
    }

    res = sv_query(sv, SV_QUERY_PROD_TC, -1, &tmp_timecode);
    sv_query(sv, SV_QUERY_PROD_UB, -1, &tmp_userbytes);
    sv_tc2asc(sv, tmp_timecode, buffer, sizeof(buffer));
    if(res == SV_OK) {
      printf("PRODTC        %-12s  %08x\n", buffer, tmp_userbytes);
    }

    printf("INPOINT       %d\n", info.video.inpoint);
    printf("OUTPOINT      %d\n", info.video.outpoint);
    speed = (double) info.video.speed / (double) info.video.speedbase;
    printf("SPEED         %f\n",speed);
    printf("OPERATIONMODE %s\n", sv_support_devmode2string(info.video.state));
    res = sv_tc2asc(sv, info.master.timecode, buffer, sizeof(buffer));
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }
    printf("VTRTC         %s\n", buffer);
    printf("VTRERROR      %s\n", sv_geterrortext(info.master.error));

    sv_support_vtrinfo2string(buffer, sizeof(buffer), info.master.info);
    printf("VTRSTATUS     %s\n", buffer);

    printf("STANDBY       %s\n", (info.master.info & SV_MASTER_INFO_STANDBY)?"on":"off");
    printf("PULLDOWN      %d\n", info.video.flags & SV_INFO_VIDEO_FLAGS_PULLDOWN ? 1 : 0);
    // writeprotect not supported yet   
    // printf("WRITEPROTECT  %d\n", info.video.flags & SV_INFO_VIDEO_FLAGS_PROTECT  ? 1 : 0);
  } else if(strncmp(what, "slave", 5) == 0) {
    printf("SLAVEMODE %d\n",    info.video.slavemode);
  } else if(strcmp(what, "system") == 0) {
    int max = 1;

    res = sv_query(sv, SV_QUERY_DEVTYPE, 0, &temp);
    if(res == SV_OK) {
      if(temp > SV_DEVTYPE_CENTAURUS) {
        max = 5;
      }
    }
    
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_TEMPERATURE, i, &temp) == SV_OK); i++) {
      printf("Temperature     : %1.1f C\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V0, i, &temp) == SV_OK); i++) {
      printf("Voltage (1.0V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V2, i, &temp) == SV_OK); i++) {
      printf("Voltage (1.2V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V5, i, &temp) == SV_OK); i++) {
      printf("Voltage (1.5V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_1V8, i, &temp) == SV_OK); i++) {
      printf("Voltage (1.8V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_2V3, i, &temp) == SV_OK); i++) {
      printf("Voltage (2.3V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_2V5, i, &temp) == SV_OK); i++) {
      printf("Voltage (2.5V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_3V3, i, &temp) == SV_OK); i++) {
      printf("Voltage (3.3V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_5V0, i, &temp) == SV_OK); i++) {
      printf("Voltage (5.0V)  : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_VOLTAGE_12V0, i, &temp) == SV_OK); i++) {
      printf("Voltage (12.0V) : %1.2f V\n", ((double)temp)/0x10000);
    }
    for(i = 0; (i < max) && (sv_query(sv, SV_QUERY_FANSPEED, i, &temp) == SV_OK); i++) {
      printf("Fanspeed        : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_CARRIER, 0, &temp);
    if(res == SV_OK) {
      printf("Carrier         : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_GENLOCK, 0, &temp);
    if(res == SV_OK) {
      printf("Genlock         : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_INPUTRASTER, 0, &temp);
    if(res == SV_OK) {
      printf("Inputraster     : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_INPUTRASTER_GENLOCK, 0, &temp);
    if(res == SV_OK) {
      printf("Genlockraster   : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_VITCINPUTLINE, 0, &temp);
    if(res == SV_OK) {
      if(SV_VITCLINE_DUPLICATE_GET(temp)) {
        printf("Vitcinputline   : %d/%d\n", temp, SV_VITCLINE_DUPLICATE_GET(temp));
      } else {
        printf("Vitcinputline   : %d\n", temp);
      }
    }
    res = sv_query(sv, SV_QUERY_DISPLAY_LINENR, 0, &temp);
    if(res == SV_OK) {
      printf("Displaylinenr   : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_RECORD_LINENR, 0, &temp);
    if(res == SV_OK) {
      printf("Recordlinenr    : %d\n", temp);
    }
    res = sv_query(sv, SV_QUERY_AUDIOINERROR, 0, &temp);
    if(res == SV_OK) {
      printf("AudioInError    : %s\n", sv_geterrortext(temp));
    }
    res = sv_query(sv, SV_QUERY_VIDEOINERROR, 0, &temp);
    if(res == SV_OK) {
      printf("VideoInError    : %s\n", sv_geterrortext(temp));
    }
    res = sv_query(sv, SV_QUERY_AUDIO_AIVCHANNELS, 0, &temp);
    if(res == SV_OK) {
      printf("AiVChannels     : 0x%08x\n", temp);
    }
    res = sv_query(sv, SV_QUERY_AUDIO_AESCHANNELS, 0, &temp);
    if(res == SV_OK) {
      printf("AESChannels     : 0x%08x\n", temp);
    }
  } else if(strncmp(what, "setup", 5) == 0) {
    if(sv_query(sv, SV_QUERY_INTERLACEID_STORAGE, 0, &intid) != SV_OK) {
      intid = 0;
    }
    if(sv_query(sv, SV_QUERY_DEVTYPE, 0, &type) != SV_OK) {
      type = SV_DEVTYPE_UNKNOWN;
    }
    printf("DEVICENAME    %s\n", sv_query_value2string(sv, SV_QUERY_DEVTYPE, type));
    if (info.colormode < arraysize(str_colormode)) {
      printf("COLORMODE     %s\n", str_colormode[info.colormode]);
    }
    if (info.nbit == 10) {
      printf("VIDEOMODE     %dx%dx10b\n", info.xsize, info.ysize);
    } else if (intid == 1) {
      printf("VIDEOMODE     %dx%dp\n", info.xsize, info.ysize);
    } else {
      printf("VIDEOMODE     %dx%d\n", info.xsize, info.ysize);
    }

    printf("FRAMERATE     %2.3f\n", (double)info.video.framerate_mHz / 1000);

    sv_support_syncmode2string(info.sync, buffer, sizeof(buffer));
    printf("VIDEOSYNC     %s\n", buffer);

    sv_query(sv, SV_QUERY_SYNCOUT, 0, &i);
    sv_support_syncout2string(i, buffer, sizeof(buffer));
    printf("SYNCOUT       %s\n", buffer);

    sv_query(sv, SV_QUERY_SYNCOUTDELAY, 0, &i);
    sv_query(sv, SV_QUERY_SYNCOUTVDELAY, 0, &j);
    printf("SYNCOUTDELAY  %d %d\n", i, j);

    printf("GENLOCK       %d\n", info.setup.genlock);
    printf("DISKSIZE      %d\n", info.setup.disksize);
    printf("DRAMSIZE      %d\n", info.setup.ramsize);
    printf("NFRAMES       %d\n", info.setup.nframes);
    printf("AUDIO         %d\n", info.setup.audio);
    printf("KEY           %d\n", info.setup.key);
    printf("MONO          %d\n", info.setup.mono);
    printf("RGBMODUL      %d\n", info.setup.rgbm);
    sv_support_iomode2string(info.video.iomode, buffer, sizeof(buffer));
    printf("IOMODE        %s\n", buffer);
    sv_query(sv, SV_QUERY_REPEATMODE, 0, &i);
    if((i >= 0) && (i < arraysize(str_repeatmode))) {
      printf("REPEATMODE    %s\n", str_repeatmode[i]);
    }
    sv_query(sv, SV_QUERY_SLOWMOTION, 0, &i);
    if((i >= 0) && (i < arraysize(str_slowmotion))) {
      printf("SLOWMOTION    %s\n", str_slowmotion[i]);
    }
    sv_query(sv, SV_QUERY_FASTMOTION, 0, &i);
    if((i >= 0) && (i < arraysize(str_fastmode))) {
      printf("FASTMOTION    %s\n", str_fastmode[i]);
    }
    sv_query(sv, SV_QUERY_LOOPMODE, 0, &i);
    if((i >= 0) && (i <= arraysize(str_loopmode))) {
      printf("LOOPMODE      %s\n", str_loopmode[i]);
    }
    sv_query(sv, SV_QUERY_AUDIOMODE, 0, &i);
    if((i >= 0) && (i < arraysize(str_audiomode))) {
      printf("AUDIOMODE     %s\n", str_audiomode[i]);
    }
    sv_query(sv, SV_QUERY_AUDIOMUTE, 0, &i);
    printf("AUDIOMUTE     %d\n", i);
    
    res = sv_query(sv, SV_QUERY_INPUTPORT, 0, &i);
    if(res == SV_OK) {
      printf("INPUTPORT     %s\n", sv_query_value2string(sv, SV_QUERY_INPUTPORT, i));
    }
    res = sv_query(sv, SV_QUERY_OUTPUTPORT, 0, &i);
    if(res == SV_OK) {
      printf("OUTPUTPORT    %s\n", sv_query_value2string(sv, SV_QUERY_OUTPUTPORT, i));
    }
    sv_query(sv, SV_QUERY_PULLDOWN, 0, &i);
    if (i != 0) {
      printf("PULLDOWN      1\n");
    }
    sv_query(sv, SV_QUERY_AUTOPULLDOWN, 0, &i);
    if (i != 0) {
      printf("AUTOPULLDOWN  1\n");
    }
    /* The following is to tell the GUI which factor should be used to      */
    /* enlarge playlist segment length with pulldown material. This ratio   */
    /* is always one to one with current SDIO and HDIO software.            */
    printf("PLAYLISTPDRATIO  1/1\n");
    sv_query(sv, SV_QUERY_WORDCLOCK, 0, &i);
    if (i != 0) {
      printf("WORDCLOCK     1\n");
    }
    sv_query(sv, SV_QUERY_FSTYPE, 0, &i);
    if((i >= 0) && (i < arraysize(str_fstype))) {
      printf("FSTYPE        %s\n", str_fstype[i]);
    }
    sv_query(sv, SV_QUERY_ANALOG, 0, &i);
    sv_support_analog2string(i, buffer, sizeof(buffer));
    printf("ANALOG        %s\n", buffer);
#if 0
    if (info.setup.audio) {
#endif
    {
      sv_support_preset2string(info.setup.preset, tmp, sizeof(tmp));
      printf("PRESET        %s\n",    tmp);
      i = 0;
      if (info.master.preset & SV_MASTER_PRESET_VIDEO) 
        tmp[i++] = 'V';
      if (info.master.preset & SV_MASTER_PRESET_ASSEMBLE) 
        tmp[i++] = 'Q';
      if (info.master.preset & SV_MASTER_PRESET_AUDIOMASK)
        tmp[i++] = 'A';
      if (info.master.preset & SV_MASTER_PRESET_AUDIO1)  
        tmp[i++] = '1';
      if (info.master.preset & SV_MASTER_PRESET_AUDIO2)  
        tmp[i++] = '2';
      if (info.master.preset & SV_MASTER_PRESET_AUDIO3)  
        tmp[i++] = '3';
      if (info.master.preset & SV_MASTER_PRESET_AUDIO4)  
        tmp[i++] = '4';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIOMASK)
        tmp[i++] = 'D';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO1)  
        tmp[i++] = '1';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO2)  
        tmp[i++] = '2';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO3)  
        tmp[i++] = '3';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO4)  
        tmp[i++] = '4';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO5)  
        tmp[i++] = '5';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO6)  
        tmp[i++] = '6';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO7)  
        tmp[i++] = '7';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO8)  
        tmp[i++] = '8';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO9)  
        tmp[i++] = '9';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO10)  
        tmp[i++] = 'a';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO11)  
        tmp[i++] = 'b';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO12)  
        tmp[i++] = 'c';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO13)  
        tmp[i++] = 'd';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO14)  
        tmp[i++] = 'e';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO15)  
        tmp[i++] = 'f';
      if (info.master.preset & SV_MASTER_PRESET_DIGAUDIO16)  
        tmp[i++] = 'g';
      tmp[i++] = 0;
      printf("MASTER_PRESET %s\n",    tmp);
    }
    printf("SOFDELAY      %d\n", info.video.sofdelay);
    sv_query(sv, SV_QUERY_AUDIOCHANNELS, 0, &i);
    printf("AUDIOCHANNELS %d\n", i);
    sv_query(sv, SV_QUERY_AUDIOFREQ, 0, &i);
    printf("AUDIOFREQUENCY %d\n", i);
    res = sv_query(sv, SV_QUERY_FEATURE, 0, &i);
    if ((res == SV_OK) && (i & SV_FEATURE_VTRMASTER_LOCAL)) {
      printf("POLLSUPPORT   1\n"); 
    }

    if (info.master.framerate) {
      printf("FRAMERATE     %d\n", info.master.framerate);      // needs to be changed
      printf("DROPFRAME     %d\n", info.video.positiontc & 0x40000000 ? 1 : 0);
    }

    res = sv_query(sv, SV_QUERY_TIMECODE_OFFSET, 0, &i);
    if (res == SV_OK) {
      res = sv_tc2asc(sv, i, buffer, sizeof(buffer));
      if (res == SV_OK) {
        printf("TCOFFSET      %s\n", buffer);
      }
    }

    sv_query(sv, SV_QUERY_PULLDOWNFPS, 0, &i);
    if (i) {
      char * phases = "a";
      printf("PULLDOWNFPS   %d\n", i);
      switch (i) {
      case 24:
        if (info.master.framerate==30) {
          phases = "abcd";
        } else if ((info.master.framerate==36) || (info.master.framerate==60)) {
          phases = "ab";
        }
        break;
      }
      printf("PULLDOWNPHASES %s\n", phases);
    }
    res = sv_query(sv, SV_QUERY_TILEFACTOR, 0, &i);
    if ((res == SV_OK) && (i >= 1)) {
      printf("TILEFACTOR  %d\n", i); 
    }
  } else if(strncmp(what, "master", 6) == 0) {
    printf("AUTOPARK %s\n", (info.master.autopark)?"on":"off");

    printf("DEVICETYPE 0x%04x\n", info.master.device);

    printf("EDITLAG %d\n", info.master.editlag);

    if(info.master.flags & SV_MASTER_FLAG_AUTOEDIT) {
      printf("VTRFLAGS autoedit\n");
    }
    if(info.master.flags & SV_MASTER_FLAG_FORCEDROPFRAME) {
      printf("VTRFLAGS forcedropframe\n");
    }
    if(info.master.flags & SV_MASTER_FLAG_EMULATESTEPCMD) {
      printf("VTRFLAGS emulatestepcmd\n");
    }

    res = sv_tc2asc(sv, info.master.parktime, buffer, sizeof(buffer));
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }   
    printf("PARKTIME %s\n", buffer);

    res = sv_tc2asc(sv, info.master.preroll, buffer, sizeof(buffer));
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }   
    printf("PREROLL %s\n", buffer);

    res = sv_tc2asc(sv, info.master.postroll, buffer, sizeof(buffer));
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }   
    printf("POSTROLL %s\n", buffer);

    if(info.master.timecodetype < arraysize(str_master_timecodetype)) {
      printf("TIMECODE %s\n", str_master_timecodetype[info.master.timecodetype]);
    }

    if(info.master.tolerance < arraysize(str_master_tolerance)) {
      printf("TOLERANCE %s\n", str_master_tolerance[info.master.tolerance]);
    } 
    
    printf("DISOFFSET %d\n", info.master.disoffset);

    printf("RECOFFSET %d\n", info.master.recoffset);

    if (info.master.framerate) {
      switch(info.master.framerate) {
      case 48:
      case 50:
      case 60:
        info.master.framerate /= 2;
        break;
      case 72:
        info.master.framerate /= 3;
        break;
       }
      printf("VTRTCRATE %d\n", info.master.framerate);
      printf("VTRDROPFRAME %d\n", info.master.timecode & 0x40000000 ? 1 : 0);
    }

  } else if(!strcmp(what, "fileformats")) {
    fm_initialize();
    printf("FORMATS");
    for(i = 0; fm_fileformat_name(buffer, sizeof(buffer), i) == FM_OK; i++) {
      printf(" %s",buffer);
    }
    printf("\n");
    fm_deinitialize();
  } else if(!strcmp(what, "mixer")) {
    strcpy(buffer, "guiinfo");
    argv = &buffer[0];
    jpeg_mixer(sv, 1, &argv);
  } else if(!strcmp(what, "feature")) {
    i = 0;
    sv_query(sv, SV_QUERY_FEATURE, 0, &i);
    if (i & SV_FEATURE_AUTOPULLDOWN) {
      printf("AUTOPULLDOWN\n");
    }
    if (i & SV_FEATURE_CAPTURE) {
      printf("CAPTURE\n");
    }
    if (i & SV_FEATURE_DUALLINK) {
      printf("DUALLINK\n");
    }
    if (i & SV_FEATURE_HEADERTRANSFER) {
      printf("HEADERTRANSFER\n");
    }
    if (i & SV_FEATURE_KEYCHANNEL) {
      printf("KEYCHANNEL\n");
    }
    if (i & SV_FEATURE_LTC_RECORDOUT) {
      printf("LTCRECORDOUTPUT\n");
    }
    if (i & SV_FEATURE_LUTSUPPORT) {
      printf("LUTSUPPORT\n");
    }
    if (i & SV_FEATURE_INDEPENDENT_IO) {
      printf("INDEPENDENT_IO\n");
    }
    if (i & SV_FEATURE_MIXERSUPPORT) {
      printf("MIXERSUPPORT\n");
    }
    if (i & SV_FEATURE_MIXERPROCESSING) {
      printf("MIXERPROCESSING\n");
    }
    if (i & SV_FEATURE_MULTIJACK) {
      printf("MULTIJACK\n");
    }
    if (i & SV_FEATURE_NOHSWTRANSFER) {
      printf("NOHSWTRANSFER\n");
    }
    if (i & SV_FEATURE_PLAYLISTMAP) {
      printf("PLAYLISTMAP\n");
    }
    if (i & SV_FEATURE_RASTERLIST) {
      printf("RASTERLIST\n");
    }
    if (i & SV_FEATURE_SCSISWITCH) {
      printf("SCSISWITCH\n");
    }
    if (i & SV_FEATURE_VTRMASTER_LOCAL) {
      printf("POLLSUPPORT\n");
    }
    if (i & SV_FEATURE_ZOOMANDPAN) {
      printf("ZOOMANDPAN\n");
    }
    if (i & SV_FEATURE_ZOOMSUPPORT) {
      printf("ZOOMSUPPORT\n");
    }
  } else if(!strcmp(what, "init")) {
    jpeg_guiinfo_init(sv, &info);
  } else if(!strcmp(what, "menu")) {
    jpeg_guiinfo_menu(sv);
  } else if(!strcmp(what, "mode")) {
    int feature;
    res = sv_query(sv, SV_QUERY_FEATURE, 0, &feature);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
      return;
    }

    if(feature & SV_FEATURE_RASTERLIST) {
      sv_rasterheader raster, current;
      int nrasters;

      res = sv_raster_status(sv, -1, &current, sizeof(current), &nrasters, 0);

      for(i = 0; i < nrasters; i++) {
        res = sv_raster_status(sv, i, &raster, sizeof(raster), NULL, 0);
        if(res == SV_OK) {
          printf("%-32s\n", raster.name);
        }
      }
    } else {
      for(i = 0; (i < arraysize(table_rasters)); i++) {
        res = sv_query(sv, SV_QUERY_MODE_AVAILABLE, table_rasters[i].mode, &temp);
        if((res == SV_OK) && temp) {
          printf("%-32s\n", table_rasters[i].name);
        }
      }
    }
  } else if(!strcmp(what, "hardware")) {
    sv_query(sv, SV_QUERY_HW_EPLDVERSION, 0, &i);
    printf("EPLDVERSION %d.%d-%d\n", (i>>16) & 0xff, (i>>8) & 0xff, i & 0xff);
    sv_query(sv, SV_QUERY_HW_EPLDOPTIONS, 0, &i);
    printf("EPLDOPTIONS $%08x\n", i);
    sv_query(sv, SV_QUERY_HW_CARDVERSION, 0, &i);
    printf("CARDVERSION %d.%d-%d\n", (i>>16) & 0xff, (i>>8) & 0xff, i & 0xff);
    sv_query(sv, SV_QUERY_HW_CARDOPTIONS, 0, &i);
    printf("CARDOPTIONS $%08x\n", i);
  } else if(!strcmp(what, "storage")) {
    sv_storageinfo storage;
    res = sv_storage_status(sv, 0, NULL, &storage, sizeof(storage), 0);
    if(res == SV_OK) {
      jpeg_storageinfo_show(&storage);
    } else {
      jpeg_errorprint(sv, res);
    }
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv guiinfo help\n");
    printf("\tfileformats\n");
    printf("\tmaster\n");
    printf("\tmixer\n");
    printf("\tposition\n");
    printf("\tsetup\n");
    printf("\tslave\n");
    printf("\tclips\n");
    printf("\trefresh\n");
    printf("\ttc\n");
    printf("\tstorage\n");
  }
}


void jpeg_debugprint(sv_handle * sv, int infinite)
{
  char buffer[4096];
  int  count;
  int  i;
  int  res = SV_OK;

  do {
    res = sv_debugprint(sv, buffer, sizeof(buffer), &count);

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    } else {
      for(i = 0; i < count; i++) {
        switch(buffer[i]) {
        case '\r':
          break;
        case '\n':
          putchar('\n');
          break;
        default:
          putchar(buffer[i]);
        }
      }

      if(infinite) {
        if(!count) {
          sv_usleep(sv, 100000);
        } else {
          fflush(stdout);
        }
      }
    }
  } while((res == SV_OK) && (infinite || count));
}

void jpeg_step(sv_handle * sv, char * step)
{
  int res;
  int nframes;

  nframes = atoi(step);

  res = sv_position(sv, nframes, 0, SV_REPEAT_DEFAULT, SV_POSITION_FLAG_RELATIVE);
  
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_pause(sv_handle * sv)
{
  int res;

  res = sv_position(sv, -1, 0, SV_REPEAT_DEFAULT, SV_POSITION_FLAG_PAUSE);
  
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_speed(sv_handle * sv, char * speed, char * loopmode)
{
  double speed_double = 1.0;
  int    speed_int;
  int    speed_base   = 0;
  int loop            = SV_LOOPMODE_DEFAULT;
  int res             = SV_OK;
  sv_info info;
  char * p;
   
  res = sv_status(sv, &info);
 
  p = strchr(speed, '/');
  if(p) {
    speed_double = 
    speed_int    = atoi(speed);
    speed_base   = atoi(p+1);
  } else {
    speed_double = atof(speed);
    speed_int    = (int) (speed_double * info.video.speedbase);
  }

  if(loopmode) {
    if(     strcmp("default",    loopmode) == 0) {
      loop = SV_LOOPMODE_DEFAULT;
    } else if(strcmp("once",     loopmode) == 0) {
      loop = SV_LOOPMODE_ONCE;
    } else if(strcmp("shuttle",  loopmode) == 0) {
      loop = SV_LOOPMODE_SHUTTLE;
    } else if(strcmp("loop",	 loopmode) == 0) {
      loop = SV_LOOPMODE_FORWARD;
    } else if(strcmp("infinite", loopmode) == 0) {
      loop = SV_LOOPMODE_INFINITE;
    } else if((strcmp(loopmode, "?") == 0) || (strcmp(loopmode, "help") == 0)) {
      printf("sv speed %s help\n", speed);
      printf("\tdefault\n");
      printf("\tforward\n");
      printf("\tinfinite\n");
      printf("\tonce\n");
      printf("\tshuttle\n");
      return;
    } else {
      jpeg_errorprintf(sv, "sv speed: Unknown loopmode: %s\n", loopmode);
      return;
    }
  }

  res = sv_option(sv, SV_OPTION_LOOPMODE, loop);
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }

  if(speed_base) {
    res = sv_option(sv, SV_OPTION_SPEEDBASE, speed_base);     
    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
  res = sv_option(sv, SV_OPTION_SPEED, speed_int);     
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
    
}  


void jpeg_analog(sv_handle * sv, int argc, char **argv)
{
  int mode = 0;
  int res  = SV_OK;
  char *what;
  char txtbuffer[256];

  res = sv_query(sv, SV_QUERY_ANALOG, 0, &mode);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
    return;
  }

  what = argv[0];

  if(strcmp(what, "auto") == 0) {
    mode &= ~SV_ANALOG_SHOW_MASK;
    mode |=  SV_ANALOG_SHOW_AUTO;
  } else if(strcmp(what, "input") == 0) {
    mode &= ~SV_ANALOG_SHOW_MASK;
    mode |=  SV_ANALOG_SHOW_INPUT;
  } else if(strcmp(what, "output") == 0) {
    mode &= ~SV_ANALOG_SHOW_MASK;
    mode |=  SV_ANALOG_SHOW_OUTPUT;
  } else if(strcmp(what, "black") == 0) {
    mode &= ~SV_ANALOG_SHOW_MASK;
    mode |=  SV_ANALOG_SHOW_BLACK;
  } else if(strcmp(what, "colorbar") == 0) {
    mode &= ~SV_ANALOG_SHOW_MASK;
    mode |=  SV_ANALOG_SHOW_COLORBAR;
  } else if(strcmp(what, "forcenone") == 0) {
    mode &= ~SV_ANALOG_FORCE_MASK;
  } else if(strcmp(what, "forcepal") == 0) {
    mode &= ~SV_ANALOG_FORCE_MASK;
    mode |=  SV_ANALOG_FORCE_PAL;
  } else if(strcmp(what, "forcentsc") == 0) {
    mode &= ~SV_ANALOG_FORCE_MASK;
    mode |=  SV_ANALOG_FORCE_NTSC;
  } else if(strcmp(what, "blacklevel7.5") == 0) {
    mode &= ~SV_ANALOG_BLACKLEVEL_MASK;
    mode |=  SV_ANALOG_BLACKLEVEL_BLACK75;
  } else if(strcmp(what, "blacklevel0") == 0) {
    mode &= ~SV_ANALOG_BLACKLEVEL_MASK;
    mode |=  SV_ANALOG_BLACKLEVEL_BLACK0;
  } else if(strcmp(what, "YC") == 0) {
    mode &= ~SV_ANALOG_OUTPUT_MASK;
    mode |=  SV_ANALOG_OUTPUT_YC;
  } else if(strcmp(what, "YUV") == 0) {
    mode &= ~SV_ANALOG_OUTPUT_MASK;
    mode |=  SV_ANALOG_OUTPUT_YUV;
  } else if(strcmp(what, "YUVS") == 0) {
    mode &= ~SV_ANALOG_OUTPUT_MASK;
    mode |=  SV_ANALOG_OUTPUT_YUVS;
  } else if(strcmp(what, "RGB") == 0) {
    mode &= ~SV_ANALOG_OUTPUT_MASK;
    mode |=  SV_ANALOG_OUTPUT_RGB;
  } else if(strcmp(what, "RGBS") == 0) {
    mode &= ~SV_ANALOG_OUTPUT_MASK;
    mode |=  SV_ANALOG_OUTPUT_RGBS;
  } else if(strcmp(what, "CVBS") == 0) {
    mode &= ~SV_ANALOG_OUTPUT_MASK;
    mode |=  SV_ANALOG_OUTPUT_CVBS;
  } else if(strcmp(what, "info") == 0) {
    sv_support_analog2string(mode, txtbuffer, sizeof(txtbuffer));
    printf("sv analog %s\n", txtbuffer);
    return;
  } else if(strcmp(what, "guiinfo") == 0) { 
    sv_support_analog2string(mode, txtbuffer, sizeof(txtbuffer));
    printf("sv analog %s\n", txtbuffer);
    return;
  } else if((strcmp(what, "?") == 0) || (strcmp(what, "help") == 0)) {
    printf("sv analog help\n");
    printf("\tinfo\t\tShow the current settings\n");
    printf("\tauto\t\tAutomatic switch between input/output\n");
    printf("\tinput\t\tShow input signal only\n");
    printf("\toutput\t\tShow output signal only\n");
    printf("\tblack\t\tShow black signal\n");
    printf("\tcolorbar\tShow colorbar signal\n");
    printf("\tforcenone\tNormal Analog colorcarrier\n");
    printf("\tforcepal\tForce PAL colorcarrier\n");
    printf("\tforcentsc\tForce NTSC colorcarrier\n");
    printf("\tblacklevel0  \t(NTSC only, Japan)\n");
    printf("\tblacklevel7.5\t(NTSC only, USA)\n");
    printf("\tYC\tSet the analog output to Y/C mode\n");
    printf("\tYUV\tSet the analog output to YUV mode\n");
    printf("\tYUVS\tSet the analog output to YUVS mode\n");
    printf("\tRGB\tSet the analog output to RGB mode\n");
    printf("\tRGBS\tSet the analog output to RGBS mode\n");
    printf("\tCVBS\tSet the analog output to CVBS mode\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv analog: Unknown command: %s\n", what);
    return;
  }
  
  res = sv_option(sv, SV_OPTION_ANALOG, mode);

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_analogoutput(sv_handle * sv, int argc, char **argv)
{
  int res  = SV_OK;
  int analogoutput = SV_ANALOGOUTPUT_RGBFULL;

  if(argc >= 1) {
    if(       strcmp(argv[0], "rgbfull") == 0) {
      analogoutput = SV_ANALOGOUTPUT_RGBFULL;
    } else if(strcmp(argv[0], "rgbhead") == 0) {
      analogoutput = SV_ANALOGOUTPUT_RGBHEAD;
    } else if(strcmp(argv[0], "yuvfull") == 0) {
      analogoutput = SV_ANALOGOUTPUT_YUVFULL;
    } else if(strcmp(argv[0], "yuvhead") == 0) {
      analogoutput = SV_ANALOGOUTPUT_YUVHEAD;
    } else if(strcmp(argv[0], "info") == 0) {
      res = sv_option_get(sv, SV_OPTION_ANALOGOUTPUT, &analogoutput);
      if(res == SV_OK) {
        if((analogoutput >= 0) && (analogoutput < sizeof(str_analogout))) {
          printf("sv analogoutput %s\n", str_analogout[analogoutput]);
        } else {
          printf("sv analogoutput %d\n", analogoutput);
        }
      }
      return;
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv analogoutput help\n");
      printf("\tinfo\t\tShow the current settings\n");
      printf("\trgbfull\t\tRGB Full\n");
      printf("\trgbhead\t\tRGB Head\n");
      printf("\tyuvfull\t\tYUV Full\n");
      printf("\tyuvhead\t\tYUV Head\n");
      return;
    } else {
      jpeg_errorprintf(sv, "sv analogoutput: Unknown command: %s\n", argv[0]);
      return;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_ANALOGOUTPUT, analogoutput);
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_dvi(sv_handle * sv, int argc, char **argv)
{
  int res   = SV_OK;
  int value = 0;
  char buffer[256];

  if(argc >= 1) {
    if(       strcmp(argv[0], "mode") == 0) {
      if(argc >= 2) {
        if(!strcmp(argv[1], "default")) {
          value = -1;
        } else {
          value = sv_support_string2videomode(argv[1], 0);
        } 
      } else {
        value = -1;
      }
      if(res == SV_OK) {
        res = sv_option_set(sv, SV_OPTION_DVI_VIDEOMODE, value);
      }
    } else if(strcmp(argv[0], "output") == 0) {
      if(argc >= 2) {
        if(!strcmp(argv[1], "dvi16")) {
          value = SV_DVI_OUTPUT_DVI16;
        } else if(!strcmp(argv[1], "dvi12")) {
          value = SV_DVI_OUTPUT_DVI12;
        } else if(!strcmp(argv[1], "dvi8")) {
          value = SV_DVI_OUTPUT_DVI8;
        } else {
          res = SV_ERROR_PARAMETER;
        }
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if(res == SV_OK) {
        res = sv_option_set(sv, SV_OPTION_DVI_OUTPUT, value);
      }
    } else if(strcmp(argv[0], "info") == 0) {
      res = sv_option_get(sv, SV_OPTION_DVI_VIDEOMODE, &value);
      if(res == SV_OK) {
        if(value == -1) {
          printf("sv dvi mode default\n");
        } else {
          res = jpeg_support_videomode2string(sv, buffer, value);
          if(res == SV_OK) {
            printf("sv dvi mode %s\n", buffer);
          }
        }
      }
      res = sv_option_get(sv, SV_OPTION_DVI_OUTPUT, &value);
      if(res == SV_OK) {
        printf("sv dvi output %s\n", sv_option_value2string(sv, SV_OPTION_DVI_OUTPUT, value));
      }
      return;
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv dvi help\n");
      printf("\tinfo\t\tShow the current settings\n");
      printf("\tmode #\t\tSet videomode for dvi output\n");
      printf("\tmode default\t\tSet videomode to same as main output\n");
      printf("\toutput {dvi8,dvi12,dvi16}\n");
      return;
    } else {
      jpeg_errorprintf(sv, "sv dvi: Unknown command: %s\n", argv[0]);
      return;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_forcerasterdetect(sv_handle * sv, int argc, char **argv)
{
  int res  = SV_OK;
  int value;

  if(argc >= 1) {
    if(strcmp(argv[0], "info") == 0) {
      res = sv_option_get(sv, SV_OPTION_FORCERASTERDETECT, &value);
      if(res == SV_OK) {
        if(value == 0) {
          printf("sv forcerasterdetect off\n");
        } else {
          printf("sv forcerasterdetect %s\n", sv_support_videomode2string(value));
        }
      }
      return;
    } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv forcerasterdetect help\n");
      printf("\tinfo\t\tShow the current settings\n");
      printf("\t#\t\tForce raster detect to raster #\n");
      printf("\toff\t\tTurn force raster detect off\n");
      return;
    } else if(strcmp(argv[0], "off") == 0) {
      value = 0;
    } else {
      value = SV_FORCEDETECT_ENABLE | sv_support_string2videomode(argv[0], 0);
    } 
    if(value == -1) {
      jpeg_errorprintf(sv, "sv forcerasterdetect: Unknown command: %s\n", argv[0]);
      return;
    }
    res = sv_option_set(sv, SV_OPTION_FORCERASTERDETECT, value);
  } else {
    jpeg_errorprintf(sv, "sv forcerasterdetect: Unknown command: %s\n", argv[0]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_monitorinfo(sv_handle * sv, int argc, char ** argv)
{
  unsigned char buffer[256];
  char  txtbuffer[32];
  int  res;
  int i,j,crc;
  int monitor = 0;
  int arg = 1;

  if(argc > 1) {
    if(!strcmp(argv[arg], "input")) {
      monitor = 1;
      arg++;
    } else if(!strcmp(argv[arg], "output")) {
      monitor = 0;
      arg++;
    } 
  }

  res = sv_monitorinfo(sv, monitor, (char*)&buffer[0], sizeof(buffer));
  if(res == SV_OK) {
    if(argc > arg) {
      if(!strcmp(argv[arg], "hex")) {
        printf("unsigned char monitornvram[128] = {\n");
        for(i = 0; i < 128; i+=16) {
          printf(" ");
          for(j = 0; j < 16; j++) {
            printf("0x%02x, ", (unsigned char)buffer[i+j]);
          }
          printf("\n");
        }
        printf("};\n");
        return;
      } else if(!strcmp(argv[1], "help") || !strcmp(argv[1], "?")) {
        printf("sv monitorinfo help\n");
        printf("\tsv monitorinfo output\tDump dvi output monitor nvram (connected monitor, default)\n");
        printf("\tsv monitorinfo input\tDump dvi input monitor nvram\n");
        printf("\t\thex\tHex dump of C array\n");
        return;
      }
    }
  }

  if(res == SV_OK) {
    for(crc = i = 0; i < 128; i+=16) {
      printf("%02x: ", i);
      for(j = 0; j < 16; j++) {
        printf(" %02x", (unsigned char)buffer[i+j]);
      }
      printf(" - ");
      for(j = 0; j < 16; j++) {
        if((buffer[i+j] >= 32) && (buffer[i+j] < 127)) {
          printf("%c", buffer[i+j]);
        } else {
          printf(".");
        }
        crc += buffer[i+j];
      }
      printf("\n");
    }

    crc &= 0xff;

    if(buffer[0] != 0x00) crc = -1;
    if(buffer[1] != 0xff) crc = -1;
    if(buffer[2] != 0xff) crc = -1;
    if(buffer[3] != 0xff) crc = -1;
    if(buffer[4] != 0xff) crc = -1;
    if(buffer[5] != 0xff) crc = -1;
    if(buffer[6] != 0xff) crc = -1;
    if(buffer[7] != 0x00) crc = -1;

    if(crc == 0) {
      printf("Edid    : %d.%d\n", buffer[0x12], buffer[0x13]);
      printf("Date    : %d/%d\n", 1990+buffer[0x11], buffer[0x10]);
      printf("Size    : %dx%d cm\n", buffer[0x15], buffer[0x16]);
      printf("Gamma   : %d.%02d\n", (buffer[0x17] + 100) / 100, (buffer[0x17] + 100) % 100);

      for(i = 0; i < 8; i++) {
        int k = 0x26 + 2*i; 
        int l;
        if((buffer[k] != 0x01) || (buffer[k+1] != 0x01)) {
          switch(buffer[k+1] & 0xc0) {
          case 0x00:  // 180 * 16/10
            l = 288; 
            break;
          case 0x40:  // 180 * 4/3
            l = 240;
            break;
          case 0x80:  // 180 * 5/4
            l = 225;
            break;
          default:    // 180 * 16/9
            l = 320;
          }  
          printf("timing%d : %dx%d/%d\n", i, (buffer[k] + 31) * 8, (buffer[k] + 31) * 8 * 180 / l, (buffer[k+1] & 0x3f) + 60);
        }
      }
    
      for(i = 0; i < 4; i++) {
        int k = 0x36 + 18*i;

        int freq    = (buffer[k  ] |  buffer[k+1] << 8) * 10000;

        if(freq) {
          int hactive = (buffer[k+2] | (buffer[k+4] & 0xf0) << 4);
          int hblank  = (buffer[k+3] | (buffer[k+4] & 0x0f) << 8);
          int vactive = (buffer[k+5] | (buffer[k+7] & 0xf0) << 4);
          int vblank  = (buffer[k+6] | (buffer[k+7] & 0x0f) << 8);
          int fps     = -1;

          if((hactive + hblank) && (vactive + vblank)) {
            fps = freq / (hactive + hblank);
            fps = fps * 1000 / (vactive + vblank);
          }
          if(freq) {
            printf("raster  : %dx%d/%d.%03d  %d\n",  hactive, vactive, fps / 1000, fps % 1000, freq);
          }
        } else {
          switch(buffer[k+3]) {
          case 0xff:
            memset(txtbuffer, 0, sizeof(txtbuffer)); for(j = 0; j < 12; j++) { if((buffer[k+5+j] >= 0x20) && (buffer[k+5+j] < 0x7f)) txtbuffer[j] = buffer[k+5+j]; else txtbuffer[j] = 0; }
            printf("Serial  : '%s'\n", txtbuffer);
            break;
          case 0xfc:
            memset(txtbuffer, 0, sizeof(txtbuffer)); for(j = 0; j < 12; j++) { if((buffer[k+5+j] >= 0x20) && (buffer[k+5+j] < 0x7f)) txtbuffer[j] = buffer[k+5+j]; else txtbuffer[j] = 0; }
            printf("Monitor : '%s'\n", txtbuffer);
            break;
          }
        }
      }
    } else if(crc == -1) {
      printf("NVRAM Magic ID not found\n");
    } else {
      printf("NVRAM CRC wrong\n");
    }
  }

  if(res != SV_OK) {
    jpeg_errorcode(sv, res, argc, argv);
  }
}



void jpeg_pulldown(sv_handle * sv, int argc, char ** argv)
{
  int  res  = SV_OK;
  int  start;

  if((strcmp(argv[0], "startphase") == 0) && (argc==2)) { 
    int pd32 = (strlen(argv[1])==1) ? 1 : 0;
    switch (*argv[1]) {
     case 'a':
      start = pd32 ? SV_PULLDOWN_STARTPHASE_A : SV_PULLDOWN_STARTPHASE_A23;
      break;
     case 'b':
      start = pd32 ? SV_PULLDOWN_STARTPHASE_B : SV_PULLDOWN_STARTPHASE_B23;
      break;
     case 'c':
      start = pd32 ? SV_PULLDOWN_STARTPHASE_C : SV_PULLDOWN_STARTPHASE_C23;
      break;
     case 'd':
      start = pd32 ? SV_PULLDOWN_STARTPHASE_D : SV_PULLDOWN_STARTPHASE_D23;
      break;
     default:
      jpeg_errorprintf(sv, "illegal start phase definition");
      return;
    } 
    res = sv_pulldown(sv, SV_PULLDOWN_CMD_STARTPHASE, start);
  } else if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
    printf("sv pulldown help\n");
    printf("\tstartphase\t\tset current pulldown start phase (a,b,c or d)\n");
    return;
  } else {
    jpeg_errorprintf(sv, "sv partition: Unknown pulldown command: %s\n", argv[0]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_wordclock(sv_handle * sv, int argc, char ** argv)
{
  int  res   = SV_OK;
  int  ok    = TRUE;
  int  value = 0;

  if(argc == 2) {
    if(!strcmp(argv[1], "off")) { 
      value = SV_WORDCLOCK_OFF;
    } else if(!strcmp(argv[1], "on")) { 
      value = SV_WORDCLOCK_ON;
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_query(sv, SV_QUERY_WORDCLOCK, 0, &value);
      if(res == SV_OK) {
        if((value >= 0) && (value < arraysize(str_wordclock))) {
          printf("sv wordclock %s\n", str_wordclock[value]);
        } else {
          printf("sv wordclock unknown=%d\n", value);
        }
        return;
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\toff\tTurn wordclock output off\n");
      printf("\ton\tTurn wordclock output on\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv wordclock: Unknown command: %s\n", argv[0]);
    return;
  }

  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_WORDCLOCK, value);
  } 

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_autopulldown(sv_handle * sv, int argc, char ** argv)
{
  int  res   = SV_OK;
  int  ok    = TRUE;
  int  value = 0;

  if(argc == 2) {
    if(!strcmp(argv[1], "off")) { 
      value = FALSE;
    } else if(!strcmp(argv[1], "on")) { 
      value = TRUE;
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_query(sv, SV_QUERY_AUTOPULLDOWN, 0, &value);
      if(res == SV_OK) {
        printf("sv autopulldown %s\n", value?"on":"off");
      }
      return;
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\t\tAutopulldown converts a 24 hz video raster to the current video mode\n");
      printf("\t\tthis only works in a 24 Hz storage mode when the videomode has been\n");
      printf("\t\tchanged to another not 24 hz mode.\n");
      printf("\toff\tTurn autopulldown on\n");
      printf("\ton\tTurn autopulldown on\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv autopulldown: Unknown command: %s\n", argv[0]);
    return;
  }

  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_AUTOPULLDOWN, value);
  } 

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_disableswitchingline(sv_handle * sv, int argc, char ** argv)
{ 
  int res   = SV_OK;
  int state = 0;
  
  if (argc < 1) {
    res = SV_ERROR_PARAMETER;
  }
  
  if(res == SV_OK) {
    if(!strcmp(argv[0], "info")) {
      res = sv_option_get(sv, SV_OPTION_DISABLESWITCHINGLINE, &state);
      if(res == SV_OK) {
        printf("DISABLESWITCHINGLINE %s\n", state?"on":"off");
      } 
    } else if(!strcmp(argv[0], "on")) {
      state = TRUE;
    } else if(!strcmp(argv[0], "off")) {
      state = FALSE;
    } else if(!strcmp(argv[0], "help")) {
      printf("sv disableswitchingline help\n");
      printf("\tinfo\n");
      printf("\ton\n");
      printf("\toff (default)\n");
    }
  }

  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_DISABLESWITCHINGLINE, state);
  }

  if(res != SV_OK) {
    sv_errorprint(sv, res);
  }
}
 

void jpeg_ltc(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;
  int mode, offset, tmp, dropframe;
  int option = -1;
  char buf[16];

  if(argc < 1) { 
    res = SV_ERROR_PARAMETER;
  }

  if(res == SV_OK) {
    if(!strcmp(argv[0], "source") && (argc >= 2)) {
      if(!strcmp(argv[1], "disk") || !strcmp(argv[1], "default")) {
        mode = SV_LTCSOURCE_DEFAULT;
      } else if(!strcmp(argv[1], "intern")) {
        mode = SV_LTCSOURCE_INTERN;
      } else if(!strcmp(argv[1], "playlist")) {
        mode = SV_LTCSOURCE_PLAYLIST;
      } else if(!strcmp(argv[1], "master")) {
        mode = SV_LTCSOURCE_MASTER;
      } else if(!strcmp(argv[1], "freerunning")) {
        mode = SV_LTCSOURCE_FREERUNNING;
      } else if(!strcmp(argv[1], "ltcoffset")) {
        mode = SV_LTCSOURCE_LTCOFFSET;
      } else if(!strcmp(argv[1], "proxy")) {
        mode = SV_LTCSOURCE_PROXY;
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if(argc > 2) {
        if((!strcmp(argv[2], "vcount")) || (!strcmp(argv[2], "playlist"))) {
          mode |= SV_LTCSOURCE_VCOUNT;
        }
      }
      option = SV_OPTION_LTCSOURCE;
    } else if(!strcmp(argv[0], "offset") && (argc >= 2)) {
      if (argv[1][0] == '-') {
        mode = 0;
      } else {        
        res = sv_asc2tc(sv, argv[1], &mode);
      }
      option = SV_OPTION_LTCOFFSET;
    } else if(!strcmp(argv[0], "info")) {
      res = sv_query(sv, SV_QUERY_LTCSOURCE, 0, &mode); 
      tmp = mode & SV_LTCSOURCE_MASK;
      if((tmp >= 0) && (tmp < arraysize(str_ltcsource))) {
        printf("LTCSOURCE %s\n", str_ltcsource[tmp]);
      }
      res = sv_query(sv, SV_QUERY_LTCSOURCE_REC, 0, &mode); 
      if((mode >= 0) && (mode < arraysize(str_ltcsource_rec))) {
        printf("LTCRECORDOUTPUT %s\n",str_ltcsource_rec[mode]);
      }
      res = sv_query(sv,SV_QUERY_TCOUT_AUX, 0, &mode); 
      if((mode >= 0) && (mode < arraysize(str_tcout_aux))) {
        printf("TCOUT_AUX %s\n",str_tcout_aux[mode]);
      }      
      res = sv_query(sv,SV_QUERY_LTCOFFSET,0,&offset);
      res = sv_tc2asc(sv, offset, &buf[0], 16);
      printf("LTCOFFSET %s\n", &buf[0]);
      printf("LTCTIMER  %s\n",(mode & SV_LTCSOURCE_VCOUNT) ? "on" : "off");
      res = sv_query(sv,SV_QUERY_LTCDROPFRAME,0,&dropframe);
      printf("LTCDROPFRAME %s\n", (dropframe == SV_LTCDROPFRAME_OFF) ? "off" : (dropframe == SV_LTCDROPFRAME_ON ? "on" : "default"));
      res = sv_option_get(sv, SV_OPTION_LTCFILTER, &mode);
      printf("LTCFILTER %s\n", mode == SV_LTCFILTER_ENABLE ? "enable" : (mode == SV_LTCFILTER_DUPLICATE ? "duplicate" : "disable"));
      res = sv_jack_option_get(sv, 0, SV_OPTION_ASSIGN_LTC, &mode);
      printf("LTC is assigned to %d\n", mode);
      return;
    } else if(!strcmp(argv[0], "filter") && (argc >= 2)) {
      if(!strcmp(argv[1], "enable") || !strcmp(argv[1], "yes")) {
        mode = SV_LTCFILTER_ENABLE;
      } else if(!strcmp(argv[1], "duplicate")) {
        mode = SV_LTCFILTER_DUPLICATE;
      } else {
        mode = SV_LTCFILTER_DISABLE;
      }
      option = SV_OPTION_LTCFILTER;
    } else if(!strcmp(argv[0], "dropframe") && (argc >= 2)) {
      if(!strcmp(argv[1], "off")) {
        mode = SV_LTCDROPFRAME_OFF;
      } else if(!strcmp(argv[1], "on")) {
        mode = SV_LTCDROPFRAME_ON;
      } else {
        mode = SV_LTCDROPFRAME_DEFAULT;
      }
      option = SV_OPTION_LTCDROPFRAME;
    } else if(!strcmp(argv[0], "recordoutput") && (argc >= 2)) {
      if(!strcmp(argv[1], "intern")) {
        mode = SV_LTCSOURCE_REC_INTERN;
      } else if(!strcmp(argv[1], "EE")) {
        mode = SV_LTCSOURCE_REC_EE;
      } else {
        res = SV_ERROR_PARAMETER;
      }
      option = SV_OPTION_LTCSOURCE_REC;
    } else if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("sv ltc help\n");
      printf("\tsource {disk,intern,playlist,master,freerunning,ltcoffset}\n");
      printf("\trecordoutput {intern,EE}\n");
      printf("\toutaux {intern, Off}\n");
      printf("\toffset tc\n");
      printf("\tfilter {enabled,duplicate,disabled}\n");
      printf("\tdropframe {default,on,off}\n");
      return;
    } else {
      res = SV_ERROR_PARAMETER;
    }
  }
  
  if((res == SV_OK) && (option >= 0)) {
    res = sv_option(sv, option, mode);
  }

  if(res != SV_OK) {
    sv_errorprint(sv, res);
  }
}


void jpeg_vgui_feedback(sv_handle * sv, char c)
{
  if (sv->vgui) {
    fprintf(stdout, "%c\n", c);
    fflush(stdout);
  }
}


void jpeg_slaveinfo(sv_handle * sv, int argc, char ** argv)
{
  sv_slaveinfo si;
  int  res  = SV_OK;
  int  ok   = TRUE;
  int  i;
  char buffer[32];

  if(argc >= 2) {
    if(!strcmp(argv[1], "info")) { 
      res = sv_slaveinfo_get(sv, &si, NULL);
      if(res == SV_OK) {
        printf("command_tick:     %d\n", si.command_tick);
        printf("command_cmd:      %03x %d", si.command_cmd, si.command_length);
        for(i = 0; i < si.command_length; i++) {
          printf(" %02x", si.command_data[i]);
        }
        printf("\n");
        printf("command_lost:     %d\n", si.command_lost);
        printf("status_data:      ");
        for(i = 0; i < 16; i++) {
          printf("%02x ", si.status_data[i]);
        }
        printf("\n");
        sv_tc2asc(sv, si.status_timecode, buffer, sizeof(buffer));
        printf("status_timecode:  %s\n", buffer);
        printf("status_userbytes: %d\n", si.status_userbytes);
        printf("status_speed:     %d\n", si.status_speed);
        printf("edit_preset:      %04x\n", si.edit_preset);
        sv_tc2asc(sv, si.edit_inpoint, buffer, sizeof(buffer));
        printf("edit_inpoint:     %s\n", buffer);
        sv_tc2asc(sv, si.edit_outpoint, buffer, sizeof(buffer));
        printf("edit_outpoint:    %s\n", buffer);
        sv_tc2asc(sv, si.edit_ainpoint, buffer, sizeof(buffer));
        printf("edit_ainpoint:    %s\n", buffer);
        sv_tc2asc(sv, si.edit_aoutpoint, buffer, sizeof(buffer));
        printf("edit_aoutpoint:   %s\n", buffer);
        printf("setup_devicetype: %04x\n", si.setup_devicetype);
        printf("setup_preroll:    %d\n", si.setup_preroll);
        printf("setup_postroll:   %d\n", si.setup_postroll);
        printf("setup_recinhibit: %d\n", si.setup_recinhibit);
        printf("setup_timermode:  %d\n", si.setup_timermode);
        return;
      }
    } else if(!strcmp(argv[1], "dump")) { 
      do {
        res = sv_slaveinfo_get(sv, &si, NULL);
        if((res == SV_OK) && (si.command_cmd)) {
          printf("Command:     %5d %03x %2d", si.command_tick, si.command_cmd, si.command_length);
          for(i = 0; i < si.command_length; i++) {
            printf(" %02x", si.command_data[i]);
          } 
          printf("\n");
          if(si.command_lost) {
            printf("command_lost:     %d\n", si.command_lost);
          }
        }
      } while(res == SV_OK);
    } else if(!strcmp(argv[1], "devicetype") && (argc > 2)) { 
      res = sv_slaveinfo_get(sv, &si, NULL);
      if(res == SV_OK) {
        si.setup_devicetype = strtol(argv[2], NULL, 16);
        res = sv_slaveinfo_set(sv, &si);
      }
    } else if(!strcmp(argv[1], "timecode") && (argc > 2)) { 
      res = sv_slaveinfo_get(sv, &si, NULL);
      if(res == SV_OK) {
        res = sv_asc2tc(sv, argv[2], &si.status_timecode);
      } 
      if(res == SV_OK) {
        si.flags |= SV_SLAVEINFO_USESTATUSANDTC;
        res = sv_slaveinfo_set(sv, &si);
      }
    } else if(!strcmp(argv[1], "default")) { 
      res = sv_slaveinfo_get(sv, &si, NULL);
      if(res == SV_OK) {
        si.flags &= ~SV_SLAVEINFO_USESTATUSANDTC;
        res = sv_slaveinfo_set(sv, &si);
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\tinfo\t\t\tDump structure\n");
      printf("\tdevicetype 1234\t\tSet devicetype (16 bit hex)\n");
      printf("\ttimecode 01:02:03:04\tSet timecode\n");
      printf("\tdefault\t\t\tUse tc/ub/info from frame/fifoapi\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv slaveinfo: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}



void jpeg_matrix(sv_handle * sv, int argc, char ** argv)
{
  sv_matrixinfo matrix = { 0 };
  sv_matrixexinfo matrixex = { 0 };
  int  matrixmode = -1;
  int  res  = SV_OK;
  int  ok   = TRUE;
  int  i,j,count;
  char * txt;
  char * filename = "<nofile>";
  int  path;
  char buffer[256];
  int  bfloat = FALSE;
  int  divisor = 0x10000;
  int  bmatrixex = FALSE;
  int  argcount = 2;

  if(argc >= 2) {
    if(!strcmp(argv[1], "info")) { 
      res = sv_matrix(sv, SV_MATRIX_QUERY, &matrix);
      if(res == SV_OK) {
        switch(matrix.matrixmode & SV_MATRIX_MASK) {
        case SV_MATRIX_DEFAULT:
          txt = "default";
          break;
        case SV_MATRIX_CUSTOM:
          if(matrix.matrixmode & SV_MATRIX_FLAG_CGRMATRIX) {
            txt = "custom cgr";
          } else {
            txt = "custom";
          }
          break;
        case SV_MATRIX_CCIR601:
          txt = "ccir601";
          break;
        case SV_MATRIX_CCIR601CGR:
          txt = "ccir601cgr";
          break;
        case SV_MATRIX_CCIR709:
          txt = "ccir709";
          break;
        case SV_MATRIX_CCIR709CGR:
          txt = "ccir709cgr";
          break;
        case SV_MATRIX_SMPTE274:
          txt = "smpte274";
          break;
        case SV_MATRIX_SMPTE274CGR:
          txt = "smpte274cgr";
          break;
        case SV_MATRIX_CCIR601INV:
          txt = "ccir601inv";
          break;
        case SV_MATRIX_SMPTE274INV:
          txt = "smpte274inv";
          break;
        case SV_MATRIX_IDENTITY:
          txt = "identity";
          break;
        case SV_MATRIX_RGBHEAD2FULL:
          txt = "rgbhead2full";
          break;
        case SV_MATRIX_RGBFULL2HEAD:
          txt = "rgbfull2head";
          break;
        case SV_MATRIX_YUVHEAD2FULL:
          txt = "yuvhead2full";
          break;
        case SV_MATRIX_YUVFULL2HEAD:
          txt = "yuvfull2head";
          break;
        case SV_MATRIX_601TO274:
          txt = "601to274";
          break;
        case SV_MATRIX_601FTO274H:
          txt = "601fto274h";
          break;
        case SV_MATRIX_601HTO274F:
          txt = "601hto274f";
          break;
        case SV_MATRIX_274TO601:
          txt = "274to601";
          break;
        case SV_MATRIX_274FTO601H:
          txt = "274fto601h";
          break;
        case SV_MATRIX_274HTO601F:
          txt = "274hto601f";
          break;

        default:
          txt = "unknown";
        }
        printf("sv matrix %s%s\n", txt, (matrix.matrixmode & SV_MATRIX_FLAG_FORCEMATRIX)?" forcematrix":"");
      }
      if(res == SV_OK) {
        if(matrix.divisor != 0) {
          printf("matrix:\n");
          for(i = 0; i < 3; i++) {
            for(j = 0; j < 3; j++) {
              printf("% 1.5f ", ((double)matrix.matrix[i*3+j]/matrix.divisor));
            }
            printf("\n");
          } 
          printf("key % 1.5f\n", ((double)matrix.matrix[9]/matrix.divisor));
          printf("dematrix:\n");
          for(i = 0; i < 3; i++) {
            for(j = 0; j < 3; j++) {
              printf("% 1.5f ", ((double)matrix.dematrix[i*3+j]/matrix.divisor));
            }
            printf("\n");
          } 
          printf("key % 1.5f\n", ((double)matrix.dematrix[9]/matrix.divisor));
        } else {
          printf("\tdivisor == 0\n");
        }

        if((matrix.inputfilter >= 0) && (matrix.inputfilter < arraysize(str_inputfilter))) {
          printf("inputfilter  %s\n", str_inputfilter[matrix.inputfilter]);
        } else {
          printf("inputfilter  %d unknown\n", matrix.inputfilter);
        }
        if((matrix.outputfilter >= 0) && (matrix.outputfilter < arraysize(str_outputfilter))) {
          printf("outputfilter %s\n", str_outputfilter[matrix.outputfilter]);
        } else {
          printf("outputfilter %d unknown\n", matrix.outputfilter);
        }
      }
    } else if(!strcmp(argv[1], "infoex")) { 
      res = sv_matrixex(sv, SV_MATRIX_QUERY, NULL, &matrixex);
      if(res == SV_OK) {
        if(matrixex.divisor != 0) {
          printf("matrix:\n");
          for(i = 0; i < 3; i++) {
            for(j = 0; j < 3; j++) {
              printf("% 1.5f ", ((double)matrixex.matrix[i*3+j]/matrixex.divisor));
            }
            printf("\n");
          } 
          printf("key % 1.5f\n", ((double)matrixex.matrix[9]/matrixex.divisor));
          printf("inoffset  % 1.5f % 1.5f % 1.5f % 1.5f\n",
            ((double)matrixex.matrix[10]/matrixex.divisor),
            ((double)matrixex.matrix[11]/matrixex.divisor),
            ((double)matrixex.matrix[12]/matrixex.divisor),
            ((double)matrixex.matrix[13]/matrixex.divisor)
          );
          printf("outoffset % 1.5f % 1.5f % 1.5f % 1.5f\n",
            ((double)matrixex.matrix[14]/matrixex.divisor),
            ((double)matrixex.matrix[15]/matrixex.divisor),
            ((double)matrixex.matrix[16]/matrixex.divisor),
            ((double)matrixex.matrix[17]/matrixex.divisor)
          );
          printf("dematrix:\n");
          for(i = 0; i < 3; i++) {
            for(j = 0; j < 3; j++) {
              printf("% 1.5f ", ((double)matrixex.dematrix[i*3+j]/matrixex.divisor));
            }
            printf("\n");
          } 
          printf("key % 1.5f\n", ((double)matrixex.dematrix[9]/matrixex.divisor));
          printf("inoffset  % 1.5f % 1.5f % 1.5f % 1.5f\n",
            ((double)matrixex.dematrix[10]/matrixex.divisor),
            ((double)matrixex.dematrix[11]/matrixex.divisor),
            ((double)matrixex.dematrix[12]/matrixex.divisor),
            ((double)matrixex.dematrix[13]/matrixex.divisor)
          );
          printf("outoffset % 1.5f % 1.5f % 1.5f % 1.5f\n",
            ((double)matrixex.dematrix[14]/matrixex.divisor),
            ((double)matrixex.dematrix[15]/matrixex.divisor),
            ((double)matrixex.dematrix[16]/matrixex.divisor),
            ((double)matrixex.dematrix[17]/matrixex.divisor)
          );
        } else {
          printf("\tdivisor == 0\n");
        }
      }
    } else if(!strcmp(argv[1], "custom")) { 
      matrixmode = SV_MATRIX_CUSTOM;
      if(argc >= 3) {
        filename = argv[2];
        argcount = 3;
      } else {
        res = SV_ERROR_PARAMETER;
      }
    } else if(!strcmp(argv[1], "customex")) { 
      bmatrixex = TRUE;
      matrixmode = SV_MATRIX_CUSTOM;
      if(argc >= 3) {
        filename = argv[2];
        argcount = 3;
      } else {
        res = SV_ERROR_PARAMETER;
      }
    } else if(!strcmp(argv[1], "ccir601")) { 
      matrixmode = SV_MATRIX_CCIR601;
    } else if(!strcmp(argv[1], "ccir601cgr")) { 
      matrixmode = SV_MATRIX_CCIR601CGR;
    } else if(!strcmp(argv[1], "ccir709")) { 
      matrixmode = SV_MATRIX_CCIR709;
    } else if(!strcmp(argv[1], "ccir709cgr")) {
      matrixmode = SV_MATRIX_CCIR709CGR;
    } else if(!strcmp(argv[1], "smpte274")) { 
      matrixmode = SV_MATRIX_SMPTE274;
    } else if(!strcmp(argv[1], "smpte274cgr")) { 
      matrixmode = SV_MATRIX_SMPTE274CGR;
    } else if(!strcmp(argv[1], "ccir601inv")) {
      matrixmode = SV_MATRIX_CCIR601INV;
    } else if(!strcmp(argv[1], "smpte274inv")) {
      matrixmode = SV_MATRIX_SMPTE274INV;
    } else if(!strcmp(argv[1], "identity")) {
      matrixmode = SV_MATRIX_IDENTITY;
    } else if(!strcmp(argv[1], "rgbhead2full")) {
      matrixmode = SV_MATRIX_RGBHEAD2FULL;
    } else if(!strcmp(argv[1], "rgbfull2head")) {
      matrixmode = SV_MATRIX_RGBFULL2HEAD;
    } else if(!strcmp(argv[1], "yuvhead2full")) {
      matrixmode = SV_MATRIX_YUVHEAD2FULL;
    } else if(!strcmp(argv[1], "yuvfull2head")) {
      matrixmode = SV_MATRIX_YUVFULL2HEAD;
    } else if(!strcmp(argv[1], "601to274")) {
      matrixmode = SV_MATRIX_601TO274;
    } else if(!strcmp(argv[1], "601fto274h")) {
      matrixmode = SV_MATRIX_601FTO274H;
    } else if(!strcmp(argv[1], "601hto274f")) {
      matrixmode = SV_MATRIX_601HTO274F;
    } else if(!strcmp(argv[1], "274to601")) {
      matrixmode = SV_MATRIX_274TO601;
    } else if(!strcmp(argv[1], "274fto601h")) {
      matrixmode = SV_MATRIX_274FTO601H;
    } else if(!strcmp(argv[1], "274hto601f")) {
      matrixmode = SV_MATRIX_274HTO601F;
    } else if(!strcmp(argv[1], "default")) { 
      matrixmode = SV_MATRIX_DEFAULT;
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\tinfo\t\tDump structure\n");
      printf("\tccir601\t\tSet ccir601 matrix\n");
      printf("\tccir601cgr\tSet ccir601cgr matrix\n");
      printf("\tccir709\t\tSet ccir709 matrix\n");
      printf("\tccir709cgr\tSet ccir709cgr matrix\n");
      printf("\tsmpte274\tSet smpte274 matrix\n");
      printf("\tsmpte274cgr\tSet smpte274cgr matrix\n");
      printf("\tccir601inv\tSet ccir601inv matrix\n");
      printf("\tsmpte274inv\tSet smpte274inv matrix\n");
      printf("\tidentity\tSet identity matrix\n");
      printf("\trgbhead2full\tSet rgb head2full matrix\n");
      printf("\trgbfull2head\tSet rgb full2head matrix\n");
      printf("\tyuvhead2full\tSet yuv head2full matrix\n");
      printf("\tyuvfull2head\tSet yuv full2head matrix\n");
      printf("\t601to274\tSet 601to274 matrix\n");
      printf("\t601fto274h\tSet 601fto274h (full2head) matrix\n");
      printf("\t601hto274f\tSet 601hto274f (head2full) matrix\n");
      printf("\t274to601\tSet 274to601 matrix\n");
      printf("\t274fto601h\tSet 274fto601h (full2head) matrix\n");
      printf("\t274hto601f\tSet 274hto601f (head2full) matrix\n");
      printf("\tcustom <filename>\t\t\tSet custom matrix from file\n");
      printf("\t\t#MATRIX\n");
      printf("\t\tdivisor\n");
      printf("\t\tMAT11 MAT12 MAT13\n");
      printf("\t\tMAT21 MAT22 MAT23\n");
      printf("\t\tMAT31 MAT32 MAT33\n");
      printf("\t\tMATKey\n");
      printf("\t\tDEM11 DEM12 DEM13\n");
      printf("\t\tDEM21 DEM22 DEM23\n");
      printf("\t\tDEM31 DEM32 DEM33\n");
      printf("\t\tDEMKey\n");
      printf("flags:\n");
      printf("\tinputfilter {default,5taps,9taps,13taps,17taps,nofilter}\tSet input filtering\n");
      printf("\toutputfilter {default,5taps,9taps,13taps,17taps,nofilter}\tSet output filtering\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
    argcount = 0;
  }

  if((matrixmode != -1) && (argc >= 3)) {
    i = argcount;
    while(ok && (argc - i >= 1)) {
      if(!strcmp(argv[i], "forcematrix")) {
        matrixmode |= SV_MATRIX_FLAG_FORCEMATRIX;
      } else if(!strcmp(argv[i], "inputfilter") && (argc - i >= 2)) {
        i++;
        if(!strcmp(argv[i], "default")) {
          matrix.inputfilter = SV_INPUTFILTER_DEFAULT;
          matrixmode         |= SV_MATRIX_FLAG_SETINPUTFILTER;
        } else if(!strcmp(argv[i], "nofilter")) {
          matrix.inputfilter = SV_INPUTFILTER_NOFILTER;
          matrixmode         |= SV_MATRIX_FLAG_SETINPUTFILTER;
        } else if(!strcmp(argv[i], "5taps")) {
          matrix.inputfilter = SV_INPUTFILTER_5TAPS;
          matrixmode         |= SV_MATRIX_FLAG_SETINPUTFILTER;
        } else if(!strcmp(argv[i], "9taps")) {
          matrix.inputfilter = SV_INPUTFILTER_9TAPS;
          matrixmode         |= SV_MATRIX_FLAG_SETINPUTFILTER;
        } else if(!strcmp(argv[i], "13taps")) {
          matrix.inputfilter = SV_INPUTFILTER_13TAPS;
          matrixmode         |= SV_MATRIX_FLAG_SETINPUTFILTER;
        } else if(!strcmp(argv[i], "17taps")) {
          matrix.inputfilter = SV_INPUTFILTER_17TAPS;
          matrixmode         |= SV_MATRIX_FLAG_SETINPUTFILTER;
        } else {
          ok = FALSE;
        }
      } else if(!strcmp(argv[i], "outputfilter") && (argc - i >= 2)) {
        i++;
        if(!strcmp(argv[i], "default")) {
          matrix.outputfilter = SV_OUTPUTFILTER_DEFAULT;
          matrixmode          |= SV_MATRIX_FLAG_SETOUTPUTFILTER;
        } else if(!strcmp(argv[i], "nofilter")) {
          matrix.outputfilter = SV_OUTPUTFILTER_NOFILTER;
          matrixmode          |= SV_MATRIX_FLAG_SETOUTPUTFILTER;
        } else if(!strcmp(argv[i], "5taps")) {
          matrix.outputfilter = SV_OUTPUTFILTER_5TAPS;
          matrixmode          |= SV_MATRIX_FLAG_SETOUTPUTFILTER;
        } else if(!strcmp(argv[i], "9taps")) {
          matrix.outputfilter = SV_OUTPUTFILTER_9TAPS;
          matrixmode          |= SV_MATRIX_FLAG_SETOUTPUTFILTER;
        } else if(!strcmp(argv[i], "13taps")) {
          matrix.outputfilter = SV_OUTPUTFILTER_13TAPS;
          matrixmode          |= SV_MATRIX_FLAG_SETOUTPUTFILTER;
        } else if(!strcmp(argv[i], "17taps")) {
          matrix.outputfilter = SV_OUTPUTFILTER_17TAPS;
          matrixmode          |= SV_MATRIX_FLAG_SETOUTPUTFILTER;
        } else {
          ok = FALSE;
        }
      } else {
        ok = FALSE;
      }
      i++;
    }
    if(!ok) {
      jpeg_errorprintf(sv, "sv matrix: Unknown command: %s\n", argv[i-1]);
      return;
    }
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv matrix: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res == SV_OK) {
    if((matrixmode & SV_MATRIX_MASK) == SV_MATRIX_CUSTOM) {   
      
      path = open(filename, O_RDONLY);

      if(path < 0) {
        res = SV_ERROR_FILEOPEN;
      }

      if(res == SV_OK) {
        jpeg_read_line(path, buffer);
        if(strncmp(buffer, "#MATRIX", 7)) {
          fprintf(stderr, "matrix file '%s' does not start with #MATRIX\n", argv[1]);
          res = SV_ERROR_PARAMETER;
        }
      }

      if(res == SV_OK) {
        do {
          count = jpeg_read_line(path, buffer);
        } while(buffer[0] == '#');
      }

      if(res == SV_OK) {
        memset(&matrix, 0, sizeof(matrix));
        memset(&matrixex, 0, sizeof(matrixex));

        if(strchr(buffer, '.')) {
          matrix.divisor = 0x10000 * atof(buffer);
          bfloat = TRUE;
          if(matrix.divisor > 0x10000) {
            divisor = (int)((double)0x10000 / matrix.divisor);
            matrix.divisor = 0x10000;
          } else {
            divisor = matrix.divisor;
          }
        } else {
          divisor = matrix.divisor = atoi(buffer);
        }
      }

      for(j = 0; (res == SV_OK) && (j < 9); j+=3) {
        count = jpeg_read_line(path, buffer);
        for(i = 0; count && buffer[i] && ((buffer[i] == ' ') || (buffer[i] == '\t')); i++);
        if(bfloat) {
          matrix.matrix[j] = (int)(atof(&buffer[i]) * divisor);
        } else {
          matrix.matrix[j] = atoi(&buffer[i]);
        }
        for(; count && buffer[i] && (buffer[i] != ' ') && (buffer[i] != '\t'); i++);
        for(; count && buffer[i] && ((buffer[i] == ' ') || (buffer[i] == '\t')); i++);
        if(buffer[i]) {
          if(bfloat) {
            matrix.matrix[j+1] = (int)(atof(&buffer[i]) * divisor);
          } else {
            matrix.matrix[j+1] = atoi(&buffer[i]);
          }
          for(; count && buffer[i] && (buffer[i] != ' ') && (buffer[i] != '\t'); i++);
          for(; count && buffer[i] && ((buffer[i] == ' ') || (buffer[i] == '\t')); i++);
          if(buffer[i]) {
            if(bfloat) {
              matrix.matrix[j+2] = (int)(atof(&buffer[i]) * divisor);
            } else {
              matrix.matrix[j+2] = atoi(&buffer[i]);
            }
          } else {
            res = SV_ERROR_PARAMETER;
          }
        } else {
          res = SV_ERROR_PARAMETER;
        }
      }
      if(res == SV_OK) {
        count = jpeg_read_line(path, buffer);
        if(bfloat) {
          matrix.matrix[9] = (int)(atof(&buffer[0]) * divisor);
        } else {
          matrix.matrix[9] = atoi(&buffer[0]);
        }
      }
      for(j = 10 ; bmatrixex && (res == SV_OK) && (j < 18); j++) {
        count = jpeg_read_line(path, buffer);
        if(bfloat) {
          matrixex.matrix[j] = (int)(atof(&buffer[0]) * divisor);
        } else {
          matrixex.matrix[j] = atoi(&buffer[0]);
        }
      }

      for(i = j = 0; (res == SV_OK) && (j < 9); j+=3) {
        count = jpeg_read_line(path, buffer);
        for(i = 0; count && buffer[i] && ((buffer[i] == ' ') || (buffer[i] == '\t')); i++);
        if(bfloat) {
          matrix.dematrix[j] = (int)(atof(&buffer[i]) * divisor);
        } else {
          matrix.dematrix[j] = atoi(&buffer[i]);
        }
        for(; count && buffer[i] && (buffer[i] != ' ') && (buffer[i] != '\t'); i++);
        for(; count && buffer[i] && ((buffer[i] == ' ') || (buffer[i] == '\t')); i++);
        if(buffer[i]) {
          if(bfloat) {
            matrix.dematrix[j+1] = (int)(atof(&buffer[i]) * divisor);
          } else {
            matrix.dematrix[j+1] = atoi(&buffer[i]);
          }
          for(; count && buffer[i] && (buffer[i] != ' ') && (buffer[i] != '\t'); i++);
          for(; count && buffer[i] && ((buffer[i] == ' ') || (buffer[i] == '\t')); i++);
          if(buffer[i]) {
            if(bfloat) {
              matrix.dematrix[j+2] = (int)(atof(&buffer[i]) * divisor);
            } else {
              matrix.dematrix[j+2] = atoi(&buffer[i]);
            }
          } else {
            res = SV_ERROR_PARAMETER;
          }
        } else {
          res = SV_ERROR_PARAMETER;
        }
      }

      if(res == SV_OK) {
        count = jpeg_read_line(path, buffer);
        if(bfloat) {
          matrix.dematrix[9] = (int)(atof(&buffer[0]) * divisor);
        } else {
          matrix.dematrix[9] = atoi(&buffer[0]);
        }
      }
      for(j = 10 ; bmatrixex && (res == SV_OK) && (j < 18); j++) {
        count = jpeg_read_line(path, buffer);
        if(bfloat) {
          matrixex.dematrix[j] = (int)(atof(&buffer[0]) * divisor);
        } else {
          matrixex.dematrix[j] = atoi(&buffer[0]);
        }
      }

      if(res == SV_OK) {
        if(bmatrixex) {
          matrixex.divisor = matrix.divisor;
          memcpy(&matrixex.matrix, &matrix.matrix, 10 * sizeof(int));
          memcpy(&matrixex.dematrix, &matrix.dematrix, 10 * sizeof(int));

          res = sv_matrixex(sv, matrixmode, &matrixex, &matrixex);
        } else {
          res = sv_matrix(sv, matrixmode, &matrix);
        }
      }

      if(path >= 0) {
        close(path);
      }
    } else if(matrixmode != -1) {
      res = sv_matrix(sv, matrixmode, (matrixmode&SV_MATRIX_FLAG_MASK)?&matrix:NULL);
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_recordmode(sv_handle * sv, int argc, char ** argv)
{
  int  res  = SV_OK;
  int  ok   = TRUE;
  int  val;

  if(argc >= 2) {
    if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_RECORDMODE, &val);
      if(res == SV_OK) {
        if((val >= 0) && (val < arraysize(str_multidevice))) {
          printf("sv %s %s\n", argv[0], str_recordmode[val]);
        } else {
          printf("sv %s %d\n", argv[0], val);
        }
      }
    } else if(!strcmp(argv[1], "normal")) { 
      sv_option_set(sv, SV_OPTION_RECORDMODE, SV_RECORDMODE_NORMAL);
    } else if(!strcmp(argv[1], "gpi")) { 
      sv_option_set(sv, SV_OPTION_RECORDMODE, SV_RECORDMODE_GPI_CONTROLLED);
    } else if(!strcmp(argv[1], "variframe")) { 
      sv_option_set(sv, SV_OPTION_RECORDMODE, SV_RECORDMODE_VARIFRAME_CONTROLLED);
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\tinfo\t\tShow current setting\n");
      printf("\tnormal\t\tNormal record mode\n");
      printf("\tgpi\t\tRecord controlled by gpi input\n");
      printf("\tvariframe\t\tRecord controlled by variframe input\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv %s: Unknown command: %s\n", argv[0], argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_anccomplete(sv_handle * sv, int argc, char ** argv)
{
  int  res  = SV_OK;
  int  ok   = TRUE;
  int  value;
  int  force_switchingline = 0;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off") || !strcmp(argv[1], "disable")) { 
      res = sv_option_set(sv, SV_OPTION_ANCCOMPLETE, SV_ANCCOMPLETE_OFF);
    } else if(!strcmp(argv[1], "on")) { 
      res = sv_option_set(sv, SV_OPTION_ANCCOMPLETE, SV_ANCCOMPLETE_ON);
    } else if(!strcmp(argv[1], "streamer")) {
      if(argc >= 3){
        if(!strcmp(argv[2], "forcesl")) {
          force_switchingline = SV_ANCCOMPLETE_FLAG_FORCE_SWITCHINGLINE;
        }
      }
      res = sv_option_set(sv, SV_OPTION_ANCCOMPLETE, SV_ANCCOMPLETE_STREAMER | force_switchingline);
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_ANCCOMPLETE, &value);
      if(res == SV_OK) {
        if(value != SV_ANCCOMPLETE_OFF) {
          if(value == SV_ANCCOMPLETE_STREAMER) {
            printf("sv anccomplete streamer\n");
          } else {
            printf("sv anccomplete on\n");
          }
        } else {
          printf("sv anccomplete off\n");
        }
        res = sv_query(sv, SV_QUERY_ANC_MINLINENR, 0, &value);
      }
      if(res == SV_OK) {
        printf("   ancminlinenr  %d\n", value);
        res = sv_query(sv, SV_QUERY_ANC_MAXVANCLINENR, 0, &value);
      }
      if(res == SV_OK) {
        printf("   ancmaxvanc    %d\n", value);
        res = sv_query(sv, SV_QUERY_ANC_MAXHANCLINENR, 0, &value);
      }
      if(res == SV_OK) {
        printf("   ancmaxhanc    %d\n", value);
        return;
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\toff                Disable anc complete\n");
      printf("\ton                 Enable anc complete\n");
      printf("\tstreamer           Enable anc streamer mode\n");
      printf("\tstreamer forcesl   Enables the usage of the switching line in anc streamer mode\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv anccomplete: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_rp165(sv_handle * sv, int argc, char ** argv)
{
  int res = SV_OK;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off") || !strcmp(argv[1], "disable")) { 
      res = sv_option_set(sv, SV_OPTION_ANCGENERATOR_RP165, FALSE);
    } else if(!strcmp(argv[1], "on") || !strcmp(argv[1], "enable")) { 
      res = sv_option_set(sv, SV_OPTION_ANCGENERATOR_RP165, TRUE);
    } else if (!strcmp(argv[1], "info")) {
      int rp165;
      res = sv_option_get(sv, SV_OPTION_ANCGENERATOR_RP165, &rp165);
      if(res == SV_OK) {
        printf("RP165 is %s\n", rp165 ? "on" : "off");
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("svram rp165 on/off/info\n");
    }
  } else {
    printf("svram rp165 on/off/info\n");
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_ancgenerator(sv_handle * sv, int argc, char ** argv)
{
  int  res  = SV_OK;
  int  ok   = TRUE;
  int  value;
  char buffer[256];
  int  i;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off") || !strcmp(argv[1], "disable")) { 
      value = SV_ANCDATA_DISABLE;
    } else if(!strcmp(argv[1], "rp188")) { 
      value = SV_ANCDATA_RP188;
    } else if(!strcmp(argv[1], "rp196")) { 
      value = SV_ANCDATA_RP196;
    } else if(!strcmp(argv[1], "rp196ltc")) { 
      value = SV_ANCDATA_RP196LTC;
    } else if(!strcmp(argv[1], "rp201")) { 
      value = SV_ANCDATA_RP201;
    } else if(!strcmp(argv[1], "rp215")) { 
      value = SV_ANCDATA_RP215;
    } else if(!strcmp(argv[1], "userdef")) { 
      value = SV_ANCDATA_USERDEF;
    } else if(!strcmp(argv[1], "default")) { 
      value = SV_ANCDATA_DEFAULT;
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_ANCGENERATOR, &value);
      if(res == SV_OK) {
        memset(buffer, 0, sizeof(buffer));
        if(value & SV_ANCDATA_FLAG_NOLTC) {
          strcat(buffer, " noltc");
        } 
        if(value & SV_ANCDATA_FLAG_NOAIV) {
          strcat(buffer, " noaiv");
        } 
        if(value & SV_ANCDATA_FLAG_NOSMPTE352) {
          strcat(buffer, " nosmpte352");
        } 
        if(((value & SV_ANCDATA_MASK) >= 0) && ((value & SV_ANCDATA_MASK) < arraysize(str_ancdata))) {
          printf("sv ancgenerator %s%s\n", str_ancdata[(value & SV_ANCDATA_MASK)], buffer);
        } else {
          printf("sv ancgenerator unknown=%d%s\n", (value & SV_ANCDATA_MASK), buffer);
        }
        return;
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\tdisable\tDisable anc output\n");
      printf("\tdefault\tSend out default data\n");
      printf("\tuserdef\tSend out userdefined data\n");
      printf("\trp188\tSend out RP188 timecode\n");
      printf("\trp196\tSend out RP196 timecode\n");
      printf("\trp196ltc\tSend out RP196 LTC timecode\n");
      printf("\trp201\tSend out RP201 filmcode\n");
      printf("\trp215\tSend out RP215 filmcode\n");
      return;
    } else {
      ok = FALSE;
    }

    for(i = 2; argc > i; i++) {
      if(!strcmp(argv[i], "noltc")) {
        value |= SV_ANCDATA_FLAG_NOLTC;
      } else if(!strcmp(argv[i], "noaiv")) {
        value |= SV_ANCDATA_FLAG_NOAIV;
      } else if(!strcmp(argv[i], "nosmpte352")) {
        value |= SV_ANCDATA_FLAG_NOSMPTE352;
      } else {
        ok = FALSE;
      }
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv ancgenerator: Unknown command: %s\n", argv[0]);
    return;
  }

  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_ANCGENERATOR, value);
  } 

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_ancreader(sv_handle * sv, int argc, char ** argv)
{
  int  res   = SV_OK;
  int  ok    = TRUE;
  int  value = 0;

  if(argc == 2) {
    if(!strcmp(argv[1], "off") || !strcmp(argv[1], "disable")) { 
      value = SV_ANCDATA_DISABLE;
    } else if(!strcmp(argv[1], "rp188")) { 
      value = SV_ANCDATA_RP188;
    } else if(!strcmp(argv[1], "rp196")) { 
      value = SV_ANCDATA_RP196;
    } else if(!strcmp(argv[1], "rp196ltc")) { 
      value = SV_ANCDATA_RP196LTC;
    } else if(!strcmp(argv[1], "rp201")) { 
      value = SV_ANCDATA_RP201;
    } else if(!strcmp(argv[1], "userdef")) { 
      value = SV_ANCDATA_USERDEF;
    } else if(!strcmp(argv[1], "default")) { 
      value = SV_ANCDATA_DEFAULT;
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_ANCREADER, &value);
      if(res == SV_OK) {
        if(((value & SV_ANCDATA_MASK) >= 0) && ((value & SV_ANCDATA_MASK) < arraysize(str_ancdata))) {
          printf("sv ancreader %s\n", str_ancdata[(value & SV_ANCDATA_MASK)]);
        } else {
          printf("sv ancreader unknown=%d\n", (value & SV_ANCDATA_MASK));
        }
        return;
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\tdisable\tDisable anc output\n");
      printf("\tdefault\tRead default data\n");
      printf("\tuserdef\tRead userdefined data\n");
      printf("\trp188\tRead RP188 timecode\n");
      printf("\trp196\tRead RP196 timecode\n");
      printf("\trp196ltc\tRead RP196 LTC timecode\n");
      printf("\trp201\tRead RP201 filmcode\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv ancreader: Unknown command: %s\n", argv[0]);
    return;
  }

  if(res == SV_OK) {
    res = sv_option(sv, SV_OPTION_ANCREADER, value);
  } 

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_rs422(sv_handle * sv, int argc, char ** argv)
{
  char buffer[16];
  int  count;
  int  res = SV_OK;
  int  device;
  int  porttype = 0; 
  int  i;

  if(argc >= 1) {
    if(!strcmp(argv[1], "help")) {
      printf("sv rs422 help\n");
      printf("sv rs422 read #port #bytes\n");
      printf("sv rs422 write #port #bytes\n");
      return;
    }
  }

  if(argc >= 3) {
    device = atoi(argv[2]);
    if(!strcmp(argv[1], "port")) {
      if(!strcmp(argv[3], "default")) {
        porttype = SV_PORTTYPE_DEFAULT;
      } else if(!strcmp(argv[3], "mixer")) {
        porttype = SV_PORTTYPE_MIXER;
      } else {
        res = SV_ERROR_PARAMETER;
      }
      if(res == SV_OK) {
        res = sv_rs422_port(sv, device, 38400, porttype);
      }
    } else if(!strcmp(argv[1], "read")) {
      res = sv_rs422_open(sv, device, 38400, 0);
      if(res == SV_OK) {
        do {
          res = sv_rs422_rw(sv, device, FALSE, buffer, sizeof(buffer), &count, 0);
          if(res == SV_OK) {
            if(count) {
              for(i = 0; i < count; i++) {
                printf("%02x ", 0xff & buffer[i]);
              }
              printf("\n");
            } else {
              sv_usleep(sv, 4000);
            }
          }
        } while(res == SV_OK);
      }
      if(res == SV_OK) {
        res = sv_rs422_close(sv, device);
      } else {
        sv_rs422_close(sv, device);
      }
    } else if(!strcmp(argv[1], "write")) {
      res = sv_rs422_open(sv, device, 38400, 0);
      if(res == SV_OK) {
        count = argc - 3;
        for(i = 0; i < count; i++) {
          buffer[i] = strtol(argv[3+i], 0, 16);
        }
        res = sv_rs422_rw(sv, device, TRUE, buffer, count, &count, 0);
      }
      if(res == SV_OK) {
        res = sv_rs422_close(sv, device);
      } else {
        sv_rs422_close(sv, device);
      }
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

      
  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_jack(sv_handle * sv, int argc, char ** argv)
{
  char buffer[64];
  int  cmd     = -1;
  int  res     = SV_OK;
  int  value   = 0;
  int  jack    = 0;
  int  iospeed = SV_IOSPEED_UNKNOWN;
  sv_jack_info info;
  sv_storageinfo storage_info;

  if(argc >= 2) {
    if(!strcmp(argv[1], "help") || !strcmp(argv[1], "?")) {
      printf("sv jack help\n");
      printf("sv jack list\n");
      printf("sv jack #jack assign ltc [info]\n");
      printf("sv jack #jack assign vtr [info]\n");
      printf("sv jack #jack mode #videomode\n");
      printf("sv jack #jack info\n");
      printf("sv jack #jack iomode #iomode\n");
      printf("sv jack #jack mixer {ab|ab_pre|off}\n");
      return;
    } else if(!strcmp(argv[1], "list")) {
      jack = 0;
      printf("Jacks\n");
      do {
        res = sv_jack_find(sv, jack, &buffer[0], sizeof(buffer), NULL);
        if(res == SV_OK) {
          res = sv_jack_status(sv, jack, &info);
          if(res == SV_OK) {
            printf("\t%8s => %s\n", buffer, sv_support_channel2string(info.channel));
          } else {
            printf("\t%8s\n", buffer);
          }
        }
        jack++;
      } while(res == SV_OK);
      return;
    } else {
      res = sv_jack_find(sv, -1, argv[1], 0, &jack);
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if((res == SV_OK) && (argc >= 3)) {
    if(!strcmp(argv[2], "assign") && (argc >= 4)) {
      if(!strcmp(argv[3], "ltc")) {
        cmd   = SV_OPTION_ASSIGN_LTC;
        value = 0;
        if(argc >= 5) {
          if(!strcmp(argv[4], "info")) {
            sv_jack_option_get(sv, 0, SV_OPTION_ASSIGN_LTC, &value);
            printf("LTC is assigned to %d\n", value);
            return;
          }
        }
      } else if(!strcmp(argv[3], "vtr")) {
        cmd   = SV_OPTION_ASSIGN_VTR;
        value = 0;
        if(argc >= 5) {
          if(!strcmp(argv[4], "info")) {
            sv_jack_option_get(sv, 0, SV_OPTION_ASSIGN_VTR, &value);
            printf("VTR is assigned to %d\n", value);
            return;
          }
        }
      } else {
        res = SV_ERROR_PARAMETER;
      }
    } else if(!strcmp(argv[2], "mode") && (argc >= 4)) {
      cmd   = SV_OPTION_VIDEOMODE;
      value = jpeg_support_string2videomode(sv, argv[3]);
    } else if(!strcmp(argv[2], "iomode") && (argc >= 4)) {
      cmd   = SV_OPTION_IOMODE;
      value = sv_support_string2iomode(argv[3], 0);
    } else if(!strcmp(argv[2], "mixer") && (argc >= 4)) {
      cmd   = SV_OPTION_ALPHAMIXER;
      if(!strcmp(argv[3], "ab")) {
        value = SV_ALPHAMIXER_AB;
      } else if(!strcmp(argv[3], "ab_pre")) {
        value = SV_ALPHAMIXER_AB_PREMULTIPLIED;
      } else if(!strcmp(argv[3], "ba")) {
        value = SV_ALPHAMIXER_BA;
      } else if(!strcmp(argv[3], "ba_pre")) {
        value = SV_ALPHAMIXER_BA_PREMULTIPLIED;
      } else {
        value = SV_ALPHAMIXER_OFF;
      }
    } else if(!strcmp(argv[2], "info")) {
      cmd   = -2;
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else if(res == SV_OK) {
    res = SV_ERROR_PARAMETER;
  }

  if(res == SV_OK) {
    if(cmd == -2) {
      res = sv_jack_option_get(sv, jack, SV_OPTION_VIDEOMODE, &value);
      if(res == SV_OK) {
        res = jpeg_support_videomode2string(sv, buffer, value);
        if(res == SV_OK) {
          printf("Video mode      : %s\n", buffer);
        }
      }
      res = sv_jack_option_get(sv, jack, SV_OPTION_IOMODE, &value);
      if(res == SV_OK) {
        sv_support_iomode2string(value, buffer, sizeof(buffer));
        printf("Video iomode    : %s\n", buffer);
      }

      res = sv_storage_status( sv, jack, NULL, &storage_info, sizeof(storage_info), SV_STORAGEINFO_COOKIEISJACK);
      if(res == SV_OK){
        // Video raster
        if((storage_info.storagexsize != storage_info.xsize) || (storage_info.storageysize != storage_info.ysize)) {
          printf("Video raster    : %dx%d (storage:%dx%d)\n", storage_info.xsize, storage_info.ysize, storage_info.storagexsize, storage_info.storageysize);
        } else {
          printf("Video raster    : %dx%d\n", storage_info.xsize, storage_info.ysize);
        }
        printf("Buffersize      : %d bytes\n", storage_info.buffersize);
      }

      res = sv_jack_status(sv, jack, &info);
      if(res == SV_OK) {
        if(info.card2host) {
          int audioinput = 0;
          int tmp;

          // Check video input
          res = sv_jack_query(sv, jack, SV_QUERY_INPUTRASTER, 0, &value);
          if(sv_jack_query(sv, jack, SV_QUERY_IOSPEED, 0, &tmp) == SV_OK) {
            iospeed = tmp;
          }
          if(res == SV_OK) {
            printf("\nInput Raster    : %s %s\n", sv_support_videomode2string(value), sv_option_value2string(sv, SV_OPTION_IOSPEED, iospeed));
          }
          res = sv_jack_query(sv, jack, SV_QUERY_VIDEOINERROR, 0, &value);
          if(res == SV_OK) {
            printf("VideoInError    : %s\n", sv_geterrortext(value));
          }

          // Check audio input
          res = sv_jack_query(sv, jack, SV_QUERY_AUDIOINPUT, 0, &audioinput);
          if(res == SV_OK) {
            printf("\nAudio Input     : %s\n", audioinput==SV_AUDIOINPUT_AIV ? "AIV" : "AES" );
          }
          res = sv_jack_query(sv, jack, SV_QUERY_AUDIOINERROR, 0, &value);
          if(res == SV_OK) {
            printf("AudioInError    : %s\n", sv_geterrortext(value));
          }
          res = sv_jack_query(sv, jack, SV_QUERY_AUDIO_AIVCHANNELS, 0, &value);
          if(res == SV_OK) {
            printf("%sChannels     : 0x%08x\n", "AIV", value);
          }
          res = sv_jack_query(sv, jack, SV_QUERY_AUDIO_AESCHANNELS, 0, &value);
          if(res == SV_OK) {
            printf("%sChannels     : 0x%08x\n", "AES", value);
          }
          res = sv_jack_query(sv, jack, SV_QUERY_VALIDTIMECODE, 0, &value);
          if(res == SV_OK) {
            printf("Timecodes       : %s%s%s%s%s%s%s%s%s%s%s%s%s%s\n", 
              (value & SV_VALIDTIMECODE_VTR)?"VTR ":"",
              (value & SV_VALIDTIMECODE_DLTC)?"DLTC ":"",
              (value & SV_VALIDTIMECODE_LTC)?"LTC ":"",
              (value & SV_VALIDTIMECODE_RP215)?"RP215 ":"",
              (value & SV_VALIDTIMECODE_VITC_F1)?"VITC/F1 ":"",
              (value & SV_VALIDTIMECODE_DVITC_F1)?"DVITC/F1 ":"",
              (value & SV_VALIDTIMECODE_RP201_F1)?"RP201/F1 ":"",
              (value & SV_VALIDTIMECODE_CC_F1)?"CC/F1 ":"",
              (value & SV_VALIDTIMECODE_ARP201_F1)?"ARP201/F1 ":"",
              (value & SV_VALIDTIMECODE_VITC_F2)?"VITC/F2 ":"",
              (value & SV_VALIDTIMECODE_DVITC_F2)?"DVITC/F2 ":"",
              (value & SV_VALIDTIMECODE_RP201_F2)?"RP201/F2 ":"",
              (value & SV_VALIDTIMECODE_CC_F2)?"CC/F2 ":"",
              (value & SV_VALIDTIMECODE_ARP201_F2)?"ARP201/F2 ":"");
          }
        } else {
          res = sv_jack_option_get(sv, jack, SV_OPTION_ALPHAMIXER, &value);
          if(res == SV_OK) {
            printf("\nMixer           : ");
            if(value == SV_ALPHAMIXER_AB) {
              printf("ab");
            } else if(value == SV_ALPHAMIXER_AB_PREMULTIPLIED) {
              printf("ab_pre");
            } else if(value == SV_ALPHAMIXER_BA) {
              printf("ba");
            } else if(value == SV_ALPHAMIXER_BA_PREMULTIPLIED) {
              printf("ba_pre");
            } else {
              printf("off");
            }
            printf("\n");
          }
        }
      }
    } else {
      res = sv_jack_option_set(sv, jack, cmd, value);
    }
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_test(sv_handle * sv, int argc, char ** argv)
{
  char buffer[64];
  int  res   = SV_OK;
  int  value = 0;

  if(argc >= 2) {
    if(!strcmp(argv[1], "help") || !strcmp(argv[1], "?")) {
      printf("sv test help\n");
      printf("sv test timecode #timecode\n");
      return;
    } else if(!strcmp(argv[1], "timecode") && (argc >= 2)) {
      res = sv_asc2tc(sv, argv[2], &value);
      if(res != SV_OK) {
        printf("sv_asc2tc failed %d/%s\n", res, sv_geterrortext(res));
      } else {
        res = sv_tc2asc(sv, value, buffer, sizeof(buffer));
        if(res != SV_OK) {
          printf("sv_tc2asc failed %d/%s\n", res, sv_geterrortext(res));
        } else {
          printf("timecode input  '%s'\n", argv[2]);
          printf("timecode hex    %08x\n", value);
          printf("timecode return '%s'\n", buffer);
        }
      }
    } else {
      res = SV_ERROR_PARAMETER;
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_multichannel(sv_handle * sv, int argc, char ** argv)
{
  int res  = SV_OK;
  int ok   = TRUE;
  int value;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off")) { 
      res = sv_option_set(sv, SV_OPTION_MULTICHANNEL, SV_MULTICHANNEL_OFF);
    } else if(!strcmp(argv[1], "on")) { 
      res = sv_option_set(sv, SV_OPTION_MULTICHANNEL, SV_MULTICHANNEL_ON);
    } else if(!strcmp(argv[1], "default")) {
      res = sv_option_set(sv, SV_OPTION_MULTICHANNEL, SV_MULTICHANNEL_DEFAULT);
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_MULTICHANNEL, &value);
      if(res == SV_OK) {
        if(value == SV_MULTICHANNEL_ON) {
          printf("sv multichannel on\n");
        } else {
          printf("sv multichannel off\n");
        }
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\ton\tEnable multichannel\n");
      printf("\toff\tDisable multichannel\n");
      printf("\tinfo\tShow current setting\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv multichannel: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_alphamixer(sv_handle * sv, int argc, char ** argv)
{
  int res  = SV_OK;
  int ok   = TRUE;
  int value;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off")) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAMIXER, SV_ALPHAMIXER_OFF);
    } else if(!strcmp(argv[1], "ab") || !strcmp(argv[1], "on")) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAMIXER, SV_ALPHAMIXER_AB);
    } else if(!strcmp(argv[1], "ab_pre")) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAMIXER, SV_ALPHAMIXER_AB_PREMULTIPLIED);
    } else if(!strcmp(argv[1], "ba")) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAMIXER, SV_ALPHAMIXER_BA);
    } else if(!strcmp(argv[1], "ba_pre")) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAMIXER, SV_ALPHAMIXER_BA_PREMULTIPLIED);
    } else if(!strcmp(argv[1], "gain") && (argc >= 3)) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAGAIN, atof(argv[2]) * 0x10000);
    } else if(!strcmp(argv[1], "offset") && (argc >= 3)) { 
      res = sv_option_set(sv, SV_OPTION_ALPHAOFFSET, atof(argv[2]) * 0x10000);
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_ALPHAMIXER, &value);
      if(res == SV_OK) {
        if(value == SV_ALPHAMIXER_AB) {
          printf("sv alphamixer ab\n");
        } else if(value == SV_ALPHAMIXER_AB_PREMULTIPLIED) {
          printf("sv alphamixer ab_pre\n");
        } else if(value == SV_ALPHAMIXER_BA) {
          printf("sv alphamixer ba\n");
        } else if(value == SV_ALPHAMIXER_BA_PREMULTIPLIED) {
          printf("sv alphamixer ba_pre\n");
        } else {
          printf("sv alphamixer off\n");
        }
      }
      res = sv_option_get(sv, SV_OPTION_ALPHAGAIN, &value);
      if(res == SV_OK) {
        printf("sv alphamixer gain %1.5f\n", (float)value / 0x10000);
      }
      res = sv_option_get(sv, SV_OPTION_ALPHAOFFSET, &value);
      if(res == SV_OK) {
        printf("sv alphamixer offset %1.5f\n", (float)value / 0x10000);
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\tab\tEnable alpha mixer in A/B mode\n");
      printf("\tab_pre\tEnable alpha mixer in A/B mode (with premultiplied channel A)\n");
      printf("\tgain\tSpecify alpha gain value.\n");
      printf("\toffset\tSpecify alpha offset value.\n");
      printf("\toff\tDisable alpha mixer\n");
      printf("\tinfo\tShow current setting\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv alphamixer: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_routing(sv_handle * sv, int argc, char ** argv)
{
  int res  = SV_OK;
  int ok   = TRUE;
  int value;

  if(argc >= 2) {
    if(!strcmp(argv[1], "default")) { 
      res = sv_option_set(sv, SV_OPTION_ROUTING, SV_ROUTING_DEFAULT);
    } else if((argv[1][0] >= '0') && (argv[1][0] <= '9')) { 
      res = sv_option_set(sv, SV_OPTION_ROUTING, atoi(argv[1]));
    } else if(!strcmp(argv[1], "info")) { 
      res = sv_option_get(sv, SV_OPTION_ROUTING, &value);
      if(res == SV_OK) {
        switch(value) {
        case SV_ROUTING_DEFAULT:
          printf("sv routing default\n");
          break;
        default:
          printf("sv routing %d\n", value);
        }
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\t12\tUse 2 outputs\n");
      printf("\t1234\tUse 4 outputs\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv routing: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_hwwatchdog(sv_handle * sv,  int argc, char ** argv)
{
  int  res   = SV_OK;
  int  value = 0;

  if(argc >= 1) {
    if(!strcmp(argv[0], "help") || !strcmp(argv[0], "?")) {
      printf("svram hwwatchdog help\n");
      printf("svram hwwatchdog relaydelay #ms\n");
      printf("svram hwwatchdog relaydelay info\n");
      printf("svram hwwatchdog relay on/off/info\n");
      printf("svram hwwatchdog gpi on/off/info\n");
      return;
    } else if(!strcmp(argv[0], "relaydelay") && (argc >= 2)) {
      if(!strcmp(argv[1], "info")) {
        res = sv_option_get(sv, SV_OPTION_HWWATCHDOG_RELAY_DELAY, &value);
        if(res != SV_OK) {
          printf("sv_option_get(SV_OPTION_HWWATCHDOG_RELAY_DELAY) failed %d/%s\n", res, sv_geterrortext(res));
        } else {
          printf("The hwwatchdog relaydelay is set to %d ms\n", value);
        }
      } else {
        value = atoi(argv[1]);

        res = sv_option_set(sv, SV_OPTION_HWWATCHDOG_RELAY_DELAY, value);
        if(res != SV_OK) {
          printf("sv_option_set(SV_OPTION_HWWATCHDOG_RELAY_DELAY) failed %d/%s\n", res, sv_geterrortext(res));
        }
      }
    } else if(!strcmp(argv[0], "relay") && (argc >= 2)) {
      res = sv_option_get(sv, SV_OPTION_HWWATCHDOG_TRIGGER, &value);
      if(res != SV_OK) {
        printf("sv_option_get(SV_OPTION_HWWATCHDOG_TRIGGER) failed %d/%s\n", res, sv_geterrortext(res));
      }
      if( res == SV_OK ) {
        if(!strcmp(argv[1], "on")) {
          value |= SV_HWWATCHDOG_RELAY;
        } else if(!strcmp(argv[1], "off")) {
          value &= ~SV_HWWATCHDOG_RELAY;
        } else {
          printf("Relay: %s\n", (value & SV_HWWATCHDOG_RELAY) ? "on" : "off" );
        }
        res = sv_option_set(sv, SV_OPTION_HWWATCHDOG_TRIGGER, value);
        if(res != SV_OK) {
          printf("sv_option_set(SV_OPTION_HWWATCHDOG_TRIGGER) failed %d/%s\n", res, sv_geterrortext(res));
        }
      }
    } else if(!strcmp(argv[0], "gpi") && (argc >= 2)) {
      res = sv_option_get(sv, SV_OPTION_HWWATCHDOG_TRIGGER, &value);
      if(res != SV_OK) {
        printf("sv_option_get(SV_OPTION_HWWATCHDOG_TRIGGER) failed %d/%s\n", res, sv_geterrortext(res));
      }
      if( res == SV_OK ) {
        if(!strcmp(argv[1], "on")) {
          value |= SV_HWWATCHDOG_GPI2;
        } else if(!strcmp(argv[1], "off")){
          value &= ~SV_HWWATCHDOG_GPI2;
        } else {
          printf("GPI: %s\n", (value & SV_HWWATCHDOG_GPI2) ? "on" : "off" );
        }
        res = sv_option_set(sv, SV_OPTION_HWWATCHDOG_TRIGGER, value);
        if(res != SV_OK) {
          printf("sv_option_set(SV_OPTION_HWWATCHDOG_TRIGGER) failed %d/%s\n", res, sv_geterrortext(res));
        }
      }
    } else {
      res = SV_ERROR_PARAMETER;
    }

    if(res != SV_OK) {
      jpeg_errorprint(sv, res);
    }
  }
}

void jpeg_linkencrypt(sv_handle * sv, int argc, char ** argv)
{
  int res  = SV_OK;
  int ok   = TRUE;
  int value;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off")) {
      res = sv_option_set(sv, SV_OPTION_LINKENCRYPT, 0);
    } else if(!strcmp(argv[1], "on")) {
      res = sv_option_set(sv, SV_OPTION_LINKENCRYPT, SV_LINKENCRYPT_A | SV_LINKENCRYPT_B);
    } else if(!strcmp(argv[1], "a")) {
      res = sv_option_set(sv, SV_OPTION_LINKENCRYPT, SV_LINKENCRYPT_A);
    } else if(!strcmp(argv[1], "b")) {
      res = sv_option_set(sv, SV_OPTION_LINKENCRYPT, SV_LINKENCRYPT_B);
    } else if(!strcmp(argv[1], "test")) {
      res = sv_option_set(sv, SV_OPTION_LINKENCRYPT, SV_LINKENCRYPT_TEST);
    } else if(!strcmp(argv[1], "info")) {
      res = sv_option_get(sv, SV_OPTION_LINKENCRYPT, &value);
      if(res == SV_OK) {
        if(value == (SV_LINKENCRYPT_A | SV_LINKENCRYPT_B)) {
          printf("sv linkencrypt on\n");
        } else if(value == SV_LINKENCRYPT_A) {
          printf("sv linkencrypt a\n");
        } else if(value == SV_LINKENCRYPT_B) {
          printf("sv linkencrypt b\n");
        } else if(value == SV_LINKENCRYPT_TEST) {
          printf("sv linkencrypt test\n");
        } else {
          printf("sv linkencrypt off\n");
        }
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\ton\tEnable link encryption for both links\n");
      printf("\ta\tEnable link encryption for link A only\n");
      printf("\tb\tEnable link encryption for link B only\n");
      printf("\toff\tDisable link encryption\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv linkencrypt: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_watermark(sv_handle * sv, int argc, char ** argv)
{
  int res  = SV_OK;
  int ok   = TRUE;
  int value;

  if(argc >= 2) {
    if(!strcmp(argv[1], "info")) {
      res = sv_option_get(sv, SV_OPTION_WATERMARK, &value);
      if(res == SV_OK) {
        printf("sv watermark %d\n", value);
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\t<location_id>\tEnable watermarking with given Location ID.\n");
      printf("\t0\t\tDisable watermarking\n");
      return;
    } else {
      value = atoi(argv[1]);
      res = sv_option_set(sv, SV_OPTION_WATERMARK, value);
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv watermark: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}

void jpeg_render(sv_handle * sv, int argc, char ** argv)
{
  int res  = SV_OK;
  int ok   = TRUE;
  int value;

  if(argc >= 2) {
    if(!strcmp(argv[1], "off")) {
      res = sv_option_set(sv, SV_OPTION_FIFO_RENDER, 0);
    } else if(!strcmp(argv[1], "on")) {
      res = sv_option_set(sv, SV_OPTION_FIFO_RENDER, 1);
    } else if(!strcmp(argv[1], "info")) {
      res = sv_option_get(sv, SV_OPTION_FIFO_RENDER, &value);
      if(res == SV_OK) {
        if(value) {
          printf("sv render on\n");
        } else {
          printf("sv render off\n");
        }
      }
    } else if((strcmp(argv[1], "?") == 0) || (strcmp(argv[1], "help") == 0)) {
      printf("sv %s help\n", argv[0]);
      printf("\ton\tEnable FIFO render mode\n");
      printf("\toff\tDisable FIFO render mode, this reduces the delay of the output pipeline.\n");
      return;
    } else {
      ok = FALSE;
    }
  } else {
    ok = FALSE;
  }

  if(!ok) {
    jpeg_errorprintf(sv, "sv render: Unknown command: %s\n", argv[1]);
    return;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}


void jpeg_fifo_memorymode(sv_handle * sv, char* mode, char* size)
{
  int current = 0;
  int res = SV_OK;
  int timeout = 2000 / 25; // 2 sec 
  sv_fifo_memory memory;
  memory.mode = SV_FIFO_MEMORYMODE_DEFAULT;
  memory.size = 0;

  if(!strcmp(mode, "help")) {
    printf("svram fifo memorymode #mode #size\n");
    printf("svram fifo memorymode none\n");
    printf("svram fifo memorymode share_render #size\n");
    printf("svram fifo memorymode all\n");
  } else if (!strcmp(mode, "info")) {
    sv_query(sv, SV_QUERY_FIFO_MEMORYMODE, 0, &current);
    printf("Current fifo memorymode = %s\n", sv_support_memorymode2string(current));  
  } else if (!strcmp(mode, "all") || !strcmp(mode, "default") ) {
    memory.mode = SV_FIFO_MEMORYMODE_DEFAULT;
    sv_fifo_memorymode(sv, &memory); 
  } else if (!strcmp(mode, "share_render")) {
    memory.mode = SV_FIFO_MEMORYMODE_SHARE_RENDER;
    if(strlen(size) > 0) {
      memory.size = atoi(size);
    }
    do {
      res = sv_fifo_memorymode(sv, &memory);
      if(res == SV_ERROR_NOTREADY) {
        sv_usleep(sv, 25 * 1000); // Wait 25 ms
      }
    } while((res == SV_ERROR_NOTREADY) && timeout--);

    // Reset to default
    if(res == SV_ERROR_NOTREADY) {
      printf("Failed to allocate the needed memory in the render API\n");
      memory.mode = SV_FIFO_MEMORYMODE_DEFAULT;
      sv_fifo_memorymode(sv, &memory);
    } else if( res != SV_OK) {
      printf("sv_fifo_memorymode() failed. res %d\n", res);
    }
  } else if (!strcmp(mode, "none")) {
    memory.mode = SV_FIFO_MEMORYMODE_FIFO_NONE;
    sv_fifo_memorymode(sv, &memory); 
  }
}


void jpeg_option(sv_handle * sv, int argc, char **argv)
{
  int res  = SV_OK;
  int option;
  int value;

  printf("argc %d\n", argc);
  printf("argv %s\n", argv[0]);
  printf("argv %s\n", argv[1]);

  if(argc >= 1) {
    if((strcmp(argv[0], "?") == 0) || (strcmp(argv[0], "help") == 0)) {
      printf("sv option help\n");
      printf("\t#code\t\tCall sv_option_get(sv,code) and print value\n");
      printf("\t#code #value\t\tCall sv_option_set(sv,code,value)#\n");
      printf("\toff\t\tTurn force raster detect off\n");
      return;
    }
    option = atoi(argv[0]);
    if(argc >= 2) {
      res = sv_option_set(sv, option, atoi(argv[1]));
    } else {
      res = sv_option_get(sv, option, &value);
      if(res == SV_OK) {
        printf("sv_option_get(%d) = %d\n", option, value);
      }
    }
  } else {
    res = SV_ERROR_PARAMETER;
  }

  if(res != SV_OK) {
    jpeg_errorprint(sv, res);
  }
}
