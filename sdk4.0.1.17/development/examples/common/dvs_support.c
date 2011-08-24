/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv/svram program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/

/*
//      SDK: Note that these functions are used by the sv program,
//           the calling convention can change, copy the routines
//           into your application if you want to use them.
*/

#ifndef _DVS_CLIB_H
# include "../../header/dvs_setup.h"
# include "../../header/dvs_clib.h"
# include "../../header/dvs_fifo.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>


void sv_support_preset2string(int preset, char * tmp, int tmpsize)
{
  int i = 0;

  if(preset & SV_PRESET_VIDEO) {
    tmp[i++] = 'V';
  }
  if(preset & SV_PRESET_TIMECODE) {
    tmp[i++] = 'T';
  }
  if(preset & SV_PRESET_AUDIO12) { 
    tmp[i++] = '1';
    tmp[i++] = '2';
  }
  if(preset & SV_PRESET_AUDIO34) { 
    tmp[i++] = '3';
    tmp[i++] = '4';
  }
  if(preset & SV_PRESET_AUDIO56) { 
    tmp[i++] = '5';
    tmp[i++] = '6';
  }
  if(preset & SV_PRESET_AUDIO78) { 
    tmp[i++] = '7';
    tmp[i++] = '8';
  }
  if(preset & SV_PRESET_AUDIO9a) { 
    tmp[i++] = '9';
    tmp[i++] = 'a';
  }
  if(preset & SV_PRESET_AUDIObc) { 
    tmp[i++] = 'b';
    tmp[i++] = 'c';
  }
  if(preset & SV_PRESET_AUDIOde) { 
    tmp[i++] = 'd';
    tmp[i++] = 'e';
  }
  if(preset & SV_PRESET_AUDIOfg) { 
    tmp[i++] = 'f';
    tmp[i++] = 'g';
  }
  if(preset & SV_PRESET_KEY) { 
    tmp[i++] = 'K';
  }
  tmp[i++] = 0;
}


int sv_support_string2preset(char * preset)
{
  int i;
  int par = 0;

  for(i = 0; i < (int)strlen(preset); i++) {
    switch(preset[i]) {
     case 'V':
     case 'v':
      par |= SV_PRESET_VIDEO;
      break;
     case 'S':
     case 's':
      par |= SV_PRESET_SECOND_VIDEO;
      break;
     case '1':
     case '2':
      par |= SV_PRESET_AUDIO12;
      break;
     case '3':
     case '4':
      par |= SV_PRESET_AUDIO34;
      break;
     case '5':
     case '6':
      par |= SV_PRESET_AUDIO56;
      break;
     case '7':
     case '8':
      par |= SV_PRESET_AUDIO78;
      break;
     case '9':
     case 'a':
      par |= SV_PRESET_AUDIO9a;
      break;
     case 'b':
     case 'c':
      par |= SV_PRESET_AUDIObc;
      break;
     case 'd':
     case 'e':
      par |= SV_PRESET_AUDIOde;
      break;
     case 'f':
     case 'g':
      par |= SV_PRESET_AUDIOfg;
      break;
     case 'k':
     case 'K':
      par |= SV_PRESET_KEY;
      break;
     case 't':
     case 'T':
      par |= SV_PRESET_TIMECODE;
      break;
     case 'D':  // VGUI compatibility
       break;
     default:
      return -1;
    }
  }
 
  return par;
}


char * sv_support_videomode2string(int videomode)
{
  char * mode;

  switch(videomode & SV_MODE_RASTERMASK) {
  case SV_MODE_PAL:                       mode = "PAL";                   break;
  case SV_MODE_PAL_24I:                   mode = "PAL/24I";               break;
  case SV_MODE_PAL_25P:                   mode = "PAL/25P";               break;
  case SV_MODE_PAL_50P:                   mode = "PAL/50P";               break;
  case SV_MODE_PAL_100P:                  mode = "PAL/100P";              break;
  case SV_MODE_PALHR:                     mode = "PALHR";                 break;

  case SV_MODE_NTSC:                      mode = "NTSC";                  break;
  case SV_MODE_NTSC_29P:                  mode = "NTSC/29P";              break;
  case SV_MODE_NTSC_59P:                  mode = "NTSC/59P";              break;
  case SV_MODE_NTSC_119P:                 mode = "NTSC/119P";             break;
  case SV_MODE_NTSCHR:                    mode = "NTSCHR";                break;

  case SV_MODE_HD360:                     mode = "HD360";                 break;
  case SV_MODE_SMPTE293_59P:              mode = "SMPTE293/59P";          break;
  case SV_MODE_TEST:                      mode = "TEST";                  break;

  case SV_MODE_EUREKA:                    mode = "EUREKA";                break;

  case SV_MODE_SMPTE240_29I:              mode = "SMPTE240/29I";          break;
  case SV_MODE_SMPTE240_30I:              mode = "SMPTE240/30I";          break;

  case SV_MODE_SMPTE274_23I:              mode = "SMPTE274/23I";          break;
  case SV_MODE_SMPTE274_24I:              mode = "SMPTE274/24I";          break;
  case SV_MODE_SMPTE274_25I:              mode = "SMPTE274/25I";          break;
  case SV_MODE_SMPTE274_29I:              mode = "SMPTE274/29I";          break;
  case SV_MODE_SMPTE274_30I:              mode = "SMPTE274/30I";          break;

  case SV_MODE_SMPTE274_23P:              mode = "SMPTE274/23P";          break;
  case SV_MODE_SMPTE274_24P:              mode = "SMPTE274/24P";          break;
  case SV_MODE_SMPTE274_25P:              mode = "SMPTE274/25P";          break;
  case SV_MODE_SMPTE274_29P:              mode = "SMPTE274/29P";          break;
  case SV_MODE_SMPTE274_30P:              mode = "SMPTE274/30P";          break;

  case SV_MODE_SMPTE274_47P:              mode = "SMPTE274/47P";          break;
  case SV_MODE_SMPTE274_48P:              mode = "SMPTE274/48P";          break;
  case SV_MODE_SMPTE274_50P:              mode = "SMPTE274/50P";          break;
  case SV_MODE_SMPTE274_59P:              mode = "SMPTE274/59P";          break;
  case SV_MODE_SMPTE274_60P:              mode = "SMPTE274/60P";          break;
  case SV_MODE_SMPTE274_71P:              mode = "SMPTE274/71P";          break;
  case SV_MODE_SMPTE274_72P:              mode = "SMPTE274/72P";          break;

  case SV_MODE_SMPTE274_23sF:             mode = "SMPTE274/23sF";         break;
  case SV_MODE_SMPTE274_24sF:             mode = "SMPTE274/24sF";         break;
  case SV_MODE_SMPTE274_25sF:             mode = "SMPTE274/25sF";         break;
  case SV_MODE_SMPTE274_29sF:             mode = "SMPTE274/29sF";         break;
  case SV_MODE_SMPTE274_30sF:             mode = "SMPTE274/30sF";         break;

  case SV_MODE_SMPTE274_2560_23P:         mode = "SMPTE274_2560/23P";     break;
  case SV_MODE_SMPTE274_2560_24P:         mode = "SMPTE274_2560/24P";     break;

  case SV_MODE_SMPTE295_25I:              mode = "SMPTE295/25I";          break;
  case SV_MODE_1980x1152_25I:             mode = "1980x1152_25I/25I";     break;

  case SV_MODE_SMPTE296_23P:              mode = "SMPTE296/23P";          break;
  case SV_MODE_SMPTE296_24P:              mode = "SMPTE296/24P";          break;
  case SV_MODE_SMPTE296_24P_30MHZ:        mode = "SMPTE296/24P/30MHZ";    break;
  case SV_MODE_SMPTE296_25P:              mode = "SMPTE296/25P";          break;
  case SV_MODE_SMPTE296_29P:              mode = "SMPTE296/29P";          break;
  case SV_MODE_SMPTE296_30P:              mode = "SMPTE296/30P";          break;
  case SV_MODE_SMPTE296_50P:              mode = "SMPTE296/50P";          break;
  case SV_MODE_SMPTE296_59P:              mode = "SMPTE296/59P";          break;
  case SV_MODE_SMPTE296_60P:              mode = "SMPTE296/60P";          break;
  case SV_MODE_SMPTE296_71P:              mode = "SMPTE296/71P";          break;
  case SV_MODE_SMPTE296_72P:              mode = "SMPTE296/72P";          break;
  case SV_MODE_SMPTE296_71P_89MHZ:        mode = "SMPTE296/71P/89MHZ";    break;
  case SV_MODE_SMPTE296_72P_89MHZ:        mode = "SMPTE296/72P/89MHZ";    break;
  case SV_MODE_SMPTE296_100P:             mode = "SMPTE296/100P";         break;
  case SV_MODE_SMPTE296_119P:             mode = "SMPTE296/119P";         break;
  case SV_MODE_SMPTE296_120P:             mode = "SMPTE296/120P";         break;

  case SV_MODE_ARRI_1920x1080_47P:        mode = "ARRI_1920x1080/47P";    break;
  case SV_MODE_ARRI_1920x1080_48P:        mode = "ARRI_1920x1080/48P";    break;
  case SV_MODE_ARRI_1920x1080_50P:        mode = "ARRI_1920x1080/50P";    break;
  case SV_MODE_ARRI_1920x1080_59P:        mode = "ARRI_1920x1080/59P";    break;
  case SV_MODE_ARRI_1920x1080_60P:        mode = "ARRI_1920x1080/60P";    break;


  case SV_MODE_FILM2K_1998x1080_23sF:     mode = "FILM2K_1998x1080/23sF"; break;
  case SV_MODE_FILM2K_1998x1080_24sF:     mode = "FILM2K_1998x1080/24sF"; break;
  case SV_MODE_FILM2K_1998x1080_23P:      mode = "FILM2K_1998x1080/23P";  break;
  case SV_MODE_FILM2K_1998x1080_24P:      mode = "FILM2K_1998x1080/24P";  break;

  case SV_MODE_FILM2K_2048x858_23sF:      mode = "FILM2K_2048x858/23sF";  break;
  case SV_MODE_FILM2K_2048x858_24sF:      mode = "FILM2K_2048x858/24sF";  break;
  case SV_MODE_FILM2K_2048x858_23P:       mode = "FILM2K_2048x858/23P";   break;
  case SV_MODE_FILM2K_2048x858_24P:       mode = "FILM2K_2048x858/24P";   break;

  case SV_MODE_FILM2K_2048x1080_23sF:     mode = "FILM2K_2048x1080/23sF"; break;
  case SV_MODE_FILM2K_2048x1080_24sF:     mode = "FILM2K_2048x1080/24sF"; break;
  case SV_MODE_FILM2K_2048x1080_25sF:     mode = "FILM2K_2048x1080/25sF"; break;

  case SV_MODE_FILM2K_2048x1080_23P:      mode = "FILM2K_2048x1080/23P";  break;
  case SV_MODE_FILM2K_2048x1080_24P:      mode = "FILM2K_2048x1080/24P";  break;
  case SV_MODE_FILM2K_2048x1080_25P:      mode = "FILM2K_2048x1080/25P";  break;
  case SV_MODE_FILM2K_2048x1080_29P:      mode = "FILM2K_2048x1080/29P";  break;
  case SV_MODE_FILM2K_2048x1080_30P:      mode = "FILM2K_2048x1080/30P";  break;
  case SV_MODE_FILM2K_2048x1080_47P:      mode = "FILM2K_2048x1080/47P";  break;
  case SV_MODE_FILM2K_2048x1080_48P:      mode = "FILM2K_2048x1080/48P";  break;
  case SV_MODE_FILM2K_2048x1080_50P:      mode = "FILM2K_2048x1080/50P";  break;
  case SV_MODE_FILM2K_2048x1080_59P:      mode = "FILM2K_2048x1080/59P";  break;
  case SV_MODE_FILM2K_2048x1080_60P:      mode = "FILM2K_2048x1080/60P";  break;

  case SV_MODE_FILM2K_2048x1536_24P:      mode = "FILM2K_2048x1536/24P";  break;
  case SV_MODE_FILM2K_2048x1536_48P:      mode = "FILM2K_2048x1536/48P";  break;
  case SV_MODE_FILM2K_2048x1536_24sF:     mode = "FILM2K_2048x1536/24sF"; break;

  case SV_MODE_FILM2K_2048x1556_14sF:     mode = "FILM2K_2048x1556/14sF"; break;
  case SV_MODE_FILM2K_2048x1556_15sF:     mode = "FILM2K_2048x1556/15sF"; break;
  case SV_MODE_FILM2K_2048x1556_19sF:     mode = "FILM2K_2048x1556/19sF"; break;
  case SV_MODE_FILM2K_2048x1556_20sF:     mode = "FILM2K_2048x1556/20sF"; break;
  case SV_MODE_FILM2K_2048x1556_24sF:     mode = "FILM2K_2048x1556/24sF"; break;
  case SV_MODE_FILM2K_2048x1556_29sF:     mode = "FILM2K_2048x1556/29sF"; break;
  case SV_MODE_FILM2K_2048x1556_30sF:     mode = "FILM2K_2048x1556/30sF"; break;
  case SV_MODE_FILM2K_2048x1556_36sF:     mode = "FILM2K_2048x1556/36sF"; break;

  case SV_MODE_FILM2K_2048x1556_24P:      mode = "FILM2K_2048x1556/24P";  break;
  case SV_MODE_FILM2K_2048x1556_48P:      mode = "FILM2K_2048x1556/48P";  break;

  case SV_MODE_FILM2K_2048x1744_24P:      mode = "FILM2K_2048x1744/24P";  break;

  case SV_MODE_FILM4K_2160_24P:           mode = "FILM4K_4096x2160/24P";  break;

  case SV_MODE_FILM4K_3112_5sF:           mode = "FILM4K_4096x3112/5sF";  break;
  case SV_MODE_FILM4K_3112_24sF:          mode = "FILM4K_4096x3112/24sF"; break;
  case SV_MODE_FILM4K_3112_24P:           mode = "FILM4K_4096x3112/24P";  break;

  case SV_MODE_VESA_640x480_59P:          mode = "VESA_640x480/59P";      break;
  case SV_MODE_VESA_640x480_60P:          mode = "VESA_640x480/60P";      break;
  case SV_MODE_VESA_640x480_71P:          mode = "VESA_640x480/71P";      break;
  case SV_MODE_VESA_640x480_72P:          mode = "VESA_640x480/72P";      break;
  
  case SV_MODE_VESA_800x600_59P:          mode = "VESA_800x600/59P";      break;
  case SV_MODE_VESA_800x600_60P:          mode = "VESA_800x600/60P";      break;
  case SV_MODE_VESA_800x600_71P:          mode = "VESA_800x600/71P";      break;
  case SV_MODE_VESA_800x600_72P:          mode = "VESA_800x600/72P";      break;
  
  case SV_MODE_VESA_1024x768_29I:         mode = "VESA_1024x768/29I";     break;
  case SV_MODE_VESA_1024x768_30I:         mode = "VESA_1024x768/30I";     break;
  case SV_MODE_VESA_1024x768_59P:         mode = "VESA_1024x768/59P";     break;
  case SV_MODE_VESA_1024x768_60P:         mode = "VESA_1024x768/60P";     break;
  case SV_MODE_VESA_1024x768_71P:         mode = "VESA_1024x768/71P";     break;
  case SV_MODE_VESA_1024x768_72P:         mode = "VESA_1024x768/72P";     break;
  
  case SV_MODE_VESA_1280x1024_29I:        mode = "VESA_1280x1024/29I";    break;
  case SV_MODE_VESA_1280x1024_30I:        mode = "VESA_1280x1024/30I";    break;
  case SV_MODE_VESA_1280x1024_59P:        mode = "VESA_1280x1024/59P";    break;
  case SV_MODE_VESA_1280x1024_60P:        mode = "VESA_1280x1024/60P";    break;
  case SV_MODE_VESA_1280x1024_71P:        mode = "VESA_1280x1024/71P";    break;
  case SV_MODE_VESA_1280x1024_72P:        mode = "VESA_1280x1024/72P";    break;

  case SV_MODE_VESA_1600x1200_29I:        mode = "VESA_1600x1200/29I";    break;
  case SV_MODE_VESA_1600x1200_30I:        mode = "VESA_1600x1200/30I";    break;
  case SV_MODE_VESA_1600x1200_59P:        mode = "VESA_1600x1200/59P";    break;
  case SV_MODE_VESA_1600x1200_60P:        mode = "VESA_1600x1200/60P";    break;
  case SV_MODE_VESA_1600x1200_71P:        mode = "VESA_1600x1200/71P";    break;
  case SV_MODE_VESA_1600x1200_72P:        mode = "VESA_1600x1200/72P";    break;

  case SV_MODE_VESASDI_1024x768_60P:      mode = "VESASDI_1024x768/60P";  break;

  case SV_MODE_1920x1200_24P:             mode = "LCD_1920x1200/24P";     break;
  case SV_MODE_1920x1200_60P:             mode = "LCD_1920x1200/60P";     break;
  case SV_MODE_3840x2400_12P:             mode = "LCD_3840x2400/12P";     break;
  case SV_MODE_3840x2400_24sF:            mode = "LCD_3840x2400/24P";     break;
  case SV_MODE_3840x2400_24P:             mode = "LCD_3840x2400/24sF";    break;

  case SV_MODE_WXGA_1366x768_50P:         mode = "WXGA_1366x768/50P";     break;
  case SV_MODE_WXGA_1366x768_59P:         mode = "WXGA_1366x768/59P";     break;
  case SV_MODE_WXGA_1366x768_60P:         mode = "WXGA_1366x768/60P";     break;
  case SV_MODE_WXGA_1366x768_90P:         mode = "WXGA_1366x768/90P";     break;
  case SV_MODE_WXGA_1366x768_120P:        mode = "WXGA_1366x768/120P";    break;

  case SV_MODE_ANALOG_1920x1080_47P:      mode = "ANALOG_1920x1080/47P";   break;
  case SV_MODE_ANALOG_1920x1080_48P:      mode = "ANALOG_1920x1080/48P";   break;
  case SV_MODE_ANALOG_1920x1080_50P:      mode = "ANALOG_1920x1080/50P";   break;
  case SV_MODE_ANALOG_1920x1080_59P:      mode = "ANALOG_1920x1080/59P";   break;
  case SV_MODE_ANALOG_1920x1080_60P:      mode = "ANALOG_1920x1080/60P";   break;

  default:
    mode = "?";
  }

  return mode;
}



int sv_support_string2videomode(char * string, int offset)
{
  int  mode = -1;
  char tmp[8][32];
  int arraycount,i,pos;
  int next = 1;

  /* For clean start conditions 'tmp' variable has to be cleared for each new function call. */
  memset(tmp, 0, sizeof(tmp));

  /* videomode is already known via rasterindex */
  if(offset) {
    while(string[offset] == '/') {
      /* skip trailing slashes and proceed */
      offset++;
    }
    if(offset && (string[offset]) == '\0') {
      /* nothing more to evaluate */
      return 0;
    }
  }

  strcpy(tmp[0], "PAL"); // SV_MODE_PAL -> 0

  arraycount = (offset != 0);
  pos = offset;
  while(string[pos] && (arraycount < sizeof(tmp)/sizeof(tmp[0]))) {
    i = 0;
    while(string[pos] && (string[pos] != '/') && (i < sizeof(tmp[0]) - 1)) {
      tmp[arraycount][i++] = string[pos++];
    }
    tmp[arraycount][i++] = 0;
    arraycount++; 
    if(string[pos]) {
      pos++;
    }
  }

  if(offset) {
    /* videomode is already known via rasterindex */
    mode = 0;
  } else {
    if(!strcmp(tmp[0], "PAL")) {
      next = 2;
      if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_PAL;
      } else if(!strcmp(tmp[1], "24I")) {
        mode = SV_MODE_PAL_24I;
      } else if(!strcmp(tmp[1], "25P")) {
        mode = SV_MODE_PAL_25P;
      } else if(!strcmp(tmp[1], "50P")) {
        mode = SV_MODE_PAL_50P;
      } else if(!strcmp(tmp[1], "100P")) {
        mode = SV_MODE_PAL_100P;
      } else {
        mode = SV_MODE_PAL;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "PAL10B")) {
      next = 2;
      if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_PAL | SV_MODE_NBIT_10B;
      } else {
        mode = SV_MODE_PAL | SV_MODE_NBIT_10B;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "PALHR")) {
      next = 2;
      if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_PALHR;
      } else {
        mode = SV_MODE_PALHR;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "PALHR10B")) {
      next = 2;
      if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_PALHR | SV_MODE_NBIT_10B;
      } else {
        mode = SV_MODE_PALHR | SV_MODE_NBIT_10B;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "NTSC")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_NTSC;
      } else if(!strcmp(tmp[1], "29P")) {
        mode = SV_MODE_NTSC_29P;
      } else if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_NTSC_59P;
      } else if(!strcmp(tmp[1], "119P")) {
        mode = SV_MODE_NTSC_119P;
      } else {
        mode = SV_MODE_NTSC;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "NTSC10B")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_NTSC | SV_MODE_NBIT_10B;
      } else {
        mode = SV_MODE_NTSC | SV_MODE_NBIT_10B;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "NTSCHR")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_NTSCHR;
      } else {
        mode = SV_MODE_NTSCHR;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "NTSCHR10B")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_NTSCHR | SV_MODE_NBIT_10B;
      } else {
        mode = SV_MODE_NTSCHR | SV_MODE_NBIT_10B;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "EUREKA")) {
      next = 2;
      if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_EUREKA;
      } else {
        mode = SV_MODE_EUREKA;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "HD360") || !strcmp(tmp[0], "NTSCHRC")) {
      mode = SV_MODE_HD360;
    } else if(!strcmp(tmp[0], "SMPTE240")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_SMPTE240_29I;
      } else if(!strcmp(tmp[1], "30I")) {
        mode = SV_MODE_SMPTE240_30I;
      } else if(!strcmp(tmp[1], "30I")) {
        mode = SV_MODE_SMPTE240_30I;
      } else {
        mode = SV_MODE_SMPTE240_29I; 
        next = 1;
      }
    } else if(!strcmp(tmp[0], "SMPTE274")) {
      next = 2;
      if(!strcmp(tmp[1], "23I")) {
        mode = SV_MODE_SMPTE274_23I;
      } else if(!strcmp(tmp[1], "24I")) {
        mode = SV_MODE_SMPTE274_24I;
      } else if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_SMPTE274_25I;
      } else if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_SMPTE274_29I;
      } else if(!strcmp(tmp[1], "30I")) {
        mode = SV_MODE_SMPTE274_30I;
      } else if(!strcmp(tmp[1], "23sF")) {
        mode = SV_MODE_SMPTE274_23sF;
      } else if(!strcmp(tmp[1], "24sF")) {
        mode = SV_MODE_SMPTE274_24sF;
      } else if(!strcmp(tmp[1], "25sF")) {
        mode = SV_MODE_SMPTE274_25sF;
      } else if(!strcmp(tmp[1], "29sF")) {
        mode = SV_MODE_SMPTE274_29sF;
      } else if(!strcmp(tmp[1], "30sF")) {
        mode = SV_MODE_SMPTE274_30sF;
      } else if(!strcmp(tmp[1], "23P")) {
        mode = SV_MODE_SMPTE274_23P;
      } else if(!strcmp(tmp[1], "24P")) {
        mode = SV_MODE_SMPTE274_24P;
      } else if(!strcmp(tmp[1], "25P")) {
        mode = SV_MODE_SMPTE274_25P;
      } else if(!strcmp(tmp[1], "29P")) {
        mode = SV_MODE_SMPTE274_29P;
      } else if(!strcmp(tmp[1], "30P")) {
        mode = SV_MODE_SMPTE274_30P;
      } else if(!strcmp(tmp[1], "47P")) {
        mode = SV_MODE_SMPTE274_47P;
      } else if(!strcmp(tmp[1], "48P")) {
        mode = SV_MODE_SMPTE274_48P;
      } else if(!strcmp(tmp[1], "50P")) {
        mode = SV_MODE_SMPTE274_50P;
      } else if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_SMPTE274_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_SMPTE274_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        mode = SV_MODE_SMPTE274_71P;
      } else if(!strcmp(tmp[1], "72P")) {
        mode = SV_MODE_SMPTE274_72P;
      } else {
        mode = SV_MODE_SMPTE274_29I;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "SMPTE293")) {
      next = 2;
      if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_SMPTE293_59P;
      } else {
        mode = SV_MODE_SMPTE293_59P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "SMPTE295")) {
      next = 2;
      if(!strcmp(tmp[1], "25I")) {
        mode = SV_MODE_SMPTE295_25I;
      } else {
        mode = SV_MODE_SMPTE295_25I;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "SMPTE296")) {
      next = 2;
      if(!strcmp(tmp[1], "23P")) {
        mode = SV_MODE_SMPTE296_23P;
      } else if(!strcmp(tmp[1], "24P")) {
        if(!strcmp(tmp[2], "30MHZ")) {
          mode = SV_MODE_SMPTE296_24P_30MHZ;
          next = 3;
        } else if(!strcmp(tmp[2], "74MHZ")) {
          mode = SV_MODE_SMPTE296_24P;
          next = 3;
        } else {
          mode = SV_MODE_SMPTE296_24P;
        }
      } else if(!strcmp(tmp[1], "25P")) {
        mode = SV_MODE_SMPTE296_25P;
      } else if(!strcmp(tmp[1], "29P")) {
        mode = SV_MODE_SMPTE296_29P;
      } else if(!strcmp(tmp[1], "30P")) {
        mode = SV_MODE_SMPTE296_30P;
      } else if(!strcmp(tmp[1], "50P")) {
        mode = SV_MODE_SMPTE296_50P;
      } else if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_SMPTE296_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_SMPTE296_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        if(!strcmp(tmp[2], "89MHZ")) {
          mode = SV_MODE_SMPTE296_71P_89MHZ;
          next = 3;
        } else if(!strcmp(tmp[2], "74MHZ")) {
          mode = SV_MODE_SMPTE296_71P;
          next = 3;
        } else {
          mode = SV_MODE_SMPTE296_71P;
        }
      } else if(!strcmp(tmp[1], "72P")) {
        if(!strcmp(tmp[2], "89MHZ")) {
          mode = SV_MODE_SMPTE296_72P_89MHZ;
          next = 3;
        } else if(!strcmp(tmp[2], "74MHZ")) {
          mode = SV_MODE_SMPTE296_72P;
          next = 3;
        } else {
          mode = SV_MODE_SMPTE296_72P;
        }
      } else {
        mode = SV_MODE_SMPTE296_59P; 
        next = 1;
      }
    } else if(!strcmp(tmp[0], "SMPTE274_2560")) {
      next = 2;
      if(!strcmp(tmp[1], "23P")) {
        mode = SV_MODE_SMPTE274_2560_23P;
      } else if(!strcmp(tmp[1], "24P")) {
        mode = SV_MODE_SMPTE274_2560_24P;
      } else {
        mode = SV_MODE_SMPTE274_2560_24P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "VESA_640x480")) {
      next = 2;
      if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_VESA_640x480_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_VESA_640x480_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        mode = SV_MODE_VESA_640x480_71P;
      } else if(!strcmp(tmp[1], "72P")) {
        mode = SV_MODE_VESA_640x480_72P;
      } else {
        mode = SV_MODE_VESA_640x480_72P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "VESA_800x600")) {
      next = 2;
      if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_VESA_800x600_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_VESA_800x600_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        mode = SV_MODE_VESA_800x600_71P;
      } else if(!strcmp(tmp[1], "72P")) {
        mode = SV_MODE_VESA_800x600_72P;
      } else {
        mode = SV_MODE_VESA_800x600_72P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "VESA_1024x768")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_VESA_1024x768_29I;
      } else if(!strcmp(tmp[1], "30I")) {
        mode = SV_MODE_VESA_1024x768_30I;
      } else if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_VESA_1024x768_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_VESA_1024x768_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        mode = SV_MODE_VESA_1024x768_71P;
      } else if(!strcmp(tmp[1], "72P")) {
        mode = SV_MODE_VESA_1024x768_72P;
      } else {
        mode = SV_MODE_VESA_1024x768_72P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "VESASDI_1024x768")) {
      next = 2;
      if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_VESASDI_1024x768_60P;
      } else {
        mode = SV_MODE_VESASDI_1024x768_60P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "VESA_1280x1024")) {
      next = 2;
      if(!strcmp(tmp[1], "29I")) {
        mode = SV_MODE_VESA_1280x1024_29I;
      } else if(!strcmp(tmp[1], "30I")) {
        mode = SV_MODE_VESA_1280x1024_30I;
      } else if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_VESA_1280x1024_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_VESA_1280x1024_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        mode = SV_MODE_VESA_1280x1024_71P;
      } else if(!strcmp(tmp[1], "72P")) {
        mode = SV_MODE_VESA_1280x1024_72P;
      } else {
        mode = SV_MODE_VESA_1280x1024_72P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "VESA_1600x1200")) {
      next = 2;
      if(!strcmp(tmp[1], "59P")) {
        mode = SV_MODE_VESA_1600x1200_59P;
      } else if(!strcmp(tmp[1], "60P")) {
        mode = SV_MODE_VESA_1600x1200_60P;
      } else if(!strcmp(tmp[1], "71P")) {
        mode = SV_MODE_VESA_1600x1200_71P;
      } else if(!strcmp(tmp[1], "72P")) {
        mode = SV_MODE_VESA_1600x1200_72P;
      } else {
        mode = SV_MODE_VESA_1280x1024_72P;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "FILM2K") || !strcmp(tmp[0], "FILM2K_2048x1536")) {
      next = 2;
      if(!strcmp(tmp[1], "24sF")) {
        mode = SV_MODE_FILM2K_1536_24sF;
      } else if(!strcmp(tmp[1], "24P")) {
        mode = SV_MODE_FILM2K_1536_24P;
      } else if(!strcmp(tmp[1], "48P")) {
        mode = SV_MODE_FILM2K_1536_48P;
      } else {
        mode = SV_MODE_FILM2K_1536_24sF;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "FILM2K_1556") || !strcmp(tmp[0], "FILM2K_2048x1556")) {
      next = 2;
      if(!strcmp(tmp[1], "14sF")) {
        mode = SV_MODE_FILM2K_1556_14sF;
      } else if(!strcmp(tmp[1], "15sF")) {
        mode = SV_MODE_FILM2K_1556_15sF;
      } else if(!strcmp(tmp[1], "19sF")) {
        mode = SV_MODE_FILM2K_1556_19sF;
      } else if(!strcmp(tmp[1], "20sF")) {
        mode = SV_MODE_FILM2K_1556_20sF;
      } else if(!strcmp(tmp[1], "24sF")) {
        mode = SV_MODE_FILM2K_1556_24sF;
      } else if(!strcmp(tmp[1], "29sF")) {
        mode = SV_MODE_FILM2K_1556_29sF;
      } else if(!strcmp(tmp[1], "30sF")) {
        mode = SV_MODE_FILM2K_1556_30sF;
      } else if(!strcmp(tmp[1], "36sF")) {
        mode = SV_MODE_FILM2K_1556_36sF;
      } else if(!strcmp(tmp[1], "24P")) {
        mode = SV_MODE_FILM2K_1556_24P;
      } else if(!strcmp(tmp[1], "48P")) {
        mode = SV_MODE_FILM2K_1556_48P;
      } else {
        mode = SV_MODE_FILM2K_1556_24sF;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "FILM2K_1080") || !strcmp(tmp[0], "FILM2K_2048x1080")) {
      next = 2;
      if(!strcmp(tmp[1], "23sF")) {
        mode = SV_MODE_FILM2K_2048x1080_23sF;
      } else if(!strcmp(tmp[1], "24sF")) {
        mode = SV_MODE_FILM2K_2048x1080_24sF;
      } else if(!strcmp(tmp[1], "23P")) {
        mode = SV_MODE_FILM2K_2048x1080_23P;
      } else if(!strcmp(tmp[1], "24P")) {
        mode = SV_MODE_FILM2K_2048x1080_24P;
      } else if(!strcmp(tmp[1], "25P")) {
        mode = SV_MODE_FILM2K_2048x1080_25P;
      } else {
        mode = SV_MODE_FILM2K_2048x1080_24sF;
        next = 1;
      }
    } else if(!strcmp(tmp[0], "STREAMER")) {
      mode = SV_MODE_STREAMER;
      next = 1;
    } else if(!strcmp(tmp[0], "STREAMERDF")) {
      mode = SV_MODE_STREAMERDF;
      next = 1;
    } else if(!strcmp(tmp[0], "STREAMERSD")) {
      mode = SV_MODE_STREAMERSD;
      next = 1;
    } else if(!strcmp(tmp[0], "TEST")) {
      mode = SV_MODE_TEST;
      next = 1;
    } else if((tmp[0][0] >= '0') && (tmp[0][0] <= '9')) {
      mode = atoi(tmp[0]);
    }
  }
  
  for(i = next; (mode != -1) && (i < arraycount); i++) {
    if(!strcmp(tmp[i], "10B")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10B;
    } else if(!strcmp(tmp[i], "10BRALE")) {	// same as 10B
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BRALE;
    } else if(!strcmp(tmp[i], "10BDPX")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BDPX;
    } else if(!strcmp(tmp[i], "10BLABE")) {	// same as 10BDPX
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BLABE;
    } else if(!strcmp(tmp[i], "10BDVS")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BDVS;
    } else if(!strcmp(tmp[i], "10BLALE")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BLALE;
    } else if(!strcmp(tmp[i], "10BRABE")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BRABE;
    } else if(!strcmp(tmp[i], "10BRALEV2")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BRALEV2;
    } else if(!strcmp(tmp[i], "10BLABEV2")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BLABEV2;
    } else if(!strcmp(tmp[i], "10BLALEV2")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BLALEV2;
    } else if(!strcmp(tmp[i], "10BRABEV2")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_10BRABEV2;
    } else if(!strcmp(tmp[i], "8B")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_8B;
    } else if(!strcmp(tmp[i], "12B")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_12B;
    } else if(!strcmp(tmp[i], "12BDPX")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_12BDPX;
    } else if(!strcmp(tmp[i], "12BDPXLE")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_12BDPXLE;
    } else if(!strcmp(tmp[i], "16BBE")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_16BBE;
    } else if(!strcmp(tmp[i], "16BLE") || !strcmp(tmp[i], "16B")) {
      mode &= ~SV_MODE_NBIT_MASK;
      mode |=  SV_MODE_NBIT_16BLE;
    } else if(!strcmp(tmp[i], "CHROMA")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_CHROMA;
    } else if(!strcmp(tmp[i], "ALPHA")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_ALPHA;
    } else if(!strcmp(tmp[i], "ALPHA_422A")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_ALPHA_422A;
    } else if(!strcmp(tmp[i], "ALPHA_444A")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_ALPHA_444A;
    } else if(!strcmp(tmp[i], "ALPHA_A444")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_ALPHA_A444;
    } else if(!strcmp(tmp[i], "LUMA") || !strcmp(tmp[i], "MONO")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_LUMA;
    } else if(!strcmp(tmp[i], "BGR")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_RGB_BGR;
    } else if(!strcmp(tmp[i], "BGRA")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_BGRA;
    } else if(!strcmp(tmp[i], "BAYER_BGGR")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_BAYER_BGGR;
    } else if(!strcmp(tmp[i], "BAYER_GBRG")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_BAYER_GBRG;
    } else if(!strcmp(tmp[i], "BAYER_GRBG")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_BAYER_GRBG;
    } else if(!strcmp(tmp[i], "BAYER_RGGB")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_BAYER_RGGB;
    } else if(!strcmp(tmp[i], "RGB")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_RGB_RGB;
    } else if(!strcmp(tmp[i], "ABGR")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_ABGR;
    } else if(!strcmp(tmp[i], "ARGB")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_ARGB;
    } else if(!strcmp(tmp[i], "RGBA")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_RGBA;
    } else if(!strcmp(tmp[i], "YUV422_UYVY") || !strcmp(tmp[i], "YUV422")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YUV422;
    } else if(!strcmp(tmp[i], "YUV422_YUYV") || !strcmp(tmp[i], "YUYV")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YUV422_YUYV;
    } else if(!strcmp(tmp[i], "YUV422A")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YUV422A;
    } else if(!strcmp(tmp[i], "YUV444")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YUV444;
    } else if(!strcmp(tmp[i], "YUV444_VYU")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YUV444_VYU;
    } else if(!strcmp(tmp[i], "YUV444A")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YUV444A;
    } else if(!strcmp(tmp[i], "XYZ")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_XYZ;
    } else if(!strcmp(tmp[i], "YCC")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YCC;
    } else if(!strcmp(tmp[i], "YCC422")) {
      mode &= ~SV_MODE_COLOR_MASK;
      mode |=  SV_MODE_COLOR_YCC422;
    } else if(!strcmp(tmp[i], "FRAME")) {
      mode |=  SV_MODE_STORAGE_FRAME;
    } else if(!strcmp(tmp[i], "BOTTOM2TOP")) {
      mode |=  SV_MODE_STORAGE_BOTTOM2TOP ;
    } else if(!strcmp(tmp[i], "FIELD")) {
      mode &= ~SV_MODE_STORAGE_FRAME;
    } else if(!strcmp(tmp[i], "NOAUDIO")) {
      mode &= ~SV_MODE_AUDIO_MASK;
      mode |=  SV_MODE_AUDIO_NOAUDIO;
    } else if(!strcmp(tmp[i], "1CH") || !strcmp(tmp[i], "AUDIO1CH")) {
      mode &= ~SV_MODE_AUDIO_MASK;
      mode |=  SV_MODE_AUDIO_1CHANNEL;
    } else if(!strcmp(tmp[i], "2CH") || !strcmp(tmp[i], "AUDIO2CH")) {
      mode &= ~SV_MODE_AUDIO_MASK;
      mode |=  SV_MODE_AUDIO_2CHANNEL;
    } else if(!strcmp(tmp[i], "4CH") || !strcmp(tmp[i], "AUDIO4CH")) {
      mode &= ~SV_MODE_AUDIO_MASK;
      mode |=  SV_MODE_AUDIO_4CHANNEL;
    } else if(!strcmp(tmp[i], "6CH") || !strcmp(tmp[i], "AUDIO6CH")) {
      mode &= ~SV_MODE_AUDIO_MASK;
      mode |=  SV_MODE_AUDIO_6CHANNEL;
    } else if(!strcmp(tmp[i], "8CH") || !strcmp(tmp[i], "AUDIO8CH")) {
      mode &= ~SV_MODE_AUDIO_MASK;
      mode |=  SV_MODE_AUDIO_8CHANNEL;
    } else if(!strcmp(tmp[i], "16")) {
      mode &= ~SV_MODE_AUDIOBITS_MASK;
      mode |=  SV_MODE_AUDIOBITS_16;
    } else if(!strcmp(tmp[i], "32")) {
      mode &= ~SV_MODE_AUDIOBITS_MASK;
      mode |=  SV_MODE_AUDIOBITS_32;
    } else if(!strcmp(tmp[i], "PACKED")) {
      mode |=  SV_MODE_FLAG_PACKED;
    } else {
      mode = -1;
    }
  }

  return mode;
}

int sv_support_string2iomode(char * string, int offset)
{
  int  mode = -1;
  char tmp[8][32];
  int arraycount,i,pos;
  int next = 1;

  strcpy(tmp[0], "YUV422");

  arraycount = (offset != 0);
  pos = offset;
  while(string[pos] && (arraycount < sizeof(tmp)/sizeof(tmp[0]))) {
    i = 0;
    while(string[pos] && (string[pos] != '/') && (i < sizeof(tmp[0]) - 1)) {
      tmp[arraycount][i++] = string[pos++];
    }
    tmp[arraycount][i++] = 0;
    arraycount++; 
    if(string[pos]) {
      pos++;
    }
  }

  next = 1;
  if(!strcmp(tmp[0], "YUV") || !strcmp(tmp[0], "YUV422")) {
    if(!strcmp(tmp[1], "12")) {
      mode = SV_IOMODE_YUV422_12;
      next = 2;
    } else {
      mode = SV_IOMODE_YUV422;
    }
  } else if(!strcmp(tmp[0], "YUV444")) {
    if(!strcmp(tmp[1], "12")) {
      mode = SV_IOMODE_YUV444_12;
      next = 2;
    } else {
      mode = SV_IOMODE_YUV444;
    }
  } else if(!strcmp(tmp[0], "RGB")) {
    if(!strcmp(tmp[1], "12")) {
      mode = SV_IOMODE_RGB_12;
      next = 2;
    } else if(!strcmp(tmp[1], "8")) {
      mode = SV_IOMODE_RGB_8;
      next = 2;
    } else {
      mode = SV_IOMODE_RGB;
    }
  } else if(!strcmp(tmp[0], "YUV422A")) {
    mode = SV_IOMODE_YUV422A;
  } else if(!strcmp(tmp[0], "YUV444A")) {
    mode = SV_IOMODE_YUV444A;
  } else if(!strcmp(tmp[0], "RGBA")) {
    if(!strcmp(tmp[1], "12")) {
      mode = SV_IOMODE_RGBA_SHIFT2;
      next = 2;
    } else if(!strcmp(tmp[1], "8")) {
      mode = SV_IOMODE_RGBA_8;
      next = 2;
    } else {
      mode = SV_IOMODE_RGBA;
    }
  } else if(!strcmp(tmp[0], "YUV422STEREO")) {
    mode = SV_IOMODE_YUV422STEREO;
  } else if(!strcmp(tmp[0], "YUV422_12")) {
    mode = SV_IOMODE_YUV422_12;
  } else if(!strcmp(tmp[0], "YUV444_12")) {
    mode = SV_IOMODE_YUV444_12;
  } else if(!strcmp(tmp[0], "RGB_12")) {
    mode = SV_IOMODE_RGB_12;
  } else if(!strcmp(tmp[0], "RGB_8")) {
    mode = SV_IOMODE_RGB_8;
  } else if(!strcmp(tmp[0], "RGBA_8")) {
    mode = SV_IOMODE_RGBA_8;
  } else if(!strcmp(tmp[0], "RGBA_SHIFT2")) {
    mode = SV_IOMODE_RGBA_SHIFT2;
  } else if(!strcmp(tmp[0], "XYZ")) {
    if(!strcmp(tmp[1], "12")) {
      mode = SV_IOMODE_XYZ_12;
      next = 2;
    } else {
      mode = SV_IOMODE_XYZ;
    }
  } else if(!strcmp(tmp[0], "YCC")) {
    if(!strcmp(tmp[1], "12")) {
      mode = SV_IOMODE_YCC_12;
      next = 2;
    } else {
      mode = SV_IOMODE_YCC;
    }
  } else if(!strcmp(tmp[0], "YCC422")) {
    mode = SV_IOMODE_YCC422;
  } else if(!strcmp(tmp[0], "XYZ_12")) {
    mode = SV_IOMODE_XYZ_12;
  } else if(!strcmp(tmp[0], "YCC_12")) {
    mode = SV_IOMODE_YCC_12;
  } else {
    mode = -1;
  }

  
  for(i = next; (mode != -1) && (i < arraycount); i++) {
    if(!strcmp(tmp[i], "HEAD")) {
      mode |=  SV_IOMODE_RANGE_HEAD;
    } else if(!strcmp(tmp[i], "FULL")) {
      mode |=  SV_IOMODE_RANGE_FULL;
    } else if(!strcmp(tmp[i], "601")) {
      mode |=  SV_IOMODE_MATRIX_601;
    } else if(!strcmp(tmp[i], "274")) {
      mode |=  SV_IOMODE_MATRIX_274;
    } else if(!strcmp(tmp[i], "CLIP")) {
      mode |=  SV_IOMODE_CLIP;
    } else {
      mode = -1;
    }
  }

  return mode;
}


int sv_support_string2syncmode(char * svname, char * postfix)
{
  int syncmode = -1;
  int sdtvsync;
  char *name;

  if(!strncmp(svname,"sdtv-",5)) {
    sdtvsync = SV_SYNC_FLAG_SDTV;
    name = svname+5;
  } else if(!strncmp(svname,"ntsc-",5)) {
    sdtvsync = SV_SYNC_FLAG_SDTV;
    name = svname+5;
  } else if(!strncmp(svname,"pal-",4)) {
    sdtvsync = SV_SYNC_FLAG_SDTV;
    name = svname+4;
  } else {
    sdtvsync = 0;
    name = svname;
  }

  if(!strcmp(name, "int")) { 
    syncmode = SV_SYNC_INTERNAL; 
  } else if(!strcmp(name, "internal")) { 
    syncmode = SV_SYNC_INTERNAL;
  } else if(!strcmp(name, "ext")) {
    syncmode = SV_SYNC_EXTERNAL; 
  } else if(!strcmp(name, "external")) {
    syncmode = SV_SYNC_EXTERNAL; 
  } else if(!strcmp(name, "device")) {
    syncmode = SV_SYNC_EXTERNAL; 
  } else if(!strcmp(name, "analog")) {
    syncmode = SV_SYNC_GENLOCK_ANALOG;
  } else if(!strcmp(name, "digital")) {
    syncmode = SV_SYNC_GENLOCK_DIGITAL;
  } else if(!strcmp(name, "slave")) {
    syncmode = SV_SYNC_SLAVE;
  } else if(!strcmp(name, "auto")) {
    syncmode = SV_SYNC_AUTO;
  } else if(!strcmp(name, "bilevel")) {
    syncmode = SV_SYNC_BILEVEL; 
  } else if(!strcmp(name, "trilevel")) {
    syncmode = SV_SYNC_TRILEVEL ; 
  } else if(!strcmp(name, "hvttl")) {
    syncmode = SV_SYNC_HVTTL; 
    if(postfix) {
      if(!strcmp(postfix, "hfvf")) {
        syncmode |= SV_SYNC_HVTTL_HFVF;
      } else if(!strcmp(postfix, "hfvr")) {
        syncmode |= SV_SYNC_HVTTL_HFVR;
      } else if(!strcmp(postfix, "hrvf")) {
        syncmode |= SV_SYNC_HVTTL_HRVF;
      } else if(!strcmp(postfix, "hrvr")) {
        syncmode |= SV_SYNC_HVTTL_HRVR;
      } else {
        syncmode = -1;
      }
    }
  } else if(!strcmp(name, "ltc")) {
    syncmode = SV_SYNC_LTC; 
  } else {
    return -1;
  }

  switch(syncmode) {
  case SV_SYNC_INTERNAL:
  case SV_SYNC_GENLOCK_ANALOG:
  case SV_SYNC_GENLOCK_DIGITAL:
  case SV_SYNC_SLAVE:
  case SV_SYNC_AUTO:
  case SV_SYNC_MODULE:
    if(postfix) {
      if(strlen(postfix)) {
        syncmode = -1;
      }
    }
    if(sdtvsync) {
      syncmode = -1;
    }
    break;
  }

  return syncmode | sdtvsync;
}

char * sv_support_syncmode2guistring(int mode)
{
  switch(mode & SV_SYNC_MASK)
  {
  case SV_SYNC_INTERNAL:
    return "Freerun";
  case SV_SYNC_EXTERNAL:
    return "Video In";
  case SV_SYNC_GENLOCK_ANALOG:
  case SV_SYNC_GENLOCK_DIGITAL:
  case SV_SYNC_BILEVEL:
  case SV_SYNC_TRILEVEL:
    return "Ref In";
  default:
    return "unknown";
  }
}

void sv_support_syncmode2string(int syncmode, char * buffer, int buffersize)
{
  int mode = syncmode & SV_SYNC_MASK;

  if(syncmode & SV_SYNC_FLAG_SDTV) {
    strcpy(buffer,"sdtv-");
  } else {
    strcpy(buffer,"");
  }

  switch(mode) {
  case SV_SYNC_INTERNAL:
    strcat(buffer, "internal");
    break;
  case SV_SYNC_EXTERNAL:
    strcat(buffer, "external");
    break;
  case SV_SYNC_GENLOCK_ANALOG:
    strcat(buffer, "genlock analog");
    break;
  case SV_SYNC_GENLOCK_DIGITAL:
    strcat(buffer, "genlock digital");
    break;
  case SV_SYNC_SLAVE:
    strcat(buffer, "slave");
    break;
  case SV_SYNC_LTC:
    strcat(buffer, "ltc");
    break;
  case SV_SYNC_AUTO:
    strcat(buffer, "auto");
    break;
  case SV_SYNC_BILEVEL:
    strcat(buffer, "bilevel");
    break;
  case SV_SYNC_TRILEVEL:
    strcat(buffer, "trilevel");
    break;
  case SV_SYNC_HVTTL:
    switch(syncmode & SV_SYNC_HVTTL_MASK) {
    case SV_SYNC_HVTTL_HFVF:
      strcat(buffer, "hvttl hfvf");
      break;
    case SV_SYNC_HVTTL_HFVR:
      strcat(buffer, "hvttl hfvr");
      break;
    case SV_SYNC_HVTTL_HRVF:
      strcat(buffer, "hvttl hrvf");
      break;
    case SV_SYNC_HVTTL_HRVR:
      strcat(buffer, "hvttl hrvr");
      break;
    default:
      strcat(buffer, "hvttl");
    }
    break;
  default:
    sprintf(buffer, "mode=%d(%08x)", mode, syncmode);
  }
}


int sv_support_string2syncout(int argc, char ** argv)
{
  int mode = -1;
  int tmp;
  int i;

  for(i = 0; i < argc; i++) {
    if(!strcmp(argv[i], "default")) { 
      mode = SV_SYNCOUT_DEFAULT; 
    } else if(!strcmp(argv[i], "off")) {
      mode = SV_SYNCOUT_OFF; 
    } else if(!strcmp(argv[i], "bilevel")) {
      mode = SV_SYNCOUT_BILEVEL; 
    } else if(!strcmp(argv[i], "trilevel")) {
      mode = SV_SYNCOUT_TRILEVEL; 
    } else if(!strcmp(argv[i], "hvttl")) {
      mode = SV_SYNCOUT_HVTTL_HFVF; 
      if(argc > i + 1) {
        if(!strcmp(argv[i+1], "hfvf")) {
          mode = SV_SYNCOUT_HVTTL_HFVF; i++;
        } else if(!strcmp(argv[i+1], "hfvr")) {
          mode = SV_SYNCOUT_HVTTL_HFVR; i++;
        } else if(!strcmp(argv[i+1], "hrvf")) {
          mode = SV_SYNCOUT_HVTTL_HRVF; i++;
        } else if(!strcmp(argv[i+1], "hrvr")) {
          mode = SV_SYNCOUT_HVTTL_HRVR; i++;
        }
      }     
    } else if(!strcmp(argv[i], "default")) {
      mode = SV_SYNCOUT_DEFAULT; 
    } else if(!strcmp(argv[i], "user")) {
      mode = SV_SYNCOUT_USERDEF; 
    } else if(!strcmp(argv[i], "auto") || !strcmp(argv[i], "automatic")) {
      mode = SV_SYNCOUT_AUTOMATIC; 
    } else if(!strcmp(argv[i], "green") || !strcmp(argv[i], "ongreen")) {
      mode |=  SV_SYNCOUT_OUTPUT_GREEN; 
    } else if(!strcmp(argv[i], "module")) {
      mode |=  SV_SYNCOUT_OUTPUT_MODULE; 
    } else if(!strcmp(argv[i], "main")) {
      mode |=  SV_SYNCOUT_OUTPUT_MAIN; 
    } else if(isdigit(argv[i][0])) {
      tmp = (int)(atof(argv[i]) * 10);
      if((tmp >= 0) && (tmp < 0xfff)) {
        mode &= ~SV_SYNCOUT_LEVEL_MASK;
        mode |=  SV_SYNCOUT_LEVEL_SET(tmp);
      } else {
        mode = -1;
      }
    } else {
      return -1;
    }
  }

  return mode;
}


void sv_support_syncout2string(int syncout, char * buffer, int buffersize)
{

  switch(syncout & SV_SYNCOUT_MASK) {
  case SV_SYNCOUT_OFF:
    strcpy(buffer, "off");
    break;
  case SV_SYNCOUT_BILEVEL:
    strcpy(buffer, "bilevel");
    break;
  case SV_SYNCOUT_TRILEVEL:
    strcpy(buffer, "trilevel");
    break;
  case SV_SYNCOUT_HVTTL_HFVF:
    strcpy(buffer, "hvttl hfvf");
    break;
  case SV_SYNCOUT_HVTTL_HFVR:
    strcpy(buffer, "hvttl hfvr");
    break;
  case SV_SYNCOUT_HVTTL_HRVF:
    strcpy(buffer, "hvttl hrvf");
    break;
  case SV_SYNCOUT_HVTTL_HRVR:
    strcpy(buffer, "hvttl hrvr");
    break;
  case SV_SYNCOUT_USERDEF:
    strcpy(buffer, "userdef");
    break;
  case SV_SYNCOUT_AUTOMATIC:
    strcpy(buffer, "automatic");
    break;
  case SV_SYNCOUT_DEFAULT:
    strcpy(buffer, "default");
    break;
  default:
    sprintf(buffer, "mode=%d(%08x)", syncout & SV_SYNCOUT_MASK, syncout);
  }


  if((syncout & SV_SYNCOUT_OUTPUT_MASK) == SV_SYNCOUT_OUTPUT_MAIN) {
    strcat(buffer, " main");
  }
  if((syncout & SV_SYNCOUT_OUTPUT_MASK) == SV_SYNCOUT_OUTPUT_MODULE) {
    strcat(buffer, " module");
  }
  if((syncout & SV_SYNCOUT_OUTPUT_MASK) == SV_SYNCOUT_OUTPUT_GREEN) {
    strcat(buffer, " green");
  }

  if((syncout & SV_SYNCOUT_LEVEL_MASK) != SV_SYNCOUT_LEVEL_DEFAULT) {  
    sprintf(&buffer[strlen(buffer)], " %d.%dv", SV_SYNCOUT_LEVEL_GET(syncout) / 10, SV_SYNCOUT_LEVEL_GET(syncout) % 10);
  }
}


void sv_support_analog2string(int mode, char * string, int size)
{
  sprintf(string, "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s", 
      ((mode & SV_ANALOG_SHOW_MASK) == SV_ANALOG_SHOW_AUTO)?"auto ":"",
      ((mode & SV_ANALOG_SHOW_MASK) == SV_ANALOG_SHOW_OUTPUT)?"output ":"",
      ((mode & SV_ANALOG_SHOW_MASK) == SV_ANALOG_SHOW_INPUT)?"input ":"",
      ((mode & SV_ANALOG_SHOW_MASK) == SV_ANALOG_SHOW_BLACK)?"black ":"",
      ((mode & SV_ANALOG_SHOW_MASK) == SV_ANALOG_SHOW_COLORBAR)?"colorbar ":"",
      ((mode & SV_ANALOG_FORCE_MASK) == SV_ANALOG_FORCE_PAL)?"forcepal ":"",
      ((mode & SV_ANALOG_FORCE_MASK) == SV_ANALOG_FORCE_NTSC)?"forcentsc ":"",
      ((mode & SV_ANALOG_BLACKLEVEL_MASK) == SV_ANALOG_BLACKLEVEL_BLACK75)?"blacklevel7.5 ":"",
      ((mode & SV_ANALOG_BLACKLEVEL_MASK) == SV_ANALOG_BLACKLEVEL_BLACK0)?"blacklevel0 ":"",
      ((mode & SV_ANALOG_OUTPUT_MASK) == SV_ANALOG_OUTPUT_YC)?"YC ":"",
      ((mode & SV_ANALOG_OUTPUT_MASK) == SV_ANALOG_OUTPUT_YUV)?"YUV ":"",
      ((mode & SV_ANALOG_OUTPUT_MASK) == SV_ANALOG_OUTPUT_YUVS)?"YUVS ":"",
      ((mode & SV_ANALOG_OUTPUT_MASK) == SV_ANALOG_OUTPUT_RGB)?"RGB ":"",
      ((mode & SV_ANALOG_OUTPUT_MASK) == SV_ANALOG_OUTPUT_RGBS)?"RGBS ":"",
      ((mode & SV_ANALOG_OUTPUT_MASK) == SV_ANALOG_OUTPUT_CVBS)?"CVBS ":"");
}



void sv_support_vtrinfo2string(char * string, int size, int info)
{
  sprintf(string, "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
      ((SV_MASTER_INFO_OFFLINE & info)?"Offline ":""),
      ((SV_MASTER_INFO_STANDBY & info)?"Standby ":""),
      ((SV_MASTER_INFO_STOP    & info)?"Stop ":""),
      ((SV_MASTER_INFO_EJECT   & info)?"Eject ":""),
      ((SV_MASTER_INFO_REWIND  & info)?"Rewind ":""),
      ((SV_MASTER_INFO_FORWARD & info)?"Forward ":""), 
      ((SV_MASTER_INFO_RECORD  & info)?"Record ":""),
      ((SV_MASTER_INFO_PLAY    & info)?"Play ":""),
      ((SV_MASTER_INFO_JOG     & info)?"Jog ":""),
      ((SV_MASTER_INFO_VAR     & info)?"Var ":""),
      ((SV_MASTER_INFO_SHUTTLE & info)?"Shuttle ":""),
      ((SV_MASTER_INFO_CASSETTEOUT & info)?"CassetteOut ":""),
      ((SV_MASTER_INFO_REFVIDEOMISSING & info)?"RefVideoMissing ":""),
      ((SV_MASTER_INFO_TAPETROUBLE & info)?"TapeTrouble ":""),
      ((SV_MASTER_INFO_HARDERROR & info)?"HardError ":""),
      ((SV_MASTER_INFO_LOCAL  & info)?"Local ":""));
}



char * sv_support_devmode2string(int devmode)
{
  switch(devmode) {
  case SV_DEVMODE_BLACK:
    return "Black";
  case SV_DEVMODE_COLORBAR:
    return "Colorbar";
  case SV_DEVMODE_LIVE:
    return "Live";
  case SV_DEVMODE_DISPLAY:
    return "Display";
  case SV_DEVMODE_RECORD:
    return "Record";
  case SV_DEVMODE_VTRLOAD:
    return "VtrLoad";
  case SV_DEVMODE_VTRSAVE:
    return "VtrSave";
  case SV_DEVMODE_VTREDIT:
    return "VtrEdit";
  case SV_DEVMODE_TIMELINE:
    return "Timeline";
  }

  return "Unknown";
}


char * sv_support_iomode2string_mode(int iomode)
{
  switch(iomode & SV_IOMODE_IO_MASK) {
  case SV_IOMODE_YUV422:
    return "YUV422";
  case SV_IOMODE_RGB:
    return "RGB";
  case SV_IOMODE_YUV444:
    return "YUV444";
  case SV_IOMODE_YUV422A:
    return "YUV422A";
  case SV_IOMODE_RGBA:
    return "RGBA";
  case SV_IOMODE_YUV444A:
    return "YUV444A";
  case SV_IOMODE_RGBA_SHIFT2:
    return "RGBA_SHIFT2";
  case SV_IOMODE_YUV422_12:
    return "YUV422/12";
  case SV_IOMODE_YUV444_12:
    return "YUV444/12";
  case SV_IOMODE_RGB_12:
    return "RGB/12";
  case SV_IOMODE_RGB_8:
    return "RGB/8";
  case SV_IOMODE_RGBA_8:
    return "RGBA/8";
  case SV_IOMODE_YUV422STEREO:
    return "YUV422STEREO";
  case SV_IOMODE_XYZ:
    return "XYZ";
  case SV_IOMODE_XYZ_12:
    return "XYZ/12";
  case SV_IOMODE_YCC:
    return "YCC";
  case SV_IOMODE_YCC_12:
    return "YCC/12";
  case SV_IOMODE_YCC422:
    return "YCC422";
  }

  return "unknown";
}

char * sv_support_colormode2string_mode( int videomode )
{
  switch(videomode & SV_MODE_COLOR_MASK) {
  case SV_MODE_COLOR_CHROMA:
    return "CHROMA";
  case SV_MODE_COLOR_LUMA:
    return "LUMA";
  case SV_MODE_COLOR_RGB_BGR:
    return "BGR";
  case SV_MODE_COLOR_BGRA:
    return "BGRA";
  case SV_MODE_COLOR_RGB_RGB:
    return "RGB";
  case SV_MODE_COLOR_ABGR:
    return "ABGR";
  case SV_MODE_COLOR_ARGB:
    return "ARGB";
  case SV_MODE_COLOR_RGBA:
    return "RGBA";
  case SV_MODE_COLOR_YUV2QT:
    return "YUV2QT";
  case SV_MODE_COLOR_YUV422:
    return "YUV422";
  case SV_MODE_COLOR_YUV422_YUYV:
    return "YUV422_YUYV";
  case SV_MODE_COLOR_YUV422A:
    return "YUV422A";
  case SV_MODE_COLOR_YUV444:
    return "YUV444";
  case SV_MODE_COLOR_YUV444_VYU:
    return "YUV444_VYU";
  case SV_MODE_COLOR_YUV444A:
    return "YUV444A";
  case SV_MODE_COLOR_ALPHA:
    return "ALPHA";
  case SV_MODE_COLOR_ALPHA_422A:
    return "ALPHA_422A";
  case SV_MODE_COLOR_ALPHA_444A:
    return "ALPHA_444A";
  case SV_MODE_COLOR_ALPHA_A444:
    return "ALPHA_A444";
  case SV_MODE_COLOR_XYZ:
    return "XYZ";
  case SV_MODE_COLOR_YCC:
    return "YCC";
  case SV_MODE_COLOR_YCC422:
    return "YCC422";
  case SV_MODE_COLOR_BAYER_BGGR:
    return "BAYER_BGGR";
  case SV_MODE_COLOR_BAYER_GBRG:
    return "BAYER_GBRG";
  case SV_MODE_COLOR_BAYER_GRBG:
    return "BAYER_GRBG";
  case SV_MODE_COLOR_BAYER_RGGB:
    return "BAYER_RGGB";
  default:
    break;
  }
  
  return "?colormode?";
}

char * sv_support_bit2string_mode( int videomode )
{
  switch(videomode & SV_MODE_NBIT_MASK) {
  case SV_MODE_NBIT_8B:
    return "8B";
  case SV_MODE_NBIT_10B:
    return "10B";
  case SV_MODE_NBIT_10BDVS:
    return "10BDVS";
  case SV_MODE_NBIT_10BDPX:
    return "10BDPX";
  case SV_MODE_NBIT_10BLALE:
    return "10BLALE";
  case SV_MODE_NBIT_10BRABE:
    return "10BRABE";
  case SV_MODE_NBIT_10BRALEV2:
    return "10BRALEV2";
  case SV_MODE_NBIT_10BLABEV2:
    return "10BLABEV2";
  case SV_MODE_NBIT_10BLALEV2:
    return "10BLALEV2";
  case SV_MODE_NBIT_10BRABEV2:
    return "10BRABEV2";
  case SV_MODE_NBIT_12B:
    return "12B";
  case SV_MODE_NBIT_12BDPX:
    return "12BDPX";
  case SV_MODE_NBIT_12BDPXLE:
    return "12BDPXLE";
  case SV_MODE_NBIT_16BBE:
    return "16BBE";
  case SV_MODE_NBIT_16BLE:
    return "16BLE";
  default:
    break;
  }
  return "?nbits?";
}

static char * sv_support_iomode2string_range(int iomode)
{
  switch(iomode & SV_IOMODE_RANGE_MASK) {
  case SV_IOMODE_RANGE_HEAD:
    return "/HEAD";
  case SV_IOMODE_RANGE_FULL:
    return "/FULL";
  }

  return "";
}

static char * sv_support_iomode2string_matrix(int iomode)
{
  switch(iomode & SV_IOMODE_MATRIX_MASK) {
  case SV_IOMODE_MATRIX_601:
    return "/601";
  case SV_IOMODE_MATRIX_274:
    return "/274";
  }

  return "";
}

static char * sv_support_iomode2string_clip(int iomode)
{
  switch(iomode & SV_IOMODE_CLIP_MASK) {
  case SV_IOMODE_CLIP:
    return "/CLIP";
  }

  return "";
}

void sv_support_iomode2string(int iomode, char * buffer, int buffersize)
{
  strcpy(buffer, sv_support_iomode2string_mode(iomode));

  if(iomode & SV_IOMODE_RANGE_MASK) {
    strcat(buffer, sv_support_iomode2string_range(iomode));
  }
  if(iomode & SV_IOMODE_MATRIX_MASK) {
    strcat(buffer, sv_support_iomode2string_matrix(iomode));
  }
  if(iomode & SV_IOMODE_CLIP_MASK) {
    strcat(buffer, sv_support_iomode2string_clip(iomode));
  }

  if(iomode & SV_IOMODE_OUTPUT_ENABLE) {
    strcat(buffer, sv_support_iomode2string_mode(SV_IOMODE_OUTPUT2MODE(iomode)));
    if(iomode & SV_IOMODE_OUTPUT_RANGE_MASK) {
      strcat(buffer, sv_support_iomode2string_range(SV_IOMODE_OUTPUT2RANGE(iomode)));
    }
    if(iomode & SV_IOMODE_OUTPUT_MATRIX_MASK) {
      strcat(buffer, sv_support_iomode2string_matrix(SV_IOMODE_OUTPUT2RANGE(iomode)));
    }
  }
}

char * sv_support_iospeed2string(int iospeed)
{
  switch(iospeed) {
  case SV_IOSPEED_1GB5:
    return "1GB5";
  case SV_IOSPEED_3GBA:
    return "3GBA";
  case SV_IOSPEED_3GBB:
    return "3GBB";
  case SV_IOSPEED_SDTV:
    return "SDTV";
  default:
    return "unknown";
  }
}

char * sv_support_channel2string(int channel)
{
  switch(channel) {
  case SV_JACK_CHANNEL_DISCONNECTED:
    return "disconnected";
  case SV_JACK_CHANNEL_OUT:
    return "out";
  case SV_JACK_CHANNEL_IN:
    return "in";
  case SV_JACK_CHANNEL_OUTB:
    return "out2";
  case SV_JACK_CHANNEL_INB:
    return "in2";
  }

  return "unknown";
}

char * sv_support_memorymode2string(int mode)
{
  switch(mode) {
  case SV_FIFO_MEMORYMODE_DEFAULT:
    return "all";
  case SV_FIFO_MEMORYMODE_SHARE_RENDER:
    return "share_render";
  case SV_FIFO_MEMORYMODE_FIFO_NONE:
    return "none";
  }

  return "unknown";
}

char * sv_support_devtype2string(int devtype)
{
  switch(devtype) {
  case SV_DEVTYPE_UNKNOWN:
    return "Unknown";
  case SV_DEVTYPE_SCSIVIDEO_1:
    return "SCSIVideo-1";
  case SV_DEVTYPE_SCSIVIDEO_2:
    return "SCSIVideo-2";
  case SV_DEVTYPE_PRONTOVIDEO:
    return "ProntoVideo";
  case SV_DEVTYPE_PRONTOVIDEO_RGB:
    return "ProntoVideo-RGB";
  case SV_DEVTYPE_PRONTOVIDEO_PICO:
    return "PicoVideo";
  case SV_DEVTYPE_PCISTUDIO:
    return "PCIStudio";
  case SV_DEVTYPE_CLIPSTATION:
    return "ClipStation";
  case SV_DEVTYPE_MOVIEVIDEO:
    return "MovieVideo";
  case SV_DEVTYPE_PRONTOVISION:
    return "ProntoVision";
  case SV_DEVTYPE_PRONTOSERVER:
    return "ProntoServer";
  case SV_DEVTYPE_CLIPBOARD:
    return "ClipBoard";
  case SV_DEVTYPE_HDSTATIONPRO:
    return "HDStationPRO";
  case SV_DEVTYPE_HDBOARD:
    return "HDBoard";
  case SV_DEVTYPE_SDSTATIONPRO:
    return "SDStationPRO";
  case SV_DEVTYPE_SDBOARD:
    return "SDBoard";
  case SV_DEVTYPE_HDXWAY:
    return "HDXWay";
  case SV_DEVTYPE_SDXWAY:
    return "SDXWay";
  case SV_DEVTYPE_CLIPSTER:
    return "Clipster";
  case SV_DEVTYPE_CENTAURUS:
    return "Centaurus";
  case SV_DEVTYPE_HYDRA:
    return "Hydra";
  case SV_DEVTYPE_DIGILAB:
    return "Digilab";
  case SV_DEVTYPE_HYDRAX:
    return "HydraX";
  case SV_DEVTYPE_ATOMIX:
    return "Atomix";
  case SV_DEVTYPE_ATOMIXLT:
    return "AtomixLT";
  }

  return "Unknown card";
}

char * sv_support_audioinput2string(int input)
{
  switch(input) {
    case SV_AUDIOINPUT_AESEBU:
      return "AES/EBU";
    case SV_AUDIOINPUT_AIV:
      return "AIV";
  }

  return "unknown";
}

int sv_support_string2audioinput(char * inputString)
{
  int mode = -1;

  if(!strcmp(inputString, "AES") || !strcmp(inputString, "AES/EBU")) {
    mode = SV_AUDIOINPUT_AESEBU;
  } else if(!strcmp(inputString, "AIV")) {
    mode = SV_AUDIOINPUT_AIV;
  }

  return mode;
}

char * sv_support_audiochannels2string(int channels)
{
  switch(channels) {
    case SV_MODE_AUDIO_NOAUDIO:
      return "No Audio";
    case SV_MODE_AUDIO_8CHANNEL:
      return "8 Stereo Channels";
  }

  return "unknown";
}

int sv_support_string2audiochannels(char * channelString)
{
  int mode = -1;

  if(!strcmp(channelString, "No Audio")) {
    mode = SV_MODE_AUDIO_NOAUDIO;
  } else if(!strcmp(channelString, "8 Stereo Channels")) {
    mode = SV_MODE_AUDIO_8CHANNEL;
  }

  return mode;
}

char * sv_support_audiofreq2string(int frequency)
{
  switch(frequency) {
    case 48000:
      return "48000 Hz";
    case 96000:
      return "96000 Hz";
  }

  return "unknown";
}

int sv_support_string2audiofreq(char * freqString)
{
  int frequency = 0;

  if(!strcmp(freqString, "48000") || !strcmp(freqString, "48000 Hz")) {
    frequency = 48000;
  } else if(!strcmp(freqString, "96000") || !strcmp(freqString, "96000 Hz")) {
    frequency = 96000;
  }

  return frequency;
}

