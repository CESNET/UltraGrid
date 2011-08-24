/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/



#include "svprogram.h"


int user_pipe(sv_handle * sv)
{
  char   buffer[256]; 
  char * argv[32];
  int    state;
  int    argc;
  int    pos;

  if(sv->vgui) {
    /*
    //	The gui can not handle output to both stderr and stdout 
    //	in the pipe mode. No error handling is done, the failure
    //	will only effect errors.
    */
    (void)dup2(1, 2);
  }

  argv[0] = "sv";

  do {
    fflush(stderr); 
    printf("<pipe>\n"); 
    fflush(stdout); 
    fgets(buffer, sizeof(buffer), stdin);

    if(feof(stdin)) {
      strcpy(buffer, "quit");
    }

    buffer[sizeof(buffer) - 1] = '\0';
    
    pos   = 0;
    state = 0;
    argc  = 1;

    while(buffer[pos]) {
      switch(buffer[pos]) {
      case '\"':
        if(state != 2) {
          argv[argc++] = &buffer[pos+1];
          state = 2;
        } else {
          buffer[pos] = '\0';
          state = 0;
        }
        break;
      case ' ':
      case '\n':
      case '\t':
        if(state != 2) {
          buffer[pos] = '\0';
          state = 0;
          break;
        }
      /* FALLTHROUGH */
      default:
        if(state == 0) {
          argv[argc++] = &buffer[pos];
          state = 1;
        }
      }
      pos++;
    }
    
    user_parse(sv, argc, argv);

  } while((strcmp(buffer, "quit") != 0));

  return 0;
}


int user_help(int argc, char ** argv)
{
  if(argc <= 1) {
    printf("sv help\n");
    printf("       svram help audio\n");
    printf("       svram help display\n");
    printf("       svram help dvi\n");
    printf("       svram help guiinfo\n");
    printf("       svram help fifo\n");
    printf("       svram help transfer\n");
    printf("       svram help setup\n");
    printf("       svram help timecode\n");
    printf("       svram help video\n");
    printf("       svram help vtr\n");
    printf("\n");
    printf("       svram man help\n");
    printf("\n");
    printf("SCSIVIDEO_CMD Syntax\n");
#ifdef WIN32
    printf("       'set SCSIVIDEO_CMD=options'\n");
#else
    printf("       'setenv SCSIVIDEO_CMD options'\n");
#endif
    printf("\n");
    printf("SCSIVIDEO_CMD Options\n");
    printf("       PCI,card:0            For 'sv' to select PCI card.\n");
    printf("       ,channel:1            Use video channel 1\n");
  } else {
    if(!strcmp(argv[1], "display") || !strcmp(argv[1], "record")) {
      printf("       svram clip help\n");
      printf("       svram display {ram,disk,{{sgi,bmp,yuv,...} filename}} [#start=0 [#nframes=1]]\n");
      printf("       svram record  {ram,disk,{{sgi,bmp,yuv,...} filename}} [#start=0 [#nframes=1]]\n");
      printf("       svram speed [#speed=1.0 [loopmode=#nop,once,shuttle,loop]\n");
      printf("       svram black\tShow black output\n");
      printf("       svram colorbar\tShow colorbar output\n");
      printf("       svram live\n");
      printf("       svram stop\n");
      printf("       svram repeat help\n");
      printf("       svram slowmotion help\n");
      printf("       svram sequence [options] start [frames [objects [...]]] [options]\n");
      printf("       svram playlist filename [timecode]\n");
      printf("       svram preset help\n");
    } else if(!strcmp(argv[1], "video")) {
      printf("       svram inputport help\tSet input port\n");
      printf("       svram iomode help\tSet cable transfer format\n");
      printf("       svram iospeed help\tSet cable link speed\n");
      printf("       svram lut help\tLoad gamma correction from file\n");
      printf("       svram mixer help\n");
      printf("       svram zoom help\n");
    } else if(!strcmp(argv[1], "misc") || !strcmp(argv[1], "timecode") || !strcmp(argv[1], "tc")) {
      printf("       svram gpi {on, off}\n");
      printf("       svram ltc help\n");
    } else if(!strcmp(argv[1], "vtr")) {
      printf("       svram master {stop, play, ...}\n");
      printf("       svram vtrload {ram,disk,{{sgi,bmp,yuv,...} filename}} [#start=0 [#nframes=1]] timecode\n");
      printf("       svram vtrsave {ram,disk,{{sgi,bmp,yuv,...} filename}} [#start=0 [#nframes=1]] timecode\n");
      printf("\n");
      printf("       Timecode format: hh:mm:ss:ff\n");
    } else if(!strcmp(argv[1], "transfer")) {
      printf("       svram load {sgi,bmp,yuv,...} filename #frame [#start=0 [#nframes=1]]\n");
      printf("       svram save {sgi,bmp,yuv,...} filename #frame [#start=0 [#nframes=1]]\n");
    } else if(!strcmp(argv[1], "dvi") || !strcmp(argv[1], "analog")) {
      printf("       svram dvi help\tSet Analog/DVI output\n");
      printf("       svram analog help\tSet Analog monitor sdtv output\n");
      printf("       svram analogoutput help\tSet Analog/DVI output\n");
      printf("       svram monitorinfo help\n");
      printf("       svram syncout help\tSet sync output mode#\n");
    } else if(!strcmp(argv[1], "setup")) {
      printf("       svram licence {show,key1,key2,key3}\n");
      printf("       svram info\n");
      printf("       svram info clock\tShow realtime clock\n");
      printf("       svram info hardware\n");
      printf("       svram info publickey\n");
      printf("       svram info random\n");
      printf("       svram info timecode\n");
      printf("       svram mode help\n");
      printf("       svram jack help\n");
      printf("       svram sync help\tSet sync input mode\n");
      printf("       svram timecode help\tChange dropframe and tc offset\n");
    } else if(!strcmp(argv[1], "audio")) {
      printf("       svram audioanalogout help\n");
      printf("       svram audiochannels help\n");
      printf("       svram audiofrequency help\n");
      printf("       svram audioinput help\n");
      printf("       svram audiomaxaiv help\n");
      printf("       svram audiomode help\n");
      printf("       svram audiomute help\n");
      printf("       svram audiooffset #offset\n");
      printf("       svram audio_speed_compensation #speed\n");
      printf("       svram wordclock help\n");
    } else if(!strcmp(argv[1], "guiinfo")) {
      printf("       svram guiinfo allclips\n");
      printf("       svram guiinfo clips\n");
      printf("       svram guiinfo fileformats\n");
      printf("       svram guiinfo help\n");
      printf("       svram guiinfo init\n");
      printf("       svram guiinfo master\n");
      printf("       svram guiinfo position\n");
      printf("       svram guiinfo refresh\n");
      printf("       svram guiinfo setup\n");
      printf("       svram guiinfo zoom\n");
    } else if(!strcmp(argv[1], "fifo")) {
      printf("       svram fifo memorymode #mode #size\n");
    } else {
      printf("svram help %s unknown\n", argv[1]);  
    }
  }

  return 0;
}



int user_date(int argc, char ** argv) 
{
#if defined(__DATE__) && defined(__TIME__)
  printf("Compile time: %s %s\n", __DATE__, __TIME__);
#else
  printf("Compile time: Compiler did not have date and time option\n");
#endif

  printf("Version: %d.%d.%d.%d\n", DVS_VERSION_MAJOR, DVS_VERSION_MINOR, DVS_VERSION_PATCH, DVS_VERSION_FIX);

  return 0;
}

#ifndef CONFIG_LIB
// EXEC_WIN32 is defined in the sv_winapi project. It's used to called from any windows application
// as a command-line program, but without the console window
#ifdef EXEC_WIN32
donttag int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int argc, char ** argv)
#endif
{
  sv_handle * sv = 0;
  int res = SV_OK;
  int pipe = 0;
  int tmp;
  int i;

#ifdef EXEC_WIN32
  int argc;
  char ** argv;

  argc = __argc;
  argv = __argv;
#endif

  if(argc > 1) {
    if(!strcmp(argv[1], "help") || !strcmp(argv[1], "man")|| !strcmp(argv[1], "?")) {
      user_help(argc-1, argv+1);
      return 0;
    } else if((strcmp(argv[1], "dpf") == 0) && (argc == 2)) {
      sv = jpeg_open(FALSE, TRUE);
      jpeg_debugprint(sv, TRUE);
      jpeg_close(sv);
    } else if(!strcmp(argv[1], "info") || !strcmp(argv[1], "version")) { // List of query only commands
      sv = jpeg_open(FALSE, TRUE);
    } else if(!strcmp(argv[1], "date")) {
      user_date(argc-1, argv+1);
      return 0;
    } else if(!strcmp(argv[1], "sleep")) {
      if(strchr(argv[2], '.')) {
        tmp = atof(argv[2]) * 1000;
      } else {
        tmp = atoi(argv[2]) * 1000; 
      }
      i = 0;
      if(tmp > 1000000) {
        printf("sv sleep %d to long, max 1000\n", tmp);
        return 0;
      } else if(tmp <= 0) {
        printf("sv sleep %d <= 0\n", tmp);
        return 0;
      }
      if((tmp > 1000) && (tmp <= 60000)) {
        for(i = 0; i < tmp; i+=1000) {
          printf(".");
        }
        printf("\n");
      }
      for(i = 0; tmp > i; i++) {
#ifdef WIN32
        if(tmp >= 1000) {
          Sleep(1000); i+=999;
        } else if(tmp >= 100) {
          Sleep(100); i+=99;
        } else {
          Sleep(10); i+=9;
        }
#else
        sleep(1); i+=999;
#endif
        if(tmp > 1000) {
          if((i % 1000) == 0) {
            printf(".");
          }
          if(tmp > 60000) {
            if((i % 10) == 9) {
              printf(" %d\n", i + 1);
            }
          }
          fflush(stdout);
        }
      }
      return 0;
    } else if(!strcmp(argv[1], "delay")) {
      tmp = atoi(argv[2]); i = 0;
      for(i = 0; tmp > i; i++) {
#ifdef WIN32
        Sleep(1000);
#else
        sleep(1);
#endif
      }
      return 0;
    } else if(!strcmp(argv[1], "check")) {
      sv = jpeg_open(TRUE, FALSE);
      if(sv) {
        sv_close(sv);
      } 
      return (sv != NULL);
    }
  }
  
  if(sv == NULL) {
    sv = jpeg_open(FALSE, FALSE);
    if(sv == NULL) {
      return -1;
    }
  }
  
  if(getenv("VGUI")) {
    sv->vgui = 1;
#ifdef WIN32
    (void)dup2(1, 2);
#endif
  }

  if(argc == 2 && strcmp(argv[1], "pipe") == 0) {
    pipe = 1;
  }
  if(pipe) {
    res = user_pipe(sv);
  } else {
    res = user_parse(sv, argc, argv);
  }

  jpeg_close(sv);

  return res;
}
#endif //CONFIG_LIB







int user_parse(sv_handle * sv, int argc, char ** argv)
{ 
  int        ok      = 0;
  int        start   = 0;
  int        nframes = 1;

#if defined(SV_DEBUG_GUI) && !defined(WIN32)
  if(sv_debug & SV_DEBUG_GUI) {
    int i;
    for(i = 0; i < argc; i++) {
      printf("%s ", argv[i]);
    }
    printf("\n");
  }
#endif

  if(argc > 1) {
    if((strcmp(argv[1], "guiinfo") == 0) && (argc > 2)) {
      if (argc == 3) {
        jpeg_guiinfo(sv, argv[2]);
        ok = 1;
      }
    } else if(strcmp(argv[1], "date") == 0) {
      user_date(argc-2, &argv[2]);
      return 0;
    } else if(strcmp(argv[1], "goto") == 0) {
      jpeg_goto(sv, argc-2, &argv[2]);
      ok = 1;
    } else if ((strcmp(argv[1], "preset") == 0) && (argc <= 3)) {
      if (argc == 3) {
        jpeg_preset(sv, argv[2]);
      } else {
        jpeg_preset(sv, "");
      }
      ok = 1;
    } else if((strcmp(argv[1], "record") == 0) && (argc > 2)) {
      if((argc >= 3) && ((!strcmp(argv[2], "ram")) ||
                         (!strcmp(argv[2], "disk")))) {
        if(argc > 3)
          start = atoi(argv[3]);
        if(argc > 4)
          nframes = atoi(argv[4]);
        if(argc <= 5) {
          jpeg_record(sv, argv[2], 0, start, nframes, 0, "1.0", "once");
          ok = 1;
        } else if (argc == 6) {
          jpeg_record(sv, argv[2], 0, start, nframes, 0, argv[5], "once");
          ok = 1;
        } else if (argc == 7) {
          jpeg_record(sv, argv[2], 0, start, nframes, 0, argv[5], argv[6]);
          ok = 1;
        }
      } else if(argc == 4) {
        jpeg_record(sv, argv[2], argv[3], 0, 1, 0, "1.0", "once");
        ok = 1;
      }
    } else if((strcmp(argv[1], "vtrload") == 0) && (argc > 2)) {
      if((argc >= 4) && ((!strcmp(argv[2], "ram")) ||
                         (!strcmp(argv[2], "disk")))) {
	if(argc > 4)
	  start = atoi(argv[3]);
	if(argc > 5)
	  nframes = atoi(argv[4]);
	if(argc <= 6) {
	  jpeg_record(sv, argv[2], 0, start, nframes, 
                                           argv[argc - 1],  "1.0", "once");
	  ok = 1;
	}
      } else if(argc == 5) {
	jpeg_record(sv, argv[2], argv[3], 0, 1, argv[4], "1.0", "once");
	ok = 1;
      }
    } else if((strcmp(argv[1], "display") == 0) && (argc > 2)) {
      if((argc > 2) && ((!strcmp(argv[2], "ram")) ||
                        (!strcmp(argv[2], "disk")))) {
	if(argc > 3)
	  start = atoi(argv[3]);
	if(argc > 4)
	  nframes = atoi(argv[4]);
 	if(argc <= 5) {
	  jpeg_display(sv, argv[2], 0, start, nframes, 0);
	  ok = 1;
	}
      } else if(argc == 4) {
	jpeg_display(sv, argv[2], argv[3], 0, 1, 0);
	ok = 1;
      } 
    } else if((strcmp(argv[1], "vtrsave") == 0) && (argc > 2)) {
      if((argc > 3) && ((!strcmp(argv[2], "ram")) ||
                        (!strcmp(argv[2], "disk")))) {
	if(argc > 4)
	  start = atoi(argv[3]);
	if(argc > 5)
	  nframes = atoi(argv[4]);
	if(argc <= 6) {
	  jpeg_display(sv, argv[2], 0, start, nframes, argv[argc - 1]);
	  ok = 1;
	}
      } else if(argc == 5) {
  	jpeg_display(sv, argv[2], argv[3], 0, 1, argv[4]);
	ok = 1;
      }
    } else if((strcmp(argv[1], "step") == 0)) {
      jpeg_step(sv, argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "pause") == 0)) {
      jpeg_pause(sv);
      ok = 1;
    } else if((strcmp(argv[1], "speed") == 0)) {
      char *speed    = "1.0";
      char *loopmode = "default";

      switch(argc) {
      case 4:
        loopmode = argv[3];
	/* FALLTHROUGH */
      case 3:
        speed = argv[2];
	/* FALLTHROUGH */
      case 2:
        jpeg_speed(sv, speed, loopmode);
        ok = 1;
      }
    } else if(((strncmp(argv[1], "load", 4) == 0) || (strncmp(argv[1], "cmp", 3) == 0))&& (argc >= 5) && (argc <= 8)) {
      jpeg_host2isp(sv, argv[1], argv[2], argv[3], argc>=6?atoi(argv[5]):0, 
           atoi(argv[4]), argc>=7?atoi(argv[6]):1, argc>=8?argv[7]:"vk");
      
      ok = 1;
    } else if((strncmp(argv[1], "save", 4) == 0) && (argc >= 5) && (argc <= 8)) {
      jpeg_isp2host(sv, argv[1], argv[2], argv[3], argc>=6?atoi(argv[5]):0, 
        atoi(argv[4]), argc>=7?atoi(argv[6]):1, argc>=8?argv[7]:"vk");
      
      ok = 1;
    } else if(!strcmp(argv[1], "black")) {  
      jpeg_black(sv);
      ok = 1;
    } else if(!strcmp(argv[1], "colorbar")) {    
      jpeg_colorbar(sv);
      ok = 1;
    } else if((!strcmp(argv[1], "live")) || (!strcmp(argv[1], "fullee")) || (!strcmp(argv[1], "showinput"))){      
      if(argc > 2) {
        jpeg_live(sv, argv[2]);
      } else {
        jpeg_live(sv, NULL);
      }
      ok = 1;
    } else if(strcmp(argv[1], "info") == 0) {
      if(argc > 2) {
        if(!strcmp(argv[2], "input")) {
          jpeg_info_input(sv);
        } else if(!strcmp(argv[2], "hardware") || !strcmp(argv[2], "hw")) {
          jpeg_info_hardware(sv);
        } else if(!strcmp(argv[2], "closedcaption") || !strcmp(argv[2], "cc")) {
          jpeg_info_closedcaption(sv);
        } else if(!strcmp(argv[2], "clock")) {
          jpeg_info_clock(sv);
        } else if(!strcmp(argv[2], "random")) {
          jpeg_info_random(sv);
        } else if(!strcmp(argv[2], "publickey")) {
          jpeg_info_publickey(sv);
        } else if(!strcmp(argv[2], "timecode") || !strcmp(argv[2], "tc")) {
          if(argc>3) {
            jpeg_info_timecode(sv, argv[3]);
          } else {
            jpeg_info_timecode(sv, "help");
          }
        } else if(!strcmp(argv[2], "setup") || !strcmp(argv[2], "card")) {
          jpeg_info_card(sv);
        } else {
          jpeg_info(sv);
        }
      } else {
        jpeg_info(sv);
      }
      ok = 1;
    } else if(strcmp(argv[1], "fifo") == 0) {
      if(argc > 2) {
        if(!strcmp(argv[2], "memorymode")) {
          if(argc > 3 && argc <=4) {
            jpeg_fifo_memorymode(sv, argv[3], "");
            ok = 1;
          } else if(argc > 3 && argc <=5) {
            jpeg_fifo_memorymode(sv, argv[3], argv[4]);
            ok = 1;
          }
        }
      }
    } else if((!strcmp(argv[1], "mode") || !strcmp(argv[1], "init")) && (argc > 2)) {
      jpeg_mode(sv, argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "color")) && (argc > 2)) {
      jpeg_mode(sv, argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "hwwatchdog")) && (argc > 2)) {
      jpeg_hwwatchdog(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "croppingmode")) && (argc > 2)) {
      jpeg_croppingmode(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "sync")) && (argc > 2)) {
      jpeg_sync(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "syncout")) && (argc > 2)) {
      jpeg_syncout(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "timecode")) && (argc > 2)) {
      jpeg_timecode(sv, argc-2, &argv[2]);
      ok = 1;
    } else if(!strcmp(argv[1], "stop")) {
      jpeg_stop(sv);
      ok = 1;
    } else if(!strcmp(argv[1], "memsetup")) {
      jpeg_memsetup(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "repeat")) && (argc > 2)) {
      jpeg_repeat(sv, argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "dropmode")) && (argc > 2)) {
      jpeg_dropmode(sv, argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "slowmotion") == 0) && (argc > 2)) {
      jpeg_slowmotion(sv, argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "framesync") == 0) && (argc > 2)) {
      jpeg_framesync(sv, argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "master") == 0)) {
      ok = jpeg_master(sv, argc - 2,  &argv[2]);
    } else if((strcmp(argv[1], "rs422pinout") == 0)) {
      ok = jpeg_rs422pinout(sv, argc - 2,  &argv[2]);
    } else if((strcmp(argv[1], "version") == 0) && (argc >= 2)) {
      jpeg_version(sv, argc-2, (argc>2) ? &argv[2] : NULL);
      ok = 1;
    } else if((argc == 4) && (strcmp(argv[1], "vtredit") == 0)) {
      jpeg_vtredit(sv, argv[2], atoi(argv[3]));
      ok = 1;
    } else if((strcmp(argv[1], "slave") == 0)) {
      if (argc == 3) {
        jpeg_slave(sv, argv[2]);
        ok = 1;
      }
    } else if((strcmp(argv[1], "inpoint") == 0) && (argc == 3)) {
      jpeg_inpoint(sv, atoi(argv[2]));
      ok = 1;
    } else if((strcmp(argv[1], "outpoint") == 0) && (argc == 3)) {
      jpeg_outpoint(sv, atoi(argv[2]));
      ok = 1;
    } else if((strcmp(argv[1], "debug") == 0) && (argc == 3)) {
      jpeg_debug(sv, atoi(argv[2]));
      ok = 1;
    } else if((strcmp(argv[1], "debugvalue") == 0) && (argc == 3)) {
      jpeg_debugvalue(sv, atoi(argv[2]));
      ok = 1;
    } else if((strcmp(argv[1], "trace") == 0) && (argc == 3)) {
      jpeg_trace(sv, argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "debugprint") == 0) && (argc == 2)) {
      jpeg_debugprint(sv, FALSE);
      ok = 1;
    } else if((!strcmp(argv[1], "svhs")) && (argc == 3)) {
      jpeg_analog(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "analog")) && (argc >= 3)) {
      jpeg_analog(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "analogoutput")) && (argc >= 3)) {
      jpeg_analogoutput(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "dvi")) && (argc >= 3)) {
      jpeg_dvi(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((!strcmp(argv[1], "option")) && (argc >= 3)) {
      jpeg_option(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "forcerasterdetect") == 0)) {
      jpeg_forcerasterdetect(sv, argc-2, argv+2);
      ok = 1;
    } else if((strcmp(argv[1], "pulldown") == 0) && (argc >= 3)) {     
      jpeg_pulldown(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "fastmotion") == 0) && (argc == 3)) {
      jpeg_fastmode(sv, argv[2]);
      ok = 1;
    } else if(strcmp(argv[1], "proxy") == 0) {
      jpeg_proxy(sv, argc-2, &argv[2]);
      ok = 1;
    } else if(strcmp(argv[1], "lut") == 0) {
      jpeg_lut(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "iomode") == 0) && (argc >= 3)) {
      jpeg_iomode(sv, argv[2], (argc == 3)?NULL:argv[3]);
      ok = 1;
    } else if((strcmp(argv[1], "iospeed") == 0) && (argc >= 3)) {
      jpeg_iospeed(sv, argv[2]);
      ok = 1;
    } else if(!strcmp(argv[1], "licence") || !strcmp(argv[1], "license")) {
      if((argc == 3) && (!strcmp(argv[2], "show") || !strcmp(argv[2], "info"))) {
        jpeg_showlicence(sv);
        ok = 1;
      } else if(argc >= 3) {
        jpeg_licence(sv, argc-1, &argv[1]);
        ok = 1;
      }
    } else if((argc >= 3) && !strcmp(argv[1], "audioanalogout")) {
      jpeg_audioanalogout(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((argc >= 3) && !strcmp(argv[1], "audiofrequency")) {
      jpeg_audiofrequency(sv, argv[2]);
      ok = 1;
    } else if((argc >= 3) && !strcmp(argv[1], "audiochannels")) {
      jpeg_audiochannels(sv, argv[2]);
      ok = 1;
    } else if((argc == 3) && !strcmp(argv[1], "audioinput")) {
      jpeg_audioinput(sv, argv[2]);
      ok = 1;
    } else if((argc == 3) && !strcmp(argv[1], "audiomaxaiv")) {
      jpeg_audiomaxaiv(sv, argv[2]);
      ok = 1;
    } else if((argc >= 3) && !strcmp(argv[1], "audioaesrouting")) {
      jpeg_audioaesrouting(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((argc == 3) && !strcmp(argv[1], "audiomute")) {
      jpeg_audiomute(sv, argv[2]);
      ok = 1;
    } else if ( argc==3 && SV_STRMATCH(argv[1],"audio_speed_compensation") ) {
      jpeg_audio_speed_compensation(sv,argv[2]);
      ok = 1;
    } else if ( argc==3 && (SV_STRMATCH(argv[1],"audio") || SV_STRMATCH(argv[1],"audiomode"))) {
      jpeg_audiomode(sv,argv[2]);
      ok = 1;
    } else if(!strcmp(argv[1], "gpi")) {
      jpeg_gpi(sv,argc-2,&argv[2]);
      ok = 1;
    } else if ( argc==3 && SV_STRMATCH(argv[1],"inputport") ) {
      jpeg_inputport(sv,argv[2]);
      ok = 1;
    } else if ( argc==3 && !strcmp(argv[1],"outputport")) {
      jpeg_outputport(sv, argv[2]);
      ok = 1;
    } else if ( argc>=3 && !strcmp(argv[1], "mainoutput")) {
      jpeg_mainoutput(sv, argc-2, &argv[2]);
      ok = 1;
    } else if(!strcmp(argv[1], "mixer") && (argc > 2)) {
      jpeg_mixer(sv, argc-2, &argv[2]);
      ok = 1;
    } else if(!strcmp(argv[1], "zoom") && (argc > 2)) {
      jpeg_zoom(sv, argc-2, &argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "disableswitchingline") == 0) && (argc > 2)) {
      jpeg_disableswitchingline(sv, argc-2, &argv[2]);
      ok = 1;
    } else if ( argc>=2 && strcmp(argv[1], "sequence") == 0) {
      jpeg_sequence(sv, argc-1, argv+1);
      ok = 1;
    } else if ( argc>=2 && strcmp(argv[1], "ltc") == 0) {
      jpeg_ltc(sv, argc-2, argv+2);
      ok = 1;
    } else if((argc > 2) && (strcmp(argv[1], "rs422") == 0)) {
      jpeg_rs422(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "wordclock") == 0)) {
      jpeg_wordclock(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "anccomplete") == 0)) {
      jpeg_anccomplete(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "ancgenerator") == 0)) {
      jpeg_ancgenerator(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "rp165") == 0)) {
      jpeg_rp165(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "ancreader") == 0)) {
      jpeg_ancreader(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "autopulldown") == 0)) {
      jpeg_autopulldown(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "slaveinfo") == 0)) {
      jpeg_slaveinfo(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 2) && (strcmp(argv[1], "matrix") == 0)) {
      jpeg_matrix(sv, argc-1, argv+1);
      ok = 1;
    } else if((strcmp(argv[1], "vitcline") == 0) && (argc > 2)) {
      jpeg_vitcline(sv, argc-2, argv+2);
      ok = 1;
    } else if((strcmp(argv[1], "vitcreaderline") == 0) && (argc > 2)) {
      jpeg_vitcreaderline(sv, argv[2]);
      ok = 1;
    } else if((strcmp(argv[1], "recordmode") == 0) && (argc > 2)) {
      jpeg_recordmode(sv, argc-1, argv+1);
      ok = 1;
    } else if((strcmp(argv[1], "monitorinfo") == 0)) {
      jpeg_monitorinfo(sv, argc-1, argv+1);
      ok = 1;
    } else if((strcmp(argv[1], "jack") == 0)) {
      jpeg_jack(sv, argc-1, argv+1);
      ok = 1;
    } else if((strcmp(argv[1], "test") == 0)) {
      jpeg_test(sv, argc-1, argv+1);
      ok = 1;
    } else if(strcmp(argv[1], "quit") == 0) {
      ok = 1;
    } else if((argc == 3) && !strcmp(argv[1], "dominance")) {
      ok = jpeg_dominance(sv, argc-2, &argv[2]);
    } else if((argc >= 2) && (strcmp(argv[1], "multichannel") == 0)) {
      jpeg_multichannel(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 3) && (strcmp(argv[1], "routing") == 0)) {
      jpeg_routing(sv, argc-1, argv+1);
    } else if((argc >= 2) && (strcmp(argv[1], "alphamixer") == 0)) {
      jpeg_alphamixer(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 3) && (strcmp(argv[1], "linkencrypt") == 0)) {
      jpeg_linkencrypt(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 3) && (strcmp(argv[1], "watermark") == 0)) {
      jpeg_watermark(sv, argc-1, argv+1);
      ok = 1;
    } else if((argc >= 3) && (strcmp(argv[1], "render") == 0)) {
      jpeg_render(sv, argc-1, argv+1);
      ok = 1;
    }
  }

  if(!ok) {
    if(!sv->vgui) {
#ifdef WIN32
      printf("For help type: sv help or sv ?\n");
#else
      printf("For help type: sv help\n");
#endif
    } else {
      printf("Unknown sv command: ");
      for(start = 0; start < argc; start++) {
        printf("%s ", argv[start]);  
      }
      printf("\n");
      return -1;
    }
  } else {
    return 0;
  }

  return 0;
}


