/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    dpxio - Shows the use of the FIFO API to display and record video from/to DPX
//            files and audio from/to an AIFF file.
//
*/


#include "dpxio.h"

/**
//  \defgroup svdpxio Example - dpxio
//
//  This chapter shows the use of the FIFO API (see \ref fifoapi) to display and record video from/to DPX
//  files and audio from/to an AIFF file.
//  @{
*/


static uint16 ancparity[512] = {
  0x200, 0x101, 0x102, 0x203, 0x104, 0x205, 0x206, 0x107,
  0x108, 0x209, 0x20a, 0x10b, 0x20c, 0x10d, 0x10e, 0x20f,
  0x110, 0x211, 0x212, 0x113, 0x214, 0x115, 0x116, 0x217,
  0x218, 0x119, 0x11a, 0x21b, 0x11c, 0x21d, 0x21e, 0x11f,
  0x120, 0x221, 0x222, 0x123, 0x224, 0x125, 0x126, 0x227,
  0x228, 0x129, 0x12a, 0x22b, 0x12c, 0x22d, 0x22e, 0x12f,
  0x230, 0x131, 0x132, 0x233, 0x134, 0x235, 0x236, 0x137,
  0x138, 0x239, 0x23a, 0x13b, 0x23c, 0x13d, 0x13e, 0x23f,
  0x140, 0x241, 0x242, 0x143, 0x244, 0x145, 0x146, 0x247,
  0x248, 0x149, 0x14a, 0x24b, 0x14c, 0x24d, 0x24e, 0x14f,
  0x250, 0x151, 0x152, 0x253, 0x154, 0x255, 0x256, 0x157,
  0x158, 0x259, 0x25a, 0x15b, 0x25c, 0x15d, 0x15e, 0x25f,
  0x260, 0x161, 0x162, 0x263, 0x164, 0x265, 0x266, 0x167,
  0x168, 0x269, 0x26a, 0x16b, 0x26c, 0x16d, 0x16e, 0x26f,
  0x170, 0x271, 0x272, 0x173, 0x274, 0x175, 0x176, 0x277,
  0x278, 0x179, 0x17a, 0x27b, 0x17c, 0x27d, 0x27e, 0x17f,
  0x180, 0x281, 0x282, 0x183, 0x284, 0x185, 0x186, 0x287,
  0x288, 0x189, 0x18a, 0x28b, 0x18c, 0x28d, 0x28e, 0x18f,
  0x290, 0x191, 0x192, 0x293, 0x194, 0x295, 0x296, 0x197,
  0x198, 0x299, 0x29a, 0x19b, 0x29c, 0x19d, 0x19e, 0x29f,
  0x2a0, 0x1a1, 0x1a2, 0x2a3, 0x1a4, 0x2a5, 0x2a6, 0x1a7,
  0x1a8, 0x2a9, 0x2aa, 0x1ab, 0x2ac, 0x1ad, 0x1ae, 0x2af,
  0x1b0, 0x2b1, 0x2b2, 0x1b3, 0x2b4, 0x1b5, 0x1b6, 0x2b7,
  0x2b8, 0x1b9, 0x1ba, 0x2bb, 0x1bc, 0x2bd, 0x2be, 0x1bf,
  0x2c0, 0x1c1, 0x1c2, 0x2c3, 0x1c4, 0x2c5, 0x2c6, 0x1c7,
  0x1c8, 0x2c9, 0x2ca, 0x1cb, 0x2cc, 0x1cd, 0x1ce, 0x2cf,
  0x1d0, 0x2d1, 0x2d2, 0x1d3, 0x2d4, 0x1d5, 0x1d6, 0x2d7,
  0x2d8, 0x1d9, 0x1da, 0x2db, 0x1dc, 0x2dd, 0x2de, 0x1df,
  0x1e0, 0x2e1, 0x2e2, 0x1e3, 0x2e4, 0x1e5, 0x1e6, 0x2e7,
  0x2e8, 0x1e9, 0x1ea, 0x2eb, 0x1ec, 0x2ed, 0x2ee, 0x1ef,
  0x2f0, 0x1f1, 0x1f2, 0x2f3, 0x1f4, 0x2f5, 0x2f6, 0x1f7,
  0x1f8, 0x2f9, 0x2fa, 0x1fb, 0x2fc, 0x1fd, 0x1fe, 0x2ff,
  0x100, 0x201, 0x202, 0x103, 0x204, 0x105, 0x106, 0x207,
  0x208, 0x109, 0x10a, 0x20b, 0x10c, 0x20d, 0x20e, 0x10f,
  0x210, 0x111, 0x112, 0x213, 0x114, 0x215, 0x216, 0x117,
  0x118, 0x219, 0x21a, 0x11b, 0x21c, 0x11d, 0x11e, 0x21f,
  0x220, 0x121, 0x122, 0x223, 0x124, 0x225, 0x226, 0x127,
  0x128, 0x229, 0x22a, 0x12b, 0x22c, 0x12d, 0x12e, 0x22f,
  0x130, 0x231, 0x232, 0x133, 0x234, 0x135, 0x136, 0x237,
  0x238, 0x139, 0x13a, 0x23b, 0x13c, 0x23d, 0x23e, 0x13f,
  0x240, 0x141, 0x142, 0x243, 0x144, 0x245, 0x246, 0x147,
  0x148, 0x249, 0x24a, 0x14b, 0x24c, 0x14d, 0x14e, 0x24f,
  0x150, 0x251, 0x252, 0x153, 0x254, 0x155, 0x156, 0x257,
  0x258, 0x159, 0x15a, 0x25b, 0x15c, 0x25d, 0x25e, 0x15f,
  0x160, 0x261, 0x262, 0x163, 0x264, 0x165, 0x166, 0x267,
  0x268, 0x169, 0x16a, 0x26b, 0x16c, 0x26d, 0x26e, 0x16f,
  0x270, 0x171, 0x172, 0x273, 0x174, 0x275, 0x276, 0x177,
  0x178, 0x279, 0x27a, 0x17b, 0x27c, 0x17d, 0x17e, 0x27f,
  0x280, 0x181, 0x182, 0x283, 0x184, 0x285, 0x286, 0x187,
  0x188, 0x289, 0x28a, 0x18b, 0x28c, 0x18d, 0x18e, 0x28f,
  0x190, 0x291, 0x292, 0x193, 0x294, 0x195, 0x196, 0x297,
  0x298, 0x199, 0x19a, 0x29b, 0x19c, 0x29d, 0x29e, 0x19f,
  0x1a0, 0x2a1, 0x2a2, 0x1a3, 0x2a4, 0x1a5, 0x1a6, 0x2a7,
  0x2a8, 0x1a9, 0x1aa, 0x2ab, 0x1ac, 0x2ad, 0x2ae, 0x1af,
  0x2b0, 0x1b1, 0x1b2, 0x2b3, 0x1b4, 0x2b5, 0x2b6, 0x1b7,
  0x1b8, 0x2b9, 0x2ba, 0x1bb, 0x2bc, 0x1bd, 0x1be, 0x2bf,
  0x1c0, 0x2c1, 0x2c2, 0x1c3, 0x2c4, 0x1c5, 0x1c6, 0x2c7,
  0x2c8, 0x1c9, 0x1ca, 0x2cb, 0x1cc, 0x2cd, 0x2ce, 0x1cf,
  0x2d0, 0x1d1, 0x1d2, 0x2d3, 0x1d4, 0x2d5, 0x2d6, 0x1d7,
  0x1d8, 0x2d9, 0x2da, 0x1db, 0x2dc, 0x1dd, 0x1de, 0x2df,
  0x2e0, 0x1e1, 0x1e2, 0x2e3, 0x1e4, 0x2e5, 0x2e6, 0x1e7,
  0x1e8, 0x2e9, 0x2ea, 0x1eb, 0x2ec, 0x1ed, 0x1ee, 0x2ef,
  0x1f0, 0x2f1, 0x2f2, 0x1f3, 0x2f4, 0x1f5, 0x1f6, 0x2f7,
  0x2f8, 0x1f9, 0x1fa, 0x2fb, 0x1fc, 0x2fd, 0x2fe, 0x1ff,
};


int dpxio_dump_ancstream(short * anc, int size )
{
  int hancsize = 536 >> 1;
  int vancsize = 3840 >> 1;
  int line;
  int position;
  int value;
  short * start = anc;
  
  // skip one VANC line
  anc += vancsize;

  // Iterate during all lines
  for(line = 8; line < 8 + 12; line++)
  {
    // dump HANC
    printf("HANC line:%d\n", line);
    for(position = 0; position < hancsize; position++, anc++) {
      value = (*anc) & 0x3ff;
      if(value != 0x040) {
        printf("%03x ", value);
      }
    }
    printf("\n");
 
    // dump VANC
    printf("VANC line:%d\n", line);
    for(position = 0; position < vancsize; position++, anc++) {
      value = (*anc) & 0x3ff;
      if( ((anc[0]&0x3ff)==0x0) && ((anc[1]&0x3ff)==0x3ff) && ((anc[2]&0x3ff)==0x3ff) ) {
        printf("\n000 3ff 3ff: ");
        anc+=2;
        position+=2;
      } else { 
        if(value != 0x040) {
          printf("%03x ", value);
        }
      }
    }
    printf("\n\n\n");
  }

  if( ((anc - start) * 2) > size ) {
    printf("Error checked %d bytes but the buffer is only %d bytes big.\n", ((anc - start) * 2), size );
  } else {
   printf("Checked %d bytes and the buffer is %d bytes big.\n", ((anc - start) * 2), size );
  }

  return SV_OK;
}

void dpxio_generate_ancpacket( char * anc_packet, int size, int did, int sdid, int line, int field, int packet )
{
#define HEADERSIZE     6    //0x000, 0x3FF, 0x3FF, DID, SDID, DC (data count)
#define USERDATASIZE   255  //User data
#define CHECKSUMSIZE   1    //Sum of all data before inclusive the header.
#define DCPOSITION     5

  int data_pos = 0;
  int dc       = 0;
  int crc      = 0;
  int count    = 0;

  //Header
  ////////
  if( size >= HEADERSIZE ) {
    anc_packet[data_pos++] = 0x000;           //The beginning of one ANC packet.
    anc_packet[data_pos++] = 0x3ff;           //The beginning of one ANC packet.
    anc_packet[data_pos++] = 0x3ff;           //The beginning of one ANC packet.
    anc_packet[data_pos++] = ancparity[did];  //DID
    anc_packet[data_pos++] = ancparity[sdid]; //SDID
    anc_packet[data_pos++] = ancparity[dc];   //Data Count
  }

  //Userdata
  //////////
  if( size >= HEADERSIZE + 2 ) {
    anc_packet[data_pos++] = ancparity[(line&0xff)   ]; //LSB byte of the line number
    anc_packet[data_pos++] = ancparity[((packet<<4)&0xff) ]; //Packet Number and Field ...
    dc+=2;
  }
  if( size >= (HEADERSIZE + USERDATASIZE) ) {
    for(count = 0; data_pos < (HEADERSIZE + USERDATASIZE); count++) {
      anc_packet[data_pos++] = ancparity[count];
      dc++;
    }
  }
 
  //Overwrite datacount
  if( size >= HEADERSIZE ) {
    anc_packet[DCPOSITION] = ancparity[dc];   //Data Count
  }

  //CRC - generate checksum
  /////////////////////////
  for( data_pos = 0; data_pos < (HEADERSIZE + USERDATASIZE); data_pos++) {
    crc += anc_packet[data_pos];
  }
  if(crc & 0x100) {
    anc_packet[data_pos] = crc & 0x1ff;
  } else {
    anc_packet[data_pos] = (crc & 0x0ff) | 0x200;
  }
}

int dpxio_generate_ancstream(short * anc, int size, int npackets)
{
  // buffer for one
  char anc_packet[ HEADERSIZE + USERDATASIZE + CHECKSUMSIZE ];

  int hancsize = 536 /*byte*/ >> 1;
  int vancsize = 3840/*byte*/ >> 1;
  int line   = 0;
  int packet = 0;
  int pos    = 0;
  short * start = anc;

  // Fill VANC with zero
  for(pos = 0; pos < vancsize; pos++) {
    *(anc++) = 0x040;
  }

  for(line = 8; line < 8 + 12; line++) 
  {
    // Fill HANC with zero
    for(pos = 0; pos < hancsize; pos++) {
      *(anc++) = 0x040;
    }

    // Fill VANC with the ancpackets
    for(packet = 0; packet < npackets; packet++)
    {
      // Generate packet
      dpxio_generate_ancpacket( (char*)&anc_packet, sizeof(anc_packet), 0x5f, 0xfa, line, 0, packet+1 );

      for(pos = 0; pos < sizeof(anc_packet); pos++) {
        *(anc++) = anc_packet[pos];
      }
    }
    // Fill the rest of VANC with zero
    for( pos = 0; pos < vancsize - ((int)sizeof(anc_packet)*npackets); pos++) {
      *(anc++) = 0x040;
    }
  }

  if( ((anc - start) * 2) > size ) {
    printf("Error filled %d bytes but the buffer is only %d bytes big.\n", ((anc - start) * 2), size );
  } else {
   printf("Filled %d bytes and the buffer is %d bytes big.\n", ((anc - start) * 2), size );
  }

  return SV_OK;
}


/**
//  \ingroup svdpxio
//
//  This function is used internally by the example program and sets explicit timecode values for a tick.
//
//  \param dpxio       --- Application handle.
//  \param frame       --- Frame value to be used for timecode.
//  \param tick        --- Tick at which the specific timecode should appear.
//  \param fieldcount  --- Number of fields for which timecodes should be set.
*/
void dpxio_set_timecodes(dpxio_handle * dpxio, int frame, int tick, int fieldcount)
{
  int value;

  for(; fieldcount; fieldcount--, tick++) {
    value = ((tick & 1) ? 0x80000000 : 0) | frame;

    if(!(tick & 1)) {
      sv_option_setat(dpxio->sv, SV_OPTION_VTR_UB, value, tick);
      sv_option_setat(dpxio->sv, SV_OPTION_LTC_UB, value, tick);
    }
    sv_option_setat(dpxio->sv, SV_OPTION_VITC_UB, value, tick);

    if(dpxio->config.verbosetc) {
      printf("SETAT %3d %d %08x\n", frame, tick, value);
    }
  }
}

/**
//  \ingroup svdpxio
//
//  This function is the execution function of the example program. All FIFO calls are performed in this function.
//
//  \param dpxio       --- Application handle.
//  \param framecount  --- Number of frames to record/display.
//  \return            The number of frames that were actually recorded/displayed.
*/
int dpxio_exec(dpxio_handle * dpxio, int framecount)
{
  sv_fifo *             pfifo     = NULL;
  sv_fifo_info          info      = {0};
  sv_fifo_info          infostart = {0};
  sv_fifo_buffer *      pbuffer;
  sv_fifo_bufferinfo    bufferinfo = {0};
  sv_fifo_bufferinfo *  pbufferinfo;
  sv_fifo_bufferinfo    putbufferinfo = {0};
  sv_storageinfo        storage = {0};
  char *                videobuffer[DPXIO_MAXBUFFER];
  char *                audiobuffer[DPXIO_MAXBUFFER];
  char *                mallocbuffer[DPXIO_MAXBUFFER] = { 0 };
  unsigned char         lut[6][32768];

  unsigned int vtimecode;
  unsigned int atimecode;
  unsigned int xtimecode;

  int   aoffset;
  int   running   = TRUE;
  int   bstarted  = FALSE;
  int   badjusted = FALSE;
  int   res       = SV_OK;
  int   res_a     = SV_OK;
  int   when      = 0;
  int   when_org  = 0;
  int   flagbase  = 0;
  int   nbits     = 10;
  int   dpxtype   = 50;
  int   offset    = 0x2000;
  int   padding   = 0;
  int   vsize     = 0;
  int   xsize     = 256;
  int   ysize     = 256;
  int   lineoffset= 0;
  int   field     = 0;
  int   frame;
  int   local_frame = 0;
  int   local_tick = 0;
  int   fieldcount = 1;
  int   starttick = 0;
  int   fifoflags = 0;
  int   i,j;
  sv_fifo_configinfo config = {0};
  short * anc     = 0;
  short * anc_org = 0;
  int     syncstate = 0;
  int     timeout;
  int   audioInputErrorCount = 0;

  char  pdphase = 0;

  if(dpxio->config.lut) {
    dpxio_3dlut_identity(lut[0]);
    dpxio_3dlut_invert(lut[1]);
    dpxio_1dlut_identity(lut[2]);
    dpxio_1dlut_invert(lut[3]);
    dpxio_1dlutsmall_identity(lut[4]);
    dpxio_1dlutsmall_invert(lut[5]);
  }

  /**
  //  <BR>First, we want to check the video raster settings that are currently initialized to be able to 
  //  set the size for the captured frame. For this  we start by querying the hardware with the function \e sv_storage_status().
  //  
  //  In all error logging functions we use the function \e sv_geterrortext() to get a readable form
  //  of the SV error codes.
  //
  //  \verbatim
  //  res = sv_storage_status(dpxio->sv, dpxio->binput ? 1 : 0, NULL, &storage, sizeof(storage), SV_STORAGEINFO_COOKIEISJACK);
  //  if(res != SV_OK) {
  //    printf("ERROR: sv_storage_status() failed = %d '%s'\n", res, sv_geterrortext(res));
  //    running = FALSE;
  //  }
  //  \endverbatim
  */
  res = sv_storage_status(dpxio->sv, dpxio->binput ? 1 : 0, NULL, &storage, sizeof(storage), SV_STORAGEINFO_COOKIEISJACK);
  if(res != SV_OK) {
    printf("ERROR: sv_storage_status() failed = %d '%s'\n", res, sv_geterrortext(res));
    running = FALSE;
  }

  if(dpxio->binput) {
    /**
    //  For the input we use the size returned by the function \e sv_storage_status(), for the output the size of the 
    //  read DPX frames will be used. We also set the color mode of the DPX frames
    //  to the proper format.
    */
    xsize = storage.storagexsize;
    ysize = storage.storageysize;
    if(dpxio->config.fieldbased) {
      ysize /= 2;
    }
    nbits = storage.nbits;
    switch(storage.colormode) {
    case SV_COLORMODE_YUV422:
      dpxtype = 100;
      break;
    default:
      dpxtype = 50;
    }
  } 

  if(dpxio->config.fieldbased) {
    if(dpxio->binput) {
      // flag needs to be specified at sv_fifo_init already
      flagbase |= SV_FIFO_FLAG_FIELD;
    } else {
      // flag can be set with each sv_fifo_getbuffer call
      fifoflags |= SV_FIFO_FLAG_FIELD;
    }
  }

  
  fifoflags |= SV_FIFO_FLAG_NODMAADDR;
  if(dpxio->hw.audiochannels > 1) {
    fifoflags |= SV_FIFO_FLAG_AUDIOINTERLEAVED;
  }
  if(dpxio->config.ancstream) {
    flagbase |= SV_FIFO_FLAG_ANC;
    fifoflags |= SV_FIFO_FLAG_ANC;
  }
  if(!dpxio->binput && !dpxio->io.dryrun && dpxio->bvideo) {
    fifoflags |= SV_FIFO_FLAG_STORAGEMODE;
  }

  xtimecode = ((dpxio->vtr.tc & 0xff000000) >> 24) | 
              ((dpxio->vtr.tc & 0x00ff0000) >> 8)  |
              ((dpxio->vtr.tc & 0x0000ff00) << 8)  |
              ((dpxio->vtr.tc & 0x000000ff) << 24);
   
  if(dpxio->pulldown) {
    /**
    //  <b>Pulldown</b>
    //  <br>The pulldown flag is needed in the parameter \e flagbase for pulldown removal on input.
    */
    flagbase |= SV_FIFO_FLAG_PULLDOWN;
    fifoflags |= SV_FIFO_FLAG_PULLDOWN;

    /*
    // Set pulldown startphase for next record/display.
    */
    res = sv_pulldown(dpxio->sv, SV_PULLDOWN_CMD_STARTPHASE, dpxio->pulldownphase);
    if(res != SV_OK) {
      printf("ERROR: sv_pulldown(SV_PULLDOWN_CMD_STARTPHASE) failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    }

    if(dpxio->config.vtrcontrol) {
      if((dpxio->pulldownphase == SV_PULLDOWN_STARTPHASE_C) || (dpxio->pulldownphase == SV_PULLDOWN_STARTPHASE_D)) {
        // Need to start edit on second field
        res = sv_vtrmaster(dpxio->sv, SV_MASTER_EDITFIELD_START, 2);
        if((res != SV_OK) && (res != SV_ERROR_VTR_OFFLINE)) {
          printf("ERROR: Setting editfield start failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
        res = sv_vtrmaster(dpxio->sv, SV_MASTER_EDITFIELD_END,   2);
        if((res != SV_OK) && (res != SV_ERROR_VTR_OFFLINE)) {
          printf("ERROR: Setting editfield end failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
      }
    }
  }
  if(dpxio->config.vtrcontrol) {
    fifoflags |= SV_FIFO_FLAG_TIMEDOPERATION;
  }

  switch(dpxio->config.repeatfactor) {
  case -1: /* 2.5 */
    break;
  case 2:
    fifoflags |= SV_FIFO_FLAG_REPEAT_2TIMES;
    break;
  case 3:
    fifoflags |= SV_FIFO_FLAG_REPEAT_3TIMES;
    break;
  case 4:
    fifoflags |= SV_FIFO_FLAG_REPEAT_4TIMES;
    break;
  }

  if(running) {
    if(dpxio->baudio) {
      if(dpxio->binput) {
        int audioformat = AUDIO_FORMAT_AIFF;
        if(strstr(dpxio->audioio.filename, ".wav")) {
          audioformat = AUDIO_FORMAT_WAVE;
        }
        if(dpxio_audio_create(dpxio, dpxio->audioio.filename, audioformat, 48000, 2, 16, (fifoflags & SV_FIFO_FLAG_AUDIOINTERLEAVED)?16:2) == AUDIO_FORMAT_INVALID) {
          printf("ERROR: Could not create audiofile\n");
          running = FALSE;
        }
      } else {
        if(dpxio_audio_open(dpxio, dpxio->audioio.filename, (fifoflags & SV_FIFO_FLAG_AUDIOINTERLEAVED)?16:2, NULL, NULL) == AUDIO_FORMAT_INVALID) {
          printf("ERROR: can't open audio file %s\n", dpxio->audioio.filename);
          running = FALSE;
        }
      }
    }
  }

  /*
  // Check if output pipeline is already locked to sync signal before starting the output FIFO.
  // This is important when the syncmode or videomode was reconfigured right before.
  */
  if(running && !dpxio->binput) {
    res = sv_query(dpxio->sv, SV_QUERY_SYNCSTATE, 0, &syncstate);
    if((res == SV_OK) && !syncstate) {
      printf("Waiting for output pipeline synchronization");

      timeout = 200;
      do {
        printf(".");
        res = sv_vsyncwait(dpxio->sv, SV_VSYNCWAIT_DISPLAY, NULL);

        if(res == SV_OK) {
          res = sv_query(dpxio->sv, SV_QUERY_SYNCSTATE, 0, &syncstate);
        }
      } while((res == SV_OK) && !syncstate && (timeout-- > 0));

      printf("\n");

      if(timeout <= 0) {
        printf("ERROR: No synchronization, make sure to connect a proper sync signal.\n");
        running = FALSE;
      }
    }
  }

  if(running) {
    if(!dpxio->bvideo) {
      fifoflags |= SV_FIFO_FLAG_AUDIOONLY;
    } 
    if(!dpxio->baudio) {
      fifoflags |= SV_FIFO_FLAG_VIDEOONLY;
    }

    /** 
    //  The function \e sv_fifo_init() is then called to initialize the input or output FIFO.
    //  For the input FIFO we may have set the FIFO flagbase as the record is done before
    //  the FIFO input get-/putbuffer pair.
    //
    //  \verbatim
    //  res = sv_fifo_init(dpxio->sv, &pfifo, dpxio->binput, TRUE, TRUE, flagbase, 0);
    //  if(res != SV_OK)  {
    //    printf("ERROR: sv_fifo_init(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
    //    running = FALSE;
    //  }
    //  \endverbatim
    */
    res = sv_fifo_init(dpxio->sv, &pfifo, dpxio->binput, FALSE, TRUE, flagbase, 0);
    if(res != SV_OK)  {
      printf("ERROR: sv_fifo_init(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    } 
  }

  if(running) {
    res = sv_fifo_configstatus(dpxio->sv, pfifo, &config);
    if(res != SV_OK)  {
      printf("ERROR: sv_fifo_configstatus(sv) failed %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    }
  }

  if(dpxio->config.ancstream && config.ancbuffersize) {
    anc_org = malloc(config.ancbuffersize + (config.dmaalignment-1));
    anc = (short *)((uintptr)(anc_org + (config.dmaalignment-1)) & ~(uintptr)(config.dmaalignment-1));
    printf("ANC Buffersize: %x\n", config.ancbuffersize );
    if(!anc_org) {
      printf("ERROR: malloc(%d) anc failed\n", config.ancbuffersize + (config.dmaalignment-1));
      running = FALSE;
    } else {
      memset(anc, 0, config.ancbuffersize);
    }
  }

  if(running) {
    if(dpxio->config.rp215 || dpxio->config.rp215a) {
      if(!dpxio->pulldown) {
        res = sv_option(dpxio->sv, SV_OPTION_ANCUSER_LINENR, dpxio->config.rp215a?dpxio->config.rp215a:dpxio->config.rp215); // Line nr 9
        if(res != SV_OK)  {
          printf("ERROR: sv_option(sv,SV_OPTION_ANCUSER_LINENR) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
        res = sv_option(dpxio->sv, SV_OPTION_ANCUSER_DID, 0x51); // rp215 -> 0x151
        if(res != SV_OK)  {
          printf("ERROR: sv_option(sv,SV_OPTION_ANCUSER_DID) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }  
        res = sv_option(dpxio->sv, SV_OPTION_ANCUSER_SDID, 0x01); // rp215 -> 0x101
        if(res != SV_OK)  {
          printf("ERROR: sv_option(sv,SV_OPTION_ANCUSER_SDID) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }  
        res = sv_option(dpxio->sv, SV_OPTION_ANCUSER_FLAGS, SV_ANCUSER_FLAG_VANC); // rp215 is stored as VANC
        if(res != SV_OK)  {
          printf("ERROR: sv_option(sv,SV_OPTION_ANCUSER_FLAGS) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }  
        res = sv_option(dpxio->sv, SV_OPTION_ANCGENERATOR, SV_ANCDATA_USERDEF);
        if(res != SV_OK)  {
          printf("ERROR: sv_option(sv,SV_OPTION_ANCGENERATOR) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }  
        res = sv_option(dpxio->sv, SV_OPTION_ANCREADER, SV_ANCDATA_USERDEF);
        if(res != SV_OK)  {
          printf("ERROR: sv_option(sv,SV_OPTION_ANCREADER) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
      }
    }
  }

  /**
  //  <b>Memory Allocation</b>
  //  <br>As a next step we allocate DMA buffers because it is a good idea to page-align any buffers. This is not really needed, but
  //  due to the large sizes of video there will be no scatter/gather block for the DMA to start
  //  the transfer. All DMA buffers need at least to be at aligned to the size returned from the driver
  //  for the board's minimum DMA alignment, for example, for Centaurus this is 16&nbsp;bytes and for the SDStationOEM it is 8&nbsp;bytes.
  //
  */
  if(running) {
    for(i = 0; i < DPXIO_MAXBUFFER; i++) {
      if((mallocbuffer[i] = malloc(DPXIO_BUFFERVIDEOSIZE + DPXIO_BUFFERAUDIOSIZE + PAGESIZE)) == NULL) {
        running = FALSE;
      } else {
        if(dpxio->hw.dmaalignment > PAGESIZE) {
          videobuffer[i] = (void*)(((uintptr)mallocbuffer[i] + (dpxio->hw.dmaalignment-1)) & ~(uintptr)(dpxio->hw.dmaalignment-1));
        } else {
          videobuffer[i] = (void*)(((uintptr)mallocbuffer[i] + (PAGESIZE-1)) & ~(uintptr)(PAGESIZE-1));
        }
        audiobuffer[i] = videobuffer[i] + DPXIO_BUFFERVIDEOSIZE;
      }
    }

    if((dpxio->audio.tmpbuffer_org = malloc(DPXIO_BUFFERAUDIOSIZE + PAGESIZE)) == NULL) {
      running = FALSE;
    } else {
      if(dpxio->hw.dmaalignment > PAGESIZE) {
        dpxio->audio.tmpbuffer = (void*)(((uintptr)dpxio->audio.tmpbuffer_org + (dpxio->hw.dmaalignment-1)) & ~(uintptr)(dpxio->hw.dmaalignment-1));
      } else {
        dpxio->audio.tmpbuffer = (void*)(((uintptr)dpxio->audio.tmpbuffer_org + (PAGESIZE-1)) & ~(uintptr)(PAGESIZE-1));
      }
    }
  }


  if(dpxio->config.vtrcontrol) {
    /**
    //  <b>VTR Control</b>
    //  <br>Before a VTR transfer is performed you should set the edit settings to appropriate values:
    //
    //  \verbatim
    //  res = sv_vtrmaster(dpxio->sv, SV_MASTER_EDITFIELD_START, 1);
    //  if((res != SV_OK) && (res != SV_ERROR_VTR_OFFLINE)) {
    //    printf("ERROR: Setting edit field start failed = %d '%s'\n", res, sv_geterrortext(res));
    //    running = FALSE;
    //  }
    //  res = sv_vtrmaster(dpxio->sv, SV_MASTER_EDITFIELD_END,   1);
    //  if((res != SV_OK) && (res != SV_ERROR_VTR_OFFLINE)) {
    //    printf("ERROR: Setting edit field end failed = %d '%s'\n", res, sv_geterrortext(res));
    //    running = FALSE;
    //  }
    //  \endverbatim
    */
    res = sv_vtrmaster(dpxio->sv, SV_MASTER_EDITFIELD_START, 1);
    if((res != SV_OK) && (res != SV_ERROR_VTR_OFFLINE)) {
      printf("ERROR: Setting edit field start failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    }
    res = sv_vtrmaster(dpxio->sv, SV_MASTER_EDITFIELD_END,   1);
    if((res != SV_OK) && (res != SV_ERROR_VTR_OFFLINE)) {
      printf("ERROR: Setting edit field end failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    }

    /**
    //
    //  VTR control for the FIFO API is performed via the \e sv_vtrcontrol() function. The first 
    //  call with a set \e init parameter should be done with the VTR timecode and the number 
    //  of frames that should be edited. This call will instruct the driver to preroll the VTR 
    //  and commence a play once the VTR has reached the preroll point. Slightly before 
    //  the inpoint the VTR will be up to speed (if the preroll time was enough) and the function
    //  will return the tick (i.e. the parameter \e when) when the edit should commence. This can
    //  be fed into the structure <i>\ref sv_fifo_bufferinfo</i> of the \e sv_fifo_getbuffer() function to
    //  start the FIFO with a timed operation which will start the input or
    //  output at the correct position.
    //
    //  These two functions are mostly needed for edits on the VTR. An edit from the VTR can also be implemented
    //  with these functions, or by just scanning the appropriate fields in the FIFO \e pbuffer structure
    //  during a permanent capture.
    //
    //  \verbatim
    //  res = sv_vtrcontrol(dpxio->sv, dpxio->binput, TRUE, dpxio->vtr.tc, dpxio->vtr.nframes, &when, NULL, 0);
    //  while((res == SV_OK) && (when == 0)) {
    //    res = sv_vtrcontrol(dpxio->sv, dpxio->binput, FALSE, 0, 0, &when, &timecode, 0);
    //    sv_usleep(dpxio->sv, 50000);
    //  }
    //  \endverbatim
    //
    //  
    */
    res = sv_vtrcontrol(dpxio->sv, dpxio->binput, TRUE, dpxio->vtr.tc, dpxio->vtr.nframes, &when, NULL, 0);
    while((res == SV_OK) && (when == 0)) {
      char buffer[64];
      int  timecode;
      res = sv_vtrcontrol(dpxio->sv, dpxio->binput, FALSE, 0, 0, &when, &timecode, 0);
      sv_usleep(dpxio->sv, 50000);
      sv_tc2asc(dpxio->sv, timecode, buffer, sizeof(buffer));
      if(dpxio->verbose) {
        printf("VTRTC %s when:%d \n", buffer, when);
      }
    }

    if(res != SV_OK)  {
      printf("ERROR: sv_vtrcontrol(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    }

    when_org = when;
  } 


  if(running) {
    if(dpxio->binput) {
      if(dpxio->verbose) {
        printf("dpxio: sv_fifo_start Called\n");
      }
      res = sv_fifo_start(dpxio->sv, pfifo);

      if(res != SV_OK)  {
        printf("ERROR: sv_fifo_start(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
      }
      bstarted = TRUE;
    } else {
      res = sv_fifo_reset(dpxio->sv, pfifo);
      if(res != SV_OK)  {
        printf("ERROR: sv_fifo_reset(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
      } 
    }
  }

  if(running) {
    res = sv_fifo_status(dpxio->sv, pfifo, &infostart);
    if(res != SV_OK) {
      printf("ERROR: sv_fifo_status() failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    }
  }


  for(frame = 0; (running && (frame < framecount)); frame++) {

    if(frame < DPXIO_MAX_TRACE) {
      QueryPerformanceCounter(&dpxio->trace[frame].start);
    }

    if(!bstarted) {
      int prebuffer_max = infostart.availbuffers < 4 ? infostart.availbuffers : 4;

      if(dpxio->config.vtrcontrol && (frame > 0) && (info.tick >= when_org)) {
        /**
        //  The function \e sv_fifo_start() starts the output of the FIFO, since for an
        //  output it is a good idea to prefill the FIFO with a couple of frames.
        //
        //  When a VTR control is performed together with slow disks, the prebuffering might take too long.
        //  In this case the FIFO may not be started before reaching <i>when</i> which will result
        //  eventually in a drop of ALL frames.
        //
        //  \verbatim
        //  res = sv_fifo_startex(dpxio->sv, pfifo, &tick, &clock_high, &clock_low, NULL);
        //  if(res != SV_OK)  {
        //  printf("ERROR: sv_fifo_start(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
        //  running = FALSE;
        //  continue;
        //  }
        //  \endverbatim
        */
        printf("dpxio: Prebuffering took too long (possibly caused by slow disks).\n");
        running = FALSE;
      } else if((frame >= prebuffer_max) || dpxio->config.nopreroll) {
        int clock_high, clock_low;
        if(dpxio->verbose) {
          printf("dpxio: sv_fifo_start Called\n");
        }

        res = sv_fifo_startex(dpxio->sv, pfifo, &starttick, &clock_high, &clock_low, NULL);
        if(res != SV_OK)  {
          printf("ERROR: sv_fifo_start(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
          continue;
        } 
        if(dpxio->config.verbosetiming) {
          printf("start tick:%6d time %08x:%08x 2:%d 1:%d\n", starttick, clock_high, clock_low, starttick+2*frame, starttick+frame);
        }
        bstarted = TRUE;
      } 
    }
  
    if(dpxio->config.vtrcontrol) {
      memset(&bufferinfo, 0, sizeof(bufferinfo));
      pbufferinfo = &bufferinfo;
      pbufferinfo->version = SV_FIFO_BUFFERINFO_VERSION_1;
      pbufferinfo->size    = sizeof(bufferinfo);
      pbufferinfo->when    = when;
      when                 = 0;
    } else {
      pbufferinfo = NULL;
    }

    if(dpxio->config.repeatfactor == -1) { // 2.5
      if((fifoflags & SV_FIFO_FLAG_REPEAT_MASK) == SV_FIFO_FLAG_REPEAT_3TIMES) {
        fifoflags = (fifoflags & ~SV_FIFO_FLAG_REPEAT_MASK) | SV_FIFO_FLAG_REPEAT_2TIMES;
      } else {
        fifoflags = (fifoflags & ~SV_FIFO_FLAG_REPEAT_MASK) | SV_FIFO_FLAG_REPEAT_3TIMES;
      }
    }

    /** 
    //  The functions \e sv_fifo_getbuffer() and \e sv_fifo_putbuffer() are the main work horse of the FIFO API.
    //  The function <i>%sv_fifo_getbuffer()</i> returns a buffer structure that can be filled by the user and
    //  afterwards returned to the driver using the <i>%sv_fifo_putbuffer()</i> function. In the function <i>%sv_fifo_putbuffer()</i>
    //  the DMA to transfer the frame to or from the hardware will be done.
    //
    //  \verbatim
    //  res = sv_fifo_getbuffer(dpxio->sv, pfifo, &pbuffer, pbufferinfo, fifoflags);
    //  if(res != SV_OK)  {
    //    printf("ERROR: sv_fifo_getbuffer(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
    //    running = FALSE;
    //    continue;
    //  }
    //  \endverbatim
    //  
    //  For an output of the image the code to set the \e pbuffer values must be ready for transfer at this point.
    //  For an input the buffer where the image should be stored must be given into the API.
    //  
    //  \verbatim
    //  res = sv_fifo_putbuffer(dpxio->sv, pfifo, pbuffer, &putbufferinfo);
    //  if(res != SV_OK) {
    //    printf("ERROR: sv_fifo_putbuffer(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
    //    running = FALSE;
    //  }
    //  \endverbatim
    //
    //  For an input the writing of the recorded image should be done here.
    //
    */
    res = sv_fifo_getbuffer(dpxio->sv, pfifo, &pbuffer, pbufferinfo, fifoflags);
    if(res != SV_OK)  {
      // Do not stop recording in case there is an audio input error.
      if(res == SV_ERROR_INPUT_AUDIO_NOAESEBU || res == SV_ERROR_INPUT_AUDIO_NOAIV) {

        // In case of audio error we skip audio DMA by using following flag to save DMA time.
        if(pbuffer) {
          pbuffer->flags |= SV_FIFO_FLAG_VIDEOONLY;
        }
        
        // For statistic use.
        audioInputErrorCount++;
        
        // Store audio error for later DMA address setting but continue normally.
        res_a = res;
        res = SV_OK;
      } else {
        printf("ERROR: sv_fifo_getbuffer(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
        continue;
      }
    } else {
      // Remove possible error state in order to fill valid audio into file in case an
      // audio error is gone now (e.g. because audio signal is connected again).
      res_a = SV_OK;
    }


    /** 
    //  If you want to perform pulldown in conjunction with RP215, the <code><I>dpxio</I></code>
    //  example shows how to use the \e sv_fifo_anc() function to achieve  this.
    //  In case pulldown is not needed but RP215 nonetheless, it shows how to use the
    //  \e sv_fifo_ancdata() function.
    */
    if(!dpxio->binput) {
      if(dpxio->config.rp215 || dpxio->config.rp215a) {
        if(dpxio->pulldown) {
          int field;
          sv_fifo_ancbuffer ancbuffer;

          if(frame == 0) {
            switch(dpxio->pulldownphase) {
              case SV_PULLDOWN_STARTPHASE_A:
                pdphase = 'A';
                break;
              case SV_PULLDOWN_STARTPHASE_B:
                pdphase = 'B';
                break;
              case SV_PULLDOWN_STARTPHASE_C:
                pdphase = 'C';
                break;
              case SV_PULLDOWN_STARTPHASE_D:
                pdphase = 'D';
                break;
            }
          }

          if(frame > 0) {
            pdphase++;
            if(pdphase == 'E') {
              pdphase = 'A';
            }
          }

          for(field = 0; ((field < 3) && (res == SV_OK)); field++) {
            memset(&ancbuffer, 0, sizeof(ancbuffer));

            ancbuffer.did        = 0x51;
            ancbuffer.sdid       = 0x01;
            ancbuffer.vanc       = 1;
            ancbuffer.field      = field;
            if(dpxio->config.rp215a) {
              ancbuffer.linenr   = dpxio->config.rp215a;
              ancbuffer.datasize = rp215a_create(ancbuffer.data, frame, dpxio->config.rp215alength);
            } else {
              ancbuffer.linenr   = dpxio->config.rp215;
              ancbuffer.datasize = rp215_create(ancbuffer.data, frame, pdphase, ancbuffer.field);
            }

            if((field < 2) || (pdphase == 'B') || (pdphase == 'D')) {
              res = sv_fifo_anc(dpxio->sv, pfifo, pbuffer, &ancbuffer);
              if(res != SV_OK) {
                printf("ERROR: sv_fifo_anc(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
                running = FALSE;
              }
            }
          }
        } else {
          unsigned char ancdata[256];
          int count;

          if(dpxio->config.rp215a) {
            count = rp215a_create(ancdata, frame, dpxio->config.rp215alength);
          } else {
            count = rp215_create(ancdata, frame, 0, 0);
          }

          res = sv_fifo_ancdata(dpxio->sv, pfifo, &ancdata[0], count, NULL);
          if(res != SV_OK)  {
            printf("ERROR: sv_fifo_ancdata(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
            running = FALSE;
          }
        }
      }

      if(dpxio->config.ancfill) {
        int line = dpxio->config.ancfill;
        int value = 0;
        int i;
        sv_fifo_ancbuffer ancbuffer;

        memset(&ancbuffer, 0, sizeof(ancbuffer));

        do {
          ancbuffer.did       = 0xc0 + line;
          ancbuffer.sdid      = frame & 0xff;
          ancbuffer.datasize  = line;
          ancbuffer.vanc      = line & 1;
          ancbuffer.linenr    = line;
          for(i = 0; i < line; i++) {
            ancbuffer.data[i] = value++;
          }
          res = sv_fifo_anc(dpxio->sv, pfifo, pbuffer, &ancbuffer);
        } while((res == SV_OK) && (++line < dpxio->config.ancfill + dpxio->config.anccount));
      }

      if(dpxio->config.ancstream) {
        res = dpxio_generate_ancstream(anc, config.ancbuffersize, dpxio->config.ancstream);
        if(res != SV_OK) {
          printf("ERROR: dpxio_generate_ancstream failed\n");
          running = FALSE;
        }
        dpxio_dump_ancstream(anc, config.ancbuffersize);
      }
    }

    if(dpxio->config.ancstream) {
      pbuffer->anc[0].addr = (char *)anc;
      pbuffer->anc[1].addr = (char *)anc;
    }

    if(dpxio->binput) {
      if(dpxio->config.rp215 || dpxio->config.rp215a) {
        unsigned char ancdata[248];
        int  count;

        memset(&ancdata[0], 0, sizeof(ancdata));

        res = sv_fifo_ancdata(dpxio->sv, pfifo, &ancdata[0], sizeof(ancdata), &count);
        if(res != SV_OK) {
          printf("ERROR: sv_fifo_ancdata(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        } else {
          rp215_dump(ancdata, count);
        }
      }

      if(dpxio->config.ccverbose || dpxio->config.cctext) {
        int cha = (pbuffer->anctimecode.closedcaption[0]   ) & 0xff;
        int chb = (pbuffer->anctimecode.closedcaption[0]>>8) & 0xff;
        int chc = (pbuffer->anctimecode.closedcaption[1]   ) & 0xff;
        int chd = (pbuffer->anctimecode.closedcaption[1]>>8) & 0xff;

        if(dpxio->config.ccverbose) {
          if(cha & 0x7f) printf("%c %02x %02x\n", cha & 0x7f, cha & 0x7f, cha);
          if(chb & 0x7f) printf("%c %02x %02x\n", chb & 0x7f, chb & 0x7f, chb);
          if(chc & 0x7f) printf("%c %02x %02x\n", chc & 0x7f, chc & 0x7f, chc);
          if(chd & 0x7f) printf("%c %02x %02x\n", chd & 0x7f, chd & 0x7f, chd);
        } else {
          if(cha & 0x7f) printf("%c", cha & 0x7f);
          if(chb & 0x7f) printf("%c", chb & 0x7f);
          if(chc & 0x7f) printf("%c", chc & 0x7f);
          if(chd & 0x7f) printf("%c", chd & 0x7f);
        }
        fflush(stdout);
      }

      if(dpxio->config.ancdump) {
        sv_fifo_ancbuffer ancbuffer;

        do {
          res = sv_fifo_anc(dpxio->sv, pfifo, pbuffer, &ancbuffer);
          if(res == SV_OK) {
            printf("ANC Field:%d Line:%d %c DID:%02x SDID:%02x datasize:%d\n", ancbuffer.field, ancbuffer.linenr, ancbuffer.vanc?'V':'H', ancbuffer.did, ancbuffer.sdid, ancbuffer.datasize);
            for(i = 0; i < ancbuffer.datasize; i+=16) {
              printf("%02x: ", i);
              for(j = 0; (j < 16) && (i + j < ancbuffer.datasize); j++) {
                printf("%02x ", ancbuffer.data[i+j]);
              }
              for(; (j < 16); j++) {
                printf("   ");
              }
              for(j = 0; (j < 16) && (i + j < ancbuffer.datasize); j++) {
                if((ancbuffer.data[i+j] >= ' ') && (ancbuffer.data[i+j] < 127)) {
                  printf("%c", ancbuffer.data[i+j]);
                } else {
                  printf(".");
                }
              }
              printf("\n");
            }
          }
        } while(res == SV_OK);

        if((res != SV_OK) && (res != SV_ERROR_NOTAVAILABLE)) {
          printf("ERROR: sv_fifo_anc(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
          running = FALSE;
        }
      }
    }

    
    fifoflags &= ~(SV_FIFO_FLAG_TIMEDOPERATION | SV_FIFO_FLAG_STORAGEMODE);

    if(!dpxio->binput) {
      pbuffer->timecode.ltc_ub         = frame;
      pbuffer->timecode.vtr_tc         = frame;
      pbuffer->timecode.vtr_ub         = frame;
      pbuffer->timecode.vitc_ub        = frame;
      pbuffer->timecode.vitc_ub2       = frame | 0x80000000;
      pbuffer->anctimecode.afilm_ub[0] = frame;
      pbuffer->anctimecode.afilm_ub[1] = frame | 0x80000000;
      pbuffer->anctimecode.aprod_ub[0] = frame;
      pbuffer->anctimecode.aprod_ub[1] = frame | 0x80000000;

      pbuffer->anctimecode.dltc_ub     = frame;
      pbuffer->anctimecode.dvitc_tc[0] = frame;
      pbuffer->anctimecode.dvitc_tc[1] = frame;
      pbuffer->anctimecode.dvitc_ub[0] = frame;
      pbuffer->anctimecode.dvitc_ub[1] = frame | 0x80000000;
      pbuffer->anctimecode.film_ub[0]  = frame;
      pbuffer->anctimecode.film_ub[1]  = frame | 0x80000000;
      pbuffer->anctimecode.prod_ub[0]  = frame;
      pbuffer->anctimecode.prod_ub[1]  = frame | 0x80000000;
    }

    /*
    //  Call read/write function, but during record with dma this must be called after putbuffer.
    */
    if(!dpxio->binput) {
      if(frame < DPXIO_MAX_TRACE) {
        QueryPerformanceCounter(&dpxio->trace[frame].getbuffer);
      }
      if(dpxio->bvideo && running) {
        vsize = dpxio_video_readframe(dpxio, frame, pbuffer, videobuffer[0], DPXIO_BUFFERVIDEOSIZE, &offset, &xsize, &ysize, &dpxtype, &nbits, &lineoffset);
        if(!vsize) {
          printf("ERROR: dpxio_video_readframe(sv) failed\n");
          running = FALSE;
        }
        if(frame < DPXIO_MAX_TRACE) {
          dpxio->trace[frame].framesize = vsize;
        }
      }
      if((frame == 0) && !dpxio->io.dryrun && dpxio->bvideo) {
    
        /**
        //  <b>Setting the Storage Format</b>
        //  <br>The format of the output FIFO can be dynamically changed as long as the FIFO
        //  memory size allows this. For all DVS video boards the FIFO can be changed
        //  in format from one vertical sync to the next without any other reinitialization.
        //  One example when this will be useful 
        //  is the play-out of file system files that may change in format. Please note that this is 
        //  not supported by the <code><I>dpxio</I></code> example program.
        //  To see an example for this it is suggested to take a look at the <code><I>cmodtst</I></code> program.
        //
        //  Some fields in the <i>\ref sv_fifo_buffer</i> structure need to be set and the flag
        //  \c #SV_FIFO_FLAG_STORAGEMODE has to be set in the \e sv_fifo_getbuffer() call. In the 
        //  \e sv_fifo_putbuffer() call the required storage parameters have to be set.
        //
        //  \verbatim
        //  struct {
        //      ...
        //    struct {
        //      int storagemode;      ///< Image data storage mode.
        //      int xsize;            ///< Image data x-size.
        //      int ysize;            ///< Image data y-size.
        //      int xoffset;          ///< Image data x-offset from center.
        //      int yoffset;          ///< Image data y-offset from center.
        //      int dataoffset;       ///< Offset to the first pixel in the buffer.
        //      int lineoffset;       ///< Offset from line to line or zero (0) for default.
        //    } storage;
        //      ...
        //  } sv_fifo_buffer;
        //  \endverbatim
        //  
        */

        pbuffer->storage.storagemode  = dpxio_getstoragemode(dpxio, dpxtype, nbits);
        pbuffer->storage.xsize        = xsize;
        pbuffer->storage.ysize        = ysize;
        pbuffer->storage.lineoffset   = lineoffset;
        pbuffer->storage.dataoffset   = offset & 0xfff;
        pbuffer->storage.xoffset      = 0;
        pbuffer->storage.yoffset      = 0;
        pbuffer->storage.matrixtype   = dpxio->config.matrixtype;
      }
      if(dpxio->baudio && running) {
        running = dpxio_audio_read(dpxio, frame, pbuffer, dpxio->audiochannel, audiobuffer[0], pbuffer->audio[0].size / 2 / 4);
        if( !running ){
          printf("ERROR: dpxio_audio_read(sv) failed\n");
        }
      }
      if(frame < DPXIO_MAX_TRACE) {
        QueryPerformanceCounter(&dpxio->trace[frame].dpxio);
      }
    } else {
      vsize = dpxio_framesize(dpxio, xsize, ysize, dpxtype, nbits, &offset, &padding);
    }

    if(dpxio->bvideo) {
      pbuffer->video[field].addr =  videobuffer[0] + (offset & ~0xfff);
      pbuffer->video[field].size = (vsize - (offset & ~0xfff) + 0xff) & ~0xff;
    } else {
      pbuffer->video[field].addr = 0;
      pbuffer->video[field].size = 0;
    }

    /*
    // There is no valid data in the other field.
    // Setting the size to 0 skips the DMA transfer for that field.
    */
    pbuffer->video[!field].size = 0;

    if(dpxio->config.fieldbased) {
      /*
      // In field-based mode, toggle through both video fields.
      // Only one of the two fields in a getbuffer/putbuffer cycle contains
      // defined data.
      */
      field = !field;
    }

    if(dpxio->baudio && (res_a == SV_OK)) {
      pbuffer->audio[0].addr[0] = audiobuffer[0];
      pbuffer->audio[0].size    = dpxio->hw.audiochannels * pbuffer->audio[0].size;
    } else {
      pbuffer->audio[0].addr[0] = 0;
      pbuffer->audio[0].size    = 0;
    }
    pbuffer->audio[1].size = 0;

    if(running) {
      if(!dpxio->binput && dpxio->config.lut) {
        printf("frame %d activating ", frame);
        switch((frame / dpxio->config.lut) % 6) {
        case 0:
          printf("3D LUT => identity\n");
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, lut[0], 32768, 0, SV_FIFO_LUT_TYPE_3D);
          break;
        case 1:
          printf("3D LUT => inverted\n");
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, lut[1], 32768, 0, SV_FIFO_LUT_TYPE_3D);
          break;
        case 2:
          printf("1D LUT (rgba) => identity\n");
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, lut[2], 16384, 0, SV_FIFO_LUT_TYPE_1D_RGBA);

          // disable other 1D LUT
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, NULL, 0, 0, SV_FIFO_LUT_TYPE_1D_RGB);
          break;
        case 3:
          printf("1D LUT (rgba) => inverted\n");
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, lut[3], 16384, 0, SV_FIFO_LUT_TYPE_1D_RGBA);

          // disable other 1D LUT
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, NULL, 0, 0, SV_FIFO_LUT_TYPE_1D_RGB);
          break;
        case 4:
          printf("1D LUT (rgb) => identity\n");
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, lut[4], 6144, 0, SV_FIFO_LUT_TYPE_1D_RGB);

          // disable other 1D LUT
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, NULL, 0, 0, SV_FIFO_LUT_TYPE_1D_RGBA);
          break;
        case 5:
          printf("1D LUT (rgb) => inverted\n");
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, lut[5], 6144, 0, SV_FIFO_LUT_TYPE_1D_RGB);

          // disable other 1D LUTs
          sv_fifo_lut(dpxio->sv, pfifo, pbuffer, NULL, 0, 0, SV_FIFO_LUT_TYPE_1D_RGBA);
          break;
        }
      }
    }

    if(running) {
      res = sv_fifo_putbuffer(dpxio->sv, pfifo, pbuffer, &putbufferinfo);
      if(res != SV_OK) {
        printf("ERROR: sv_fifo_putbuffer(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
      }     

      if(dpxio->config.verbosetiming) {
        printf("Frame %3d when:%6d time %08x:%08x\n", frame, putbufferinfo.when, putbufferinfo.clock_high, putbufferinfo.clock_low);
      } 

      if(dpxio->config.setat) {
        if(!dpxio->binput) {
          if(bstarted) {
            int chunksize = dpxio->pulldown ? 5 : 4;

            if(!badjusted) {
              /*
              // Conclude if the current raster is interlaced or progressive.
              */
              fieldcount = (putbufferinfo.when - starttick == chunksize) ? 1 : 2;

              /*
              // Set timecodes for prebuffered frames.
              */
              for(local_tick = starttick; local_tick < putbufferinfo.when; local_tick += fieldcount, local_frame++) {

                dpxio_set_timecodes(dpxio, local_frame, local_tick, fieldcount);
              }

              badjusted = TRUE;
            } 
            if(frame % 4 == 0) {
              /*
              // Set timecodes for next chunk of frames.
              */
              for(local_tick = putbufferinfo.when; chunksize > 0; chunksize--, local_tick += fieldcount, local_frame++) {
                dpxio_set_timecodes(dpxio, local_frame, local_tick, fieldcount);
              }
            }
          }
        }
      }

      if(dpxio->binput) {
        if(dpxio->config.ancstream) {
          res = dpxio_dump_ancstream(anc, config.ancbuffersize);
          if(res != SV_OK) {
            printf("ERROR: dpxio_dump_ancstream failed\n");
            running = FALSE;
          }
        }
      }

      vtimecode = 0;
      atimecode = 0;
      aoffset   = 0;
      if(dpxio->config.verbosetc || (dpxio->config.verifytimecode && ((frame == 0) || (dpxio->config.verifytimecode == 2)))) {
        vtimecode = dpxio_dpx_timecode((unsigned char *)pbuffer->video[0].addr, dpxtype, xsize, NULL);
        if(dpxio->baudio) {
          atimecode = dpxio_audio_timecode((unsigned char *)audiobuffer[0], pbuffer->audio[0].size / 4 / (2 * dpxio->hw.audiochannels), dpxio->hw.audiochannels, &aoffset);
        }
      }

      if((dpxio->config.verifytimecode && ((frame == 0) || (dpxio->config.verifytimecode == 2)))) {
        int tcerror = FALSE;
        int frames;
        int seconds; 
        int minutes;
        int hours;

        if(xtimecode != vtimecode) {
          tcerror = TRUE;
        }
        if(dpxio->baudio) {
          if(xtimecode != atimecode) {
            tcerror = TRUE;
          }
        }
        if(dpxio->hw.devtype != SV_DEVTYPE_SDBOARD) {
          if(xtimecode != (unsigned int)pbuffer->anctimecode.dvitc_tc[0]) {
            tcerror = TRUE;
          }
        }
        if(xtimecode != (unsigned int)pbuffer->timecode.vtr_tc) {
          tcerror = TRUE;
        }
        //if(xtimecode != (unsigned int)pbuffer->timecode.ltc_tc) {
        //  tcerror = TRUE;
        //}
        if(tcerror) {
          printf("ERROR: ltc:%08x dvitc:%08x vtr:%08x video:%08x audio:%08x/%d - %08x\n", pbuffer->timecode.ltc_tc, pbuffer->anctimecode.dvitc_tc[0], pbuffer->timecode.vtr_tc, vtimecode, atimecode, aoffset, xtimecode);
        }

        frames  = 10 * ((xtimecode & 0xf0) >> 4) + (xtimecode & 0xf); xtimecode >>= 8;
        seconds = 10 * ((xtimecode & 0xf0) >> 4) + (xtimecode & 0xf); xtimecode >>= 8;
        minutes = 10 * ((xtimecode & 0xf0) >> 4) + (xtimecode & 0xf); xtimecode >>= 8;
        hours   = 10 * ((xtimecode & 0xf0) >> 4) + (xtimecode & 0xf); 

        frames++;
        if(frames >= storage.fps) {
          frames = 0;
          seconds++;
        }
        if(seconds > 60) {
          seconds = 0;
          minutes++;
        }
        if(minutes > 60) {
          minutes = 0;
          hours++;
        }
        if(hours > 24) {
          hours = 0;
        }

        xtimecode  = (0x10 * (frames  / 10) + (frames  % 10));
        xtimecode |= (0x10 * (seconds / 10) + (seconds % 10)) << 8;
        xtimecode |= (0x10 * (minutes / 10) + (minutes % 10)) << 16;
        xtimecode |= (0x10 * (hours   / 10) + (hours   % 10)) << 24;
      }
      if(dpxio->config.verbosetc) {
        printf("      %3d ltc:%08x dvitc:%08x vtr:%08x video:%08x audio:%08x/%d\n", frame, pbuffer->timecode.ltc_tc, pbuffer->anctimecode.dvitc_tc[0], pbuffer->timecode.vtr_tc, vtimecode, atimecode, aoffset);
      }

      res = sv_fifo_status(dpxio->sv, pfifo, &info);
      if(res != SV_OK) {
        printf("ERROR: sv_fifo_status() failed = %d '%s'\n", res, sv_geterrortext(res));
        running = FALSE;
      }

      if(frame < DPXIO_MAX_TRACE) {
        QueryPerformanceCounter(&dpxio->trace[frame].putbuffer);
      }

      if(dpxio->binput) {
        int timecode;

        if(dpxio->bvideo && running) {
          if(dpxio->hw.validtimecode & SV_VALIDTIMECODE_DLTC) {
            timecode = pbuffer->anctimecode.dltc_tc;
          } else if(dpxio->hw.validtimecode & SV_VALIDTIMECODE_DVITC_F1) {
            timecode = pbuffer->anctimecode.dvitc_tc[0];
          } else if(dpxio->hw.validtimecode & SV_VALIDTIMECODE_LTC) {
            timecode = pbuffer->timecode.ltc_tc;
          } else {
            printf("No valid timecode found\n");
            timecode = 0;
          }

          dpxio_video_writeframe(dpxio, frame, pbuffer, videobuffer[0], vsize, xsize, ysize, dpxtype, nbits, offset, padding, timecode);
        }
        if(dpxio->baudio && running && (res_a == SV_OK)) {
          dpxio_audio_write(dpxio, frame, dpxio->audiochannel, audiobuffer[0], pbuffer->audio[0].size / 4 / ((fifoflags & SV_FIFO_FLAG_AUDIOINTERLEAVED)?16:2));
        }
      }

      if(dpxio->verbose) {
        if(res_a != SV_OK) {
          printf("tick: %06d / frame: %4d / buffer: avail: %2d dropped: %2d / dropped_audio:%2d\n", putbufferinfo.when, frame, info.availbuffers, info.dropped, audioInputErrorCount);
        } else {
          printf("tick: %06d / frame: %4d / buffer: avail: %2d dropped: %2d\n", putbufferinfo.when, frame, info.availbuffers, info.dropped);
        }
      }

      if(frame < DPXIO_MAX_TRACE) {
        dpxio->trace[frame].tick     = info.tick;
        dpxio->trace[frame].dropped  = info.dropped;
        dpxio->trace[frame].nbuffers = info.availbuffers;
        QueryPerformanceCounter(&dpxio->trace[frame].finished);
      }
    }
  }

  if(pfifo) {
    /**
    //  The \e sv_fifo_status() function returns the number of total buffers, free buffers and whether any 
    //  frames were dropped.
    */
    res = sv_fifo_status(dpxio->sv, pfifo, &info);
    if(res != SV_OK) {
      printf("ERROR: sv_fifo_status() failed = %d '%s'\n", res, sv_geterrortext(res));
    } else {
      if(dpxio->verbose) {
        printf("nbuffers  : %d\n", info.nbuffers);
        printf("avail     : %d\n", info.availbuffers);
        printf("Dropped   : %d\n", info.dropped);
      }
      if(info.dropped) {
        printf("WARNING: Dropped %d frames\n", info.dropped);
      }
      if(audioInputErrorCount) {
        printf("\nWARNING: Audio input samples were missing. Function sv_fifo_getbuffer() detected %d audio input %s.\n", audioInputErrorCount, audioInputErrorCount==1?"error":"errors");
      }
    }
  }

  if(running && !bstarted) {
    int tick, clock_high, clock_low;
    res = sv_fifo_startex(dpxio->sv, pfifo, &tick, &clock_high, &clock_low, NULL);
    if(res != SV_OK)  {
      printf("ERROR: sv_fifo_start(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
      running = FALSE;
    } 
    bstarted = TRUE;
    if(dpxio->config.verbosetiming) {
      printf("start tick:%6d time %08x:%08x\n", tick, clock_high, clock_low);
    }
  }

  if(pfifo) {
    /**
    //  <b>Wait</b>
    //  <br>If you want to wait until all frames have been transmitted for an output FIFO
    //  call the function \e sv_fifo_wait().
    */
    res = sv_fifo_wait(dpxio->sv, pfifo);
    if(running && (res != SV_OK))  {
      printf("ERROR: sv_fifo_wait(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
    }
  }

  if(pfifo) {
    res = sv_fifo_stop(dpxio->sv, pfifo, 0);
    if(running && (res != SV_OK))  {
      printf("ERROR: sv_fifo_stop(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
    }
    
    /**
    //  <b>Closing</b>
    //  <br>After finishing using the FIFO it should be closed. The FIFO is closed and freed for other
    //  usages with the function \e sv_fifo_free(). Once this function is called, the \e pfifo handle should be discarded
    //  and not used anymore.
    */
    res = sv_fifo_free(dpxio->sv, pfifo);
    if(running && (res != SV_OK))  {
      printf("ERROR: sv_fifo_free(sv) failed = %d '%s'\n", res, sv_geterrortext(res));
    }
  }

  if(dpxio->baudio) {
    dpxio_audio_close(dpxio);
  }

  for(i = 0; i < DPXIO_MAXBUFFER; i++) {
    if(mallocbuffer[i]) {
      free(mallocbuffer[i]);
    }
  }

  if(anc_org) {
    free((void*)anc_org);
    anc_org = 0;
    anc = 0;
  }

  return frame;
}

/**
//  \ingroup svdpxio
//
//  This function is used internally by the example program and dumps timing measurement results.
//
//  \param dpxio  --- Application handle.
//  \param start  --- First frame to dump.
//  \param count  --- Number of frames to dump.
*/
void dpxio_tracelog(dpxio_handle * dpxio, int start, int count)
{
#ifdef WIN32
  char speedbuffer[128];
  LARGE_INTEGER f;
  int i;
  
  QueryPerformanceFrequency(&f);
  if(start >= DPXIO_MAX_TRACE) {
    return;
  }
  if(count >= DPXIO_MAX_TRACE) {
    count = DPXIO_MAX_TRACE;
  }
  for(i = 0; i < count; i++) {
    if(dpxio->trace[i].finished.QuadPart == dpxio->trace[i].start.QuadPart) {
      // Avoid divide by 0.
      dpxio->trace[i].finished.QuadPart++;
    }

    if(dpxio->trace[i].framesize) {
      double speed = (double)dpxio->trace[i].framesize * (double)f.QuadPart / (dpxio->trace[i].finished.QuadPart - dpxio->trace[i].start.QuadPart);
      sprintf(speedbuffer, "%d MB/s", (int)(speed / 1000000));
    } else {
      sprintf(speedbuffer, "");
    }
    printf("%3d %06d %2d %2d : %6d %6d %6d %6d %s\n", i + start, dpxio->trace[i].tick, dpxio->trace[i].nbuffers, dpxio->trace[i].dropped, 
      (int)(1000000 * (dpxio->trace[i].getbuffer.QuadPart - dpxio->trace[i].start.QuadPart)/f.QuadPart),
      (int)(1000000 * (dpxio->trace[i].dpxio.QuadPart - dpxio->trace[i].getbuffer.QuadPart)/f.QuadPart),
      (int)(1000000 * (dpxio->trace[i].putbuffer.QuadPart - dpxio->trace[i].dpxio.QuadPart)/f.QuadPart),
      (int)(1000000 * (dpxio->trace[i].finished.QuadPart - dpxio->trace[i].putbuffer.QuadPart)/f.QuadPart), speedbuffer);
  }
#endif
}


/**
//  \ingroup svdpxio
//
//  The main function of the application.
//
//  \param argc --- Argument count.
//  \param argv --- Argument vector.
//  \return Return code.
*/
int main(int argc, char ** argv)
{
  int error = FALSE;
  int count = 0;
  int noprint = FALSE;
  char vtrtc[64];
  int res = SV_OK;
  int framecount = 0;
  int tmp = 0;
  int i   = 0;
  int key_current = 0;

  dpxio_handle   gdpxio = { 0 };
  dpxio_handle * dpxio  = &gdpxio;

  memset(dpxio, 0, sizeof(dpxio_handle));

  for(i = 0; i < argc; i++) {
    printf("%s ", argv[i]);
  }
  printf("\n");

  while(!error && (argc > 2) && (argv[1][0] == '-')) {
    switch(argv[1][1]) {
    case 'a':
      switch(argv[1][2]) {
      case 'a':
        dpxio->config.ancdump = TRUE;
        break;
      case 'c':
        if(argv[1][3] == 'c') {
          dpxio->config.ccverbose = TRUE;
        } else {
          dpxio->config.cctext = TRUE;
        }
        break;
      case 'f':
        dpxio->config.ancfill = atoi(&argv[1][3]);
        break;
      case 'l':
        dpxio->config.anccount = atoi(&argv[1][3]);
        break;
      case 's':
        dpxio->config.rp215 = atoi(&argv[1][3]);
        if(!dpxio->config.rp215) {
          dpxio->config.rp215 = 9;
        } else {
          printf("Using rp215 on line %d\n", dpxio->config.rp215);
        }
        break;
      case 't':
        dpxio->config.rp215a = 9;
        dpxio->config.rp215alength = atoi(&argv[1][3]);
        if(!dpxio->config.rp215alength) {
          dpxio->config.rp215alength = 255;
        } else {
          printf("Using rp215A on line %d\n", dpxio->config.rp215a);
        }
        break;
      case 'x':
        dpxio->config.ancstream = 1;
        if(atoi(&argv[1][3]) > 0) {
          dpxio->config.ancstream = atoi(&argv[1][3]);
        }
        if(dpxio->config.ancstream > 4) {
          dpxio->config.ancstream = 4;
        }
        break;
      default:
        error = TRUE;
      }
      break;
    case 'b':
      switch(argv[1][2]) {
      case '0':
      case 'a':
      case 'A':
        dpxio->audiochannel = 0;
        break;
      case '1':
      case 'b':
      case 'B':
        dpxio->audiochannel = 1;
        break;
      case '2':
      case 'c':
      case 'C':
        dpxio->audiochannel = 2;
        break;
      case '3':
      case 'd':
      case 'D':
        dpxio->audiochannel = 3;
        break;
      case '4':
      case 'e':
      case 'E':
        dpxio->audiochannel = 4;
        break;
      case '5':
      case 'f':
      case 'F':
        dpxio->audiochannel = 5;
        break;
      case '6':
      case 'g':
      case 'G':
        dpxio->audiochannel = 6;
        break;
      case '7':
      case 'h':
      case 'H':
        dpxio->audiochannel = 7;
        break;
      default:
        error = TRUE;
      }
      break;
    case 'd':
      dpxio->io.dryrun = TRUE;
      break;
    case 'f':
      dpxio->config.fieldbased = TRUE;
      break;
    case 'k':
      if(key_current < arraysize(dpxio->config.keys)) {
        int fd = open(&argv[1][3], O_RDONLY | O_BINARY);
        char buffer[512];
        char key[16];
        char tmp[3];
        int nbytes = 0;

        if(fd == -1) {
          printf("ERROR: Could not open key file.\n");
          error = TRUE;
        } else {
          nbytes = read(fd, buffer, sizeof(buffer));

          if(nbytes < 256) {
            for(i = 0; i < nbytes; i++) {
              if((buffer[i] == '\n') || (buffer[i] == '\r')) {
                buffer[i] = '\0';
                break;
              }
            }
            buffer[nbytes] = 0;

            if(strlen(buffer) == 16 * 2) {
              // Convert from ascii to binary.
              for(i = sizeof(key)-1; i >= 0; i--) {
                tmp[0] = buffer[i*2];
                tmp[1] = buffer[i*2+1];
                tmp[2] = '\0';
                key[sizeof(key)-1-i] = strtoul(tmp, NULL, 16);
              }
              memcpy(dpxio->config.keys[key_current].key, key, sizeof(key));
              nbytes = sizeof(key);
            }
          } else {
            memcpy(dpxio->config.keys[key_current].key, buffer, nbytes);
          }

          close(fd);
        }
        dpxio->config.keys[key_current].decrypt = nbytes;

        key_current++;
      }
      break;
    case 'K':
      dpxio->config.key_base = atoi(&argv[1][3]);
      break;
    case 'l':
      if(!strncmp(&argv[1][1], "lut", 3)) {
        dpxio->config.lut = atoi(&argv[1][4]);
        if(dpxio->config.lut == 0) {
          dpxio->config.lut = 10;
        }
      } else {
        dpxio->config.loopmode = TRUE;
      }
      break;
    case 'L':
      dpxio->config.tracelog = TRUE;
      break;
    case 'm':
      switch(atoi(&argv[1][2])) {
      case SV_MATRIXTYPE_RGBFULL:
      case SV_MATRIXTYPE_RGBHEAD:
      case SV_MATRIXTYPE_601FULL:
      case SV_MATRIXTYPE_601HEAD:
      case SV_MATRIXTYPE_274FULL:
      case SV_MATRIXTYPE_274HEAD:
        dpxio->config.matrixtype = atoi(&argv[1][2]);
        break;
      default:
        dpxio->config.matrixtype = SV_MATRIXTYPE_DEFAULT;
      }
      break;
    case 'p':
      dpxio->pulldown = TRUE;
      switch(argv[1][2]) {
      case 'a':
      case 'A':
        dpxio->pulldownphase = SV_PULLDOWN_STARTPHASE_A;
        break;
      case 'b':
      case 'B':
        dpxio->pulldownphase = SV_PULLDOWN_STARTPHASE_B;
        break;
      case 'c':
      case 'C':
        dpxio->pulldownphase = SV_PULLDOWN_STARTPHASE_C;
        break;
      case 'd':
      case 'D':
        dpxio->pulldownphase = SV_PULLDOWN_STARTPHASE_D;
        break;
      default:
        dpxio->pulldownphase = SV_PULLDOWN_STARTPHASE_A;
      }
      break;
    case 'P':
      dpxio->config.nopreroll = TRUE;
      break;
    case 's':
      dpxio->config.setat = TRUE;
      break;
    case 't':
      dpxio->config.vtrcontrol = TRUE;
      if(argv[1][2] == 't') {
        if(argv[1][3] == 't') {
          dpxio->config.verifytimecode = 2;
        } else {
          dpxio->config.verifytimecode = 1;
        }
      }
      strcpy(vtrtc, argv[2]); argv++; argc--;
      break;
    case 'T':
      dpxio->config.verbosetiming = TRUE;
      break;
    case 'u':
      dpxio->config.verbosetc = TRUE;
      break;
    case 'v':
      dpxio->verbose = TRUE;
      break;
    case 'x':
      if(!strcmp(argv[2], "2.5")) {
        dpxio->config.repeatfactor = -1; argv++; argc--;
      } else {
        dpxio->config.repeatfactor = atoi(argv[2]); argv++; argc--;
        switch(dpxio->config.repeatfactor) {
        case 2:
        case 3:
        case 4:
         break;
        default:
          printf("ERROR: Wrong repeat factor, allowed is 2,2.5,3 or 4.\n");
          error = TRUE;
        }
      }
      break;
    default:
      error = TRUE;
    }
    argv++; argc--;
  }

  if(!error) {
    if(argc >= 5) {
      dpxio->bvideo = TRUE;
      if(argc >= 6) {
        dpxio->baudio = TRUE;
      }
      if(!strcmp(argv[1], "rec")){
        dpxio->binput = TRUE;
      } else if(!strcmp(argv[1], "dis")){
        dpxio->binput = FALSE;
      } else {
        error = TRUE;
      }

      if(dpxio->bvideo) {
        strcpy(dpxio->videoio.filename, argv[2]);
        argv++; argc--;
        if (!strcmp(dpxio->videoio.filename, "dummy")) {
          if(dpxio->config.verbosetc) {
            printf("ERROR: Illegal combination of special filename 'dummy' with option '-u'\n");
            error = TRUE;
          } else {
            dpxio->bvideo = FALSE;
          }
        }
      }

      if(dpxio->bvideo) {
        // do not allow field-based output
        if (!dpxio->binput && dpxio->config.fieldbased) {
          printf("ERROR: Field-based output is not implemented.\n");
          error = TRUE;
          noprint = TRUE;
        }
      } 

      if(dpxio->baudio) {
        strcpy(dpxio->audioio.filename, argv[2]);
        argv++; argc--;
        if (!strcmp(dpxio->audioio.filename, "dummy")) {
          if(!dpxio->bvideo) {
            printf("ERROR: Using special filename 'dummy' for video and audio at the same time is not allowed.\n");
            error = TRUE;
          } else {
            dpxio->baudio = FALSE;
          }
        }
      }

      dpxio->videoio.framenr    = atoi(argv[2]);
      framecount = atoi(argv[3]);
    } else {
      error = TRUE;
    }
  }

  if(!error) {
    int opentype = 0;

    if(dpxio->pulldown) {
      // sv_pulldown() is always setting pulldown phase for input and output.
      // That is why there is a more global opentype needed.
      opentype = SV_OPENTYPE_DEFAULT;
    } else {
      if(dpxio->binput) {
        opentype = SV_OPENTYPE_INPUT;
      } else {
        opentype = SV_OPENTYPE_OUTPUT;
      }
    }
    res = sv_openex(&dpxio->sv, "", SV_OPENPROGRAM_DEMOPROGRAM, opentype, 0, 0);
    if(res != SV_OK) {
      printf("ERROR: Error '%s' opening video device", sv_geterrortext(res));
      noprint = error = TRUE;
    } 

    if(!error) {
      res = sv_jack_query(dpxio->sv, dpxio->binput ? 1 : 0, SV_QUERY_MODE_CURRENT, 0, &tmp);
      if(res != SV_OK) {
        printf("ERROR: sv_query(SV_QUERY_MODE_CURRENT) failed = %d '%s'\n", res, sv_geterrortext(res));
        noprint = error = TRUE;
      } 

      res = sv_query(dpxio->sv, SV_QUERY_DMAALIGNMENT, 0, &dpxio->hw.dmaalignment);
      if(res != SV_OK) {
        printf("ERROR: sv_query(SV_QUERY_DMAALIGNMENT) failed = %d '%s'\n", res, sv_geterrortext(res));
        noprint = error = TRUE;
      } 

      res = sv_query(dpxio->sv, SV_QUERY_DEVTYPE, 0, &dpxio->hw.devtype);
      if(res != SV_OK) {
        printf("ERROR: sv_query(SV_QUERY_DEVTYPE) failed = %d '%s'\n", res, sv_geterrortext(res));
        noprint = error = TRUE;
      } 

      res = sv_query(dpxio->sv, SV_QUERY_VALIDTIMECODE, 0, &dpxio->hw.validtimecode);
      if(res != SV_OK) {
        printf("ERROR: sv_query(SV_QUERY_VALIDTIMECODE) failed = %d\n", res);
        noprint = error = TRUE;
      } 

      res = sv_jack_query(dpxio->sv, dpxio->binput ? 1 : 0, SV_QUERY_AUDIOCHANNELS, 0, &dpxio->hw.audiochannels);
      if(res != SV_OK) {
        printf("ERROR: sv_query(SV_QUERY_AUDIOCHANNELS) failed = %d\n", res);
        noprint = error = TRUE;
      } 
      if(dpxio->baudio && !dpxio->hw.audiochannels) {
        printf("ERROR: No audio configured on device\n");
        noprint = error = TRUE;
      }

      if(dpxio->binput) {
        if(((tmp & (SV_MODE_NBIT_MASK | SV_MODE_COLOR_MASK)) != (SV_MODE_NBIT_10BDPX | SV_MODE_COLOR_RGB_RGB)) &&
           ((tmp & (SV_MODE_NBIT_MASK | SV_MODE_COLOR_MASK)) != (SV_MODE_NBIT_8B | SV_MODE_COLOR_YUV422))) {
          printf("ERROR: Not running in RGB/10B dpx or YUV422/8B mode\n");
          noprint = error = TRUE;
        }
      }
      
      if(dpxio->config.fieldbased) {
        if((tmp & SV_MODE_STORAGE_FRAME)) {
          printf("ERROR: Dpxio does not work in fieldmode when in framestorage\n");
          noprint = error = TRUE;
        }
      } else {
        if(!(tmp & SV_MODE_STORAGE_FRAME)) {
          printf("ERROR: Dpxio does only work in fieldstorage when using fieldmode\n");
          noprint = error = TRUE;
        }
      }
    }

    if(!error) {
      if(dpxio->pulldown && dpxio->config.fieldbased) {
        printf("WARNING: Usage of pulldown overrides fieldmode\n");
        noprint = TRUE;
        error = FALSE;
      }
    }

    if(!error) {
      switch(dpxio->hw.devtype) {
      case SV_DEVTYPE_SDBOARD:
        if(dpxio->hw.audiochannels > 1) {
          printf("ERROR: Dpxio does not work with more than one stereo channel on SDBoard\n");
          noprint = error = TRUE;
        }
        break;
      default:;
      }
    }

    if(dpxio->config.vtrcontrol) {
      sv_asc2tc(dpxio->sv, vtrtc, &dpxio->vtr.tc);
      dpxio->vtr.nframes = framecount;
    }
    for(key_current = 0; key_current < arraysize(dpxio->config.keys); key_current++) {
      if(dpxio->config.keys[key_current].decrypt) {
        if(dpxio->config.keys[key_current].decrypt == 16) {
          res = sv_kdmkey_load(dpxio->sv, dpxio->config.key_base + key_current, dpxio->config.keys[key_current].key, dpxio->config.keys[key_current].decrypt, SV_KDMKEY_FORMAT_PLAIN_R2L | SV_KDMKEY_ENCRYPTION_NONE);
        } else {
          res = sv_kdmkey_load(dpxio->sv, dpxio->config.key_base + key_current, dpxio->config.keys[key_current].key, dpxio->config.keys[key_current].decrypt, SV_KDMKEY_FORMAT_BASE64 | SV_KDMKEY_ENCRYPTION_RSA);
        }
        if(res != SV_OK) {
          printf("ERROR: sv_kdmkey_load() failed = %d '%s'\n", res, sv_geterrortext(res));
          noprint = error = TRUE;
        }

        if(res == SV_OK) {
          if(dpxio->config.keys[key_current].decrypt != 16) {
            unsigned char keydata[138];
            char string[26];
            int keyid;
            int i;

            res = sv_kdmkey_readback(dpxio->sv, &keyid, (char *)keydata, sizeof(keydata));
            if(res == SV_OK) {
              printf("structure id:      ");
              for(i = 0; i < 16; i++) printf("%02x ", keydata[i]);
              printf("\ncert. thumbprint:  ");
              for(i = 16; i < 36; i++) printf("%02x ", keydata[i]);
              printf("\ncomp. playlist id: ");
              for(i = 36; i < 52; i++) printf("%02x ", keydata[i]);
              printf("\nkey type:          ");
              for(i = 52; i < 56; i++) printf("%02x ", keydata[i]);
              printf("\nkey id:            ");
              for(i = 56; i < 72; i++) printf("%02x ", keydata[i]);
              strncpy(string, (char *)&keydata[72], 25); string[25] = '\0';
              printf("\nnot valid before:  %s", string);
              strncpy(string, (char *)&keydata[97], 25); string[25] = '\0';
              printf("\nnot valid after:   %s", string);
              printf("\n");
            } else {
              printf("ERROR: sv_kdmkey_readback() failed = %d '%s'\n", res, sv_geterrortext(res));
            }
          }
        }
      }
    }

    do {
      if(dpxio->bvideo) {
        if(!error) {
          if(!dpxio_verifyformat(dpxio, dpxio->videoio.filename, dpxio->videoio.framenr)) {
            printf("ERROR: Could not determine video file format\n");
            noprint = error = TRUE;
          }
        }

        if(!error) {
          if(!dpxio_video_opensequence(dpxio, dpxio->videoio.filename, dpxio->videoio.framenr, dpxio->videoio.framenr + framecount)) {
            printf("ERROR: Error opening video files");
            noprint = error = TRUE;
          }
        }
      }

      if(!error) {
        count = dpxio_exec(dpxio, framecount);

        if(dpxio->bvideo) {
          if(!dpxio_video_closesequence(dpxio)) {
            printf("ERROR: Error closing video files");
            noprint = error = TRUE;
          }
        }
      }
    } while(!error && dpxio->config.loopmode);

    if(!error && dpxio->config.tracelog) {
      dpxio_tracelog(dpxio, 0, count);
    }

    if(dpxio->sv) {
      sv_close(dpxio->sv);
    }
  } else {
    error = TRUE;
  }

  if(error && !noprint) { 
    printf("\ndpxio [options] {rec,dis} dpxfiles audiofile #start #nframes\n");
    printf("                     For audio the start is in seconds of the clip\n");
    printf("        rec,dis    - Records and Displays from video card.\n");
    printf("        filenames  - The special filename 'dummy' omits disk i/o.\n");
    printf("                     NOTE! Don't use 'dummy' for audio and video at the same time.\n");
    printf(" options:\n");
    printf("         -aa         Display anc captured packets.\n");
    printf("         -ac         Display closed caption captured.\n");
    printf("         -acc        Display closed caption captured (verbose).\n");
    printf("         -as#        Use anc userembedder to embedd/deembedd rp215 timecode. # linenr, default 9\n");
    printf("         -at#        Use anc userembedder to embedd/deembedd rp215a timecode. #bytecount, max 255\n");
    printf("         -af#        Fill anc data from line #\n");
    printf("         -al#        Fill # lines of ancdata\n");
    printf("         -ax#        Fill/dump anc stream.\n");
    printf("         -bX         Record/Display from stereo audio channel X.\n");
    printf("                       X -> {0,1,2,3,4,5,6,7}\n");
    printf("         -d          Dry-run\n");
    printf("         -f          Field based\n");
    printf("         -k=file     Use decryption key from file.\n");
    printf("         -K=#        Base index for loading decryption keys.\n");
    printf("         -l          Loop forever.\n");
    printf("         -lut#       Demonstrate LUT toggle (every # frames).\n");
    printf("         -L          Display datarate.\n");
    printf("         -p          Playout with Pulldown.\n");
    printf("         -pX         Playout with Pulldown. X - Startphase\n");
    printf("                       X -> {a,b,c,d}\n");
    printf("         -s          Use set-at mechanism.\n");
    printf("         -t <TC>     Enable vtrcontrol.\n");
    printf("         -u          Verbose timing / LTC / VTR. (record)\n");
    printf("                     NOTE! Don't use option '-u'with special filename 'dummy'!\n");
    printf("         -v          Verbose.\n");
    printf("         -x <z>      Playout with repeat factor.\n");
    printf("                       z = {2, 2.5, 3, 4}\n");
  }

  return error;
}
/**
// @} 
// doxygen end - do not remove
*/
