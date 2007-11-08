/*
 * FILE:     audio_types.h
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson
 *
 * Copyright (c) 1998-2001 University College London
 * All rights reserved.
 *
 * $Id: audio_types.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

#ifndef _RAT_AUDIO_TYPES_H_
#define _RAT_AUDIO_TYPES_H_

/* This version of the code can only work with a constant device      */
/* encoding. To use different encodings the parameters below have     */
/* to be changed and the program recompiled.                          */
/* The clock translation problems have to be taken into consideration */
/* if different users use different base encodings...                 */

typedef enum {
	DEV_PCMU, /* mu-law (8 bits) */
        DEV_PCMA, /*  a-law (8 bits) */
	DEV_S8,   /* signed 8 bits   */
        DEV_U8,   /* unsigned 8 bits */
	DEV_S16   /* signed 16 bits  */
} deve_e;

typedef struct s_audio_format {
  deve_e encoding;
  unsigned int	sample_rate; 		/* Should be one of 8000, 16000, 24000, 32000, 48000 	*/
  int    	bits_per_sample;	/* Should be 8 or 16 					*/
  int    	channels;  		/* Should be 1 or 2  					*/
  int   	bytes_per_block;	/* size of unit we will read/write in 			*/
} audio_format;

typedef short sample;       	/* Sample representation 16 bit signed 			*/
typedef int audio_desc_t;   	/* Unique handle for identifying audio devices 		*/

#define AUDIO_DEVICE_NAME_LENGTH 63

typedef struct {
        audio_desc_t 	 descriptor;
        char  		*name;
} audio_device_details_t;

typedef uint32_t audio_port_t;

#define AUDIO_PORT_NAME_LENGTH 20

#define AUDIO_PORT_PHONE      "Phone"
#define AUDIO_PORT_SPEAKER    "Speaker"
#define AUDIO_PORT_HEADPHONE  "Headphone"
#define AUDIO_PORT_LINE_OUT   "Line-Out"
#define AUDIO_PORT_MICROPHONE "Microphone"
#define AUDIO_PORT_LINE_IN    "Line-In"
#define AUDIO_PORT_CD         "CD"

typedef struct {
        audio_port_t port;
        char         name[AUDIO_PORT_NAME_LENGTH + 1];
} audio_port_details_t;

#define BYTES_PER_SAMPLE sizeof(sample)
#define PCMU_AUDIO_ZERO	127
#define PCMA_AUDIO_ZERO	213
#define L16_AUDIO_ZERO	0
#define MAX_AMP		100
#define DEVICE_REC_BUF	16000
#define DEVICE_BUF_UNIT	320

#endif /* _RAT_AUDIO_TYPES_H_ */
