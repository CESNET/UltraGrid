/*
 * FILE:     auddev.c
 * PROGRAM:  RAT
 * AUTHOR:   Orion Hodson 
 *
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:58 $
 */
 
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw.h"
#include "audio_hw/null.h"
#include "audio_hw/freebsd_luigi.h"
#include "audio_hw/freebsd_newpcm.h"
#include "audio_hw/freebsd_pca.h"
#include "audio_hw/solaris.h"
#include "audio_hw/solaris_osprey.h"
#include "audio_hw/linux_oss.h"
#include "audio_hw/linux_alsa.h"
#include "audio_hw/linux_ixj.h"
#include "audio_hw/irix.h"
#include "audio_hw/win32.h"
#include "audio_hw/macosx.h"

typedef struct {
        int  (*audio_if_init)(void);                 /* Test and initialize audio interface (OPTIONAL)    */
        int  (*audio_if_free)(void);                 /* Free audio interface (OPTIONAL)                   */
        int  (*audio_if_dev_cnt)(void);              /* Device count for interface (REQUIRED)             */
        char* 
             (*audio_if_dev_name)(int);              /* Device name query (REQUIRED)                      */

        int  (*audio_if_open)(int, audio_format *ifmt, audio_format *ofmt); /* Open device with formats   */
        void (*audio_if_close)(int);                 /* Close device (REQUIRED)                           */
        void (*audio_if_drain)(int);                 /* Drain device (REQUIRED)                           */
        int  (*audio_if_duplex)(int);                /* Device full duplex (REQUIRED)                     */

        int  (*audio_if_read) (int, u_char*, int);   /* Read samples (REQUIRED)                           */
        int  (*audio_if_write)(int, u_char*, int);   /* Write samples (REQUIRED)                          */
        void (*audio_if_non_block)(int);             /* Set device non-blocking (REQUIRED)                */
        void (*audio_if_block)(int);                 /* Set device blocking (REQUIRED)                    */

        void (*audio_if_set_igain)(int,int);          /* Set input gain (REQUIRED)                        */
        int  (*audio_if_get_igain)(int);              /* Get input gain (REQUIRED)                        */
        void (*audio_if_set_ogain)(int,int);          /* Set output gain (REQUIRED)                       */
        int  (*audio_if_get_ogain)(int);              /* Get output gain (REQUIRED)                       */
        void (*audio_if_loopback)(int, int);          /* Enable hardware loopback (OPTIONAL)              */

        void (*audio_if_set_oport)(int, audio_port_t); /* Set output port (REQUIRED)                      */
        audio_port_t
             (*audio_if_get_oport)(int);               /* Get output port (REQUIRED)                      */
        const audio_port_details_t*
             (*audio_if_get_oport_details)(int, int);  /* Get details of port (REQUIRED)                  */
        int  (*audio_if_get_oport_count)(int);         /* Get number of output ports (REQUIRED)           */

        void (*audio_if_set_iport)(int, audio_port_t); /* Set input port (REQUIRED)                       */
        audio_port_t
             (*audio_if_get_iport)(int);               /* Get input port (REQUIRED)                       */
        const audio_port_details_t*
             (*audio_if_get_iport_details)(int, int);  /* Get details of port (REQUIRED)                  */
        int  (*audio_if_get_iport_count)(int);          /* Get number of input ports (REQUIRED)            */

        int  (*audio_if_is_ready)(int);              /* Poll for audio availability (REQUIRED)            */
        void (*audio_if_wait_for)(int, int);         /* Wait until audio is available (REQUIRED)          */
        int  (*audio_if_format_supported)(int, audio_format *);
} audio_if_t;

audio_if_t audio_if_table[] = {
#ifdef HAVE_MACOSX_AUDIO
        {
                macosx_audio_init,
                NULL,
                macosx_audio_device_count,
                macosx_audio_device_name,
                macosx_audio_open,
                macosx_audio_close,
                macosx_audio_drain,
                macosx_audio_duplex,
                macosx_audio_read,
                macosx_audio_write,
                macosx_audio_non_block,
                macosx_audio_block,
                macosx_audio_set_igain,
                macosx_audio_get_igain,
                macosx_audio_set_ogain,
                macosx_audio_get_ogain,
                macosx_audio_loopback,
                macosx_audio_oport_set,
                macosx_audio_oport_get,
                macosx_audio_oport_details,
                macosx_audio_oport_count,
                macosx_audio_iport_set,
                macosx_audio_iport_get,
                macosx_audio_iport_details,
                macosx_audio_iport_count,
                macosx_audio_is_ready,
                macosx_audio_wait_for,
                macosx_audio_supports
        },
#endif /* HAVE_MACOSX_AUDIO */
#ifdef HAVE_SGI_AUDIO
        {
                NULL, 
                NULL, 
                sgi_audio_device_count,
                sgi_audio_device_name,
                sgi_audio_open,
                sgi_audio_close,
                sgi_audio_drain,
                sgi_audio_duplex,
                sgi_audio_read,
                sgi_audio_write,
                sgi_audio_non_block,
                sgi_audio_block,
                sgi_audio_set_igain,
                sgi_audio_get_igain,
                sgi_audio_set_ogain,
                sgi_audio_get_ogain,
                sgi_audio_loopback,
                sgi_audio_oport_set,
                sgi_audio_oport_get,
                sgi_audio_oport_details,
                sgi_audio_oport_count,
                sgi_audio_iport_set,
                sgi_audio_iport_get,
                sgi_audio_iport_details,
                sgi_audio_iport_count,
                sgi_audio_is_ready,
                sgi_audio_wait_for,
                NULL
        },
#endif /* SGI_AUDIO */
#ifdef HAVE_SPARC_AUDIO
        {
                NULL,
                NULL,
                sparc_audio_device_count,
                sparc_audio_device_name,
                sparc_audio_open,
                sparc_audio_close,
                sparc_audio_drain,
                sparc_audio_duplex,
                sparc_audio_read,
                sparc_audio_write,
                sparc_audio_non_block,
                sparc_audio_block,
                sparc_audio_set_igain,
                sparc_audio_get_igain,
                sparc_audio_set_ogain,
                sparc_audio_get_ogain,
                sparc_audio_loopback,
                sparc_audio_oport_set,
                sparc_audio_oport_get,
                sparc_audio_oport_details,
                sparc_audio_oport_count,
                sparc_audio_iport_set,
                sparc_audio_iport_get,
                sparc_audio_iport_details,
                sparc_audio_iport_count,
                sparc_audio_is_ready,
                sparc_audio_wait_for,
                sparc_audio_supports
        },
#endif /* HAVE_SPARC_AUDIO */
#ifdef HAVE_OSPREY_AUDIO
        {
                osprey_audio_init, 
                NULL, 
                osprey_audio_device_count,
                osprey_audio_device_name,
                osprey_audio_open,
                osprey_audio_close,
                osprey_audio_drain,
                osprey_audio_duplex,
                osprey_audio_read,
                osprey_audio_write,
                osprey_audio_non_block,
                osprey_audio_block,
                osprey_audio_set_igain,
                osprey_audio_get_igain,
                osprey_audio_set_ogain,
                osprey_audio_get_ogain,
                osprey_audio_loopback,
                osprey_audio_oport_set,
                osprey_audio_oport_get,
                osprey_audio_oport_details,
                osprey_audio_oport_count,
                osprey_audio_iport_set,
                osprey_audio_iport_get,
                osprey_audio_iport_details,
                osprey_audio_iport_count,
                osprey_audio_is_ready,
                osprey_audio_wait_for,
                NULL
        },
#endif /* HAVE_OSPREY_AUDIO */
#ifdef HAVE_ALSA_AUDIO
        {
                alsa_audio_init, 
                NULL,
                alsa_get_device_count,
                alsa_get_device_name,
                alsa_audio_open,
                alsa_audio_close,
                alsa_audio_drain,
                alsa_audio_duplex,
                alsa_audio_read,
                alsa_audio_write,
                alsa_audio_non_block,
                alsa_audio_block,
                alsa_audio_set_igain,
                alsa_audio_get_igain,
                alsa_audio_set_ogain,
                alsa_audio_get_ogain,
                NULL,
                alsa_audio_oport_set,
                alsa_audio_oport_get,
                alsa_audio_oport_details,
                alsa_audio_oport_count,
                alsa_audio_iport_set,
                alsa_audio_iport_get,
                alsa_audio_iport_details,
                alsa_audio_iport_count,
                alsa_audio_is_ready,
                alsa_audio_wait_for,
                alsa_audio_supports
        },
#endif /* HAVE_ALSA_AUDIO */
#ifdef HAVE_OSS_AUDIO
        {
                oss_audio_init, 
                NULL,
                oss_get_device_count,
                oss_get_device_name,
                oss_audio_open,
                oss_audio_close,
                oss_audio_drain,
                oss_audio_duplex,
                oss_audio_read,
                oss_audio_write,
                oss_audio_non_block,
                oss_audio_block,
                oss_audio_set_igain,
                oss_audio_get_igain,
                oss_audio_set_ogain,
                oss_audio_get_ogain,
                oss_audio_loopback,
                oss_audio_oport_set,
                oss_audio_oport_get,
                oss_audio_oport_details,
                oss_audio_oport_count,
                oss_audio_iport_set,
                oss_audio_iport_get,
                oss_audio_iport_details,
                oss_audio_iport_count,
                oss_audio_is_ready,
                oss_audio_wait_for,
                oss_audio_supports
        },
#endif /* HAVE_OSS_AUDIO */
#ifdef HAVE_IXJ_AUDIO
        {
                ixj_audio_init, 
                NULL,
                ixj_get_device_count,
                ixj_get_device_name,
                ixj_audio_open,
                ixj_audio_close,
                ixj_audio_drain,
                ixj_audio_duplex,
                ixj_audio_read,
                ixj_audio_write,
                ixj_audio_non_block,
                ixj_audio_block,
                ixj_audio_set_igain,
                ixj_audio_get_igain,
                ixj_audio_set_ogain,
                ixj_audio_get_ogain,
                ixj_audio_loopback,
                ixj_audio_oport_set,
                ixj_audio_oport_get,
                ixj_audio_oport_details,
                ixj_audio_oport_count,
                ixj_audio_iport_set,
                ixj_audio_iport_get,
                ixj_audio_iport_details,
                ixj_audio_iport_count,
                ixj_audio_is_ready,
                ixj_audio_wait_for,
                ixj_audio_supports
        },
#endif /* HAVE_IXJ_AUDIO */
#ifdef WIN32
        {
                w32sdk_audio_init,
                w32sdk_audio_free, 
                w32sdk_get_device_count,
                w32sdk_get_device_name,
                w32sdk_audio_open,
                w32sdk_audio_close,
                w32sdk_audio_drain,
                w32sdk_audio_duplex,
                w32sdk_audio_read,
                w32sdk_audio_write,
                w32sdk_audio_non_block,
                w32sdk_audio_block,
                w32sdk_audio_set_igain,
                w32sdk_audio_get_igain,
                w32sdk_audio_set_ogain,
                w32sdk_audio_get_ogain,
                w32sdk_audio_loopback,
                w32sdk_audio_oport_set,
                w32sdk_audio_oport_get,
                w32sdk_audio_oport_details,
                w32sdk_audio_oport_count,
                w32sdk_audio_iport_set,
                w32sdk_audio_iport_get,
                w32sdk_audio_iport_details,
                w32sdk_audio_iport_count,
                w32sdk_audio_is_ready,
                w32sdk_audio_wait_for,
                w32sdk_audio_supports
        },
#endif /* WIN32 */
#ifdef HAVE_LUIGI_AUDIO
        {
                luigi_audio_query_devices,
                NULL,
                luigi_get_device_count,
                luigi_get_device_name,
                luigi_audio_open,
                luigi_audio_close,
                luigi_audio_drain,
                luigi_audio_duplex,
                luigi_audio_read,
                luigi_audio_write,
                luigi_audio_non_block,
                luigi_audio_block,
                luigi_audio_set_igain,
                luigi_audio_get_igain,
                luigi_audio_set_ogain,
                luigi_audio_get_ogain,
                luigi_audio_loopback,
                luigi_audio_oport_set,
                luigi_audio_oport_get,
                luigi_audio_oport_details,
                luigi_audio_oport_count,
                luigi_audio_iport_set,
                luigi_audio_iport_get,
                luigi_audio_iport_details,
                luigi_audio_iport_count,
                luigi_audio_is_ready,
                luigi_audio_wait_for,
                luigi_audio_supports
        },
#endif /* HAVE_LUIGI_AUDIO */
#ifdef HAVE_NEWPCM_AUDIO
        {
                newpcm_audio_query_devices,
                NULL,
                newpcm_get_device_count,
                newpcm_get_device_name,
                newpcm_audio_open,
                newpcm_audio_close,
                newpcm_audio_drain,
                newpcm_audio_duplex,
                newpcm_audio_read,
                newpcm_audio_write,
                newpcm_audio_non_block,
                newpcm_audio_block,
                newpcm_audio_set_igain,
                newpcm_audio_get_igain,
                newpcm_audio_set_ogain,
                newpcm_audio_get_ogain,
                newpcm_audio_loopback,
                newpcm_audio_oport_set,
                newpcm_audio_oport_get,
                newpcm_audio_oport_details,
                newpcm_audio_oport_count,
                newpcm_audio_iport_set,
                newpcm_audio_iport_get,
                newpcm_audio_iport_details,
                newpcm_audio_iport_count,
                newpcm_audio_is_ready,
                newpcm_audio_wait_for,
                newpcm_audio_supports
        },
#endif /* HAVE_NEWPCM_AUDIO */
#ifdef HAVE_PCA_AUDIO
        {
                pca_audio_init,
                NULL, 
                pca_audio_device_count,
                pca_audio_device_name,
                pca_audio_open,
                pca_audio_close,
                pca_audio_drain,
                pca_audio_duplex,
                pca_audio_read,
                pca_audio_write,
                pca_audio_non_block,
                pca_audio_block,
                pca_audio_set_igain,
                pca_audio_get_igain,
                pca_audio_set_ogain,
                pca_audio_get_ogain,
                pca_audio_loopback,
                pca_audio_oport_set,
                pca_audio_oport_get,
                pca_audio_oport_details,
                pca_audio_oport_count,
                pca_audio_iport_set,
                pca_audio_iport_get,
                pca_audio_iport_details,
                pca_audio_iport_count,
                pca_audio_is_ready,
                pca_audio_wait_for,
                pca_audio_supports
        },
#endif /* HAVE_PCA_AUDIO */
        {
                /* This is the null audio device - it should always go last so that
                 * audio_get_null_device works.  The idea being when we can't get hold
                 * of a real device we fake one.  Prevents lots of problems elsewhere.
                 */
                NULL,
                NULL, 
                null_audio_device_count,
                null_audio_device_name,
                null_audio_open,
                null_audio_close,
                null_audio_drain,
                null_audio_duplex,
                null_audio_read,
                null_audio_write,
                null_audio_non_block,
                null_audio_block,
                null_audio_set_igain,
                null_audio_get_igain,
                null_audio_set_ogain,
                null_audio_get_ogain,
                null_audio_loopback,
                null_audio_oport_set,
                null_audio_oport_get,
                null_audio_oport_details,
                null_audio_oport_count,
                null_audio_iport_set,
                null_audio_iport_get,
                null_audio_iport_details,
                null_audio_iport_count,
                null_audio_is_ready,
                null_audio_wait_for,
                null_audio_supports
        }
};

#define INITIAL_AUDIO_INTERFACES (sizeof(audio_if_table)/sizeof(audio_if_t))

/* Active interfaces is a table of entries pointing to entries in
 * audio interfaces table.  Audio open returns index to these */
static audio_desc_t active_device_desc[INITIAL_AUDIO_INTERFACES];
static int active_devices;
static uint32_t actual_devices, actual_interfaces;
static audio_device_details_t *dev_details;

#define MAX_ACTIVE_DEVICES   2

/* These are requested device formats.  */
#define AUDDEV_REQ_IFMT      0
#define AUDDEV_REQ_OFMT      1

/* These are actual device formats that are transparently converted 
 * into the required ones during reads and writes.  */
#define AUDDEV_ACT_IFMT      2
#define AUDDEV_ACT_OFMT      3

#define AUDDEV_NUM_FORMATS   4

static audio_format* fmts[MAX_ACTIVE_DEVICES][AUDDEV_NUM_FORMATS];
static sample*       convert_buf[MAX_ACTIVE_DEVICES]; /* used if conversions used */

/* Counters for samples read/written */
static uint32_t samples_read[MAX_ACTIVE_DEVICES], samples_written[MAX_ACTIVE_DEVICES];

/* We map indexes outside range for file descriptors so people don't attempt
 * to circumvent audio interface.  If something is missing it should be added
 * to the interfaces...
 */

#define AIF_GET_INTERFACE(x) ((((x) & 0x0f00) >> 8) - 1)
#define AIF_GET_DEVICE_NO(x) (((x) & 0x000f) - 1)
#define AIF_MAKE_DESC(iface,dev) (((iface + 1) << 8) | (dev + 1))

#define AIF_VALID_INTERFACE(id) ((id & 0x0f00))
#define AIF_VALID_DEVICE_NO(id) ((id & 0x000f))

/*****************************************************************************
 *
 * Code for working out how many devices are, what they are called, and what 
 * descriptor should be used to access them.
 *
 *****************************************************************************/

uint32_t
audio_get_device_count()
{
        return actual_devices;
}

const audio_device_details_t *
audio_get_device_details(uint32_t idx)
{
        assert(idx < actual_devices);

        if (idx < actual_devices) {
                return &dev_details[idx];
        }
        return NULL;
}

audio_desc_t
audio_get_null_device()
{
        audio_desc_t ad;

        /* Null audio device is only device on the last interface*/
        ad = AIF_MAKE_DESC(actual_interfaces - 1, 0);

        return ad;
}

/*****************************************************************************
 *
 * Interface code.  Maps audio functions to audio devices.
 *
 *****************************************************************************/

static int
get_active_device_index(audio_desc_t ad)
{
        int i;
        
        for (i = 0; i < active_devices; i++) {
                if (active_device_desc[i] == ad) {
                        return i;
                }
        }

	debug_msg("Device %d is not active\n", (int) ad);
        return -1;
}

int
audio_device_is_open(audio_desc_t ad)
{
        int dev = get_active_device_index(ad);
        return (dev != -1);
}

int
audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
        audio_format 	format;
        int 		iface, device, dev_idx;
        int 		success;
	char 		s[50];

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(ifmt != NULL);
        assert(ofmt != NULL);

        if (ofmt->sample_rate != ifmt->sample_rate) {
                /* Fail on things we don't support */
                debug_msg("Not supported\n");
                return 0;
        }

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        if (active_devices == MAX_ACTIVE_DEVICES) {
                debug_msg("Already have the maximum number of devices (%d) open.\n", MAX_ACTIVE_DEVICES);
                return FALSE;
        }

        dev_idx   = active_devices;

        assert(audio_if_table[iface].audio_if_open);

        if (audio_format_get_common(ifmt, ofmt, &format) == FALSE) {
                /* Input and output formats incompatible */
                return 0;
        }

        fmts[dev_idx][AUDDEV_ACT_IFMT] = audio_format_dup(&format);
        fmts[dev_idx][AUDDEV_ACT_OFMT] = audio_format_dup(&format);

        /* Formats can get changed in audio_if_open, but only sample
         * type, not the number of channels or freq 
         */
        success = audio_if_table[iface].audio_if_open(device, 
                                                          fmts[dev_idx][AUDDEV_ACT_IFMT], 
                                                          fmts[dev_idx][AUDDEV_ACT_OFMT]);

        if (success) {
                /* Add device to list of those active */
                debug_msg("Opened device: %s\n", audio_if_table[iface].audio_if_dev_name(device));
                active_device_desc[dev_idx] = ad;
                active_devices ++;

                if ((fmts[dev_idx][AUDDEV_ACT_IFMT]->sample_rate != format.sample_rate) ||
                    (fmts[dev_idx][AUDDEV_ACT_OFMT]->sample_rate != format.sample_rate) ||
                    (fmts[dev_idx][AUDDEV_ACT_IFMT]->channels    != format.channels)    ||
                    (fmts[dev_idx][AUDDEV_ACT_OFMT]->channels    != format.channels)) {
                        debug_msg("Device changed sample rate or channels - unsupported functionality.\n");
                        audio_close(ad);
                        return FALSE;
                }

                if (!audio_if_table[iface].audio_if_duplex(device)) {
                        printf("RAT v3.2.0 and later require a full duplex audio device, but \n");
                        printf("your device only supports half-duplex operation. Sorry.\n");
                        audio_close(ad);
                        return FALSE;
                }
                
                /* If we are going to need conversion between requested and 
                 * actual device formats store requested formats */
                if (!audio_format_match(ifmt, fmts[dev_idx][AUDDEV_ACT_IFMT])) {
                        fmts[dev_idx][AUDDEV_REQ_IFMT] = audio_format_dup(ifmt);
			audio_format_name(fmts[dev_idx][AUDDEV_REQ_IFMT], s, 50);
			debug_msg("Requested Input: %s\n", s);
			audio_format_name(fmts[dev_idx][AUDDEV_ACT_IFMT], s, 50);
			debug_msg("Actual Input:    %s\n", s);
                } else {
			audio_format_name(fmts[dev_idx][AUDDEV_ACT_IFMT], s, 50);
			debug_msg("Input:  %s\n", s);
		}

                if (!audio_format_match(ofmt, fmts[dev_idx][AUDDEV_ACT_OFMT])) {
                        fmts[dev_idx][AUDDEV_REQ_OFMT] = audio_format_dup(ofmt);
			audio_format_name(fmts[dev_idx][AUDDEV_REQ_OFMT], s, 50);
			debug_msg("Requested Output: %s\n", s);
			audio_format_name(fmts[dev_idx][AUDDEV_ACT_OFMT], s, 50);
			debug_msg("Actual Output:    %s\n", s);
                } else {
			audio_format_name(fmts[dev_idx][AUDDEV_ACT_OFMT], s, 50);
			debug_msg("Output: %s\n", s);
		}

                if (fmts[dev_idx][AUDDEV_REQ_IFMT] || fmts[dev_idx][AUDDEV_REQ_OFMT]) {
                        convert_buf[dev_idx] = (sample*)malloc(DEVICE_REC_BUF); /* is this in samples or bytes ? */
                }

                samples_read[dev_idx]    = 0;
                samples_written[dev_idx] = 0;

                return TRUE;
        }

        audio_format_free(&fmts[dev_idx][AUDDEV_ACT_IFMT]);
        audio_format_free(&fmts[dev_idx][AUDDEV_ACT_OFMT]);

        return FALSE;
}

void
audio_close(audio_desc_t ad)
{
        int i, j, k, iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        audio_if_table[iface].audio_if_close(device);

        /* Check device is open */
        assert(get_active_device_index(ad) != -1);

        i = j = 0;
        for(i = 0; i < active_devices; i++) {
                if (active_device_desc[i] == ad) {
                        for(k = 0; k < AUDDEV_NUM_FORMATS; k++) {
                                if (fmts[i][k] != NULL) audio_format_free(&fmts[i][k]);                                
                        }
                        if (convert_buf[i]) {
                                free(convert_buf[i]);
                                convert_buf[i] = NULL;
                        }
                        samples_written[i] = 0;
                        samples_read[i]    = 0;
                } else {
                        if (i != j) {
                                active_device_desc[j] = active_device_desc[i];
                                for(k = 0; k < AUDDEV_NUM_FORMATS; k++) {
                                        assert(fmts[j][k] == NULL);
                                        fmts[j][k] = fmts[i][k];
                                }
                                convert_buf[j]     = convert_buf[i];
                                samples_read[j]    = samples_read[i];
                                samples_written[j] = samples_written[i];
                        }
                        j++;
                }
        }

        active_devices --;
}

const audio_format*
audio_get_ifmt(audio_desc_t ad)
{
        int idx = get_active_device_index(ad);

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(idx >= 0 && idx < active_devices);

        if (fmts[idx][AUDDEV_REQ_IFMT]) {
                return fmts[idx][AUDDEV_REQ_IFMT];
        }

        return fmts[idx][AUDDEV_ACT_IFMT];
}

const audio_format*
audio_get_ofmt(audio_desc_t ad)
{
        int idx = get_active_device_index(ad);

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(idx >= 0 && idx < active_devices);

        if (fmts[idx][AUDDEV_REQ_OFMT]) {
                return fmts[idx][AUDDEV_REQ_OFMT];
        }

        return fmts[idx][AUDDEV_ACT_OFMT];
}

void
audio_drain(audio_desc_t ad)
{
        int device, iface;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));
        
        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        audio_if_table[iface].audio_if_drain(device);
}

int
audio_duplex(audio_desc_t ad)
{
        int device, iface;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        return audio_if_table[iface].audio_if_duplex(device);
}

int
audio_read(audio_desc_t ad, sample *buf, int samples)
{
        /* Samples is the number of samples to read * number of channels */
        int read_len;
        int sample_size;
        int device, iface;
        int idx = get_active_device_index(ad);

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));        
        assert(idx >= 0 && idx < active_devices);
        assert(buf != NULL);

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        if (fmts[idx][AUDDEV_REQ_IFMT] == NULL) {
                /* No conversion necessary as input format and real format are
                 * the same. [Input format only allocated if different from
                 * real format].
                 */
                sample_size = fmts[idx][AUDDEV_ACT_IFMT]->bits_per_sample / 8;
                read_len    = audio_if_table[iface].audio_if_read(device, (u_char*)buf, samples * sample_size);
                samples_read[idx] += read_len / (sample_size * fmts[idx][AUDDEV_ACT_IFMT]->channels);
        } else {
                assert(fmts[idx][AUDDEV_ACT_IFMT] != NULL);
                sample_size = fmts[idx][AUDDEV_ACT_IFMT]->bits_per_sample / 8;
                read_len = samples * sample_size * fmts[idx][AUDDEV_ACT_IFMT]->channels;
                read_len    = audio_if_table[iface].audio_if_read(device, (u_char*)convert_buf[idx], read_len);
                read_len    = audio_format_buffer_convert(fmts[idx][AUDDEV_ACT_IFMT], (u_char*) convert_buf[idx],  read_len, fmts[idx][AUDDEV_REQ_IFMT], (u_char*) buf, DEVICE_REC_BUF);
                sample_size = fmts[idx][AUDDEV_REQ_IFMT]->bits_per_sample / 8;
                samples_read[idx] += read_len / (sample_size * fmts[idx][AUDDEV_REQ_IFMT]->channels);
        }

        return read_len / sample_size;
}

int
audio_write(audio_desc_t ad, sample *buf, int len)
{
        int write_len ,sample_size;
        int iface, device;
        int idx = get_active_device_index(ad);
        
        assert(idx >= 0 && idx < active_devices);
        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        if (fmts[idx][AUDDEV_REQ_OFMT] == NULL) {
                /* No conversion necessary as output format and real format are
                 * the same. [Output format only allocated if different from
                 * real format].
                 */
                sample_size = fmts[idx][AUDDEV_ACT_OFMT]->bits_per_sample / 8;
                write_len   = audio_if_table[iface].audio_if_write(device, (u_char*)buf, len * sample_size);
                samples_written[idx] += write_len / (sample_size * fmts[idx][AUDDEV_ACT_OFMT]->channels);
        } else {
                write_len = audio_format_buffer_convert(fmts[idx][AUDDEV_REQ_OFMT],
                                                        (u_char*)buf,
                                                        len,
                                                        fmts[idx][AUDDEV_ACT_OFMT],
                                                        (u_char*) convert_buf[idx],
                                                        DEVICE_REC_BUF);
                audio_if_table[iface].audio_if_write(device, (u_char*)convert_buf[idx], write_len);
                sample_size = fmts[idx][AUDDEV_ACT_OFMT]->bits_per_sample / 8;
                samples_written[idx] += write_len / (sample_size * fmts[idx][AUDDEV_REQ_OFMT]->channels);
        }

        return write_len / sample_size;
}

void
audio_non_block(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        audio_if_table[iface].audio_if_non_block(device);
}

void
audio_block(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        audio_if_table[iface].audio_if_block(device);
}

void
audio_set_igain(audio_desc_t ad, int gain)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        assert(gain >= 0);
        assert(gain <= MAX_AMP);

        audio_if_table[iface].audio_if_set_igain(device, gain);
}

int
audio_get_igain(audio_desc_t ad)
{
        int gain;
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        gain = audio_if_table[iface].audio_if_get_igain(device);
	debug_msg("GAIN=%d\n", gain);

        assert(gain >= 0);
        assert(gain <= MAX_AMP);

        return gain;
}

void
audio_set_ogain(audio_desc_t ad, int volume)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad));
	assert(AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        assert(volume >= 0);
        assert(volume <= MAX_AMP);

        audio_if_table[iface].audio_if_set_ogain(device, volume);
}

int
audio_get_ogain(audio_desc_t ad)
{
        int volume;
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        volume = audio_if_table[iface].audio_if_get_ogain(device);
        assert(volume >= 0);
        assert(volume <= MAX_AMP);

        return volume;
}

void
audio_loopback(audio_desc_t ad, int gain)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        assert(gain >= 0);
        assert(gain <= MAX_AMP);

        if (audio_if_table[iface].audio_if_loopback) {
                audio_if_table[iface].audio_if_loopback(device, gain);
        }
}

void
audio_set_oport(audio_desc_t ad, audio_port_t port)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        audio_if_table[iface].audio_if_set_oport(device, port);
}

audio_port_t
audio_get_oport(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        return (audio_if_table[iface].audio_if_get_oport(device));
}

const audio_port_details_t*
audio_get_oport_details(audio_desc_t ad, int port_idx)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        return (audio_if_table[iface].audio_if_get_oport_details(device, port_idx));
}

int
audio_get_oport_count(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        return audio_if_table[iface].audio_if_get_oport_count(device);
}

void
audio_set_iport(audio_desc_t ad, audio_port_t port)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        audio_if_table[iface].audio_if_set_iport(device, port);
}

audio_port_t
audio_get_iport(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        return (audio_if_table[iface].audio_if_get_iport(device));
}

const audio_port_details_t*
audio_get_iport_details(audio_desc_t ad, int port_idx)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        return (audio_if_table[iface].audio_if_get_iport_details(device, port_idx));
}

int
audio_get_iport_count(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);
        
        return audio_if_table[iface].audio_if_get_iport_count(device);
}

int
audio_is_ready(audio_desc_t ad)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        return (audio_if_table[iface].audio_if_is_ready(device));
}

void
audio_wait_for(audio_desc_t ad, int delay_ms)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        audio_if_table[iface].audio_if_wait_for(device, delay_ms);
}

/* Code for adding/initialising/removing audio ifaces */

int
audio_device_supports(audio_desc_t ad, uint16_t rate, uint16_t channels)
{
        int iface, device;

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
//      assert(audio_device_is_open(ad));

        iface  = AIF_GET_INTERFACE(ad);
        device = AIF_GET_DEVICE_NO(ad);

        if (((rate % 8000) && (rate % 11025)) || channels > 2) {
                debug_msg("Invalid combo %d Hz %d channels\n", rate, channels);
                return FALSE;
        }

        if (audio_if_table[iface].audio_if_format_supported) {
                audio_format tfmt;
                tfmt.encoding    = DEV_S16;
                tfmt.sample_rate = rate;
                tfmt.channels    = channels;
                return audio_if_table[iface].audio_if_format_supported(device, &tfmt);
        }

        debug_msg("Format support query function not implemented! Lying about supported formats.\n");

        return TRUE;
}

uint32_t
audio_get_device_time(audio_desc_t ad)
{
        audio_format *fmt;
        uint32_t       samples_per_block;
        int dev = get_active_device_index(ad);

        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(dev >= 0 && dev < active_devices);

        if (fmts[dev][AUDDEV_REQ_IFMT]) {
                fmt = fmts[dev][AUDDEV_REQ_IFMT]; 
        } else {
                fmt = fmts[dev][AUDDEV_ACT_IFMT]; 
        }
        
        samples_per_block = fmt->bytes_per_block * 8 / (fmt->channels * fmt->bits_per_sample);
        
        return (samples_read[dev]/samples_per_block) * samples_per_block;
}

uint32_t
audio_get_samples_read(audio_desc_t ad)
{
        int dev = get_active_device_index(ad);
        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(dev >= 0 && dev < active_devices);
        return samples_read[dev];
}

uint32_t
audio_get_samples_written(audio_desc_t ad)
{
        int dev = get_active_device_index(ad);
        assert(AIF_VALID_INTERFACE(ad) && AIF_VALID_DEVICE_NO(ad));
        assert(dev >= 0 && dev < active_devices);
        return samples_written[dev];
}

int
audio_init_interfaces(void)
{
        static int inited = 0;
        uint32_t i, j, k, n, devs[INITIAL_AUDIO_INTERFACES];

        if (inited) {
                return 0;
        }
        inited++;

        actual_devices = 0;
        actual_interfaces = INITIAL_AUDIO_INTERFACES;
        for(i = 0; i < INITIAL_AUDIO_INTERFACES; i++) {
                if (audio_if_table[i].audio_if_init) {
                        audio_if_table[i].audio_if_init(); 
                }
                assert(audio_if_table[i].audio_if_dev_cnt);
		devs[i] = audio_if_table[i].audio_if_dev_cnt();
                actual_devices += devs[i];
        }

        /* Remove interfaces where number of devs is zero.
         * This could be inside init loop above, but makes it
         * hard to read and does not save anything worthwhile.
         */
        for(i = j = 0; i < INITIAL_AUDIO_INTERFACES; i++) {
                n = INITIAL_AUDIO_INTERFACES - i - 1;
                if (devs[i] == 0 && n != 0) {
                        memmove(audio_if_table + j, audio_if_table + j + 1, n * sizeof(audio_if_t));
                        actual_interfaces --;
                } else {
                        j++;
                }
        }

        /* Create device details table and fill it in */
        dev_details = (audio_device_details_t*)malloc(sizeof(audio_device_details_t) * actual_devices);
        k = 0;
        for (i = 0; i < actual_interfaces; i++) {
                n = audio_if_table[i].audio_if_dev_cnt();
                assert(n > 0);
                for (j = 0; j < n; j++) {
                        dev_details[k].name       = strdup(audio_if_table[i].audio_if_dev_name(j));
                        dev_details[k].descriptor = AIF_MAKE_DESC(i, j);
                        k++;
                }
        }
        assert(k == actual_devices);

        return TRUE;
}

int
audio_free_interfaces(void)
{
        uint32_t i;

        for(i = 0; i < INITIAL_AUDIO_INTERFACES; i++) {
                if (audio_if_table[i].audio_if_free) {
                        audio_if_table[i].audio_if_free(); 
                }
        }

        if (dev_details != NULL) {
                for(i = 0; i < actual_devices; i++) {
                        free(dev_details[i].name);
                }
                free(dev_details);
                dev_details = NULL;
        }

        return TRUE;
}

