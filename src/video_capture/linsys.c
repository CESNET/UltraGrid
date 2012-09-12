/*
 * FILE:    linsys.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#ifndef HAVE_MACOSX
#ifdef HAVE_LINSYS              /* From config.h */

#define FMODE_MAGIC             0x9B7DA07u
#define MAX_TILES               4

#define LINSYS_AUDIO_BPS 2
#define LINSYS_AUDIO_SAMPLE_RATE 48000

#define LINSYS_AUDIO_BUFSIZE (LINSYS_AUDIO_BPS * LINSYS_AUDIO_SAMPLE_RATE *\
        audio_capture_channels * 1)

#include "video_capture/linsys.h"
#include "audio/audio.h"
#include "audio/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <semaphore.h>

/* 
   LINSYS SDK includes. We are also using a couple of utility functions from the
   LINSYS SDK Examples. That is where util.h comes from.
*/

#include "Linsys/include/sdivideo.h"
#include "Linsys/include/master.h"
#include "Linsys/Examples/util.h"

#define MAXLEN 256

static const char progname[] = "videocapture";
static const char sys_fmt[] = "/sys/class/sdivideo/sdivideo%cx%i/%s";
static const char devfile_fmt[] = "/dev/sdivideorx%1u";
static const char audio_dev[] = "/dev/sdiaudiorx0";

static volatile bool should_exit = false;

struct frame_mode {
        char  * const    name;
        unsigned int     width;
        unsigned int     height;
        double           fps;
        int              aux; /* AUX_* */
        unsigned int     magic;
};

static const struct frame_mode frame_modes[] = {
        [SDIVIDEO_CTL_UNLOCKED] =
                { "Unset", 0u, 0u, 0.0, 0, FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ] =
                { "SMPTE 125M 486i 59.94 Hz", 720u, 486u , 59.94 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_BT_601_576I_50HZ] =
                { "ITU-R BT.601 720x576i 50 Hz", 720u, 576u, 50.0 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ] =
                { "SMPTE 260M 1035i 60 Hz", 1920u, 1035u, 60.0 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ] =
                { "SMPTE 260M 1035i 59.94 Hz", 1920u, 1035u, 59.94 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ] =
                { "SMPTE 295M 1080i 50 Hz", 1920u, 1080u, 50.0 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ] =
                { "SMPTE 274M 1080i 60 Hz", 1920u, 1080u, 60.0 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080PSF_30HZ] =
                { "SMPTE 274M 1080psf 30 Hz", 1920u, 1080u, 30.0, AUX_SF,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ] =
                { "SMPTE 274M 1080i 59.94 Hz", 1920u, 1080u, 59.94 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080PSF_29_97HZ] =
                { "SMPTE 274M 1080psf 29.97 Hz", 1920u, 1080u, 29.97, AUX_SF,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ] =
                { "SMPTE 274M 1080i 50 Hz", 1920u, 1080u, 50.0 / 2.0, AUX_INTERLACED,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080PSF_25HZ] =
                { "SMPTE 274M 1080psf 25 Hz", 1920u, 1080u, 25.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ] =
                { "SMPTE 274M 1080psf 24 Hz", 1920u, 1080u, 24.0, AUX_SF,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ] =
                { "SMPTE 274M 1080psf 23.98 Hz", 1920u, 1080u, 23.98, AUX_SF,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ] =
                { "SMPTE 274M 1080p 30 Hz", 1920u, 1080u, 30.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ] =
                { "SMPTE 274M 1080p 29.97 Hz", 1920u, 1080u, 29.97, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ] =
                { "SMPTE 274M 1080p 25 Hz", 1920u, 1080u, 25.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ] =
                { "SMPTE 274M 1080p 24 Hz", 1920u, 1080u, 24.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ] =
                { "SMPTE 274M 1080p 23.98 Hz", 1920u, 1080u, 23.98, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_60HZ] =
                { "SMPTE 296M 720p 60 Hz", 1280u, 720u, 60.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ] =
                { "SMPTE 296M 720p 59.94 Hz", 1280u, 720u, 59.94, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_50HZ] =
                { "SMPTE 296M 720p 50 Hz", 1280u, 720u, 50.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_30HZ] =
                { "SMPTE 296M 720p 30 Hz", 1280u, 720u, 30.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ] =
                { "SMPTE 296M 720p 29.97 Hz", 1280u, 720u, 29.97, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_25HZ] =
                { "SMPTE 296M 720p 25 Hz", 1280u, 720u, 25.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_24HZ] =
                { "SMPTE 296M 720p 24 Hz", 1280u, 720u, 24.0, AUX_PROGRESSIVE,
                        FMODE_MAGIC },
        [SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ] =
                { "SMPTE 296M 720p 23.98 Hz", 1280u, 720u, 23.98, AUX_PROGRESSIVE,
                        FMODE_MAGIC }
};

struct vidcap_linsys_state {
        int                 devices_cnt;
        int                 fd[MAX_TILES];
        struct              pollfd pfd[MAX_TILES];
        int                 audio_fd;
        struct              pollfd audio_pfd;
        unsigned long int   bufsize;
        unsigned long int   buffers;
        unsigned long int   audio_bufsize;
        unsigned long int   audio_buffers;
        struct video_frame *frame;
        sem_t               have_item;
        sem_t               boss_waiting;
        pthread_t           grabber;
        
        struct audio_frame  audio;
        unsigned int        audio_channels_preset; // by driver
        unsigned int        audio_bytes_read;
        unsigned int        grab_audio:1; /* wheather we process audio or not */
};

static int          frames = 0;
static struct       timeval t, t0;

static void print_output_modes(void);
static void * vidcap_grab_thread(void *args);

static void
get_carrier (int fd, const char *device)
{
    int val;

    printf ("\tGetting the carrier status :  ");
    if (ioctl (fd, SDIVIDEO_IOC_RXGETCARRIER, &val) < 0) {
        fprintf (stderr, "%s: ", device);
        perror ("unable to get the carrier status");
    } else if (val) {
        printf ("Carrier detected.\n");
    } else {
        printf ("No carrier.\n");
    }
    return;
}


/* 
 * This function should autodetect input format and return it.
 * But the didn't seem worked at time of writing 
 * TODO: revide if the driver is still buggy, if so, consider removing function
 */
/*static const struct frame_mode * get_video_standard (int fd, const char *device)
{
	unsigned int val;

	if (ioctl (fd, SDIVIDEO_IOC_RXGETVIDSTATUS, &val) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("\tunable to get the receive video standard "
                        "detected.");
                return NULL;
	} else {
                if(val < sizeof(frame_modes)/sizeof(struct frame_mode)
                                && frame_modes[val].magic == FMODE_MAGIC) {
                        printf("\t%s video mode detected.\n",
                                        frame_modes[val].name);
                        return &frame_modes[val];
                } else {
                        fprintf(stderr, "Unknown video standard detected!");
                        return NULL;
                }
	}
}*/

struct vidcap_type *
vidcap_linsys_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_LINSYS_ID;
		vt->name        = "linsys";
		vt->description = "HD-SDI Linsys PCIe card";
	}
	return vt;
}

static int open_audio(struct vidcap_linsys_state *s) {
        struct stat buf;
        const char fmt[] = "/sys/class/sdiaudio/sdiaudiorx%i/%s";
        char name[MAXLEN], str[MAXLEN], *endptr;
        int num;
                
        /* Get the sysfs info */
        memset (&buf, 0, sizeof (buf));
        if (stat (audio_dev, &buf) < 0) {
                fprintf (stderr, "%s: ", audio_dev);
                perror ("unable to get the file status");
                return -1;
        }
        if (!S_ISCHR (buf.st_mode)) {
                fprintf (stderr, "%s: not a character device\n", audio_dev);
                return -1;
        }
        if (!(buf.st_rdev & 0x0080)) {
                fprintf (stderr, "%s: not a receiver\n", audio_dev);
                return -1;
        }
        num = buf.st_rdev & 0x007f;
        snprintf (name, sizeof (name), fmt, num, "dev");
        memset (str, 0, sizeof (str));
        if (util_read (name, str, sizeof (str)) < 0) {
                fprintf (stderr, "%s: ", audio_dev);
                perror ("unable to get the device number");
                return -1;
        }
        if (strtoul (str, &endptr, 0) != (buf.st_rdev >> 8)) {
                fprintf (stderr, "%s: not a SMPTE 259M-C device\n", audio_dev);
                return -1;
        }
        if (*endptr != ':') {
                fprintf (stderr, "%s: error reading %s\n",
                        audio_dev, name);
                return -1;
        }

        /* Open the file */
        if ((s->audio_fd = open (audio_dev, O_RDONLY)) < 0) {
                fprintf (stderr, "%s: ", audio_dev);
                perror ("unable to open file for reading");
                return -1;
        }

        /* Get the buffer size */
        snprintf (name, sizeof (name), fmt, num, "bufsize");
        if (util_strtoul (name, &s->audio_bufsize) < 0) {
                fprintf (stderr, "%s: ", audio_dev);
                perror ("unable to get the receiver buffer size");
                return -1;
        }

        unsigned long int channels;
        snprintf (name, sizeof (name), fmt, num, "channels");
        if (util_strtoul (name, &channels) < 0) {
                fprintf (stderr, "%s: ", audio_dev);
                perror ("unable to get the receiver buffer size");
                return -1;
        }

        s->audio_channels_preset = channels;

        /* Allocate some memory */
        /*if ((s->audio.data = (char *)malloc (s->audio_bufsize)) == NULL) {
                fprintf (stderr, "%s: ", audio_dev);
                fprintf (stderr, "unable to allocate memory\n");
                return -1;
        }*/
        
        /* Receive the data and check for errors */
        s->audio_pfd.fd = s->audio_fd;
        s->audio_pfd.events = POLLIN | POLLPRI;
        
        return 0;
}

void *
vidcap_linsys_init(char *init_fmt, unsigned int flags)
{
	struct vidcap_linsys_state *s;

        const struct frame_mode   *frame_mode;
        int                       frame_mode_number;
        int                       i;
        char                      *save_ptr = NULL;
        char                      *fmt_dup, *item;
        int                       devices[4];

	printf("vidcap_linsys_init\n");

        s = (struct vidcap_linsys_state *) malloc(sizeof(struct vidcap_linsys_state));
	if(s == NULL) {
		printf("Unable to allocate linsys state\n");
		return NULL;
	}

        if(!init_fmt || strcmp(init_fmt, "help") == 0) {
                print_output_modes();
                return NULL;
        }
        
        if(flags & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                s->grab_audio = TRUE;
                
                s->audio.bps = LINSYS_AUDIO_BPS;
                s->audio.sample_rate = LINSYS_AUDIO_SAMPLE_RATE;
                s->audio.data = (char *) malloc(LINSYS_AUDIO_BUFSIZE);
                s->audio_bytes_read = 0u;
                if(open_audio(s) != 0) {
                        s->grab_audio = FALSE;
                } else {
                        if(s->audio_channels_preset != audio_capture_channels && audio_capture_channels != 1) {
                                fprintf(stderr, "[Linsys] Unable to grab %d channels. Current value provided by driver is %d.\n"
                                                "Also grabbing 1 channel is possible.\n"
                                                "You can change this value by writing 2,4,6 or 8 to /sys/class/sdiaudio/sdiaudiotx1/bufsize.\n",
                                                audio_capture_channels, s->audio_channels_preset);
                                s->grab_audio = FALSE;
                        } else {
                                s->audio.ch_count = audio_capture_channels;
                        }
                }
        } else {
                s->grab_audio = FALSE;
        }

        fmt_dup = strdup(init_fmt);

        item = strtok_r(fmt_dup, ":", &save_ptr);
        if(item == NULL) {
                fprintf(stderr, "[Linsys] Card index not given.\n");
                return NULL;
        }
        char *devices_str = strdup(item);
        s->devices_cnt = 0;
        char *ptr, *saveptr2 = NULL;
        ptr = strtok_r(devices_str, ",", &saveptr2);
        do {
                devices[s->devices_cnt] = atoi(ptr);
                ++s->devices_cnt;
        } while ((ptr = strtok_r(NULL, ",", &saveptr2)));
        free(devices_str);

        item = strtok_r(NULL, ":", &save_ptr);
        if(item == NULL) {
                fprintf(stderr, "[Linsys] Card index or mode not given.\n");
                return NULL;
        }
	frame_mode_number = atoi(item);
	if(frame_mode_number < 0 || 
                        (unsigned int) frame_mode_number >= 
                        sizeof(frame_modes)/sizeof(struct frame_mode)) {
                return NULL;
        }
        frame_mode = &frame_modes[frame_mode_number];
        if(frame_mode == &frame_modes[SDIVIDEO_CTL_UNLOCKED]) {
                fprintf(stderr, "Please setup correct video mode "
                                "via sysfs.");
                return NULL;
        }

        free(fmt_dup);
        
        s->frame = vf_alloc(s->devices_cnt);

        gettimeofday(&t0, NULL);

	/* CHECK IF LINSYS CAN WORK CORRECTLY */
    
        /*Printing current settings from the sysfs info */

        /* Stat the file, fills the structure with info about the file
        * Get the major number from device node
        */

        memset((void *) s->fd, 0, sizeof(s->fd));
        //memset((void *) s->frame, 0, sizeof(s->frame));

        for(i = 0; i < s->devices_cnt; ++i) {
                char                      dev_name[MAXLEN];
                unsigned int              cap;
                unsigned int              val;
                struct stat               buf;
                char                      name[MAXLEN];
                char                      data[MAXLEN];
                const char                *codec_name;
                int                       codec_index;
                char                      type;
                char                      *endptr;
                int                       num;
                unsigned long int         mode;
                const struct codec_info_t *c_info;
                
                snprintf(dev_name, sizeof(dev_name), devfile_fmt, devices[i]);

                memset (&buf, 0, sizeof (buf));
                if(stat (dev_name, &buf) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the file status");
                        vidcap_linsys_done(s);
                        return NULL;
                }

                /* Check if it is a character device or not */
                if(!S_ISCHR (buf.st_mode)) {
                        fprintf (stderr, "%s: not a character device\n", dev_name);
                        vidcap_linsys_done(s);
                        return NULL;
                }
                if(!(buf.st_rdev & 0x0080)) {
                        fprintf (stderr, "%s: not a receiver\n", dev_name);
                        vidcap_linsys_done(s);
                        return NULL;
                }

                /* Check the minor number to determine if it is a receive or transmit device */
                type = (buf.st_rdev & 0x0080) ? 'r' : 't';

                /* Get the receiver or transmitter number */
                num = buf.st_rdev & 0x007f;

                /* Build the path to sysfs file */
                snprintf (name, sizeof (name), sys_fmt, type, num, "dev");
    
                memset (data, 0,sizeof(data));
                /* Read sysfs file (dev) */
                if (util_read (name,data, sizeof (data)) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the device number");
                        vidcap_linsys_done(s);
                        return NULL;
                }

                /* Compare the major number taken from sysfs file to the one taken from device node */
                if (strtoul (data, &endptr, 0) != (buf.st_rdev >> 8)) {
                        fprintf (stderr, "%s: not a SMPTE 292M/SMPTE 259M-C device\n", dev_name);
                        vidcap_linsys_done(s);
                        return NULL;
                }

                if (*endptr != ':') {
                        fprintf (stderr, "%s: error reading %s\n", dev_name, name);
                        vidcap_linsys_done(s);
                        return NULL;
                }

                snprintf (name, sizeof (name), sys_fmt, type, num, "mode");
                if (util_strtoul (name, &mode) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the pixel mode");
                        vidcap_linsys_done(s);
                        return NULL;
                }

               printf ("\tMode: %lu ", mode);
               switch (mode) {
                   case SDIVIDEO_CTL_MODE_UYVY:
                       printf ("(assume 8-bit uyvy data)\n");
                       codec_name = "UYVY";
                       break;
                   case SDIVIDEO_CTL_MODE_V210:
                       printf ("(assume 10-bit v210 synchronized data)\n");
                       codec_name = "v210";
                       break;
                   case SDIVIDEO_CTL_MODE_V210_DEINTERLACE:
                       printf ("(assume 10-bit v210 deinterlaced data)\n");
                       codec_name = "v210";
                       break;
                   case SDIVIDEO_CTL_MODE_RAW:
                       printf ("(assume raw data)\n");
                       fprintf(stderr, "Raw data not (yet) supported!");
                       return NULL;
                   default:
                       printf ("(unknown)\n");
                       fprintf(stderr, "Unknown colour space not (yet) supported!");
                       vidcap_linsys_done(s);
                       return NULL;
               }

                c_info = NULL;
                for(codec_index = 0; codec_info[codec_index].name != NULL;
                                codec_index++) {
                        if(strcmp(codec_info[codec_index].name, codec_name) == 0) {
                                c_info = &codec_info[codec_index];
                                break;
                        }
                }

                if(c_info == NULL) {
                        fprintf(stderr, "Wrong config. Unknown color space %s\n", codec_name);
                        return NULL;
                }



                snprintf (name, sizeof (name), sys_fmt, type, num, "buffers");
                if (util_strtoul (name, &s->buffers) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the number of buffers");
                        vidcap_linsys_done(s);
                        return NULL;
                }

                snprintf (name, sizeof (name), sys_fmt, type, num, "bufsize");
                if (util_strtoul (name, &s->bufsize) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the buffer size");
                        vidcap_linsys_done(s);
                        return NULL;
                }

                printf ("\t%lux%lu-byte buffers\n", s->buffers, s->bufsize);
                if(s->bufsize != frame_mode->width * frame_mode->height * 
                                c_info->bpp)
                {
                        int needed_size;

                        needed_size = frame_mode->width * frame_mode->height * 
                                c_info->bpp;
                        fprintf (stderr, "%s: ", dev_name);
                        fprintf (stderr, "Buffer size doesn't match frame size.");
                        fprintf (stderr, "Please run 'echo %d > %s' as root.", needed_size, name);
                        vidcap_linsys_done(s);
                        return NULL;
                }

                /* END OF CHECK IF LINSYS CAN WORK CORRECTLY */

    
                /* Open the file */
                if((s->fd[i] = open (dev_name, O_RDONLY,0)) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to open file for reading");
                        vidcap_linsys_done(s);
                        return NULL;
                }
    

                /* Get the receiver capabilities */
                if (ioctl (s->fd[i], SDIVIDEO_IOC_RXGETCAP, &cap) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the receiver capabilities");
                        vidcap_linsys_done(s);
                        return NULL;
                }

                /*Get carrier*/
                if(cap & SDIVIDEO_CAP_RX_CD) {
                        get_carrier (s->fd[i], dev_name);
                } 

    
                if(ioctl (s->fd[i], SDIVIDEO_IOC_RXGETSTATUS, &val) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the receiver status");
                } else {
                        fprintf (stderr, "\tReceiver is ");
                        if (val) {
                                //printf ("passing data.\n");
                                fprintf (stderr, "passing data\n");
                        } else {
                                //printf ("blocking data.\n");
                                fprintf (stderr, "blocking data\n");
                        }
                }

                /*Get video standard*/
                /*frame_mode = get_video_standard (s->fd);*/

                struct tile *tile = vf_get_tile(s->frame, i);
	
                s->frame->color_spec = c_info->codec;
                s->frame->fps = frame_mode->fps;
                switch(frame_mode->aux) {
                        case AUX_PROGRESSIVE:
                                s->frame->interlacing = PROGRESSIVE;
                                break;
                        case AUX_INTERLACED:
                                s->frame->interlacing = INTERLACED_MERGED;
                                break;
                        case AUX_SF:
                                s->frame->interlacing = SEGMENTED_FRAME;
                                break;
                }
                tile->width = frame_mode->width;
                tile->height = frame_mode->height;
                tile->linesize = vc_get_linesize(frame_mode->width, c_info->codec);
                tile->data_len = tile->linesize * tile->height;
                

                if((tile->data = (char *)
                                malloc (tile->data_len)) == NULL) {
                        fprintf (stderr, "%s: ", dev_name);
                        fprintf (stderr, "unable to allocate memory\n");
                        vidcap_linsys_done(s);
                        return NULL;
                }

                /* SET SOME VARIABLES*/
                s->pfd[i].fd = s->fd[i];
                s->pfd[i].events = POLLIN | POLLPRI;
        }

        sem_init(&s->have_item, 0, 0);
        sem_init(&s->boss_waiting, 0, 0);
        pthread_create(&s->grabber, NULL, vidcap_grab_thread, s);

	return s;
}

void
vidcap_linsys_finish(void *state)
{
	struct vidcap_linsys_state *s = (struct vidcap_linsys_state *) state;
	assert(s != NULL);

        should_exit = true;

	pthread_join(s->grabber, NULL);
}

void
vidcap_linsys_done(void *state)
{
	struct vidcap_linsys_state *s = (struct vidcap_linsys_state *) state;

	assert(s != NULL);

	if (s != NULL) {
                int i;
		for (i = 0; i < s->devices_cnt; ++i) {
			if(s->frame->tiles[i].data != NULL)
				free(s->frame->tiles[i].data);
			if(s->fd[i] != 0)
				close(s->fd[i]);
		}
	}
        
        vf_free(s->frame);
	sem_destroy(&s->boss_waiting);
	sem_destroy(&s->have_item);
}

static void * vidcap_grab_thread(void *args)
{
	struct vidcap_linsys_state 	*s = (struct vidcap_linsys_state *) args;
        struct timespec timeout;

        timeout.tv_sec = 0;
        timeout.tv_nsec = 500 * 1000;

        while(!should_exit) {
		unsigned int val;
                int cur_dev;
		int return_vec = 0u;

                if(sem_timedwait(&s->boss_waiting, &timeout) == ETIMEDOUT)
			continue;
                if (should_exit) 
			break;

                //for(cur_dev = 0; cur_dev < s->devices_cnt; ++cur_dev)
		while (return_vec != (1 << s->devices_cnt) - 1)
                {
                        if(poll (s->pfd, s->devices_cnt, 1000/23) < 0) {
                                //fprintf (stderr, "%s: ", device);
                                perror ("unable to poll device file");
                                return NULL;
                        }

			for(cur_dev = 0; cur_dev < s->devices_cnt; ++cur_dev) {
                                struct tile *tile = vf_get_tile(s->frame, cur_dev);
                                
                                
                                if (s->pfd[cur_dev].revents & POLLIN) {
                                        unsigned int ret;
                                        ret = read(s->fd[cur_dev], tile->data, s->bufsize);
                                        assert(ret == s->bufsize);
                                        return_vec |= 1 << cur_dev;
                                }

                                if(s->pfd[cur_dev].revents & POLLPRI) {
                                        if (ioctl (s->fd[cur_dev],SDIVIDEO_IOC_RXGETEVENTS, &val) < 0) {
                                                //fprintf (stderr, "%s: ", device);
                                                perror ("unable to get receiver event flags");
                                                return NULL;
                                        }
                                        if (val & SDIVIDEO_EVENT_RX_BUFFER) {
                                                fprinttime (stderr, "");
                                                fprintf (stderr,
                                                        "driver receive buffer queue "
                                                        "overrun detected\n");
                                        }
                                        if (val &  SDIVIDEO_EVENT_RX_FIFO) {
                                                fprinttime (stderr, "");
                                                fprintf (stderr,
                                                        "onboard receive FIFO "
                                                        "overrun detected\n");
                                        }
                                        if (val & SDIVIDEO_EVENT_RX_CARRIER) {
                                                fprinttime (stderr, "");
                                                fprintf (stderr,
                                                        "carrier status "
                                                        "change detected\n");
                                        }
                                if (val & SDIVIDEO_EVENT_RX_STD) {
                                    fprinttime (stderr, progname);
                                    fprintf (stderr,
                                        "format "
                                        "change detected\n");
                                   }
                                }

                        }
                }

                while (poll (s->pfd, s->devices_cnt, 0) > 0) {
                        for(cur_dev = 0; cur_dev < s->devices_cnt; ++cur_dev) {
                                struct tile *tile = vf_get_tile(s->frame, cur_dev);
                                                
                                if (s->pfd[cur_dev].revents & POLLIN) {
                                        unsigned int ret;
                                        ret = read(s->fd[cur_dev], tile->data, s->bufsize);
                                        assert(ret == s->bufsize);
                                        return_vec |= 1 << cur_dev;
                                }

                                if(s->pfd[cur_dev].revents & POLLPRI) {
                                        if (ioctl (s->fd[cur_dev],SDIVIDEO_IOC_RXGETEVENTS, &val) < 0) {
                                                //fprintf (stderr, "%s: ", device);
                                                perror ("unable to get receiver event flags");
                                                return NULL;
                                        }
                                        if (val & SDIVIDEO_EVENT_RX_BUFFER) {
                                                fprinttime (stderr, "");
                                                fprintf (stderr,
                                                        "driver receive buffer queue "
                                                        "overrun detected\n");
                                        }
                                        if (val &  SDIVIDEO_EVENT_RX_FIFO) {
                                                fprinttime (stderr, "");
                                                fprintf (stderr,
                                                        "onboard receive FIFO "
                                                        "overrun detected\n");
                                        }
                                        if (val & SDIVIDEO_EVENT_RX_CARRIER) {
                                                fprinttime (stderr, "");
                                                fprintf (stderr,
                                                        "carrier status "
                                                        "change detected\n");
                                        }
                                if (val & SDIVIDEO_EVENT_RX_STD) {
                                    fprinttime (stderr, progname);
                                    fprintf (stderr,
                                        "format "
                                        "change detected\n");
                                   }
                                }

                        }
                }

                if(s->grab_audio) {
                        /* read all audio data that are in buffers */
                        s->audio.data_len = 0;
                        while (poll (&s->audio_pfd, 1, 0) > 0) {
                                if(s->audio.data_len + s->audio_bufsize / s->audio_channels_preset * s->audio.ch_count <= LINSYS_AUDIO_BUFSIZE) {
                                        if((int) s->audio_channels_preset == s->audio.ch_count) {
                                                s->audio.data_len += read(s->audio_fd, s->audio.data + s->audio.data_len, s->audio_bufsize);
                                        } else { //we need to demux one mono channel
                                                assert(s->audio.ch_count == 1);

                                                char *tmp = malloc(s->audio_bufsize);
                                                int data_read = read(s->audio_fd, tmp, s->audio_bufsize);
                                                demux_channel(s->audio.data + s->audio.data_len, tmp, s->audio.bps, data_read,
                                                                s->audio_channels_preset, 0);
                                                free(tmp);
                                        }
                                } else {
                                        break; // we have our buffer full
                                }
                        }
                }
                sem_post(&s->have_item);
        }
        return NULL;
}

struct video_frame *
vidcap_linsys_grab(void *state, struct audio_frame **audio)
{

	struct vidcap_linsys_state 	*s = (struct vidcap_linsys_state *) state;

        sem_post(&s->boss_waiting);
        sem_wait(&s->have_item);

        if(s->grab_audio && s->audio.data_len > 0)
                *audio = &s->audio;
        else
                *audio = NULL;

        frames++;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, t0);    
        if (seconds >= 5) {
            float fps  = frames / seconds;
            fprintf(stderr, "[Linsys] %d frames in %g seconds = %g FPS\n", frames, seconds, fps);
            t0 = t;
            frames = 0;
        }  

	return s->frame;
}

static void print_output_modes()
{
        unsigned int i;
        printf("usage: -t linsys:<device(s)>:<mode>\n\twhere mode is one of following.\n");
        printf("\nAvailable devices (inputs):\n");
                
        for(i = 0; i < 16; ++i) {
                char                      dev_name[MAXLEN];
                struct stat               buf;

                snprintf(dev_name, sizeof(dev_name), devfile_fmt, i);

                memset (&buf, 0, sizeof (buf));
                if(stat (dev_name, &buf) < 0) {
                        break;
                } else {
                        struct util_info *info;
                        char type;
                        int num;
                        unsigned long int id;
                        char         name[MAXLEN];

                        /* Check if it is a character device or not */
                        if(!S_ISCHR (buf.st_mode)) {
                                fprintf (stderr, "not a character device\n");
                                continue;
                        }
                        if(!(buf.st_rdev & 0x0080)) {
                                fprintf (stderr, "not a receiver\n");
                                continue;
                        }

                        /* Check the minor number to determine if it is a receive or transmit device */
                        type = (buf.st_rdev & 0x0080) ? 'r' : 't';

                        /* Get the receiver or transmitter number */
                        num = buf.st_rdev & 0x007f;

                        /* Build the path to sysfs file */
                        snprintf (name, sizeof (name), sys_fmt, type, num, "dev");

                        if (util_strtoul (name, &id) < 0) {
                                fprintf (stderr, "unable to get the firmware version");
                                continue;
                        }
                        if ((info = getinfo (id)) == NULL) {
                                printf ("\tUnknown device\n");
                        } else {
                                printf ("\t%d) %s\n", i,
                                        info->name);
                        }
                }
        }

        printf("\nAvailable output modes:\n");
        for(i = 0; i < sizeof(frame_modes)/sizeof(struct frame_mode); ++i) {
                if(frame_modes[i].magic == FMODE_MAGIC)
                        printf("\t%2u: %s\n", i, frame_modes[i].name);
        }
        printf("\nPixel mode is set via sysfs.\n");
}


#endif /* HAVE_LINSYS */
#endif /* HAVE_MACOSX */

