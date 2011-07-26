/*
 * FILE:    quad.c
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
#ifdef HAVE_QUAD		/* From config.h */

#define FMODE_MAGIC             0x9B7DA07u
#define MAX_TILES               4

#include "video_capture/quad.h"

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
   QUAD SDK includes. We are also using a couple of utility functions from the
   QUAD SDK Examples. That is where util.h comes from.
*/

#include "quad/include/sdivideo.h"
#include "quad/include/master.h"
#include "quad/Examples/util.h"

#define MAXLEN 256

extern int	should_exit;

static const char progname[] = "videocapture";
static const char sys_fmt[] = "/sys/class/sdivideo/sdivideo%cx%i/%s";
static const char devfile_fmt[] = "/dev/sdivideorx%1u";

struct frame_mode {
        char  * const    name;
        unsigned int     width;
        unsigned int     height;
        double           fps;
        int              interlacing; /* AUX_* */
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

struct vidcap_quad_state {
        int                 device_cnt;
        int                 fd[MAX_TILES];
        struct              pollfd pfd[MAX_TILES];
        unsigned long int   bufsize;
        unsigned long int   buffers;
        struct video_frame  frame[MAX_TILES];
        sem_t               have_item;
        sem_t               boss_waiting;
        pthread_t           grabber;
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
static const struct frame_mode * get_video_standard (int fd, const char *device)
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
}

struct vidcap_type *
vidcap_quad_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_QUAD_ID;
		vt->name        = "quad";
		vt->description = "HD-SDI Maste Quad/i PCIe card";
	}
	return vt;
}

void *
vidcap_quad_init(char *init_fmt)
{
	struct vidcap_quad_state *s;

        const struct frame_mode   *frame_mode;
        int                       frame_mode_number;
        int                       i;
        char                      *save_ptr = NULL;
        char                      *fmt_dup, *item;
	int			  tile_x_cnt, tile_y_cnt;

	printf("vidcap_quad_init\n");

        s = (struct vidcap_quad_state *) malloc(sizeof(struct vidcap_quad_state));
	if(s == NULL) {
		printf("Unable to allocate Quad state\n");
		return NULL;
	}

        if(strcmp(init_fmt, "help") == 0) {
                print_output_modes();
                return NULL;
        }

        fmt_dup = strdup(init_fmt);
        item = strtok_r(fmt_dup, ":", &save_ptr);

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
        if((item = strtok_r(NULL, ":", &save_ptr)) != NULL)
        {
		tile_x_cnt = atoi(item);
		item = strtok_r(NULL, ":", &save_ptr);
		if (item == NULL) {
                        fprintf(stderr, "quad: One tile dimension entered, expected zero or 2 values (see -g help)!");
			return NULL;
		}
		tile_y_cnt = atoi(item);
	        s->device_cnt = tile_x_cnt * tile_y_cnt;
        } else {
		tile_x_cnt = 1;
		tile_y_cnt = 1;
                s->device_cnt = 1;
        }

        free(fmt_dup);

	/* CHECK IF QUAD CAN WORK CORRECTLY */
    
        /*Printing current settings from the sysfs info */

        /* Stat the file, fills the structure with info about the file
        * Get the major number from device node
        */

        memset((void *) s->fd, 0, sizeof(s->fd));
        memset((void *) s->frame, 0, sizeof(s->frame));

        for(i = 0; i < s->device_cnt; ++i) {
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

                snprintf(dev_name, sizeof(dev_name), devfile_fmt, i);

                memset (&buf, 0, sizeof (buf));
                if(stat (dev_name, &buf) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the file status");
                        vidcap_quad_done(s);
                        return NULL;
                }

                /* Check if it is a character device or not */
                if(!S_ISCHR (buf.st_mode)) {
                        fprintf (stderr, "%s: not a character device\n", dev_name);
                        vidcap_quad_done(s);
                        return NULL;
                }
                if(!(buf.st_rdev & 0x0080)) {
                        fprintf (stderr, "%s: not a receiver\n", dev_name);
                        vidcap_quad_done(s);
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
                        vidcap_quad_done(s);
                        return NULL;
                }

                /* Compare the major number taken from sysfs file to the one taken from device node */
                if (strtoul (data, &endptr, 0) != (buf.st_rdev >> 8)) {
                        fprintf (stderr, "%s: not a SMPTE 292M/SMPTE 259M-C device\n", dev_name);
                        vidcap_quad_done(s);
                        return NULL;
                }

                if (*endptr != ':') {
                        fprintf (stderr, "%s: error reading %s\n", dev_name, name);
                        vidcap_quad_done(s);
                        return NULL;
                }

                snprintf (name, sizeof (name), sys_fmt, type, num, "mode");
                if (util_strtoul (name, &mode) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the pixel mode");
                        vidcap_quad_done(s);
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
                       vidcap_quad_done(s);
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
                        vidcap_quad_done(s);
                        return NULL;
                }

                snprintf (name, sizeof (name), sys_fmt, type, num, "bufsize");
                if (util_strtoul (name, &s->bufsize) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the buffer size");
                        vidcap_quad_done(s);
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
                        vidcap_quad_done(s);
                        return NULL;
                }

                /* END OF CHECK IF QUAD CAN WORK CORRECTLY */

    
                /* Open the file */
                if((s->fd[i] = open (dev_name, O_RDONLY,0)) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to open file for reading");
                        vidcap_quad_done(s);
                        return NULL;
                }
    

                /* Get the receiver capabilities */
                if (ioctl (s->fd[i], SDIVIDEO_IOC_RXGETCAP, &cap) < 0) {
                        fprintf (stderr, "%s: ", dev_name);
                        perror ("unable to get the receiver capabilities");
                        vidcap_quad_done(s);
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

	
                s->frame[i].color_spec = c_info->codec;
                s->frame[i].width = frame_mode->width;
                s->frame[i].height = frame_mode->height;
                s->frame[i].fps = frame_mode->fps;
                if(c_info->h_align) {
                   s->frame[i].src_linesize = ((s->frame[i].width + c_info->h_align - 1) / c_info->h_align) * 
                        c_info->h_align;
                } else {
                     s->frame[i].src_linesize = s->frame[i].width;
                }
                s->frame[i].src_linesize *= c_info->bpp;
                s->frame[i].data_len = s->frame[i].src_linesize * s->frame[i].height;
                s->frame[i].aux = frame_mode->interlacing;
                if(strcasecmp(c_info->name, "UYVY") == 0)
                        s->frame[i].aux |= AUX_YUV;
                if(strcasecmp(c_info->name, "v210") == 0) {
                        s->frame[i].aux |= AUX_YUV | AUX_10Bit;
                }
                if(s->device_cnt > 1) {
                        s->frame[i].aux |= AUX_TILED;
                        s->frame[i].tile_info.x_count = tile_x_cnt;
                        s->frame[i].tile_info.y_count = tile_y_cnt;
                        s->frame[i].tile_info.pos_x = i % tile_x_cnt;
                        s->frame[i].tile_info.pos_y = i / tile_x_cnt;
                }

                if((s->frame[i].data = (char *) malloc (s->frame[i].data_len)) == NULL) {
                        fprintf (stderr, "%s: ", dev_name);
                        fprintf (stderr, "unable to allocate memory\n");
                        vidcap_quad_done(s);
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
vidcap_quad_done(void *state)
{
	struct vidcap_quad_state *s = (struct vidcap_quad_state *) state;

	assert(s != NULL);

	if (s != NULL) {
                int i;
		for (i = 0; i < s->device_cnt; ++i) {
			if(s->frame[i].data != NULL)
				free(s->frame[i].data);
			if(s->fd[i] != 0)
				close(s->fd[i]);
		}
	}
	pthread_join(s->grabber, NULL);
	sem_destroy(&s->boss_waiting);
	sem_destroy(&s->have_item);
}

static void * vidcap_grab_thread(void *args)
{
	struct vidcap_quad_state 	*s = (struct vidcap_quad_state *) args;
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

                //for(cur_dev = 0; cur_dev < s->device_cnt; ++cur_dev)
		while (return_vec != (1 << s->device_cnt) - 1)
                {
                        if(poll (s->pfd, s->device_cnt, 1000/23) < 0) {
                                //fprintf (stderr, "%s: ", device);
                                perror ("unable to poll device file");
                                return NULL;
                        }

			for(cur_dev = 0; cur_dev < s->device_cnt; ++cur_dev) {
                                if (s->pfd[cur_dev].revents & POLLIN) {
                                        unsigned int ret;
                                        ret = read(s->fd[cur_dev], s->frame[cur_dev].data, s->bufsize);
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

                while (poll (s->pfd, s->device_cnt, 0) > 0) {
                        for(cur_dev = 0; cur_dev < s->device_cnt; ++cur_dev) {
                                if (s->pfd[cur_dev].revents & POLLIN) {
                                        unsigned int ret;
                                        ret = read(s->fd[cur_dev], s->frame[cur_dev].data, s->bufsize);
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
                sem_post(&s->have_item);
        }
        return NULL;
}

struct video_frame *
vidcap_quad_grab(void *state, int *count)
{

	struct vidcap_quad_state 	*s = (struct vidcap_quad_state *) state;

        sem_post(&s->boss_waiting);
        sem_wait(&s->have_item);

        frames++;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, t0);    
        if (seconds >= 5) {
            float fps  = frames / seconds;
            fprintf(stderr, "%d frames in %g seconds = %g FPS\n", frames, seconds, fps);
            t0 = t;
            frames = 0;
        }  

        *count = s->device_cnt;
	return &s->frame[0];
}

static void print_output_modes()
{
        unsigned int i;
        printf("usage: -g <mode>[<x_tiles_count>:<y_tiles_count>\n\twhere mode is one of following.\n");
        printf("Available output modes:\n");
        for(i = 0; i < sizeof(frame_modes)/sizeof(struct frame_mode); ++i) {
                if(frame_modes[i].magic == FMODE_MAGIC)
                        printf("\t%2u: %s\n", i, frame_modes[i].name);
        }
}


#endif /* HAVE_QUAD */
#endif /* HAVE_MACOSX */
