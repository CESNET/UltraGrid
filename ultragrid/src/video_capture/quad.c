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
#include "video_types.h"
#include "video_capture.h"

#include "tv.h"

#ifndef HAVE_MACOSX
#ifdef HAVE_QUAD		/* From config.h */

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

/* 
   QUAD SDK includes. We are also using a couple of utility functions from the
   QUAD SDK Examples. That is where util.h comes from.
*/

#include "sdivideo.h"
#include "master.h"
#include "util.h"

#define MAXLEN 256

extern int	should_exit;

static const char progname[] = "videocapture";
const char fmt[] = "/sys/class/sdivideo/sdivideo%cx%i/%s";
const char  device[] = "/dev/sdivideorx0";

struct vidcap_quad_state {
    int                 fd;
    struct              pollfd pfd;
	unsigned char*		data;
	unsigned long int 	bufsize;
};

int frames = 0;
struct timeval t, t0;


static void
get_carrier (int fd)
{
    int val;

    printf ("Getting the carrier status :  ");
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


struct vidcap_type *
vidcap_quad_probe(void)
{
	printf("vidcap_quad_probe\n");

	struct vidcap_type*		vt;

	/* CHECK IF QUAD CAN WORK CORRECTLY */

	

	/* END OF CHECK IF QUAD CAN WORK CORRECTLY */

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_QUAD_ID;
		vt->name        = "quad";
		vt->description = "HD-SDI Maste Quad/i PCIe card";
		vt->width       = hd_size_x;
		vt->height      = hd_size_y;
		vt->colour_mode = YUV_422;
	}
	return vt;
}

void *
vidcap_quad_init(void)
{
	struct vidcap_quad_state *s;

	struct stat        buf;
	char               name[MAXLEN];
    char               data[MAXLEN];
    char               type;
    char               *endptr;
	int                num;
    unsigned long int  mode;
    unsigned long int  buffers;
    unsigned int       cap;
    unsigned int       val;

    s = (struct vidcap_quad_state *) malloc(sizeof(struct vidcap_quad_state));
	if(s == NULL) {
		printf("Unable to allocate Quad state\n");
		return NULL;
	}


    /*Printing current settings from the sysfs info */

    /* Stat the file, fills the structure with info about the file
    * Get the major number from device node
    */
	memset (&buf, 0, sizeof (buf));
    if(stat (device, &buf) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to get the file status");
		goto NO_STAT;
	}

    /* Check if it is a character device or not */
    if(!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", device);
		goto NO_STAT;
	}
	if(!(buf.st_rdev & 0x0080)) {
		fprintf (stderr, "%s: not a receiver\n", device);
		goto NO_STAT;
	}

    /* Check the minor number to determine if it is a receive or transmit device */
    type = (buf.st_rdev & 0x0080) ? 'r' : 't';

    /* Get the receiver or transmitter number */
	num = buf.st_rdev & 0x007f;

   /* Build the path to sysfs file */
   snprintf (name, sizeof (name), fmt, type, num, "dev");
    
   memset (data, 0,sizeof(data));
   /* Read sysfs file (dev) */
   if (util_read (name,data, sizeof (data)) < 0) {
        fprintf (stderr, "%s: ", device);
        perror ("unable to get the device number");
		goto NO_STAT;
    }

   /* Compare the major number taken from sysfs file to the one taken from device node */
    if (strtoul (data, &endptr, 0) != (buf.st_rdev >> 8)) {
        fprintf (stderr, "%s: not a SMPTE 292M/SMPTE 259M-C device\n", device);
		goto NO_STAT;
    }

    if (*endptr != ':') {
        fprintf (stderr, "%s: error reading %s\n", device, name);
	    goto NO_STAT;
    }
    
    snprintf (name, sizeof (name),fmt, type, num, "buffers");
    if (util_strtoul (name, &buffers) < 0) {
        fprintf (stderr, "%s: ", device);
        perror ("unable to get the number of buffers");
	    goto NO_STAT;
    }

    snprintf (name, sizeof (name),fmt, type, num, "bufsize");
    if (util_strtoul (name, &(s->bufsize)) < 0) {
        fprintf (stderr, "%s: ", device);
        perror ("unable to get the buffer size");
	    goto NO_STAT;
    }
    printf ("\t%lu x %lu-byte buffers\n", buffers, s->bufsize);
    
    snprintf (name, sizeof (name),fmt, type, num, "mode");
    if (util_strtoul (name, &mode) < 0) {
        fprintf (stderr, "%s: ", device);
        perror ("unable to get the pixel mode");
	    goto NO_STAT;
    }

    printf ("\tMode: %lu ", mode);
    switch (mode) {
        case SDIVIDEO_CTL_MODE_UYVY:
            printf ("(assume 8-bit uyvy data)\n");
            break;
        case SDIVIDEO_CTL_MODE_V210:
            printf ("(assume 10-bit v210 synchronized data)\n");
            break;
        case SDIVIDEO_CTL_MODE_V210_DEINTERLACE:
            printf ("(assume 10-bit v210 deinterlaced data)\n");
            break;
        case SDIVIDEO_CTL_MODE_RAW:
            printf ("(assume raw data)\n");
            break;
        default:
            printf ("(unknown)\n");
            break;
    }

    /* Open the file */
	if((s->fd = open (device, O_RDONLY)) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to open file for reading");
		goto NO_STAT;
	}

    /* Get the receiver capabilities */
    if (ioctl (s->fd, SDIVIDEO_IOC_RXGETCAP, &cap) < 0) {
        fprintf (stderr, "%s: ", device);
        perror ("unable to get the receiver capabilities");
        close (s->fd);
		goto NO_STAT;
    }

    if(cap & SDIVIDEO_CAP_RX_CD) {
        get_carrier (s->fd);
    } 

    if(ioctl (s->fd, SDIVIDEO_IOC_RXGETSTATUS, &val) < 0) {
                fprintf (stderr, "%s: ", device);
                perror ("unable to get the receiver status");
        }else {
        fprintf (stderr, "Receiver is ");
        if (val) {
            //printf ("passing data.\n");
            fprintf (stderr, "passing data\n");
        }else {
            //printf ("blocking data.\n");
            fprintf (stderr, "blocking data\n");
        }
    }


	/* Allocate some memory */
	if((s->data = (unsigned char *)malloc (s->bufsize)) == NULL) {
		fprintf (stderr, "%s: ", device);
		fprintf (stderr, "unable to allocate memory\n");
		goto NO_BUFS;
	}

    
    /* SET SOME VARIABLES*/
	s->pfd.fd = s->fd;
	s->pfd.events = POLLIN | POLLPRI;

    hd_size_x=1920;
    hd_size_y=1080;
    hd_color_bpp=2;

	return s;

NO_STAT:
	return NULL;
NO_BUFS:
	close(s->fd);
	return NULL;
}

void
vidcap_quad_done(void *state)
{
	struct vidcap_quad_state *s = (struct vidcap_quad_state *) state;

	assert(s != NULL);

	if(s!= NULL) {
		free(s->data);
		close(s->fd);
	}
}

struct video_frame *
vidcap_quad_grab(void *state)
{

	struct vidcap_quad_state 	*s = (struct vidcap_quad_state *) state;
	struct video_frame		    *vf;

    unsigned int val;
    ssize_t      read_ret;
    ssize_t      bytes;
    
    /* Receive the data and check for errors */
	
	if(poll (&(s->pfd), 1, 1000) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to poll device file");
		goto NO_RUN;
	}

	if(s->pfd.revents & POLLIN) {
		if ((read_ret = read (s->fd, s->data, s->bufsize)) < 0) {
			fprintf (stderr, "%s: ", device);
			perror ("unable to read from device file");
			goto NO_RUN;
		}
		bytes = 0;

	}

	if(s->pfd.revents & POLLPRI) {
		if (ioctl (s->fd,SDIVIDEO_IOC_RXGETEVENTS, &val) < 0) {
			fprintf (stderr, "%s: ", device);
			perror ("unable to get receiver event flags");
			goto NO_RUN;
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

	if(s->data != NULL) {
		vf = (struct video_frame *) malloc(sizeof(struct video_frame));
		if (vf != NULL) {
			vf->colour_mode	= YUV_422;
			vf->width	    = hd_size_x;
			vf->height	    = hd_size_y;
			vf->data	    = (char*) s->data;
			vf->data_len	= s->bufsize;
		}

        frames++;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, t0);    
        if (seconds >= 5) {
            float fps  = frames / seconds;
            fprintf(stderr, "%d frames in %g seconds = %g FPS\n", frames, seconds, fps);
            t0 = t;
            frames = 0;
        }  

		return vf;
	}

	return NULL;

NO_RUN:
	return NULL;
}


#endif /* HAVE_QUAD */
#endif /* HAVE_MACOSX */
