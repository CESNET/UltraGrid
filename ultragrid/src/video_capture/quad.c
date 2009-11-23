#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifndef HAVE_MACOSX
#ifdef HAVE_QUAD		/* From config.h */

#include "debug.h"
#include "video_types.h"
#include "video_capture.h"

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

#include "sdi.h"
#include "master.h"
#include "../../../quad/Examples/util.h"

#define BUFLEN 256

extern int	should_exit;

const char	device[] = "/dev/sdirx0";
const char	fmt[] = "/sys/class/sdi/sdirx%i/%s";

int fd;
ssize_t ret, read_ret, bytes;
struct pollfd pfd;
struct timeval tv;
double lasttime, time_sec, dt;
unsigned int frames, last_frames, timestamp, last_timestamp;
unsigned int val;

struct vidcap_quad_state {
	char 			name[BUFLEN];
	unsigned char*		data;
	unsigned long int 	bufsize;
};

struct vidcap_type *
vidcap_quad_probe(void)
{
	printf("vidcap_quad_probe\n");

	struct vidcap_type*		vt;

	/* CHECK IF QUAD CAN WARK CORRECTLY */

	

	/* END OF CHECK IF QUAD CAN WARK CORRECTLY */

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
vidcap_quad_init(int fps)
{
	printf("vidcap_quad_init\n");

	struct vidcap_quad_state *s;

	struct stat buf;
	int num;
	char str[BUFLEN], *endptr;

	s = (struct vidcap_quad_state *) malloc(sizeof(struct vidcap_quad_state));
	if(s == NULL) {
		printf("Unable to allocate Quad state\n");
		return NULL;
	}

	/* Get the sysfs info */
	memset (&buf, 0, sizeof (buf));
	if(stat (device, &buf) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to get the file status");
		goto NO_STAT;
	}
	if(!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", device);
		goto NO_STAT;
	}
	if(!(buf.st_rdev & 0x0080)) {
		fprintf (stderr, "%s: not a receiver\n", device);
		goto NO_STAT;
	}

	num = buf.st_rdev & 0x007f;
	snprintf (s->name, sizeof (s->name), fmt, num, "dev");

	memset (str, 0, sizeof (str));
	if(util_read (s->name, str, sizeof (str)) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to get the device number");
		goto NO_STAT;
	}
	if(strtoul (str, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not a SMPTE 259M-C device\n", device);
		goto NO_STAT;
	}
	if(*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", device, s->name);
		goto NO_STAT;
	}

	/* Open the file */
	if((fd = open (device, O_RDONLY)) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to open file for reading");
		goto NO_STAT;
	}

	/* Get the buffer size */
	snprintf (s->name, sizeof (s->name), fmt, num, "bufsize");
	if(util_strtoul (s->name, &(s->bufsize)) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to get the receiver buffer size");
		goto NO_BUFS;
	}

	/* Allocate some memory */
	if((s->data = (unsigned char *)malloc (s->bufsize)) == NULL) {
		fprintf (stderr, "%s: ", device);
		fprintf (stderr, "unable to allocate memory\n");
		goto NO_BUFS;
	}

	/* SET SOME VARIABLES*/
	pfd.fd = fd;
	pfd.events = POLLIN | POLLPRI;
	last_frames = 0;
	last_timestamp = 0;

	s->data = NULL;

	if(gettimeofday (&tv, NULL) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to get time");
		return NULL;
	}
	lasttime = tv.tv_sec + (double)tv.tv_usec / 1000000;

	return s;

NO_STAT:
	return NULL;
NO_BUFS:
	close(fd);
	return NULL;
}

void
vidcap_quad_done(void *state)
{
	struct vidcap_quad_state *s = (struct vidcap_quad_state *) state;

	assert(s != NULL);

	if(s!= NULL) {
		free(s->data);
		close(fd);
	}
}

struct video_frame *
vidcap_quad_grab(void *state)
{
	printf("vidcap_quad_grab\n");

	struct vidcap_quad_state 	*s = (struct vidcap_quad_state *) state;
	struct video_frame		*vf;

	/* Receive the data and check for errors */
	
	if(poll (&pfd, 1, 1000) < 0) {
		fprintf (stderr, "%s: ", device);
		perror ("unable to poll device file");
		goto NO_RUN;
	}

	if(pfd.revents & POLLIN) {
		if ((read_ret = read (fd, s->data, s->bufsize)) < 0) {
			fprintf (stderr, "%s: ", device);
			perror ("unable to read from device file");
			goto NO_RUN;
		}
		bytes = 0;

		/* I DON'T NEED TO WRITE DATA SOMEWHERE, I WILL RETURN THEM THROUGH THIS FUNCTION*/
		/*
		while (bytes < read_ret) {
			if ((ret = write (STDOUT_FILENO, s->data + bytes, read_ret - bytes)) < 0) {
				fprintf (stderr, "%s: ", device);
				perror ("unable to write to output");
				goto NO_RUN;
			}
			bytes += ret;
		}
		*/
	}

	if(pfd.revents & POLLPRI) {
		if (ioctl (fd, SDI_IOC_RXGETEVENTS, &val) < 0) {
			fprintf (stderr, "%s: ", device);
			perror ("unable to get receiver event flags");
			goto NO_RUN;
		}
		if (val & SDI_EVENT_RX_BUFFER) {
			fprinttime (stderr, "");
			fprintf (stderr,
				"driver receive buffer queue "
				"overrun detected\n");
		}
		if (val & SDI_EVENT_RX_FIFO) {
			fprinttime (stderr, "");
			fprintf (stderr,
				"onboard receive FIFO "
				"overrun detected\n");
		}
		if (val & SDI_EVENT_RX_CARRIER) {
			fprinttime (stderr, "");
			fprintf (stderr,
				"carrier status "
				"change detected\n");
		}
	}

	gettimeofday (&tv, NULL);
	time_sec = tv.tv_sec + (double)tv.tv_usec / 1000000;
	dt = time_sec - lasttime;
		
	/* Only for HD-SDI, display timestamp and counter */

	if(dt >= 5) {	
		if(ioctl (fd, SDI_IOC_RXGET27COUNT, &frames) < 0) {
			fprintf (stderr, "%s: ", device);
			perror ("unable to get "
				"the counter");
			free (s->data);
			close (fd);
			return NULL;
		}

		if(ioctl (fd, SDI_IOC_RXGETTIMESTAMP, &timestamp) < 0) { 
			fprintf (stderr, "%s: ", device);
			perror ("unable to get "
				"the timestamp");
			free (s->data);
			close (fd);
			return NULL;
		}

		float fps  = (frames - last_frames) / (timestamp - last_timestamp);
		fprintf(stderr, "%d frames in %g seconds = %g FPS\n", (frames - last_frames), (timestamp - last_timestamp), fps);

		last_frames = frames;
		last_timestamp = timestamp;

		if(ioctl(fd, SDI_IOC_RXGETCARRIER, &val) < 0){
			fprintf (stderr, "%s: ", device);
			perror ("unable to get the carrier status");
		}else if (val) {
			fprintf (stderr, "Carrier detected, ");
		}else {
			fprintf (stderr, "No carrier, ");
		}

		if(ioctl (fd, SDI_IOC_RXGETSTATUS, &val) < 0) {
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
			
		lasttime = time_sec;
	}

	if(s->data != NULL) {
		vf = (struct video_frame *) malloc(sizeof(struct video_frame));
		if (vf != NULL) {
			vf->colour_mode	= YUV_422;
			vf->width	= hd_size_x;
			vf->height	= hd_size_y;
			vf->data	= (char*) s->data;
			vf->data_len	= s->bufsize;
			//vf->data_len	= hd_size_x * hd_size_y * hd_color_bpp;
		}

		// testing write of frames into the files
		/*
		char gn[128];
		memset(gn, 0, 128);
		sprintf(gn, "_frames/frame%04d.yuv", s->delegate->get_framecount());
		FILE *g=fopen(gn, "w+");
		fwrite(vf->data, 1, vf->data_len, g);
		fclose(g);
		*/

		return vf;
	}

	return NULL;

NO_RUN:
	return NULL;
}

#endif /* HAVE_QUAD */
#endif /* HAVE_MACOSX */
