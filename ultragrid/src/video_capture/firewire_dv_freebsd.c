/*
 * FILE:   firewire_dv_freebsd.c
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2004 University of Glasgow
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
 *    This product includes software developed by the University of Glasgow.
 * 
 * 4. The name of the University may not be used to endorse or promote 
 *    products derived from this software without specific prior written 
 *    permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 * $Revision: 1.2 $
 * $Date: 2009/11/20 19:38:23 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifdef HAVE_FIREWIRE_DV_FREEBSD

#include <dev/firewire/firewire.h>

#include "debug.h"
#include "video_types.h"
#include "video_capture.h"
#include "video_capture/firewire_dv_freebsd.h"

struct vidcap_dvbsd_state {
	int	fd;	/* File descriptor for the device */
//	int	fps;
};

void *
vidcap_dvbsd_init(struct vidcap_fmt *fmt)
{
	//int fps = atoi(fmt); //FIXME What is fps good for?
	struct vidcap_dvbsd_state 	*s;
	struct fw_isochreq 		 isoreq;
	struct fw_isobufreq		 bufreq;

	s = malloc(sizeof(struct vidcap_dvbsd_state));
	if (s == NULL) {
		return NULL;
	}

//	s->fps = fps;

	s->fd  = open("/dev/fw0.0", O_RDWR);
	if (s->fd < 0) {
		perror("Unable to open /dev/fw0.0");
		free(s);
		return NULL;
	}

	bufreq.rx.nchunk  = 8;
	bufreq.rx.npacket = 256;
	bufreq.rx.psize   = 512;
	bufreq.tx.nchunk  = 0;
	bufreq.tx.npacket = 0;
	bufreq.tx.psize   = 0;
	if (ioctl(s->fd, FW_SSTBUF, &bufreq) < 0) {
		perror("Unable to configure IEEE-1394 capture device");
		close(s->fd);
		free(s);
		return NULL;
	}

	isoreq.ch  = 63;
	isoreq.tag = 1<<6;
	if (ioctl(s->fd, FW_SRSTREAM, &isoreq) < 0) {
		perror("Unable to start IEEE-1394 capture device");
		close(s->fd);
		free(s);
		return NULL;
	}

	return s;
}

void
vidcap_dvbsd_done(void *state)
{
	struct vidcap_dvbsd_state *s = (struct vidcap_dvbsd_state *) state;

	assert(s != NULL);
	close(s->fd);
	free(s);
}

struct video_frame *
vidcap_dvbsd_grab(void *state)
{
	struct vidcap_dvbsd_state *s = (struct vidcap_dvbsd_state *) state;

	assert(s != NULL);

	return NULL;
}

struct vidcap_type *
vidcap_dvbsd_probe(void)
{
        struct vidcap_type      *vt;

        vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id          = VIDCAP_DVBSD_ID;
                vt->name        = "dv";
                vt->description = "IEEE-1394/DV";
                vt->width       = 720;		/* PAL frame size */
                vt->height      = 576;
                vt->colour_mode = YUV_422;
        }
        return vt;
}

#endif /* HAVE_FIREWIRE_DV_FREEBSD */

