/*
 * FILE:   quicktime.c
 * AUTHOR: Colin Perkins <csp@csperkins.org
 *         Alvaro Saurin <saurin@dcs.gla.ac.uk>
 *
 * Copyright (c) 2005-2006 University of Glasgow
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
 * 3. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 * $Revision: 1.1 $
 * $Date: 2007/11/22 12:14:19 $
 *
 */

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "video_types.h"
#include "video_capture.h"
#include "video_capture/quicktime.h"

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>

#define MAGIC_QT_GRABBER	VIDCAP_QUICKTIME_ID

struct qt_grabber_state {
	uint32_t		magic;
	SeqGrabComponent	grabber;
	SGChannel		video_channel;
	Rect			bounds;
	GWorldPtr		gworld;
	ImageSequence		seqID;
};


/*
 * Sequence grabber data procedure
 * The sequence grabber calls the data function whenever any of the grabbers
 * channels write digitized data to the destination movie file.
 *
 * This data function does two things, it first decompresses captured video
 * data into an offscreen GWorld, draws some status information onto the frame then
 * transfers the frame to an onscreen window.
 *
 * For more information refer to Inside Macintosh: QuickTime Components, page 5-120
 * c - the channel component that is writing the digitized data.
 * p - a pointer to the digitized data.
 * len - the number of bytes of digitized data.
 * offset - a pointer to a field that may specify where you are to write the
 * digitized data, and that is to receive a value indicating where you wrote the data.
 * chRefCon - per channel reference constant specified using SGSetChannelRefCon.
 * time    - the starting time of the data, in the channel's time scale.
 * writeType - the type of write operation being performed.
 * 	seqGrabWriteAppend  - Append new data.
 * 	seqGrabWriteReserve - Do not write data. Instead, reserve space for the amount of data
 * 	                      specified in the len parameter.
 * 	seqGrabWriteFill    - Write data into the location specified by offset. Used to
 * 	                      fill the space previously reserved with seqGrabWriteReserve. 
 *	                      The Sequence Grabber may call the DataProc several times to 
 *	                      fill a single reserved location.
 * refCon - the reference constant you specified when you assigned your data
 *          function to the sequence grabber.
 */
static pascal OSErr
qt_data_proc(SGChannel c, Ptr p, long len, long *offset, long chRefCon, TimeValue time, short writeType, long refCon)
{
	ComponentResult		 err		= noErr;
	struct qt_grabber_state	*s		= (struct qt_grabber_state*) refCon;
	CodecFlags		 ignore;
	ImageDescriptionHandle	 imageDesc;

	UNUSED (offset);
	UNUSED (chRefCon);
	UNUSED (time);
	UNUSED (writeType);

	if (s == NULL) {
		debug_msg("corrupt state\n");
		return -1;
	}

	if (c == s->video_channel) {
   		if (s->seqID == 0) {
			debug_msg("DecompressSequenceBegin\n");
			imageDesc = (ImageDescriptionHandle)NewHandle(0);
   			// set up decompression sequence

			// retrieve a channel's current sample description, the channel returns a
			// sample description that is appropriate to the type of data being captured
			err = SGGetChannelSampleDescription(c, (Handle)imageDesc);

			debug_msg("image %dx%d size=%d\n", (*imageDesc)->width, (*imageDesc)->height, (*imageDesc)->dataSize);

			// begin the process of decompressing a sequence of frames
			// this is a set-up call and is only called once for
			// the sequence - the ICM will interrogate different
			// codecs and construct a suitable decompression chain,
			// as this is a time consuming process we don't want to
			// do this once per frame (eg. by using DecompressImage)
			// see http://developer.apple.com/quicktime/icefloe/dispatch008.html
			// the destination is specified as the GWorld

			err = DecompressSequenceBegin(&(s->seqID), imageDesc, s->gworld, NULL, NULL,
							NULL, srcCopy, NULL, 0, codecNormalQuality, bestSpeedCodec);
			if (err != noErr) {
				debug_msg("DecompressSequenceBeginS\n");
				return err;
			}

			DisposeHandle ((Handle)imageDesc);
		}

		// decompress a frame into the window - can queue a frame for
		// async decompression when passed in a completion proc
		err = DecompressSequenceFrameS(s->seqID,
						p,	// pointer to compressed image data
						len,	// size of the buffer
						0,	// in flags
						&ignore,	// out flags
						NULL);		// async completion proc

		if (err != noErr) {
			debug_msg("DecompressSequenceFrameS\n");
			return err;
		}
	}

	return 0;
}

/* Initialize the QuickTime grabber */
static int
qt_open_grabber(struct qt_grabber_state *s)
{
	assert (s        != NULL);
	assert (s->magic == MAGIC_QT_GRABBER);

	/****************************************************************************************/
	/* Step 0: Initialise the QuickTime movie toolbox.                                      */
	InitCursor();
	EnterMovies();

	/****************************************************************************************/
	/* Step 1: Create an off-screen graphics world object, into which we can capture video. */
	/* Lock it into position, to prevent QuickTime from messing with it while capturing.    */
	/* FIXME: maybe should be kYUVUPixelFormat?                                             */
	if (QTNewGWorld(&(s->gworld), k2vuyPixelFormat, &(s->bounds), 0, NULL, 0) != noErr) {
		debug_msg("Unable to create GWorld\n");
		return 0;
	}

	if (!LockPixels(GetPortPixMap(s->gworld))) {
		debug_msg("Unable to lock pixels\n");
		return 0;
	}

	/****************************************************************************************/
	/* Step 2: Open and initialise the default sequence grabber.                            */
	s->grabber = OpenDefaultComponent(SeqGrabComponentType, 0);
	if (s->grabber == 0) {
		debug_msg("Unable to open grabber\n");
		return 0;
	}

	if (SGInitialize(s->grabber) != noErr) {
		debug_msg("Unable to init grabber\n");
		return 0;
	}

	/****************************************************************************************/
	/* Specify the destination data reference for a record operation tell it */
	/* we're not making a movie if the flag seqGrabDontMakeMovie is used,    */
	/* the sequence grabber still calls your data function, but does not     */
	/* write any data to the movie file writeType will always be set to      */
	/* seqGrabWriteAppend                                                    */
        if (SGSetDataRef(s->grabber, 0, 0, seqGrabDontMakeMovie) != noErr) {
		CloseComponent(s->grabber);
		debug_msg("Unable to set data ref\n");
		return 0;
	}

	if (SGSetGWorld(s->grabber, s->gworld, GetMainDevice()) != noErr) {
		debug_msg("Unable to set graphics world\n");
		return 0;
	}

	if (SGNewChannel(s->grabber, VideoMediaType, &s->video_channel) != noErr) {
		debug_msg ("Unable to open video channel\n");
		return 0;
	}

	if (SGSetChannelBounds(s->video_channel, &(s->bounds)) != noErr) {
		debug_msg("Unable to set channel bounds\n");
		return 0;
	}

	if (SGSetChannelUsage(s->video_channel, seqGrabRecord) != noErr) {
		debug_msg("Unable to set channel usage\n");
		return 0;
	}

	/****************************************************************************************/
	/* Step ?: Set the data procedure, which processes the frames as they're captured.      */
	SGSetDataProc(s->grabber, NewSGDataUPP (qt_data_proc), (long)s);

	/****************************************************************************************/
	/* Step ?: Start capturing video...                                                     */
	if (SGPrepare(s->grabber, FALSE, TRUE) != noErr) {
		debug_msg("Unable to prepare capture\n");
		return 0;
	}

        if (SGStartRecord(s->grabber) != noErr) {
		debug_msg("Unable to start recording\n");
		return 0;
	}

	return 1;
}

/*******************************************************************************
 * Public API
 ******************************************************************************/
struct vidcap_type *
vidcap_quicktime_probe (void)
{
	struct vidcap_type	*vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id			= VIDCAP_QUICKTIME_ID;
		vt->name		= "quicktime";
		vt->description		= "QuickTime device";
		vt->width		= hd_size_x;
		vt->height		= hd_size_y;
		vt->colour_mode		= YUV_422;
	}

	return vt;
}

/* Initialize the QuickTime grabbing system */
void *
vidcap_quicktime_init (int fps)
{
	struct qt_grabber_state	*s;

	UNUSED(fps);

	s = (struct qt_grabber_state*) malloc (sizeof (struct qt_grabber_state));
	if (s != NULL) {
		s->magic         = MAGIC_QT_GRABBER;
		s->grabber       = 0;
		s->video_channel = 0;
		s->seqID         = 0;
		s->bounds.top    = 0;
		s->bounds.left   = 0;
		s->bounds.bottom = 575;
		s->bounds.right  = 719;

		qt_open_grabber (s);
	}

	return s;
}

/* Finalize the grabbing system */
void
vidcap_quicktime_done (void *state)
{
	struct qt_grabber_state	*s	= (struct qt_grabber_state*) state;

	assert (s != NULL);

	if (s!= NULL) {
		assert(s->magic != MAGIC_QT_GRABBER);
		SGStop(s->grabber);
		CloseComponent (s->grabber);
		UnlockPixels(GetPortPixMap(s->gworld));
		DisposeGWorld(s->gworld);
		ExitMovies();
		free (s);
	}
}

/* Grab a frame */
struct video_frame *
vidcap_quicktime_grab (void *state)
{
	struct qt_grabber_state	*s	= (struct qt_grabber_state*) state;
	struct video_frame	*vf;

	assert (s	 != NULL);
	assert (s->magic == MAGIC_QT_GRABBER);


	/* Run the QuickTime sequence grabber idle function, which provides */
	/* processor time to out data proc running as a callback.           */
	if (SGIdle(s->grabber) != noErr) {
		debug_msg("Error in SGIDle\n");
		return NULL;
	}

	vf = malloc(sizeof(struct video_frame));
	if (vf != NULL) {
		vf->colour_mode = YUV_422;
		vf->width       = 720;
		vf->height      = 576;
		vf->data        = (unsigned char *) GetPixBaseAddr(GetGWorldPixMap(s->gworld));
		vf->data_len    = 720 * 576 * 2;
	}
	return vf;
}

#endif /* HAVE_MACOSX */

