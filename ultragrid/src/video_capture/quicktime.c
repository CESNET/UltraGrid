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
 * $Revision: 1.14 $
 * $Date: 2009/12/02 10:37:30 $
 *
 */

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "tv.h"
#include "video_types.h"
#include "video_capture.h"
#include "video_capture/quicktime.h"
#include "v_codec.h"
#include <math.h>

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>
#include <QuickTime/QuickTimeComponents.h>

#define MAGIC_QT_GRABBER	VIDCAP_QUICKTIME_ID

void nprintf(char *str);
intqt_open_grabber(struct qt_grabber_state *s, char *fmt);

struct qt_grabber_state {
    	uint32_t		magic;
    	SeqGrabComponent	grabber;
    	SGChannel		video_channel;
    	Rect			bounds;
    	GWorldPtr		gworld;
    	ImageSequence		seqID;
    	int                     sg_idle_enough;
        int                     major;
        int                     minor;
        int                     width;
        int                     height;
        int                     codec;
        struct codec_info_t     *c_info;
        unsigned                gui:1;
};

int frames = 0;
struct timeval t, t0;

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

static int linesize=0;

static pascal OSErr
qt_data_proc(SGChannel c, Ptr p, long len, long *offset, long chRefCon, TimeValue time, short writeType, long refCon)
{
	/*
	ComponentResult		 err		= noErr;
	*/
	struct qt_grabber_state	*s		= (struct qt_grabber_state*) refCon;
	FILE *out;
	/*
	CodecFlags		 ignore;
	ImageDescriptionHandle	 imageDesc;
	*/

	ImageDescriptionHandle   imageDesc;
	UNUSED (c);
	UNUSED (offset);
	UNUSED (chRefCon);
	UNUSED (time);
	UNUSED (writeType);

	if (s == NULL) {
		debug_msg("corrupt state\n");
		return -1;
	}

	/*
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
							NULL, srcCopy, NULL, 0, codecLowQuality, bestSpeedCodec);
			if (err != noErr) {
				debug_msg("DecompressSequenceBeginS\n");
				return err;
			}

			DisposeHandle ((Handle)imageDesc);
		}

		// decompress a frame into the window - can queue a frame for
		// async decompression when passed in a completion proc
		err = DecompressSequenceFrameWhen(s->seqID,
						  p,		// pointer to compressed image data
						  len,		// size of the buffer
						  0,		// in flags
						  &ignore,	// out flags
						  NULL,		// async completion proc
						  nil);		// decompress the frame imediatelly

		if (err != noErr) {
			debug_msg("DecompressSequenceFrameS\n");
			return err;
		}
	}
	*/

	/* Low latency processing */

/*	if(linesize == 0) {
		imageDesc = (ImageDescriptionHandle)NewHandle(0);
		SGGetChannelSampleDescription(c, (Handle)imageDesc);
	    printf("image %dx%d size=%d\n", (*imageDesc)->width, (*imageDesc)->height, (*imageDesc)->dataSize);
		linesize = (*imageDesc)->dataSize / (*imageDesc)->height;
		DisposeHandle ((Handle)imageDesc);
	}
	
	char *dst = GetPixBaseAddr(GetGWorldPixMap(s->gworld));
	int i;
	for(i=0; i < hd_size_y; i++) {
		memcpy(dst + hd_size_x*8*i/3, p+linesize*i, hd_size_x*8/3);
	}
*/
/*	out = fopen("/tmp/dump", "w");
	fwrite(p, len, 1, out);
	fclose(out);*/

	memcpy(GetPixBaseAddr(GetGWorldPixMap(s->gworld)), p, len);
	s->sg_idle_enough = 1;

	frames++;
	gettimeofday(&t, NULL);
	double seconds = tv_diff(t, t0);	
	if (seconds >= 5) {
		float fps  = frames / seconds;
		fprintf(stderr, "%d frames in %g seconds = %g FPS\n", frames, seconds, fps);
		t0 = t;
		frames = 0;
	}

	return 0;
}

static Boolean
SeqGrabberModalFilterProc (DialogPtr theDialog, const EventRecord *theEvent,
	short *itemHit, long refCon)
{
	UNUSED(theDialog);
	UNUSED(itemHit);

	// Ordinarily, if we had multiple windows we cared about, we'd handle
	// updating them in here, but since we don't, we'll just clear out
	// any update events meant for us

	Boolean	handled = false;

	if ((theEvent->what == updateEvt) && 
		((WindowPtr) theEvent->message == (WindowPtr) refCon))
	{
		BeginUpdate ((WindowPtr) refCon);
		EndUpdate ((WindowPtr) refCon);
		handled = true;
	}
	return (handled);
}


//  SGSettingsDialog with the "Compression" panel removed
OSErr MinimalSGSettingsDialog(SeqGrabComponent seqGrab, SGChannel sgchanVideo, WindowPtr gMonitor)
{
    OSErr err;
    Component *panelListPtr = NULL;
    UInt8 numberOfPanels = 0;

    ComponentDescription cd = {SeqGrabPanelType, VideoMediaType, 0, 0, 0 };
    Component c = 0;
    Component *cPtr = NULL;

    numberOfPanels = CountComponents(&cd);
    panelListPtr = (Component *)NewPtr(sizeof(Component) * (numberOfPanels + 1));

    cPtr = panelListPtr;
    numberOfPanels = 0;
    CFStringRef compressionCFSTR = CFSTR("Compression");
    do {
        ComponentDescription compInfo;
        c = FindNextComponent(c, &cd);
        if (c) {
            Handle hName = NewHandle(0);
            GetComponentInfo(c, &compInfo, hName, NULL, NULL);
            CFStringRef nameCFSTR = CFStringCreateWithPascalString(kCFAllocatorDefault, (unsigned char *)(*hName), kCFStringEncodingASCII);
            if (CFStringCompare(nameCFSTR, compressionCFSTR, kCFCompareCaseInsensitive) !=  kCFCompareEqualTo) {
              *cPtr++ = c;
              numberOfPanels++;
            }
            DisposeHandle(hName);
        }
    } while (c);

    if ((err = SGSettingsDialog(seqGrab, sgchanVideo, numberOfPanels, panelListPtr, seqGrabSettingsPreviewOnly, (SGModalFilterUPP)NewSGModalFilterUPP(SeqGrabberModalFilterProc), (long)gMonitor))) {
        return err;
    }
}

void
nprintf(char *str) 
{
	char tmp[((int)str[0])+1];

	strncpy(tmp, &str[1], str[0]);
	tmp[(int)str[0]]=0;
	fprintf(stdout, "%s", tmp);
}

/* Initialize the QuickTime grabber */
int
qt_open_grabber(struct qt_grabber_state *s, char *fmt)
{
	GrafPtr 		savedPort;
	WindowPtr		gMonitor;
	//SGModalFilterUPP	seqGrabModalFilterUPP;

	assert (s        != NULL);
	assert (s->magic == MAGIC_QT_GRABBER);

	/****************************************************************************************/
	/* Step 0: Initialise the QuickTime movie toolbox.                                      */
	InitCursor();
	EnterMovies();

	/****************************************************************************************/
	/* Step 1: Create an off-screen graphics world object, into which we can capture video. */
	/* Lock it into position, to prevent QuickTime from messing with it while capturing.    */
	OSType pixelFormat;
	if (bitdepth == 10) {
		pixelFormat = FOUR_CHAR_CODE('v210');
	} else {
		pixelFormat = k2vuyPixelFormat;
	}

	pixelFormat = FOUR_CHAR_CODE('BGRA');
	/****************************************************************************************/
	/* Step 2: Open and initialise the default sequence grabber.                            */
	s->grabber = OpenDefaultComponent(SeqGrabComponentType, 0);
	if (s->grabber == 0) {
		debug_msg("Unable to open grabber\n");
		return 0;
	}

	gMonitor = GetDialogWindow(GetNewDialog(1000, NULL, (WindowPtr)-1L));

	GetPort(&savedPort);
	SetPort(gMonitor);

	if (SGInitialize(s->grabber) != noErr) {
		debug_msg("Unable to init grabber\n");
		return 0;
	}

	SGSetGWorld(s->grabber, GetDialogPort(gMonitor), NULL);

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

	if (SGSetGWorld(s->grabber, NULL, NULL) != noErr) {
		debug_msg("Unable to get gworld from grabber\n");
		return 0;
	}

	if (SGNewChannel(s->grabber, VideoMediaType, &s->video_channel) != noErr) {
		debug_msg("Unable to open video channel\n");
		return 0;
	}

	/* Print available devices */
	int i;
	int j;
	SGDeviceInputList inputList;
	SGDeviceList deviceList;
	if(strcmp(fmt, "help")==0) {
		if (SGGetChannelDeviceList(s->video_channel, sgDeviceListIncludeInputs, &deviceList) == noErr) {
			fprintf(stdout, "Available capture devices:\n");
			for(i = 0; i < (*deviceList)->count; i++) {
				SGDeviceName *deviceEntry = &(*deviceList)->entry[i];
				fprintf(stdout, " Device %d: ",i );
				nprintf((char*)(deviceEntry->name));
				if (deviceEntry->flags & sgDeviceNameFlagDeviceUnavailable){
					fprintf(stdout, "  - ### NOT AVAILABLE ###");
				}
				if (i == (*deviceList)->selectedIndex){
					fprintf(stdout, " - ### ACTIVE ###");
				}
				fprintf(stdout, "\n");
				short activeInputIndex = 0;
				inputList = deviceEntry->inputs;
				if (inputList && (*inputList)->count >= 1) {
					SGGetChannelDeviceAndInputNames(s->video_channel, NULL, NULL, &activeInputIndex);
					for (j = 0; j < (*inputList)->count; j++){
						fprintf(stdout, "\t");
						fprintf(stdout, "- %d. ", j);
						nprintf((char*)((&(*inputList)->entry[j].name)));
    					        if ((i == (*deviceList)->selectedIndex)&&(j == activeInputIndex))
							fprintf(stdout, " - ### ACTIVE ###");
						fprintf(stdout, "\n");
					}
				}		
			}
			SGDisposeDeviceList(s->grabber, deviceList);
			CodecNameSpecListPtr list;
		    GetCodecNameList(&list, 1);
			printf("Compression types:\n");
			for(i=0; i < list->count; i++) {
				int fcc = list->list[i].cType;
				printf("\t%d) ", i);
				nprintf((char*)list->list[i].typeName);
				printf(" - FCC (%c%c%c%c)",
						fcc >> 24,
						(fcc >> 16)&0xff,
						(fcc >> 8)&0xff,
						(fcc)&0xff);
				printf(" - codec id %x", (unsigned int)(list->list[i].codec));
				printf(" - cType %x", (unsigned int)list->list[i].cType);
				printf("\n");
			}
		}
		return 0;
	}


	if (SGSetChannelUsage(s->video_channel, seqGrabRecord | seqGrabPreview | seqGrabAlwaysUseTimeBase | seqGrabLowLatencyCapture) != noErr) {
		debug_msg("Unable to set channel usage\n");
		return 0;
	}

	if (SGSetChannelPlayFlags(s->video_channel, channelPlayAllData) != noErr) {
		debug_msg("Unable to set channel flags\n");
		return 0;
	}

//	SGPause(s->grabber, true);

	SGStartPreview(s->grabber);

//	if (s->video_channel) { //IMHO, this should be always true, otherwise previous calls would failed
        if (strcmp(fmt, "gui") == 0) { //Use gui to select input
//		seqGrabModalFilterUPP = (SGModalFilterUPP)NewSGModalFilterUPP(SeqGrabberModalFilterProc);
//		SGSettingsDialog(s->grabber, s->video_channel, 0, nil, 0L, (SGModalFilterUPP)NewSGModalFilterUPP(SeqGrabberModalFilterProc), (long)(gMonitor));
//		DisposeSGModalFilterUPP(seqGrabModalFilterUPP);
		MinimalSGSettingsDialog(s->grabber, s->video_channel, gMonitor);
	} else { // Use input specified on cmd
                /* Select device */
		if (SGGetChannelDeviceList(s->video_channel, sgDeviceListIncludeInputs, &deviceList) != noErr) {
                        debug_msg("Unable to get list of quicktime devices\n");
                        return 0;
                }

                char *tmp;

                tmp = strtok(fmt, ":");
                if(!tmp) {
                    fprintf(stderr, "Wrong config %s\n", fmt);
                    return 0;
                }
                s->major = atoi(tmp);
                tmp = strtok(NULL, ":");
                if(!tmp) {
                    fprintf(stderr, "Wrong config %s\n", fmt);
                    return 0;
                }
                s->minor = atoi(tmp);
                tmp = strtok(NULL, ":");
                if(!tmp) {
                    fprintf(stderr, "Wrong config %s\n", fmt);
                    return 0;
                }
                s->codec = atoi(tmp);

                SGDeviceName *deviceEntry = &(*deviceList)->entry[s->major];
                printf("Quicktime: Setting device: ");
                nprintf((char*)deviceEntry->name);printf("\n");
                if (SGSetChannelDevice (s->video_channel, deviceEntry->name) != noErr) {
                    debug_msg("Setting up the selected device failed\n");			
                    return 0;
                }
        
                /* Select input */
                inputList = deviceEntry->inputs;
                printf("Quicktime: Setting input: ");
                nprintf((char*)(&(*inputList)->entry[s->minor].name)); printf("\n");
                if (SGSetChannelDeviceInput (s->video_channel, s->minor) != noErr) {
                    debug_msg("Setting up input on selected device failed\n");			
                    return 0;
                }
        }

	/* Set video size according to selected vide format */
	Rect gActiveVideoRect;
	SGGetSrcVideoBounds (s->video_channel, &gActiveVideoRect);

	hd_size_x = s->bounds.right = gActiveVideoRect.right - gActiveVideoRect.left;
	hd_size_y = s->bounds.bottom = gActiveVideoRect.bottom - gActiveVideoRect.top;

	printf("Quicktime: Video size: %dx%d\n", s->bounds.right, s->bounds.bottom);

	if (SGSetChannelBounds(s->video_channel, &(s->bounds)) != noErr) {
		debug_msg("Unable to set channel bounds\n");
		return 0;
	}

	/* Set selected fmt->codec and get pixel format of that codec */
	int pixfmt;
	if(s->codec > 0) {
	        CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
		pixfmt = list->list[s->codec].cType;
		printf("Quicktime: SetCompression: %d\n", 
                      (int)SGSetVideoCompressor(s->video_channel, 0, list->list[s->codec].codec, 0, 0, 0));
	} else {
		int codec;
		SGGetVideoCompressor(s->video_channel, NULL, &codec, NULL, NULL, NULL);
		CodecNameSpecListPtr list;
		GetCodecNameList(&list, 1);
		for(i=0; i < list->count; i++) {
			if ((unsigned)codec == list->list[i].codec) {
				pixfmt = list->list[i].cType;
				break;
			}
		}
	}

        for(i = 0; codec_info[i].name != NULL; i++) {
                if((unsigned)pixfmt == codec_info[i].fcc) {
                        s->c_info = &codec_info[i];
                }
        }

	printf("Quicktime: Selected pixel format: %c%c%c%c\n",
		pixfmt >> 24, (pixfmt >> 16)&0xff, (pixfmt >> 8)&0xff, (pixfmt)&0xff);
        int h_align = s->c_info->h_align;
	if (h_align) {
		hd_size_x = ((hd_size_x + h_align - 1) / h_align) * h_align;
		printf("Quicktime: Pixel format 'v210' was selected -> Setting hd_size_x to %d\n", 
                                hd_size_x);
	} 
        
        hd_color_bpp = ceil(s->c_info->bpp);
	
//	SGUpdate(s->grabber, 0);	

	SetPort(savedPort);

	if (QTNewGWorld(&(s->gworld), pixelFormat, &(s->bounds), 0, NULL, 0) != noErr) {
		debug_msg("Unable to create GWorld\n");
		return 0;
	}

	if (!LockPixels(GetPortPixMap(s->gworld))) {
		debug_msg("Unable to lock pixels\n");
		return 0;
	}

	/*if (SGSetGWorld(s->grabber, s->gworld, GetMainDevice()) != noErr) {
		debug_msg("Unable to set graphics world\n");
		return 0;
	}*/

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

//	SGUpdate(s->grabber, 0);	

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
vidcap_quicktime_init (char *fmt)
{
	struct qt_grabber_state	*s;

	s = (struct qt_grabber_state*) malloc (sizeof (struct qt_grabber_state));
	if (s != NULL) {
		s->magic         = MAGIC_QT_GRABBER;
		s->grabber       = 0;
		s->video_channel = 0;
		s->seqID         = 0;
		s->bounds.top    = 0;
		s->bounds.left   = 0;
		s->bounds.bottom = hd_size_y;
		s->bounds.right  = hd_size_x;
		s->sg_idle_enough = 0;

		if(qt_open_grabber (s, fmt)==0) {
			free(s);
			return NULL;
		}
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
		UnlockPixels(GetPortPixMap(s->gworld));
		CloseComponent(s->grabber);
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

	/* The while loop done in this way is also sort of nice bussy waiting */
	/* and synchronizes capturing and sending.			      */
	s->sg_idle_enough = 0;
	while(!s->sg_idle_enough) {
		if (SGIdle(s->grabber) != noErr) {
			debug_msg("Error in SGIDle\n");
			return NULL;
		}
	}

	vf = malloc(sizeof(struct video_frame));
	if (vf != NULL) {
		vf->colour_mode = YUV_422;
		vf->width       = hd_size_x;
		vf->height      = hd_size_y;
		vf->data        = (char *) GetPixBaseAddr(GetGWorldPixMap(s->gworld));
		vf->data_len    = hd_size_x * hd_size_y * hd_color_bpp;
	}
	return vf;
}

#endif /* HAVE_MACOSX */

