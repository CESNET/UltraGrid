/*
 * FILE:    video_display/quicktime.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "host.h"
#include "tv.h"
#include "video_codec.h"

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>

#include "compat/platform_semaphore.h"
#include <signal.h>
#include <pthread.h>
#include <assert.h>

#include "video_display.h"
#include "video_display/quicktime.h"

#define MAGIC_QT_DISPLAY 	DISPLAY_QUICKTIME_ID


struct state_quicktime {
    ComponentInstance	videoDisplayComponentInstance;
//    Component			videoDisplayComponent;
    GWorldPtr			gworld;
    ImageSequence		seqID;

    char			*buffers[2];
    char			*outBuffer;
    int				image_display, image_network;

    int     device;
    int     mode;
    char   *codec;
    const struct codec_info_t *cinfo;
    int     width;
    int     height;

        /* Thread related information follows... */
    pthread_t               thread_id;
    sem_t                   semaphore;

    uint32_t		magic;
};

/* Prototyping */
char * four_char_decode(int format);
void nprintf(char *str);


char * four_char_decode(int format)
{
	static char	fbuf0[32];
	static char	fbuf1[32];
	static int	count = 0;
	char		*fbuf;

	if (count & 1)
		fbuf = fbuf1;
	else
		fbuf = fbuf0;
	count++;

	if ((unsigned)format < 64) 
		sprintf(fbuf, "%d", format);
	else {
		fbuf[0] = (char)(format >> 24); 
		fbuf[1] = (char)(format >> 16); 
		fbuf[2] = (char)(format >> 8); 
		fbuf[3] = (char)(format >> 0); 
	}
	return fbuf;
}

static void*
display_thread_quicktime(void *arg)
{
	struct state_quicktime	*s = (struct state_quicktime *) arg;

	ImageDescriptionHandle	imageDesc;
	CodecFlags		ignore;
	int			ret;

	char			*line1, *line2;

	int			frames = 0;
	struct			timeval t, t0;

	
	imageDesc = (ImageDescriptionHandle)NewHandle(sizeof(ImageDescription));
	
	platform_sem_wait(&s->semaphore);

	(**(ImageDescriptionHandle)imageDesc).idSize = sizeof(ImageDescription);
    (**(ImageDescriptionHandle)imageDesc).cType = s->cinfo->fcc;
    (**(ImageDescriptionHandle)imageDesc).dataSize = hd_size_x * hd_size_y * s->cinfo->bpp; // dataSize is specified in bytes and is specified as height*width*bytes_per_luma_instant. v210 sets bytes_per_luma_instant to 8/3. See http://developer.apple.com/quicktime/icefloe/dispatch019.html#v210
	//(**(ImageDescriptionHandle)imageDesc).cType = '2Vuy'; // QuickTime specifies '2vuy' codec, however Kona3 reports it as '2Vuy'
	//(**(ImageDescriptionHandle)imageDesc).hRes = 72; // not used actually. Set to 72. See http://developer.apple.com/quicktime/icefloe/dispatch019.html#imagedesc
	//(**(ImageDescriptionHandle)imageDesc).vRes = 72; // not used actually. Set to 72. See http://developer.apple.com/quicktime/icefloe/dispatch019.html#imagedesc
	(**(ImageDescriptionHandle)imageDesc).width = s->width; // Beware: must be a multiple of horiz_align_pixels which is 2 for 2Vuy and 48 for v210. hd_size_x=1920 is a multiple of both. TODO: needs further investigation for 2K!
	(**(ImageDescriptionHandle)imageDesc).height = s->height;
	//(**(ImageDescriptionHandle)imageDesc).frameCount = 0;
	//(**(ImageDescriptionHandle)imageDesc).depth = 24; // Given by the cType. See http://developer.apple.com/quicktime/icefloe/dispatch019.html
	//(**(ImageDescriptionHandle)imageDesc).clutID = -1; // We dont use any custom color table

	ret = DecompressSequenceBeginS(&(s->seqID),
				       imageDesc,
				       s->buffers[s->image_display],
				       hd_size_x * hd_size_y * s->cinfo->bpp, // Size of the buffer, not size of the actual frame data inside
				       s->gworld,
				       NULL,
				       NULL,
				       NULL,
				       srcCopy,
				       NULL,
				       (CodecFlags)NULL,
				       codecNormalQuality,
				       bestSpeedCodec);
	if (ret != noErr) {
		fprintf(stderr, "Failed DecompressSequenceBeginS\n");
	}
	DisposeHandle((Handle)imageDesc);

	//ICMFrameTimeRecord	frameTime = {{0}};
	//TimeBase		timeBase;

	//timeBase = NewTimeBase();
	//SetTimeBaseRate(timeBase, 0);

    /* TODO frametime probably not needed */
	//memset(&frameTime, 0, sizeof(ICMFrameTimeRecord));
	//frameTime.recordSize = sizeof(frameTime);
	//frameTime.scale = 1000; // Units per second
	//frameTime.base = timeBase; // Specifying a timeBase means that DecompressSequenceFrameWhen must run asynchronously
	//frameTime.duration = 30; // Duration of one frame specified accordingly to the scale specified above
	//frameTime.frameNumber = 0; // We don't know the frame number
	//frameTime.flags = icmFrameTimeDecodeImmediately;

	while (1) {
		platform_sem_wait(&s->semaphore);

		line1 = s->buffers[s->image_display];
		line2 = s->outBuffer;
        memcpy(line2, line1, hd_size_x*hd_size_y*s->cinfo->bpp);

		/* TODO: Running DecompressSequenceFrameWhen asynchronously in this way introduces a possible race condition! */
		ret = DecompressSequenceFrameWhen(s->seqID,
					       s->outBuffer,
					       hd_size_x * hd_size_y * s->cinfo->bpp, // Size of the buffer, not size of the actual frame data inside					       
					       0,
					       &ignore,
					       -1, // If you set asyncCompletionProc to -1, the operation is performed asynchronously but the decompressor does not call the completion function.
					       NULL);
					       //&frameTime);
		if (ret != noErr) {
			fprintf(stderr, "Failed DecompressSequenceFrameWhen: %d\n", ret);
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
	}

	return NULL;
}

char *
display_quicktime_getf(void *state)
{
    struct state_quicktime *s = (struct state_quicktime *) state;
	assert(s->magic == MAGIC_QT_DISPLAY);
	return (char *)s->buffers[s->image_network];
}

int
display_quicktime_putf(void *state, char *frame)
{
    int              tmp;
    struct state_quicktime *s = (struct state_quicktime *) state;

	UNUSED(frame);
    assert(s->magic == MAGIC_QT_DISPLAY);

    /* ...and give it more to do... */
    tmp = s->image_display;
    s->image_display = s->image_network;
    s->image_network = tmp;

    /* ...and signal the worker */
    platform_sem_post(&s->semaphore);
    return 0;
}

static void
print_modes(int fullhelp)
{
	ComponentDescription	cd;
	Component		c = 0;

	cd.componentType = QTVideoOutputComponentType;
	cd.componentSubType = 0;
	cd.componentManufacturer = 0;
	cd.componentFlags = 0;
	cd.componentFlagsMask = kQTVideoOutputDontDisplayToUser;

	//fprintf(stdout, "Number of Quicktime Vido Display components %d\n", CountComponents (&cd));
	
    fprintf(stdout, "Available playback devices:\n");
	/* Print relevant video output components */
	while ((c = FindNextComponent(c, &cd))) {
		Handle componentNameHandle = NewHandle(0);
		GetComponentInfo(c, &cd, componentNameHandle, NULL, NULL);
		HLock(componentNameHandle);
		char *cName = *componentNameHandle;

        fprintf(stdout, " Device %d: ", (int)c);
        nprintf(cName);
        fprintf(stdout, "\n");

		HUnlock(componentNameHandle);
		DisposeHandle(componentNameHandle);

        /* Get display modes of selected video output component */
        QTAtomContainer		            modeListAtomContainer = NULL;
        ComponentInstance	            videoDisplayComponentInstance;

        videoDisplayComponentInstance = OpenComponent(c);

        int ret = QTVideoOutputGetDisplayModeList(videoDisplayComponentInstance, &modeListAtomContainer);
        if (ret != noErr || modeListAtomContainer == NULL) {
            fprintf(stdout, "\tNo output modes available\n");
            CloseComponent(videoDisplayComponentInstance);
            continue;
        }

        int i = 1;
        QTAtom          atomDisplay = 0, nextAtomDisplay = 0;
        QTAtomType      type;
        QTAtomID        id;

        /* Print modes of current display component */
	    while (i < QTCountChildrenOfType(modeListAtomContainer, kParentAtomIsContainer, kQTVODisplayModeItem)) {

			ret = QTNextChildAnyType(modeListAtomContainer, kParentAtomIsContainer, atomDisplay, &nextAtomDisplay);
			// Make sure its a display atom
			ret = QTGetAtomTypeAndID(modeListAtomContainer, nextAtomDisplay, &type, &id);
			if (type != kQTVODisplayModeItem) continue;

			atomDisplay = nextAtomDisplay;   

			QTAtom		atom;
			long		dataSize, *dataPtr;

            /* Print component ID */
			fprintf(stdout, "\t - %ld: ", id);

            /* Print component name */
			atom = QTFindChildByID(modeListAtomContainer, atomDisplay, kQTVOName, 1, NULL);
			ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr*)&dataPtr);
			fprintf(stdout, "  %s; ", (char *)dataPtr);

			//if (strcmp((char *)dataPtr, mode) == 0) {
			//	displayMode = id;
			//}

            /* Print component other info */
			atom = QTFindChildByID(modeListAtomContainer, atomDisplay, kQTVODimensions, 1, NULL);
			ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
			fprintf(stdout, "%dx%d px\n", (int)EndianS32_BtoN(dataPtr[0]), (int)EndianS32_BtoN(dataPtr[1]));

            /* Do not print codecs */
            if (!fullhelp) {
                i++;
                continue;
            }

            /* Print supported pixel formats */
            fprintf(stdout, "\t\t - Codec: ");
			QTAtom		decompressorsAtom;
			int		j = 1;
            int     codecsPerLine = 0;
			while ((decompressorsAtom = QTFindChildByIndex(modeListAtomContainer, atomDisplay, kQTVODecompressors, j, NULL)) != 0) {
				atom = QTFindChildByID(modeListAtomContainer, decompressorsAtom, kQTVODecompressorType, 1, NULL);
				ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
                if (!(codecsPerLine % 9)) {
                    fprintf(stdout, "\n  \t\t\t");
				    fprintf(stdout, "%s", (char *)dataPtr);
                } else {
				    fprintf(stdout, ", %s", (char *)dataPtr);
                }
                codecsPerLine++;

				//atom = QTFindChildByID(modeListAtomContainer, decompressorsAtom, kQTVODecompressorComponent, 1, NULL);
				//ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
				//fprintf(stdout, "%s\n", (char *)dataPtr);

				j++;
			}
		    fprintf(stdout, "\n\n");

			i++;
            CloseComponent(videoDisplayComponentInstance);
		}
		fprintf(stdout, "\n");

		cd.componentType = QTVideoOutputComponentType;
		cd.componentSubType = 0;
		cd.componentManufacturer = 0;
		cd.componentFlags = 0;
		cd.componentFlagsMask = kQTVideoOutputDontDisplayToUser;
	}
}

static void
show_help(int full)
{
    printf("Quicktime output options:\n");
    printf("\tdevice:mode:codec | help | fullhelp\n");
    print_modes(full);    
}


void *
display_quicktime_init(char *fmt)	
{
	struct state_quicktime *s;
	int	                    ret;
	int	                    i;

	/* Parse fmt input */
	s = (struct state_quicktime *) malloc(sizeof(struct state_quicktime));
	s->magic = MAGIC_QT_DISPLAY;

    if(fmt!=NULL) {
        if(strcmp(fmt, "help") == 0) {
            show_help(0);
            free(s);
            return NULL;
        }
        if(strcmp(fmt, "fullhelp") == 0) {
            show_help(1);
            free(s);
            return NULL;
        }
        char *tmp = strdup(fmt);
        char *tok;

        tok = strtok(tmp, ":");
        if(tok == NULL) {
            show_help(0);
            free(s);
            free(tmp);
            return NULL;
        }
        s->device = atol(tok);
        tok = strtok(NULL, ":");
        if(tok == NULL) {
            show_help(0);
            free(s);
            free(tmp);
            return NULL;
        }
        s->mode = atol(tok);
        tok = strtok(NULL, ":");
        if(tok == NULL) {
            show_help(0);
            free(s);
            free(tmp);
            return NULL;
        }
        s->codec = strdup(tok);
    }

    for(i = 0; codec_info[i].name != NULL; i++) {
        if(strcmp(s->codec, codec_info[i].name) == 0) {
            s->cinfo = &codec_info[i];
        }
    }

	s->videoDisplayComponentInstance = 0;
	s->seqID = 0;

    /* Open device */
	s->videoDisplayComponentInstance = OpenComponent((Component)s->device);

	InitCursor();
	EnterMovies();

	/* Set the display mode */
	ret = QTVideoOutputSetDisplayMode(s->videoDisplayComponentInstance, s->mode);
	if (ret != noErr) {
		fprintf(stderr, "Failed to set video output display mode.\n");
		return NULL;
	}

	/* We don't want to use the video output component instance echo port*/
	ret = QTVideoOutputSetEchoPort(s->videoDisplayComponentInstance, nil);
	if (ret != noErr) {
		fprintf(stderr, "Failed to set video output echo port.\n");
		return NULL;
	}

	/* Register Ultragrid with instande of the video outpiut */
	ret = QTVideoOutputSetClientName(s->videoDisplayComponentInstance, (ConstStr255Param)"Ultragrid");
	if (ret != noErr) {
		fprintf(stderr, "Failed to register Ultragrid with selected video output instance.\n");
		return NULL;
	}

	/* Call QTVideoOutputBegin to gain exclusive access to the video output */
	ret = QTVideoOutputBegin(s->videoDisplayComponentInstance);
	if (ret != noErr) {
		fprintf(stderr, "Failed to get exclusive access to selected video output instance.\n");
		return NULL;
	}

	/* Get a pointer to the gworld used by video output component */
	ret = QTVideoOutputGetGWorld(s->videoDisplayComponentInstance, &s->gworld);
	if (ret != noErr) {
		fprintf(stderr, "Failed to get selected video output instance GWorld.\n");
		return NULL;
	}

	ImageDescriptionHandle		gWorldImgDesc = NULL;
	PixMapHandle	    		gWorldPixmap = (PixMapHandle)GetGWorldPixMap(s->gworld);

    /* Determine width and height */
	ret = MakeImageDescriptionForPixMap(gWorldPixmap, &gWorldImgDesc);
	if (ret != noErr) {
		fprintf(stderr, "Failed to determine width and height.\n");
        return NULL;
	}
    hd_size_x = s->width  = (**gWorldImgDesc).width;
	hd_size_y = s->height = (**gWorldImgDesc).height;

    if (s->cinfo->h_align) {
        hd_size_x = ((hd_size_x + s->cinfo->h_align -1) / s->cinfo->h_align) * s->cinfo->h_align;
    }

    fprintf(stdout, "Selected mode: %d(%d)x%d, %fbpp\n", s->width, hd_size_x, hd_size_y, s->cinfo->bpp);

    platform_sem_init(&s->semaphore, 0, 0);

    s->buffers[0] = malloc(hd_size_x*hd_size_y*s->cinfo->bpp);
    s->buffers[1] = malloc(hd_size_x*hd_size_y*s->cinfo->bpp);

	s->image_network = 0;
	s->image_display = 1;

	s->outBuffer = malloc(hd_size_x*hd_size_y*s->cinfo->bpp);

    if (pthread_create(&(s->thread_id), NULL, display_thread_quicktime, (void *) s) != 0) {
        perror("Unable to create display thread\n");
        return NULL;
	}

	return (void *) s;
}

void
display_quicktime_done(void *state)
{
	struct state_quicktime *s = (struct state_quicktime *) state;
	int		ret;

	assert(s->magic == MAGIC_QT_DISPLAY);
	ret = QTVideoOutputEnd(s->videoDisplayComponentInstance);
	if (ret != noErr) {
		fprintf(stderr, "Failed to release the video output component.\n");
	}

	ret = CloseComponent(s->videoDisplayComponentInstance);
	if (ret != noErr) {
		fprintf(stderr, "Failed to close the video output component.\n");
	}

	DisposeGWorld(s->gworld);
}

display_colour_t
display_quicktime_colour(void *state)
{
    struct state_quicktime *s = (struct state_quicktime *) state;
    assert(s->magic == MAGIC_QT_DISPLAY);
    return DC_YUV;
}

display_type_t *
display_quicktime_probe(void)
{
    display_type_t          *dtype;
    display_format_t        *dformat;

    dformat = malloc(sizeof(display_format_t));
    if (dformat == NULL) {
        return NULL;
    }
    dformat->size        = DS_1920x1080;
    dformat->colour_mode = DC_YUV;
    dformat->num_images  = 1;

    dtype = malloc(sizeof(display_type_t));
    if (dtype != NULL) {
        dtype->id          = DISPLAY_QUICKTIME_ID;
        dtype->name        = "quicktime";
        dtype->description = "QuickTime display device";
        dtype->formats     = dformat;
        dtype->num_formats = 1;
    }
    return dtype;
}

#endif /* HAVE_MACOSX */

