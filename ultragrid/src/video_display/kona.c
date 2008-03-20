#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "host.h"
#include "tv.h"

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>

#include "compat/platform_semaphore.h"
#include <signal.h>
#include <pthread.h>
#include <assert.h>

#include "video_display.h"
#include "video_display/kona.h"

#define KONA_MAGIC 	DISPLAY_KONA_ID


struct state_kona {
	ComponentInstance	videoDisplayComponentInstance;
	Component		videoDisplayComponent;
	GWorldPtr		gworld;
	ImageSequence		seqID;

	char			*buffers[2];
	char			*outBuffer;
	int			image_display, image_network;

        /* Thread related information follows... */
        pthread_t               thread_id;
        sem_t                   semaphore;

	uint32_t		magic;
};

/* Prototyping */
char * four_char_decode(int format);


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
display_thread_kona(void *arg)
{
	struct state_kona	*s = (struct state_kona *) arg;

	ImageDescriptionHandle	imageDesc;
	CodecFlags		ignore;
	int			ret;
	int			i;

	char			*line1, *line2;

	int			frames = 0;
	struct			timeval t, t0;

	
	imageDesc = (ImageDescriptionHandle)NewHandle(sizeof(ImageDescription));
	
	platform_sem_wait(&s->semaphore);

	(**(ImageDescriptionHandle)imageDesc).idSize = sizeof(ImageDescription);
	if (bitdepth == 10) {
		(**(ImageDescriptionHandle)imageDesc).cType = 'v210'; // v210 seems to be a little bit different than 10-bit 4:2:2 uyvy we are sending
		(**(ImageDescriptionHandle)imageDesc).dataSize = hd_size_x * hd_size_y * 8/3; // dataSize is specified in bytes and is specified as height*width*bytes_per_luma_instant. v210 sets bytes_per_luma_instant to 8/3. See http://developer.apple.com/quicktime/icefloe/dispatch019.html#v210
	} else {
		(**(ImageDescriptionHandle)imageDesc).cType = '2Vuy'; // QuickTime specifies '2vuy' codec, however Kona3 reports it as '2Vuy'
		(**(ImageDescriptionHandle)imageDesc).dataSize = hd_size_x * hd_size_y * hd_color_bpp; // dataSize is specified in bytes 
	}
	(**(ImageDescriptionHandle)imageDesc).hRes = 72; // not used actually. Set to 72. See http://developer.apple.com/quicktime/icefloe/dispatch019.html#imagedesc
	(**(ImageDescriptionHandle)imageDesc).vRes = 72; // not used actually. Set to 72. See http://developer.apple.com/quicktime/icefloe/dispatch019.html#imagedesc
	(**(ImageDescriptionHandle)imageDesc).width = hd_size_x; // Beware: must be a multiple of horiz_align_pixels which is 2 for 2Vuy and 48 for v210. hd_size_x=1920 is a multiple of both. TODO: needs further investigation for 2K!
	(**(ImageDescriptionHandle)imageDesc).height = hd_size_y;
	(**(ImageDescriptionHandle)imageDesc).frameCount = 0;
	(**(ImageDescriptionHandle)imageDesc).depth = 24; // Given by the cType. See http://developer.apple.com/quicktime/icefloe/dispatch019.html
	(**(ImageDescriptionHandle)imageDesc).clutID = -1; // We dont use any custom color table

	ret = DecompressSequenceBeginS(&(s->seqID),
				       imageDesc,
				       s->buffers[s->image_display],
				       hd_size_x * hd_size_y * hd_color_bpp, // Size of the buffer, not size of the actual frame data inside
				       s->gworld,
				       NULL,
				       NULL,
				       NULL,
				       srcCopy,
				       NULL,
				       nil,
				       codecNormalQuality,
				       bestSpeedCodec);
	if (ret != noErr) {
		fprintf(stderr, "Failed DecompressSequenceBeginS\n");
	}
	DisposeHandle((Handle)imageDesc);

	ICMFrameTimeRecord	frameTime = {{0}};
	TimeBase		timeBase;

	timeBase = NewTimeBase();
	SetTimeBaseRate(timeBase, 0);

	memset(&frameTime, 0, sizeof(ICMFrameTimeRecord));

	frameTime.recordSize = sizeof(frameTime);
	frameTime.scale = 1000; // Units per second
	frameTime.base = timeBase; // Specifying a timeBase means that DecompressSequenceFrameWhen must run asynchronously
	frameTime.duration = 30; // Duration of one frame specified accordingly to the scale specified above
	frameTime.frameNumber = 0; // We don't know the frame number
	frameTime.flags = icmFrameTimeDecodeImmediately;

	while (1) {
		platform_sem_wait(&s->semaphore);

		line1 = s->buffers[s->image_display];
		line2 = s->outBuffer;
		if (bitdepth == 10) {
			for (i = 0; i < 1080; i += 2) {
				memcpy(line2, line1, 5120);
				memcpy(line2+5120, line1+5120*540, 5120);
				line1 += 5120;
				line2 += 2*5120;			
			}
		} else {
			for (i = 0; i < 1080; i += 2) {
				memcpy(line2, line1, hd_size_x * hd_color_bpp);
				memcpy(line2 + hd_size_x * hd_color_bpp, line1 + hd_size_x * hd_color_bpp * hd_size_y / 2, hd_size_x * hd_color_bpp);
				line1 += hd_size_x * hd_color_bpp;
				line2 += hd_size_x * hd_color_bpp * 2;
			}
		}
	

		/* TODO: Running DecompressSequenceFrameWhen asynchronously in this way introduces a possible race condition! */
		ret = DecompressSequenceFrameWhen(s->seqID,
					       s->outBuffer,
					       hd_size_x * hd_size_y * hd_color_bpp, // Size of the buffer, not size of the actual frame data inside					       
					       0,
					       &ignore,
					       -1, // If you set asyncCompletionProc to -1, the operation is performed asynchronously but the decompressor does not call the completion function.
					       &frameTime);
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
display_kona_getf(void *state)
{
        struct state_kona *s = (struct state_kona *) state;
	assert(s->magic == KONA_MAGIC);
	return (char *)s->buffers[s->image_network];

}

int
display_kona_putf(void *state, char *frame)
{
        int              tmp;
        struct state_kona *s = (struct state_kona *) state;

	UNUSED(frame);
        assert(s->magic == KONA_MAGIC);

        /* ...and give it more to do... */
        tmp = s->image_display;
        s->image_display = s->image_network;
        s->image_network = tmp;

        /* ...and signal the worker */
        platform_sem_post(&s->semaphore);
        return 0;

}

void *
display_kona_init(void)	
{

	struct state_kona	*s;

	ComponentDescription	cd;
	Component		c = 0;
	QTAtomContainer		modeListAtomContainer = NULL;

	int			ret;
	int			i;

	s = (struct state_kona *) malloc(sizeof(struct state_kona));
	s->magic = KONA_MAGIC;
	s->videoDisplayComponentInstance = 0;
	s->seqID = 0;

	InitCursor();
	EnterMovies();

	cd.componentType = QTVideoOutputComponentType;
	cd.componentSubType = 0;
	cd.componentManufacturer = 0;
	cd.componentFlags = 0;
	cd.componentFlagsMask = kQTVideoOutputDontDisplayToUser;
	
	/* Get video output component */
	while ((c = FindNextComponent(c, &cd))) {
		Handle componentNameHandle = NewHandle(0);
		GetComponentInfo(c, &cd, componentNameHandle, NULL, NULL);
		char *cName = *componentNameHandle;
		DisposeHandle(componentNameHandle);

		if (strcmp(cName, "AJA")) {
			fprintf(stdout, "Found video output component: %s\n", cName);
			s->videoDisplayComponent = c;
			s->videoDisplayComponentInstance = OpenComponent(s->videoDisplayComponent);
			break;
		} else {
			fprintf(stderr, "AJA Kona3 not found!\n");
			return NULL;
		}
	}


	long	displayMode = 0;

	/* Get display modes of selected video output component */
	ret = QTVideoOutputGetDisplayModeList(s->videoDisplayComponentInstance, &modeListAtomContainer);
	if (ret == noErr && modeListAtomContainer != NULL) {
		i = 1;
		QTAtom          atomDisplay = 0, nextAtomDisplay = 0;
		QTAtomType      type;
		QTAtomID        id;

		char		*mode;

		/* TODO: this is hardcoded right now */
		if (bitdepth == 10) {
			mode = strdup("AJA Kona 1080p29.97 10 Bit");
		} else {
			mode = strdup("AJA Kona 1080p29.97   8 Bit");
		}

		fprintf(stdout, "\nSupported video output modes:\n");
		while (i < QTCountChildrenOfType(modeListAtomContainer, kParentAtomIsContainer, kQTVODisplayModeItem)) {

			ret = QTNextChildAnyType(modeListAtomContainer, kParentAtomIsContainer, atomDisplay, &nextAtomDisplay);
			// Make sure its a display atom
			ret = QTGetAtomTypeAndID(modeListAtomContainer, nextAtomDisplay, &type, &id);
			if (type != kQTVODisplayModeItem) continue;

			atomDisplay = nextAtomDisplay;   

			QTAtom		atom;
			long		dataSize, *dataPtr;

			fprintf(stdout, "%ld: ", id);

			atom = QTFindChildByID(modeListAtomContainer, atomDisplay, kQTVOName, 1, NULL);
			ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr*)&dataPtr);
			fprintf(stdout, "  %s; ", (char *)dataPtr);

			if (strcmp((char *)dataPtr, mode) == 0) {
				displayMode = id;
			}

			atom = QTFindChildByID(modeListAtomContainer, atomDisplay, kQTVODimensions, 1, NULL);
			ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
			fprintf(stdout, "%dx%d px; ", (int)EndianS32_BtoN(dataPtr[0]), (int)EndianS32_BtoN(dataPtr[1]));

			atom = QTFindChildByID(modeListAtomContainer, atomDisplay, kQTVORefreshRate, 1, NULL);
			ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
			fprintf(stdout, "%ld Hz; ", EndianS32_BtoN(dataPtr[0]));

			atom = QTFindChildByID(modeListAtomContainer, atomDisplay, kQTVOPixelType, 1, NULL);
			ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
			fprintf(stdout, "%s\n", (char *)dataPtr);

			QTAtom		decompressorsAtom;
			int		j = 1;
			while ((decompressorsAtom = QTFindChildByIndex(modeListAtomContainer, atomDisplay, kQTVODecompressors, j, NULL)) != 0) {
				atom = QTFindChildByID(modeListAtomContainer, decompressorsAtom, kQTVODecompressorType, 1, NULL);
				ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
				fprintf(stdout, "        Decompressor: %s, ", (char *)dataPtr);

				atom = QTFindChildByID(modeListAtomContainer, decompressorsAtom, kQTVODecompressorComponent, 1, NULL);
				ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
				fprintf(stdout, "%s, ", (char *)dataPtr);

				atom = QTFindChildByID(modeListAtomContainer, decompressorsAtom, kQTVODecompressorContinuous, 1, NULL);
				ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
				fprintf(stdout, "%ld\n", EndianS32_BtoN(dataPtr[0]));

				j++;
			}

			i++;
		}
		fprintf(stdout, "\n");
	} else {
		fprintf(stderr, "Video output component AJA doesn't seem to provide any display mode!\n");
		return NULL;
	}

	/* Set the display mode */

	ret = QTVideoOutputSetDisplayMode(s->videoDisplayComponentInstance, displayMode);
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
	

        platform_sem_init(&s->semaphore, 0, 0);

        s->buffers[0] = malloc(hd_size_x*hd_size_y*hd_color_bpp);
        s->buffers[1] = malloc(hd_size_x*hd_size_y*hd_color_bpp);

	s->image_network = 0;
	s->image_display = 1;

	s->outBuffer = malloc(hd_size_x*hd_size_y*hd_color_bpp);

        if (pthread_create(&(s->thread_id), NULL, display_thread_kona, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}

	/* Register Ultragrid with instande of the video outpiut */
	ret = QTVideoOutputSetClientName(s->videoDisplayComponentInstance, (ConstStr255Param)"Ultragrid");
	if (ret != noErr) {
		fprintf(stderr, "Failed to register Ultragrid with Kona3 video output instance.\n");
		return NULL;
	}

	/* Call QTVideoOutputBegin to gain exclusive access to the video output */
	ret = QTVideoOutputBegin(s->videoDisplayComponentInstance);
	if (ret != noErr) {
		fprintf(stderr, "Failed to get exclusive access to Kona3 video output instance.\n");
		return NULL;
	}

	/* Get a pointer to the gworld used by video output component */
	ret = QTVideoOutputGetGWorld(s->videoDisplayComponentInstance, &s->gworld);
	if (ret != noErr) {
		fprintf(stderr, "Failed to get Kona3 video output instance GWorld.\n");
		return NULL;
	}

	ImageDescriptionHandle		gWorldImgDesc = NULL;
	PixMapHandle			gWorldPixmap = (PixMapHandle)GetGWorldPixMap(s->gworld);

	ret = MakeImageDescriptionForPixMap(gWorldPixmap, &gWorldImgDesc);
	if (ret == noErr) {
		fprintf(stdout, "\n\nKona3 gWorld settings:\n");
		fprintf(stdout, "\tctype: '%s'\n", four_char_decode((**gWorldImgDesc).cType));
		fprintf(stdout, "\tvendor: 0x%ld\n", (**gWorldImgDesc).vendor);
		fprintf(stdout, "\twidth: %d\n", (**gWorldImgDesc).width);
		fprintf(stdout, "\theight: %d\n", (**gWorldImgDesc).height);
		fprintf(stdout, "\thRes: %ld\n", (**gWorldImgDesc).hRes);
		fprintf(stdout, "\tvRes: %ld\n", (**gWorldImgDesc).vRes);
		fprintf(stdout, "\tdataSize: %ld\n", (**gWorldImgDesc).dataSize);
		fprintf(stdout, "\tframeCount: %d\n", (**gWorldImgDesc).frameCount);
		fprintf(stdout, "\tname: %s\n", (**gWorldImgDesc).name);
		fprintf(stdout, "\tdepth: %d\n", (**gWorldImgDesc).depth);
		fprintf(stdout, "\tclutID: %d\n", (**gWorldImgDesc).clutID);
		fprintf(stdout, "\n");
	}

	return (void *) s;
}

void
display_kona_done(void *state)
{
	struct state_kona *s = (struct state_kona *) state;
	int		ret;

	assert(s->magic == KONA_MAGIC);
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
display_kona_colour(void *state)
{
        struct state_kona *s = (struct state_kona *) state;
        assert(s->magic == KONA_MAGIC);
        return DC_YUV;
}

display_type_t *
display_kona_probe(void)
{
        display_type_t          *dtype;
        display_format_t        *dformat;

	ComponentDescription	cd;
	Component		c = 0;
	int			foundAJAKona = 0;

	cd.componentType = QTVideoOutputComponentType;
	cd.componentSubType = 0;
	cd.componentManufacturer = 0;
	cd.componentFlags = 0;
	cd.componentFlagsMask = kQTVideoOutputDontDisplayToUser;
	
	while ((c = FindNextComponent(c, &cd))) {
		Handle componentNameHandle = NewHandle(0);
		GetComponentInfo(c, &cd, componentNameHandle, NULL, NULL);
		// Print the component info
		char *cName = *componentNameHandle;
		if (strcmp(cName, "AJA")) {
			foundAJAKona = 1;
		}

		int len = *cName++;
		cName[len] = 0;

		DisposeHandle(componentNameHandle);
	}

	if (!foundAJAKona) {
		return NULL;
	}

        dformat = malloc(sizeof(display_format_t));
        if (dformat == NULL) {
                return NULL;
        }
        dformat->size        = DS_1920x1080;
        dformat->colour_mode = DC_YUV;
        dformat->num_images  = 1;

        dtype = malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id          = DISPLAY_KONA_ID;
                dtype->name        = "kona";
                dtype->description = "AJA Kona3 (1080i/60 YUV 4:2:2)";
                dtype->formats     = dformat;
                dtype->num_formats = 1;
        }
        return dtype;
}

#endif /* HAVE_MACOSX */

