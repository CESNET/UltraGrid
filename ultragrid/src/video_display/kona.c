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
	int			image_display, image_network;

	int			frames_to_process;
        /* Thread related information follows... */
        pthread_t               thread_id;
        sem_t                   semaphore;

	uint32_t		magic;
};

static void*
display_thread_kona(void *arg)
{
	struct state_kona	*s = (struct state_kona *) arg;

	ImageDescriptionHandle	imageDesc;
	CodecFlags		ignore;
	int			ret;

	int			frames = 0;
	struct			timeval t, t0;

	
	imageDesc = (ImageDescriptionHandle)NewHandle(0);
	ret = DecompressSequenceBegin(&(s->seqID), imageDesc, s->gworld, NULL, NULL,
					NULL, srcCopy, NULL, 0, codecNormalQuality, bestSpeedCodec);
	if (ret != noErr) {
		fprintf(stderr, "Failed DecompressSequenceBeginS\n");
	}
	DisposeHandle((Handle)imageDesc);

	while (1) {
		platform_sem_wait(&s->semaphore);
		s->frames_to_process--;

		/* TODO */
		// memcpy(GetPixBaseAddr(GetGWorldPixMap(s->gworld)), s->buffers[s->image_display], hd_size_x*hd_size_y*hd_color_bpp);

		ret = DecompressSequenceFrameWhen(s->seqID,
						  s->buffers[s->image_display],
						  hd_size_x*hd_size_y*hd_color_bpp,
						  0,
						  &ignore,
						  NULL,
						  nil);
		if (ret != noErr) {
			fprintf(stderr, "Failed DecompressSequenceFrameWhen\n");
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
	s->frames_to_process++;
        if(s->frames_to_process > 1)
                printf("frame drop!\n");
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
	s->frames_to_process = 0;

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
			s->videoDisplayComponent = c;
			s->videoDisplayComponentInstance = OpenComponent(s->videoDisplayComponent);
			break;
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

			/* TODO: this is hardcoded right now */
			if (strcmp((char *)dataPtr, "AJA Kona 1080i30 10 Bit") == 0) {
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
		fprintf(stdout, "Found video output component: %s\n", cName);

		DisposeHandle(componentNameHandle);
	}

	if (!foundAJAKona) {
		fprintf(stderr, "AJA Kona3 not found!\n");
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

