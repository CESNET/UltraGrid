#ifdef __cplusplus
extern "C" {
#endif

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "tv.h"

#include "debug.h"
#include "video_types.h"
#include "video_capture.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

#ifdef HAVE_DECKLINK		/* From config.h */

#include "video_capture/decklink.h"

#include "DeckLinkAPI.h" /* From DeckLink SDK */ 

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds

unsigned int init_hd_color_bpp = 2; // hd_color_bpp;

static int	device = 0; // use first BlackMagic device
static int	mode = 5; // hd_video_mode; // 0; // It should need 5 .... 1080i50 or 6 .... 1080i59.94 // 5 .... for intensity // 7 .... for blackmagic
static int	connection = 0; // the choice of BMDVideoConnection // It should be 0 .... bmdVideoConnectionSDI

int frames = 0;
struct timeval t, t0;

#ifdef __cplusplus
extern "C" {
#endif

extern int		 should_exit;

#ifdef __cplusplus
}
#endif

struct vidcap_decklink_state;

class VideoDelegate : public IDeckLinkInputCallback
{
private:
	int32_t mRefCount;
	double  lastTime;
    
public:
	int	newFrameReady;
	void*	pixelFrame;
	int	first_time;
	struct vidcap_decklink_state	*s;
	
	void set_vidcap_state(vidcap_decklink_state	*state);
	
	VideoDelegate () {
		newFrameReady = 0;
		first_time = 1;
		s = NULL;
	};

	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, LPVOID *ppv) { return E_NOINTERFACE; }
	virtual ULONG STDMETHODCALLTYPE AddRef(void)
	{
		return mRefCount++; 
	};
	virtual ULONG STDMETHODCALLTYPE  Release(void)
	{
		int32_t newRefValue;
        	
		newRefValue = mRefCount--;
		if (newRefValue == 0)
		{
			delete this;
			return 0;
		}        
        	return newRefValue;
	};
	virtual HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode*, BMDDetectedVideoInputFormatFlags)
	{
		return S_OK;
	};
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*);    
};

struct vidcap_decklink_state {
	IDeckLink*		deckLink;
	IDeckLinkInput*		deckLinkInput;
	// void*			rtp_buffer;
	int			buffer_size;
	VideoDelegate*		delegate;
	unsigned int		hd_size_x;
	unsigned int		hd_size_y;
	unsigned int		next_frame_time; // avarege time between frames
	pthread_mutex_t	 	lock;
	pthread_cond_t	 	boss_cv;
	int		 	boss_waiting;
};

HRESULT	
VideoDelegate::VideoInputFrameArrived (IDeckLinkVideoInputFrame *arrivedFrame, IDeckLinkAudioInputPacket *audioPacket)
{

	// Video

	pthread_mutex_lock(&(s->lock));
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	if(arrivedFrame)
	{
		if (arrivedFrame->GetFlags() & bmdFrameHasNoInputSource)
		{
			// fprintf(stderr, "Frame received (#%lu) - No input signal detected\n", framecount);
		}
		else{
			// printf("Frame received (#%lu) - Valid Frame (Size: %li bytes)\n", framecount, arrivedFrame->GetRowBytes() * arrivedFrame->GetHeight());
			
		}
		frames++;
	}

	arrivedFrame->GetBytes(&pixelFrame);

	if(first_time){
		first_time = 0;
	}

	newFrameReady = 1; // The new frame is ready to grab
	
	if (s->boss_waiting) {
		pthread_cond_signal(&(s->boss_cv));
	}
	
// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	pthread_mutex_unlock(&(s->lock));

	gettimeofday(&t, NULL);
	double seconds = tv_diff(t, t0);	
	if (seconds >= 5) {
		float fps  = frames / seconds;
		fprintf(stderr, "%d frames in %g seconds = %g FPS\n", frames, seconds, fps);
		t0 = t;
		frames = 0;
	}
        
	debug_msg("VideoInputFrameArrived - END\n"); /* TOREMOVE */

	return S_OK;
}

void VideoDelegate::set_vidcap_state(vidcap_decklink_state	*state){
	s = state;
}

/* External API ***************************************************************/

struct vidcap_type *
vidcap_decklink_probe(void)
{

	struct vidcap_type*		vt;
	IDeckLinkIterator*		deckLinkIterator;
	IDeckLink*			deckLink;
	IDeckLinkDisplayModeIterator*	displayModeIterator = NULL;
	IDeckLinkInput*		  	deckLinkInput = NULL;
	IDeckLinkDisplayMode*		displayMode = NULL;
	HRESULT				result;
	int				numDevices = 0;
	
	unsigned int hd_size_x;
	unsigned int hd_size_y;

	/* CHECK IF THERE IS ANY BLACKMAGIC DEVICE */

	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
	deckLinkIterator = CreateDeckLinkIteratorInstance();
	if (deckLinkIterator == NULL)
	{
		printf("A DeckLink iterator could not be created.  The DeckLink drivers may not be installed.\n");
		return NULL;
	}
	
	int dnum = 0;
	
	// Enumerate all cards in this system to find if there is any
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
		if (device != dnum) {
			dnum++;

			// Release the IDeckLink instance when we've finished with it to prevent leaks
			deckLink->Release();
			deckLink = NULL;
			continue;	
		}
		dnum++;
		
		result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput);
	  	if (result != S_OK)
    		{
			printf("Could not obtain the IDeckLinkInput interface - result = %08x\n", result);
			return NULL;
		}

		result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
		if (result != S_OK)
		{
			printf("Could not obtain the video input display mode iterator - result = %08x\n", result);
			return NULL;
    		}

		int mnum = 0;
		while (displayModeIterator->Next(&displayMode) == S_OK)
		{
			if (mode != mnum) {
				mnum++;
				// Release the IDeckLinkDisplayMode object to prevent a leak
				displayMode->Release();
				continue;
			}
			mnum++;   
                
			// Obtain the display mode's properties
			hd_size_x = displayMode->GetWidth();
			hd_size_y = displayMode->GetHeight();
		}
		
		// Release the IDeckLink instance when we've finished with it to prevent leaks
		deckLink->Release();
	}
	
	// the number of DeckLink devices
	numDevices = dnum;
	
	// If no DeckLink cards were found in the system, inform the user
	if (numDevices == 0){
		printf("No Blackmagic Design devices were found.\n");
		printf("Cannot probe Blackmagic capture device\n");
		return NULL;
	}

	/* END OF CHECK IF THERE IS ANY BLACKMAGIC DEVICE */

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DECKLINK_ID;
		vt->name        = "decklink";
		vt->description = "Blackmagic DeckLink card";
		vt->width       = hd_size_x;
		vt->height      = hd_size_y;
		vt->colour_mode = YUV_422;
	}
	return vt;
}

void *
vidcap_decklink_init(struct vidcap_fmt *fmt)
{
	//int fps; //FIXME What is it good for?
	//fps = atoi(fmt);

	debug_msg("vidcap_decklink_init\n"); /* TOREMOVE */

	struct vidcap_decklink_state *s;

	int dnum, mnum;

	IDeckLinkIterator*	deckLinkIterator;
	IDeckLink*		deckLink;
	HRESULT			result;

	IDeckLinkInput*			deckLinkInput = NULL;
	IDeckLinkDisplayModeIterator*	displayModeIterator = NULL;
	IDeckLinkDisplayMode*		displayMode = NULL;
	IDeckLinkConfiguration*		deckLinkConfiguration = NULL;

	s = (struct vidcap_decklink_state *) malloc(sizeof(struct vidcap_decklink_state));
	if (s == NULL) {
		//printf("Unable to allocate DeckLink state\n",fps);
		printf("Unable to allocate DeckLink state\n");
		return NULL;
	}

	s->deckLink = NULL;
	s->deckLinkInput = NULL;

	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
	deckLinkIterator = CreateDeckLinkIteratorInstance();
	if (deckLinkIterator == NULL)
	{
		printf("A DeckLink iterator could not be created. The DeckLink drivers may not be installed.\n");
		return NULL;
	}

	dnum = 0;
	deckLink = NULL;
    
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
		if (device != dnum) {
			dnum++;

			// Release the IDeckLink instance when we've finished with it to prevent leaks
			deckLink->Release();
			deckLink = NULL;
			continue;	
		}
		dnum++;

		s->deckLink = deckLink;

		const char *deviceNameString = NULL;
		
		// Print the model name of the DeckLink card
		result = deckLink->GetModelName(&deviceNameString);
		if (result == S_OK)
		{	
			printf("Using device [%s]\n", deviceNameString);

			// Query the DeckLink for its configuration interface
			result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput);
			if (result != S_OK)
			{
				printf("Could not obtain the IDeckLinkInput interface - result = %08x\n", result);
				goto error;
			}

			s->deckLinkInput = deckLinkInput;

			// Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
			result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
			if (result != S_OK)
			{
				printf("Could not obtain the video input display mode iterator - result = %08x\n", result);
				goto error;
			}

			mnum = 0;
			while (displayModeIterator->Next(&displayMode) == S_OK)
			{
				if (mode != mnum) {
					mnum++;
					// Release the IDeckLinkDisplayMode object to prevent a leak
					displayMode->Release();
					continue;
				}
				mnum++; 

				printf("The desired display mode is supported: %d\n",mode);  
                
				const char *displayModeString = NULL;

				result = displayMode->GetName(&displayModeString);
				if (result == S_OK)
				{
					BMDPixelFormat pf = bmdFormat8BitYUV;

					// get avarage time between frames
					BMDTimeValue	frameRateDuration;
					BMDTimeScale	frameRateScale;
					double				fps;

					// Obtain the display mode's properties
					s->hd_size_x = displayMode->GetWidth();
					s->hd_size_y = displayMode->GetHeight();

					displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
					fps = (double)frameRateScale / (double)frameRateDuration;
					s->next_frame_time = (int) (1000000 / fps); // in microseconds
					debug_msg("%-20s \t %d x %d \t %g FPS \t %d AVAREGE TIME BETWEEN FRAMES\n", displayModeString, s->hd_size_x, s->hd_size_y, fps, s->next_frame_time); /* TOREMOVE */  

					deckLinkInput->StopStreams();

					printf("Enable video input: %s\n", displayModeString);

					result = deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, 0);
					if (result != S_OK)
					{
						printf("Could not enable video input: %08x\n", result);
						goto error;
					}

					// Query the DeckLink for its configuration interface
					result = deckLinkInput->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
					if (result != S_OK)
					{
						printf("Could not obtain the IDeckLinkConfiguration interface: %08x\n", result);
						goto error;
					}

					BMDVideoConnection conn;
					switch (connection) {
					case 0:
						conn = bmdVideoConnectionSDI;
						break;
					case 1:
						conn = bmdVideoConnectionHDMI;
						break;
					case 2:
						conn = bmdVideoConnectionComponent;
						break;
					case 3:
						conn = bmdVideoConnectionComposite;
						break;
					case 4:
						conn = bmdVideoConnectionSVideo;
						break;
					case 5:
						conn = bmdVideoConnectionOpticalSDI;
						break;
					default:
						break;
					}
                    
					if (deckLinkConfiguration->SetVideoInputFormat(conn) == S_OK) {
						printf("Input set to: %d\n", connection);
					}

					// We don't want to process audio
					deckLinkInput->DisableAudioInput();

					// set Callback which returns frames
					s->delegate = new VideoDelegate();
					s->delegate->set_vidcap_state(s);
					deckLinkInput->SetCallback(s->delegate);

					// Start streaming
					printf("Start capture\n", connection);
					result = deckLinkInput->StartStreams();
					if (result != S_OK)
					{
						printf("Could not start stream: %08x\n", result);
						goto error;
					}

				}else{
					printf("Could not : %08x\n", result);
					goto error;
				}

				displayMode->Release();
				displayMode = NULL;
			}

			if (displayModeIterator != NULL){
				displayModeIterator->Release();
				displayModeIterator = NULL;
			}
		}
	}

	// init mutex
	pthread_mutex_init(&(s->lock), NULL);
	pthread_cond_init(&(s->boss_cv), NULL);
	
	s->boss_waiting = FALSE;

	// setup rtp_buffer
	s->buffer_size = s->hd_size_x * s->hd_size_y * init_hd_color_bpp;
	// s->rtp_buffer = malloc(s->buffer_size);

	printf("DeckLink capture device enabled\n");

	debug_msg("vidcap_decklink_init - END\n"); /* TOREMOVE */

	return s;
error:

	if(displayMode != NULL)
	{
		displayMode->Release();
		displayMode = NULL;
	}

	if(deckLinkInput != NULL)
	{
		deckLinkInput->Release();
		deckLinkInput = NULL;
	}

	if(deckLink != NULL)
	{
		deckLink->Release();
		deckLink = NULL;
	}

	return NULL;
}

void
vidcap_decklink_done(void *state)
{
	debug_msg("vidcap_decklink_done\n"); /* TOREMOVE */

	HRESULT	result;

	struct vidcap_decklink_state *s = (struct vidcap_decklink_state *) state;

	assert (s != NULL);

	if (s!= NULL) {
		result = s->deckLinkInput->StopStreams();
		if (result != S_OK)
		{
			printf("Could not stop stream: %08x\n", result);
		}

		if(s->deckLinkInput != NULL)
		{
			s->deckLinkInput->Release();
			s->deckLinkInput = NULL;
		}

		if(s->deckLink != NULL)
		{
			s->deckLink->Release();
			s->deckLink = NULL;
		}

		free(s);
	}
}

struct video_frame *
vidcap_decklink_grab(void *state)
{
	debug_msg("vidcap_decklink_grab\n"); /* TO REMOVE */

	struct vidcap_decklink_state 	*s = (struct vidcap_decklink_state *) state;
	struct video_frame		*vf;

	HRESULT	result;
	
	int		rc;
	struct timespec	ts;
	struct timeval	tp;

	int timeout = 0;

	pthread_mutex_lock(&(s->lock));
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	debug_msg("vidcap_decklink_grab - before while\n"); /* TOREMOVE */

	while (!s->delegate->newFrameReady) {
		s->boss_waiting = TRUE;
		debug_msg("vidcap_decklink_grab - pthread_cond_timedwait\n"); /* TOREMOVE */

		// get time for timeout
		rc =  gettimeofday(&tp, NULL);

		/* Convert from timeval to timespec */
		ts.tv_sec  = tp.tv_sec;
		ts.tv_nsec = tp.tv_usec * 1000;
		ts.tv_nsec += 2 * s->next_frame_time * 1000;
		// make it correct
		ts.tv_sec += ts.tv_nsec / 1000000000;
		ts.tv_nsec = ts.tv_nsec % 1000000000;

		debug_msg("vidcap_decklink_grab - current time: %02d:%03d\n",tp.tv_sec, tp.tv_usec/1000); /* TOREMOVE */

		rc = pthread_cond_timedwait(&(s->boss_cv), &(s->lock), &ts);
		s->boss_waiting = FALSE;
		
		debug_msg("vidcap_decklink_grab - AFTER pthread_cond_timedwait - newFrameReady: %d\n",s->delegate->newFrameReady); /* TOREMOVE */

		if (rc != 0) { //(rc == ETIMEDOUT) {
			printf("Waiting for new frame timed out!\n");
			debug_msg("Waiting for new frame timed out!\n");

			// try to restart stream
			/*
			debug_msg("Try to restart DeckLink stream!\n");
			result = s->deckLinkInput->StopStreams();
			if (result != S_OK)
			{
				debug_msg("Could not stop stream: %08x\n", result);
			}
			result = s->deckLinkInput->StartStreams();
			if (result != S_OK)
			{
				debug_msg("Could not start stream: %08x\n", result);
				return NULL; // really end ???
			}
			*/

			if((!s->delegate->first_time) || (should_exit)){
				s->delegate->newFrameReady = 1;
				timeout = 1;
			}else{
				// wait half of timeout
				usleep(s->next_frame_time);
			}
		}
	}

	/* if timeout != 0 then send clear frame */
	if(!should_exit)
	if(!timeout){
		debug_msg("vidcap_decklink_grab - memcpy START\n");
		//memcpy(s->rtp_buffer,s->delegate->pixelFrame,s->buffer_size);
		debug_msg("vidcap_decklink_grab - memcpy END\n");
	}else{
		// clear s->rtp_buffer
		debug_msg("vidcap_decklink_grab - memset START\n");
		// memset(s->rtp_buffer, 0, s->buffer_size);
		debug_msg("vidcap_decklink_grab - memset END\n");
	}
	s->delegate->newFrameReady = 0;

// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	pthread_mutex_unlock(&(s->lock));

	if (s->delegate->pixelFrame != NULL) {
		vf = (struct video_frame *) malloc(sizeof(struct video_frame));
		if (vf != NULL) {
			vf->colour_mode	= YUV_422;
			vf->width				= s->hd_size_x;
			vf->height			= s->hd_size_y;
			vf->data				= (char*) s->delegate->pixelFrame;
			vf->data_len		= s->buffer_size;
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
}

#endif /* HAVE_DECKLINK */
