/**
 * @file video_capture/DirectShowGrabber.cpp
 * @author Pavel Troubil
 * @author Martin Pulec  <martin.pulec@cesnet.cz>
 *
 * Defines the exported functions for the DLL application.
 */

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

#include <tchar.h>
#include <dshow.h>
#include <dvdmedia.h>
#include <qedit.h>
#include <oleauto.h>
//#include <Streams.h>
#include <windows.h>

#include "debug.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/string.h"
#include "utils/windows.h"
#include "video.h"
#include "video_capture.h"

#define MOD_NAME "[dshow] "

EXTERN_C const CLSID CLSID_NullRenderer;
EXTERN_C const CLSID CLSID_SampleGrabber;

typedef void (*convert_t)(int width, int height, const unsigned char *in,
                          unsigned char *out);

using namespace std;

static void DeleteMediaType(AM_MEDIA_TYPE *mediaType);
static const CHAR * GetSubtypeName(const GUID *pSubtype);
static codec_t get_ug_codec(const GUID *pSubtype);
static convert_t get_conversion(const GUID *pSubtype);
static codec_t get_ug_from_subtype_name(const char *subtype_name);
static void vidcap_dshow_probe(device_info **available_cards, int *count, void (**deleter)(void *));

static void ErrorDescription(HRESULT hr)
{ 
        if(FACILITY_WINDOWS == HRESULT_FACILITY(hr)) 
                hr = HRESULT_CODE(hr); 

        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error: %s\n", get_win32_error(hr));
}

#define DEFAULT_DEVNUM 1
#define MAX_STRING_LEN 1000
#define DEFAULT_VIDEO_WIDTH 640
#define DEFAULT_VIDEO_HEIGHT 480
#define DEFAULT_FPS 15

#define MAX_YUV_CLAMP 535
#define MIN_YUV_CLAMP -276

static int *yuv_clamp;

class SampleGrabberCallback;

struct vidcap_dshow_state {
	bool com_initialized;
	int deviceNumber;
	char *deviceName;
	int modeNumber;
	struct video_desc desc;
	bool convert_YUYV_RGB; ///< @todo check - currently newer set
	convert_t convert;

	struct video_frame *frame;
	long grabBufferLen;
	long returnBufferLen;
	bool haveNewReturnBuffer;
	//BYTE *buffer;
	BYTE *grabBuffer;
	BYTE *returnBuffer;
	BYTE *convert_buffer;

	unsigned long frames;
	bool should_exit;

	CRITICAL_SECTION returnBufferCS;
	CONDITION_VARIABLE grabWaitCV;

	SampleGrabberCallback *SGCallback;
	ICaptureGraphBuilder2 *graphBuilder;
	IGraphBuilder *filterGraph;
	IBaseFilter *captureFilter;
	IBaseFilter *sampleGrabberFilter;
	IBaseFilter *nullRenderer;
	ISampleGrabber *sampleGrabber;
	ICreateDevEnum *devEnumerator;
	IEnumMoniker *videoInputEnumerator;
	IMoniker *moniker;
	IAMStreamConfig *streamConfig;
	IMediaControl *mediaControl;
};

class SampleGrabberCallback : public ISampleGrabberCB {
public:
	unsigned long *frameNumber;
	struct vidcap_dshow_state *s;

	// We have to implement IUnknown COM interface, but can make it a little easier
	STDMETHODIMP_(unsigned long) AddRef() { return 2; }
	STDMETHODIMP_(unsigned long) Release() { return 1; }
	STDMETHODIMP QueryInterface(REFIID iface, void **pointer) {
		if (pointer == NULL) return E_POINTER;
		if (iface == IID_IUnknown || iface == IID_ISampleGrabberCB) {
			(*pointer) = this;
			return S_OK;
		}
		*pointer = NULL;
		return E_NOINTERFACE;
	}

	SampleGrabberCallback(struct vidcap_dshow_state *s, unsigned long *frameNumber) :
	frameNumber(frameNumber),
	s(s)
	{
		if (frameNumber != NULL) *frameNumber = 0;
	}
	~SampleGrabberCallback() {}

	STDMETHODIMP SampleCB(double sampleTime, IMediaSample *ms) {
		UNUSED(sampleTime);
		UNUSED(ms);

		return S_OK;
	}

	STDMETHODIMP BufferCB(double sampleTime, BYTE *buffer, long len) {
		UNUSED(sampleTime);
		
		if (len <= 0) return S_OK;

		EnterCriticalSection(&s->returnBufferCS);
                const long req_len = vc_get_datalen(
                    s->desc.width, s->desc.height, s->desc.color_spec);
                if (s->grabBufferLen != req_len) {
                        s->grabBuffer = (BYTE *) realloc((void *) s->grabBuffer, req_len);
			if (s->grabBuffer == NULL) {
				s->grabBufferLen = 0;
				return S_OK;
			}
			s->grabBufferLen = len;
		}

		// We need to make a copy, DirectShow will do something with the data
		if (s->convert != nullptr) {
			s->convert(s->desc.width, s->desc.height, buffer, s->grabBuffer);
		} else {
			memcpy((char *) s->grabBuffer, (char *) buffer, len);
		}

		bool grabMightWait = false;
		if (!s->haveNewReturnBuffer) {
			grabMightWait = true;
			s->haveNewReturnBuffer = true;
		}
		LeaveCriticalSection(&s->returnBufferCS);
		if (grabMightWait) {
			LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "WakeConditionVariable s->grabWaitCV\n";
			WakeConditionVariable(&s->grabWaitCV);
		}

		return S_OK;
	}
};

static bool cleanup(struct vidcap_dshow_state *s) {
	if (s->mediaControl != NULL) s->mediaControl->Release();
	log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Released: mediaControl\n");
	if (s->nullRenderer != NULL) s->nullRenderer->Release();
	log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Released: nullRenderer\n");
	if (s->captureFilter != NULL) s->captureFilter->Release();
	log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Released: captureFilter\n");
	if (s->sampleGrabberFilter != NULL) s->sampleGrabberFilter->Release();
	log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Released: sampleGrabberFilter\n");
	if (s->moniker != NULL) s->moniker->Release();
	log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Released: moniker\n");
	/*
	if (s->videoInputEnumerator != NULL) s->videoInputEnumerator->Release();
	fprintf(stderr, "[dshow] Released: videoInputEnum\n");
	if (s->devEnumerator != NULL) s->devEnumerator->Release();
	fprintf(stderr, "[dshow] Released: devEnum\n");
	if (s->filterGraph != NULL) s->filterGraph->Release();
	fprintf(stderr, "[dshow] Released: filterGraph\n");
	if (s->graphBuilder != NULL) s->graphBuilder->Release();
	fprintf(stderr, "[dshow] Released: graphBuilder\n");
	*/

	// COM library uninitialization
	com_uninitialize(&s->com_initialized);

	if (s->frame != NULL) vf_free(s->frame);
	if (s->grabBuffer != NULL) free(s->grabBuffer);
	if (s->returnBuffer != NULL) free(s->returnBuffer);
	if (s->convert_buffer != NULL) free(s->convert_buffer);
	if (yuv_clamp != NULL) free(yuv_clamp);
	free(s->deviceName);

	DeleteCriticalSection(&s->returnBufferCS);

	free(s);

	return true;
}

#define HANDLE_ERR_ACTION(res, action, msg, ...) \
        if (res != S_OK) { \
                MSG(ERROR, msg ": %s\n", \
                    __VA_ARGS__ __VA_OPT__(, ) hresult_to_str(res)); \
                action; \
        } else

static bool common_init(struct vidcap_dshow_state *s) {
#define HANDLE_ERR(msg, ...) \
        HANDLE_ERR_ACTION(res, goto error, \
                          "vidcap_dshow_init: " msg __VA_OPT__(, ) \
                              __VA_ARGS__)
	// set defaults
	s->deviceNumber = 1;
	s->modeNumber = 0;
	s->desc.width = DEFAULT_VIDEO_WIDTH;
	s->desc.height = DEFAULT_VIDEO_HEIGHT;
	s->desc.fps = DEFAULT_FPS;
	s->desc.tile_count = 1;
	s->desc.interlacing = PROGRESSIVE;

	s->frame = NULL;
	s->grabBufferLen = 0;
	s->returnBufferLen = 0;
	s->grabBuffer = NULL;
	s->returnBuffer = NULL;
	s->convert_buffer = NULL;
	s->convert_YUYV_RGB = false;
	yuv_clamp = NULL;

	s->graphBuilder = NULL;
	s->filterGraph = NULL;
	s->devEnumerator = NULL;
	s->videoInputEnumerator = NULL;
	s->moniker = NULL;
	s->captureFilter = NULL;
	s->nullRenderer = NULL;
	s->sampleGrabberFilter = NULL;
	s->streamConfig = NULL;
	s->mediaControl = NULL;

	HRESULT res;

	// Initialize COM library
	// COINIT_APARTMENTTHREADED is used because we do not expect any other thread to work with the object
	if (!com_initialize(&s->com_initialized, "widcap_dshow_init: ")) {
		return false;
	}

	// Create non-inheritable mutex without a name, owned by this thread
	InitializeCriticalSectionAndSpinCount(&s->returnBufferCS, 0x40);
	s->haveNewReturnBuffer = false;

	s->frames = 0;

	// create device enumerator
	res = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&s->devEnumerator));
        HANDLE_ERR("Cannot create System Device Enumerator");

        res = s->devEnumerator->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &s->videoInputEnumerator, 0);
	if (res == S_FALSE) {
		MSG(ERROR, "no devices found\n");
		goto error;
	}
        HANDLE_ERR("Cannot create Video Input Device enumerator");

	// Media processing classes (filters) are conected to a graph.
	// Create graph builder -- helper class for connecting of the graph
	res = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER,
			IID_ICaptureGraphBuilder2, (void **) &s->graphBuilder);
        HANDLE_ERR("Cannot create instance of Capture Graph Builder 2");

	// create the filter graph
	res = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER,
			IID_IGraphBuilder, (void **) &s->filterGraph);
        HANDLE_ERR("Cannot create instance of Filter Graph");

	// specify the graph builder a graph to be built
	res = s->graphBuilder->SetFiltergraph(s->filterGraph);
        HANDLE_ERR("Cannot attach Filter Graph to Graph Builder");

	return true;

error:
	cleanup(s);
	return false;
#undef HANDLE_ERR
}

static struct video_desc vidcap_dshow_get_video_desc(AM_MEDIA_TYPE *mediaType)
{
	BITMAPINFOHEADER *bmiHeader;
	struct video_desc desc{};
	if (mediaType->formattype != FORMAT_VideoInfo && mediaType->formattype != FORMAT_VideoInfo2) {
		LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unsupported format type!\n";
		return desc;
	}

	desc.color_spec = get_ug_codec(&mediaType->subtype);
	desc.tile_count = 1;
	desc.interlacing = PROGRESSIVE;
	if (mediaType->formattype == FORMAT_VideoInfo) {
		VIDEOINFOHEADER *infoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mediaType->pbFormat);
		bmiHeader = &infoHeader->bmiHeader;
		desc.fps = 10000000.0/infoHeader->AvgTimePerFrame;
	} else {
		VIDEOINFOHEADER2 *infoHeader = reinterpret_cast<VIDEOINFOHEADER2*>(mediaType->pbFormat);
		bmiHeader = &infoHeader->bmiHeader;
		desc.fps = 10000000.0/infoHeader->AvgTimePerFrame;
		if (infoHeader->dwInterlaceFlags & AMINTERLACE_IsInterlaced) {
			if (infoHeader->dwInterlaceFlags & AMINTERLACE_1FieldPerSample) {
				LOG(LOG_LEVEL_WARNING) << MOD_NAME "1 Field Per Sample is not supported! " BUG_MSG "\n";
			} else {
				desc.interlacing = INTERLACED_MERGED;
			}
		}
	}
	desc.width = bmiHeader->biWidth;
	desc.height = bmiHeader->biHeight;

	return desc;
}

static void show_help(struct vidcap_dshow_state *s) {
	printf("dshow grabber options:\n");
	col() << SBOLD(SRED("\t-t dshow") << "[:device=<DeviceNumber>|<DeviceName>][:mode=<ModeNumber>][:RGB]") "\n";
	col() << "\t    Flag " << SBOLD("RGB") << " forces use of RGB codec, otherwise native is used if possible.\n";
	printf("\tor\n");
	col() << SBOLD(SRED("\t-t dshow:[Device]<DeviceNumber>:<codec>:<width>:<height>:<fps>")) "\n\n";

	if (!common_init(s)) return;

        color_printf("Devices:\n");
        device_info *cards = nullptr;
        int count = 0;
        void (*deleter)(void *) = NULL;
        vidcap_dshow_probe(&cards, &count, &deleter);

	// Enumerate all capture devices
	for (int n = 0; n < count; ++n) {
		color_printf("Device %d) " TERM_BOLD "%s\n" TERM_RESET, n + 1, cards[n].name);

		int i = 0;
		// iterate over all capabilities
		while (strlen(cards[n].modes[i].id) > 0) {
			printf("    Mode %2d: %s", i, cards[n].modes[i].name);
			putchar(i % 2 == 1 ? '\n' : '\t');
			++i;
		}

		printf("\n\n");
	}
	deleter = IF_NOT_NULL_ELSE(deleter, (void (*)(void *)) free);
	deleter(cards);

	printf("Mode flags:\n");
        printf("C - codec not natively supported by UG; F - video format is "
               "not supported\n\n");
}

static string
get_friendly_name(IMoniker *moniker, int idx = -1)
{
#define HANDLE_ERR(msg, ...) \
        HANDLE_ERR_ACTION(res, return {}, "%s: " msg, \
                          __func__ __VA_OPT__(, ) __VA_ARGS__)
        IPropertyBag *properties = nullptr;
        const string device_id = idx == -1 ? "" : to_string(idx) + " ";
        // Attach structure for reading basic device properties
        HRESULT res = moniker->BindToStorage(0, 0, IID_PPV_ARGS(&properties));
        HANDLE_ERR("Failed to read device %s properties", device_id.c_str());

        VARIANT var;
        VariantInit(&var);
        res = properties->Read(L"FriendlyName", &var, NULL);
        HANDLE_ERR("Failed to read device %s FriendlyName", device_id.c_str());

        char buf[MAX_STRING_LEN];
        // convert to standard C string
        wcstombs(buf, var.bstrVal, sizeof buf);

        // clean up structures
        VariantClear(&var);
        properties->Release();

        return buf;
#undef HANDLE_ERR
}

static void vidcap_dshow_probe(device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
	struct vidcap_dshow_state *s = (struct vidcap_dshow_state *) calloc(1, sizeof(struct vidcap_dshow_state));

	if (!common_init(s))
                return;

        device_info *cards = nullptr;
        int card_count = 0;

	HRESULT res;
	int n = 0;
	// Enumerate all capture devices
	while ((res = s->videoInputEnumerator->Next(1, &s->moniker, NULL)) == S_OK) {
		n++;
		card_count = n;
		cards = (struct device_info *) realloc(cards, n * sizeof(struct device_info));
		memset(&cards[n - 1], 0, sizeof(struct device_info));
		snprintf(cards[n-1].dev, sizeof cards[n-1].dev - 1, ":device=%d", n);
		snprintf(cards[n-1].name, sizeof cards[n-1].name - 1, "_DSHOW_FAILED_TO_READ_NAME_%d_", n);

		string friendly_name = get_friendly_name(s->moniker, n);
		if (friendly_name.empty()) {
			log_msg(LOG_LEVEL_WARNING, MOD_NAME "vidcap_dshow_help: Failed to get device %d name.\n", n);
			// Ignore the device
			continue;
		}
		const char *name = friendly_name.c_str();;
                snprintf(cards[n - 1].name, sizeof cards[n - 1].name, "%s",
                         name);

#define HANDLE_ERR(msg, ...) \
        HANDLE_ERR_ACTION(res, continue, "vidcap_dshow_help: %s: " msg, \
                             name __VA_OPT__(, ) __VA_ARGS__)
                // bind the selected device to the capture filter
                IBaseFilter *captureFilter;
                res = s->moniker->BindToObject(NULL, NULL, IID_IBaseFilter, (void **) &captureFilter);
                HANDLE_ERR("Cannot bind capture filter to device");

                // add the capture filter to the filter graph
                res = s->filterGraph->AddFilter(captureFilter, L"Capture filter");
                HANDLE_ERR("Cannot add capture filter to filter graph");

                // connect stream config interface to the capture filter
                res = s->graphBuilder->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, captureFilter,
                                IID_IAMStreamConfig, (void **) &s->streamConfig);
		HANDLE_ERR("Cannot find interface for reading capture capabilites");

                int capCount, capSize;
                // read number of capture device capabilities
                res = s->streamConfig->GetNumberOfCapabilities(&capCount, &capSize);
                HANDLE_ERR("Cannot read number of capture capabilites");
                // check if the format of capture capabilities is the right one
                if (capSize != sizeof(VIDEO_STREAM_CONFIG_CAPS)) {
			log_msg(LOG_LEVEL_WARNING, MOD_NAME "vidcap_dshow_help: %s: Unknown format of capture capabilites.\n", name);
                        continue;
                }

                // iterate over all capabilities
                for (int i = 0; i < capCount; i++) {
			if (i >= (int) (sizeof cards[card_count - 1].modes /
						sizeof cards[card_count - 1].modes[0])) { // no space
				break;
			}

                        AM_MEDIA_TYPE *mediaType;
                        VIDEO_STREAM_CONFIG_CAPS streamCaps;

                        res = s->streamConfig->GetStreamCaps(i, &mediaType, (BYTE*) &streamCaps);
                        HANDLE_ERR("Cannot read stream capabilities #%d", i);

                        struct video_desc desc = vidcap_dshow_get_video_desc(mediaType);
                        if (desc.width == 0) {
                                continue;
                        }

			snprintf(cards[card_count - 1].modes[i].id,
					sizeof cards[card_count - 1].modes[i].id,
					"{\"mode\":\"%d\"}", i);
			snprintf(cards[card_count - 1].modes[i].name,
					sizeof cards[card_count - 1].modes[i].name,
					"%s %ux%u @%0.2lf%s %s%s", GetSubtypeName(&mediaType->subtype),
					desc.width, desc.height, desc.fps * (desc.interlacing == INTERLACED_MERGED ? 2 : 1), get_interlacing_suffix(desc.interlacing),
					desc.color_spec ? "" : "C",
					mediaType->formattype == FORMAT_VideoInfo ? "" : "F");

                        DeleteMediaType(mediaType);
                }

                s->streamConfig->Release();
                res = s ->filterGraph->RemoveFilter(captureFilter);
                HANDLE_ERR("Cannot remove capture filter from filter graph");
                captureFilter->Release();
                s->moniker->Release();
	}
	cleanup(s);
        *available_cards = cards;
        *count = card_count;
#undef HANDLE_ERR
}

static bool process_args(struct vidcap_dshow_state *s, char *init_fmt) {
	char *token;
	char *strtok_context;
	int i = 1;

	if (strchr(init_fmt, '=') == NULL) { // positional arguments
		while ((token = strtok_s(init_fmt, ":", &strtok_context)) != NULL) {
			init_fmt = NULL;
			switch (i) {
			case 1 :
				if (strstr(token, "Device") == token) token = token + strlen("Device");
				if (isdigit(token[0])) { // device specified by number
					s->deviceNumber = atoi(token);
				} else { // device specified by name
					s->deviceName = (char *) malloc(sizeof(char) * (strlen(token) + 100));
					if (s->deviceName == NULL) return false;
					strcpy_s(s->deviceName, strlen(token) + 1, token);
					s->deviceNumber = -1;
				}
				break;
			case 2 :
				if (strstr(token, "Mode") == token) token = token + strlen("Mode");
				if (isdigit(token[0])) {
					s->modeNumber = atoi(token);
				} else {
					s->modeNumber = -1;
					s->desc.color_spec = get_codec_from_name(token);
					if (s->desc.color_spec == VIDEO_CODEC_NONE) { // try Win subtype name
						s->desc.color_spec = get_ug_from_subtype_name(token);
					}
					if (s->desc.color_spec == VIDEO_CODEC_NONE) {
						log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported video format: %s. "
                                                                "Please contact us via %s if you need support for this codec.\n",
                                                                token, PACKAGE_BUGREPORT);
						return false;
					}
				}
				break;
			case 3 :
				if (s->modeNumber != -1) {
					if (strcmp(token, "RGB") == 0) s->desc.color_spec = BGR;
					else {
						log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown parameter: %s.\n", token);
						return false;
					}
					break;
				}
				s->desc.width = atoi(token);
				if (s->desc.width <= 0) {
					log_msg(LOG_LEVEL_ERROR, MOD_NAME "Invalid video width parameter: %s.\n", token);
					return false;
				}
				break;
			case 4 :
				if (s->modeNumber != -1) break;
				s->desc.height = atoi(token);
				if (s->desc.height <= 0) {
					log_msg(LOG_LEVEL_ERROR, MOD_NAME "Invalid video height parameter: %s.\n", token);
					return false;
				}
				break;
			case 5 :
				if (s->modeNumber != -1) break;
				s->desc.fps = atoi(token);
				if (s->desc.fps <= 0) {
					log_msg(LOG_LEVEL_ERROR, MOD_NAME "Invalid video fps parameter: %s.\n", token);
					return false;
				}
				break;
			default :
				log_msg(LOG_LEVEL_ERROR, MOD_NAME "More arguments than expected, ignoring.\n");
				break;
			}
			i++;
		}
	} else {
		while ((token = strtok_s(init_fmt, ":", &strtok_context)) != NULL) {
			init_fmt = NULL;
			if (IS_KEY_PREFIX(token, "device")) {
				token  = strchr(token, '=') + 1;
				if (isdigit(token[0])) { // device specified by number
					s->deviceNumber = atoi(token);
				} else { // device specified by name
					s->deviceName = (char *) malloc(sizeof(char) * (strlen(token) + 100));
					strcpy_s(s->deviceName, strlen(token) + 1, token);
					s->deviceNumber = -1;
				}
			} else if (IS_KEY_PREFIX(token, "mode")) {
				token  = strchr(token, '=') + 1;
				s->modeNumber = atoi(token);
			} else if (strcmp(token, "RGB") == 0) {
				s->desc.color_spec = BGR;
			} else {
				MSG(ERROR, "Unknown argument: %s\n", token);
				return false;
			}
		}
	}

	return true;
}

static HRESULT PinIsConnected(IPin *pin, bool *result) {
	IPin *connectedPin;

	HRESULT res = pin->ConnectedTo(&connectedPin);
	if (res == S_OK) {
		*result = true;
	} else if (res == VFW_E_NOT_CONNECTED) {
		res = S_OK;
		*result = false;
	}

	if (connectedPin != NULL) connectedPin->Release();
	return res;
}

static HRESULT PinHasDirection(IPin *pin, PIN_DIRECTION direction, bool *result) {
	PIN_DIRECTION pinDir;

	HRESULT res = pin->QueryDirection(&pinDir);
	if (res == S_OK) {
		*result = (pinDir == direction);
	}

	return res;
}

static HRESULT FindUnconnectedPin(IBaseFilter *filter, PIN_DIRECTION direction, IPin **pin) {
	IEnumPins *pinEnum = NULL;
	IPin *filterPin = NULL;
	bool pinFound;

	HRESULT res = filter->EnumPins(&pinEnum);
	if (res != S_OK) {
		goto error;
	}

	pinFound = false;
	while ((res = pinEnum->Next(1, &filterPin, NULL)) == S_OK) {
		bool isConnected;
		res = PinIsConnected(filterPin, &isConnected);
		if (res == S_OK) {
			if (isConnected) {
				continue;
			}
			bool hasDirection;
			res = PinHasDirection(filterPin, direction, &hasDirection);
			if (res == S_OK) {
				if (!hasDirection) continue;

				// We found the pin!
				*pin = filterPin;
				(*pin)->AddRef();
				pinFound = true;
				break;
			} else {
				goto error;
			}
		} else {
			goto error;
		}
		if (filterPin != NULL) filterPin->Release();
	}

	if (!pinFound) {
		res = VFW_E_NOT_FOUND;
	}

error:
	if (pinEnum != NULL) pinEnum->Release();
	if (filterPin != NULL) filterPin->Release();
	return res;
}

static HRESULT ConnectFilters(IGraphBuilder *g, IPin *fromPin, IBaseFilter *toFilter) {
	IPin *toPin = NULL;

	HRESULT res = FindUnconnectedPin(toFilter, PINDIR_INPUT, &toPin);
	if (res == S_OK) {
		res = g->Connect(fromPin, toPin);
		toPin->Release();
	}

	return res;
}

static HRESULT ConnectFilters(IGraphBuilder *g, IBaseFilter *fromFilter, IBaseFilter *toFilter) {
	IPin *fromPin = NULL;

	HRESULT res = FindUnconnectedPin(fromFilter, PINDIR_OUTPUT, &fromPin);
	if (res == S_OK) {
		res = ConnectFilters(g, fromPin, toFilter);
		fromPin->Release();
	}

	return res;
}

[[maybe_unused]] static HRESULT GraphRun(IMediaControl *mc) {
	HRESULT res;

	if ((res = mc->Run()) == S_FALSE) {
		FILTER_STATE fs;

		while ((res = mc->GetState(500, (OAFilterState*) &fs)) != VFW_S_CANT_CUE && !(res == S_OK && fs == State_Running)) {
			if (res != VFW_S_STATE_INTERMEDIATE && res != S_OK) {
				break;
			}
		}
	}

	if (res != S_OK) {
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot run filter graph.\n");
		ErrorDescription(res);
	}

	return res;
}

[[maybe_unused]] static HRESULT GraphPause(IMediaControl *mc) {
	HRESULT res;

	if ((res = mc->Pause()) == S_FALSE) {
		FILTER_STATE fs;

		while ((res = mc->GetState(500, (OAFilterState*) &fs)) != VFW_S_CANT_CUE && !(res == S_OK && fs == State_Paused)) {
			if (res != VFW_S_STATE_INTERMEDIATE && res != S_OK) break;
		}
	}

	if (res != S_OK) {
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot pause filter graph.\n");
		ErrorDescription(res);
	}

	return res;
}

[[maybe_unused]] static HRESULT GetPinCategory(IPin *pPin, GUID *pPinCategory)
{
    IKsPropertySet *pKs = NULL;

    HRESULT hr = pPin->QueryInterface(IID_PPV_ARGS(&pKs));
    if (FAILED(hr))
    {
        return hr;
    }

    // Try to retrieve the pin category.
    DWORD cbReturned = 0;
    hr = pKs->Get(AMPROPSETID_Pin, AMPROPERTY_PIN_CATEGORY, NULL, 0, 
        pPinCategory, sizeof(GUID), &cbReturned);
    
    // If this succeeded, pPinCategory now contains the category GUID.

	if (pKs != NULL) pKs->Release();
    return hr;
}

static void vidcap_dshow_should_exit(void *state) {
	auto *s = static_cast<vidcap_dshow_state *>(state);
	EnterCriticalSection(&s->returnBufferCS);
	s->should_exit = true;
	LeaveCriticalSection(&s->returnBufferCS);
	WakeConditionVariable(&s->grabWaitCV);
}

static int vidcap_dshow_init(struct vidcap_params *params, void **state) {
#define HANDLE_ERR(msg, ...) \
        HANDLE_ERR_ACTION(res, goto error, \
                          "vidcap_dshow_init: " msg __VA_OPT__(, ) __VA_ARGS__)
        struct vidcap_dshow_state *s;
        HRESULT res;

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }

	s = (struct vidcap_dshow_state *) calloc(1, sizeof(struct vidcap_dshow_state));
	if (s == NULL) {
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "vidcap_dshow_init: memory allocation error\n");
		return VIDCAP_INIT_FAIL;
	}

	InitializeConditionVariable(&s->grabWaitCV);
	InitializeCriticalSection(&s->returnBufferCS);

	if (vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "help") == 0) {
		show_help(s); 
		cleanup(s);
		return VIDCAP_INIT_NOERR;
	}

	if (!common_init(s)) {
		goto error;
	}

	if (vidcap_params_get_fmt(params) != NULL) {
                char *init_fmt = strdup(vidcap_params_get_fmt(params));
		if (!process_args(s, init_fmt)) goto error;
                free(init_fmt);
	}

	// Select video capture device
	if (s->deviceNumber != -1) { // Device was specified by number
		res = E_FAIL;
		for (int i = 1; i <= s->deviceNumber; i++) {
			// Take one device. We could take more at once, but it would require allocation of more moniker objects
			res = s->videoInputEnumerator->Next(1, &s->moniker, NULL);
                        if(i != s->deviceNumber) {
                                if (s->moniker)
                                        s->moniker->Release();
                                s->moniker = NULL;
                        }
			if (res != S_OK) break;
		}
		if (res != S_OK) {
			log_msg(LOG_LEVEL_ERROR, MOD_NAME "vidcap_dshow_init: Device number %d was not found.\n", s->deviceNumber);
			goto error;
		}
	} else { // device specified by name
		while ((res = s->videoInputEnumerator->Next(1, &s->moniker, NULL)) == S_OK) {
			string friendly_name = get_friendly_name(s->moniker);
			if (strcasecmp(s->deviceName, friendly_name.c_str()) == 0) {
				break;
			}
			s->moniker->Release();
		}

		if (res != S_OK) {
			log_msg(LOG_LEVEL_ERROR, MOD_NAME "vidcap_dshow_init: Device named %s was not found.\n", s->deviceName);
			goto error;
		}
	}

        LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Capturing from device: "
                              << get_friendly_name(s->moniker) << "\n";

	res = s->moniker->BindToObject(NULL, NULL, IID_IBaseFilter, (void **) &s->captureFilter);
        HANDLE_ERR("Cannot bind capture filter to device");

	res = s->filterGraph->AddFilter(s->captureFilter, L"Capture filter");
	HANDLE_ERR("Cannot add capture filter to filter graph");

	res = s->graphBuilder->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, s->captureFilter,
			IID_IAMStreamConfig, (void **) &s->streamConfig);
	HANDLE_ERR("Cannot find interface for reading capture capabilites");

	// create instance of sample grabber
	res = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&s->sampleGrabberFilter));
	HANDLE_ERR("Cannot create instance of sample grabber");

	// add the sample grabber to filter graph
	res = s->filterGraph->AddFilter(s->sampleGrabberFilter, L"Sample Grabber");
	HANDLE_ERR("Cannot add sample grabber to filter graph");

	// query the sample grabber filter for control interface
	res = s->sampleGrabberFilter->QueryInterface(IID_PPV_ARGS(&s->sampleGrabber));
	HANDLE_ERR("Cannot query sample grabber filter for control interface");

	// make the sample grabber buffer frames
	res = s->sampleGrabber->SetBufferSamples(TRUE);
	HANDLE_ERR("Cannot set sample grabber to buffer samples");

	// set media type for sample grabber; this is not done a very detailed setup, because it would be unneccessarily complicated to do so
	AM_MEDIA_TYPE sampleGrabberMT;
	ZeroMemory(&sampleGrabberMT, sizeof(sampleGrabberMT));
	sampleGrabberMT.majortype = MEDIATYPE_Video;
	sampleGrabberMT.subtype = MEDIASUBTYPE_RGB24;
	res = s->sampleGrabber->SetMediaType(&sampleGrabberMT);
	HANDLE_ERR("Cannot setup media type of grabber filter");

	int capCount, capSize;
	res = s->streamConfig->GetNumberOfCapabilities(&capCount, &capSize);
	HANDLE_ERR("Cannot read number of capture capabilites");
	if (capSize != sizeof(VIDEO_STREAM_CONFIG_CAPS)) {
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "vidcap_dshow_init: Unknown format of capture capabilites.\n");
		goto error;
	}

	bool format_found;
	format_found = false;
	AM_MEDIA_TYPE *mediaType;
	VIDEO_STREAM_CONFIG_CAPS streamCaps;

	if (s->modeNumber >= 0) { // mode number was set by user directly
		res = s->streamConfig->GetStreamCaps(s->modeNumber, &mediaType, (BYTE*) &streamCaps);
		// Invalid mode selected
		if (res == S_FALSE || res == E_INVALIDARG) {
			log_msg(LOG_LEVEL_ERROR, MOD_NAME "Invalid mode index!\n");
			goto error;
		}
		// Some other error occured
                HANDLE_ERR(
                    "Cannot read stream capabilities #%d (index is correct)",
                    s->modeNumber);

                assert(s->desc.color_spec == VC_NONE || s->desc.color_spec == BGR);
                if (s->desc.color_spec == VC_NONE) {
                        s->desc.color_spec = get_ug_codec(&mediaType->subtype);
                        if (s->desc.color_spec == VC_NONE) {
                                MSG(WARNING,
                                    "Pixel format %.4s not supported directly, "
                                    "capturing BGR.\n",
                                    (char *) &mediaType->subtype.Data1);
                                s->desc.color_spec = BGR;
                        } else {
                                res = s->sampleGrabber->SetMediaType(mediaType);
                                HANDLE_ERR("Cannot setup media type "
                                           "of grabber filter");
                        }
                }

		struct video_desc desc = vidcap_dshow_get_video_desc(mediaType);
		s->desc.width = desc.width;
		s->desc.height = desc.height;
		s->desc.fps = desc.fps;
		s->desc.interlacing = desc.interlacing;

		format_found = true;
	} else {
		for (int i = 0; i < capCount; i++) {
			res = s->streamConfig->GetStreamCaps(i, &mediaType, (BYTE*) &streamCaps);
			if (res != S_OK) {
				log_msg(LOG_LEVEL_WARNING, MOD_NAME "vidcap_dshow_help: Cannot read stream capabilities #%d.\n", i);
				continue;
			}
                        if (s->desc.color_spec !=
                            get_ug_codec(&mediaType->subtype)) {
                                continue;
                        }

                        struct video_desc desc = vidcap_dshow_get_video_desc(mediaType);
			if (desc.height == s->desc.height && desc.width  == s->desc.width) {
                                format_found = true;
                                res = s->sampleGrabber->SetMediaType(mediaType);
                                HANDLE_ERR("Cannot setup media type "
                                           "of grabber filter");
                                break;
                        }

			DeleteMediaType(mediaType);
		}
	}

	if (!format_found) {
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "vidcap_dshow_init: Requested format not supported by the device. Quitting.\n");
		goto error;
	}

#if 0
	if (s->modeNumber < 0) { // mode number was not set by user directly
                s->streamConfig->GetFormat(&mediaType);
                HANDLE_ERR("Cannot get current capture format");
                switch (s->desc.color_spec) {
                        case BGR : mediaType->subtype = MEDIASUBTYPE_RGB24;
                                   break;
                        case YUYV : mediaType->subtype = MEDIASUBTYPE_YUY2;
                                    break;
			default:
				// this is directly from parse_fmt, where, only 2 formats above are permissible
				abort();
                }
                VIDEOINFOHEADER *infoHeader;
                infoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mediaType->pbFormat);
                infoHeader->rcSource.bottom = s->desc.height;
                infoHeader->rcSource.right = s->desc.width;
                infoHeader->AvgTimePerFrame = (REFERENCE_TIME) (1e7 / s->desc.fps);
        }
#endif
	res = s->streamConfig->SetFormat(mediaType);
	HANDLE_ERR("Cannot set capture format");

	if (s->convert_YUYV_RGB) {
		s->convert_buffer = (BYTE *) malloc(s->desc.height * s->desc.width * 3);
		if (s->convert_buffer == NULL) {
			log_msg(LOG_LEVEL_ERROR, MOD_NAME "vidcap_dshow_init: memory allocation error\n");
			goto error;
		}

		yuv_clamp = (int *) malloc((MAX_YUV_CLAMP - MIN_YUV_CLAMP + 1) * sizeof(int));
		int i;
		for (i = 0; i + MIN_YUV_CLAMP <= 0; i++) {
			yuv_clamp[i] = 0;
		}
		for (int j = 1; j <= 254; j++, i++) {
			yuv_clamp[i] = j;
		}
		for ( ; i < MAX_YUV_CLAMP - MIN_YUV_CLAMP; i++) {
			yuv_clamp[i] = 255;
		}
	}

	// Create null renderer discarding all incoming frames
	res = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter,
			(void **) &s->nullRenderer);
        HANDLE_ERR("Cannot create NullRenderer");

	res = s->filterGraph->AddFilter(s->nullRenderer, L"NullRenderer");
        HANDLE_ERR("Cannot add null renderer to filter graph");

	IEnumPins *pinEnum;
	IPin *pin;

	res = s->captureFilter->EnumPins(&pinEnum);
        HANDLE_ERR("Error enumerating pins of capture filter");

	while (pinEnum->Next(1, &pin, NULL) == S_OK) {
		res = ConnectFilters(s->filterGraph, pin, s->sampleGrabberFilter);
		if (pin != NULL) pin->Release();
		if (res == S_OK) {
			break;
		}
	}
	HANDLE_ERR("Cannot connect capture filter to sample grabber");

        res = s->sampleGrabber->GetConnectedMediaType(&sampleGrabberMT);
        HANDLE_ERR("Cannot get current grabber format");
        MSG(INFO, "streaming type: %s, grabber type: %s, output: %s\n",
            GetSubtypeName(&mediaType->subtype),
            GetSubtypeName(&sampleGrabberMT.subtype),
            ((string) vidcap_dshow_get_video_desc(&sampleGrabberMT)).c_str());
        s->convert = get_conversion(&sampleGrabberMT.subtype);
        DeleteMediaType(mediaType);

	res = ConnectFilters(s->filterGraph, s->sampleGrabberFilter, s->nullRenderer);
	HANDLE_ERR("Cannot connect sample grabber to null renderer");

	s->SGCallback = new SampleGrabberCallback(s, &s->frames);
	s->sampleGrabber->SetCallback(s->SGCallback, 1);

	/*
	res = s->graphBuilder->RenderStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, s->captureFilter,
			NULL, s->nullRenderer);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot render stream.\n");
		goto error;
	}
	*/

	res = s->filterGraph->QueryInterface(IID_IMediaControl, (void **) &s->mediaControl);
        HANDLE_ERR("Cannot find media control interface");

	if(get_friendly_name(s->moniker).find("OBS Virtual") != std::string::npos){
		IMediaFilter *pMediaFilter;
		res = s->filterGraph->QueryInterface(IID_IMediaFilter, (void **) &pMediaFilter);
		HANDLE_ERR("Cannot find media filter interface");

		log_msg(LOG_LEVEL_WARNING, MOD_NAME "OBS virtual camera detected. Setting sync source to NULL!\n");
		pMediaFilter->SetSyncSource(NULL);
	}

	FILTER_STATE fs;
	res = s->mediaControl->Run();
	while ((res = s->mediaControl->GetState(500, (OAFilterState*) &fs)) != VFW_S_CANT_CUE && !(res == S_OK && fs == State_Running)) {
			if (res != VFW_S_STATE_INTERMEDIATE && res != S_OK) {
				break;
			}
	}
        HANDLE_ERR("Cannot run filter graph");
	res = s->sampleGrabberFilter->GetState(INFINITE, &fs);
        HANDLE_ERR("filter getstate error");

	s->frame = vf_alloc_desc(s->desc);
	register_should_exit_callback(vidcap_params_get_parent(params), vidcap_dshow_should_exit, s);

	*state = s;
	return VIDCAP_INIT_OK;

error:
	cleanup(s);
	return VIDCAP_INIT_FAIL;
#undef HANDLE_ERR
}

static void vidcap_dshow_done(void *state) {
	struct vidcap_dshow_state *s = (struct vidcap_dshow_state *) state;

	HRESULT res = s->mediaControl->Stop();
	if (res != S_OK) {
		log_msg(LOG_LEVEL_WARNING, MOD_NAME "vidcap_dshow_init: Failed to stop filter graph.\n");
	}

	cleanup(s);
}

static inline void convert_yuv_rgb(BYTE y, BYTE u, BYTE v, BYTE *dst) {
	int yy = (y - 16) * 298;
	int uu = u - 128;
	int vv = v - 128;

	dst[0] = yuv_clamp[(yy + uu * 516 + 128) >> 8];
	dst[1] = yuv_clamp[(yy - uu * 100 - vv * 208 + 128) >> 8];
	dst[2] = yuv_clamp[(yy + 409 * vv + 128) >> 8];
}

static void convert_yuyv_rgb(BYTE *src, BYTE *dst, int input_len) {
	BYTE *s = src;
	//BYTE *d = dst;

	for (int i = 0; i < input_len; i += 4) {
		convert_yuv_rgb(s[0], s[1], s[3], dst);
		convert_yuv_rgb(s[2], s[1], s[3], dst + 3);
		s += 4;
		dst += 6;
	}
}

static struct video_frame * vidcap_dshow_grab(void *state, struct audio_frame **audio) {
	struct vidcap_dshow_state *s = (struct vidcap_dshow_state *) state;
	*audio = NULL;

	LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "GRAB: enter: " << s->deviceNumber << "\n";
	EnterCriticalSection(&s->returnBufferCS);
	//fprintf(stderr, "[dshow] s: %p\n", s);
	while (!s->haveNewReturnBuffer && !s->should_exit) {
		LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "Wait CV\n";
		SleepConditionVariableCS(&s->grabWaitCV, &s->returnBufferCS, INFINITE);
		//fprintf(stderr, "[dshow] s: %p\n", s);
	}

	if (s->should_exit) {
		LeaveCriticalSection(&s->returnBufferCS);
		return nullptr;
	}

	LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "Swap buffers\n";
	//fprintf(stderr, "[dshow] s: %p\n", s);
	// switch the buffers
	BYTE *tmp = s->returnBuffer;
	s->returnBuffer = s->grabBuffer;
	s->grabBuffer = tmp;
	unsigned long tmplen = s->returnBufferLen;
	s->returnBufferLen = s->grabBufferLen;
	s->grabBufferLen = tmplen;
	s->haveNewReturnBuffer = false;
	//fprintf(stderr, "[dshow] s: %p\n", s);

	LeaveCriticalSection(&s->returnBufferCS);

	//fprintf(stderr, "[dshow] GRAB: CS leave: %d\n", s->deviceNumber);
	//fprintf(stderr, "[dshow] s: %p\n", s);


	if (s->convert_YUYV_RGB) {
		convert_yuyv_rgb(s->returnBuffer, s->convert_buffer, s->returnBufferLen);
	}
	//fprintf(stderr, "[dshow] s: %p\n", s);

	s->frame->tiles[0].data = (char *) s->returnBuffer;
	//fprintf(stderr, "[dshow] s: %p\n", s);
	//s->tile->data_len = s->width * s->height * 3;
	s->frame->tiles[0].data_len = is_codec_opaque(s->frame->color_spec) ? s->returnBufferLen :
		vc_get_datalen(s->frame->tiles[0].width, s->frame->tiles[0].height, s->frame->color_spec);

/*
	fprintf(stderr, "[dshow] s5: %p\n", s);
	fprintf(stderr, "[dshow] s6: %p\n", s);
	fprintf(stderr, "[dshow] s7: %p\n", s);
	fprintf(stderr, "[dshow] s: %p\n", s);
	fprintf(stderr, "[dshow] %lf\n", seconds);
	*/

        s->frames++;

	return s->frame;
}

static const CHAR * GetSubtypeNameA(const GUID *pSubtype);
static int LocateSubtype(const GUID *pSubtype);
static void FreeMediaType(AM_MEDIA_TYPE& mt);

static void FreeMediaType(AM_MEDIA_TYPE& mt)
{
    if (mt.cbFormat != 0) {
        CoTaskMemFree((PVOID)mt.pbFormat);

        // Strictly unnecessary but tidier
        mt.cbFormat = 0;
        mt.pbFormat = NULL;
    }
    if (mt.pUnk != NULL) {
        mt.pUnk->Release();
        mt.pUnk = NULL;
    }
}

static void DeleteMediaType(AM_MEDIA_TYPE *pmt) {
        // allow NULL pointers for coding simplicity

        if (pmt == NULL) {
                return;
        }

        FreeMediaType(*pmt);
        CoTaskMemFree((PVOID)pmt);
}

static void
nv12_to_uyvy(int width, int height, const unsigned char *in, unsigned char *out)
{
        const int uyvy_linesize = vc_get_linesize(width, UYVY);
        for (ptrdiff_t y = 0; y < height; ++y) {
                const unsigned char *src_y    = in + width * y;
                const unsigned char *src_cbcr =
                    in + (ptrdiff_t) width * height + width * (y / 2);
                unsigned char *dst = out + y * uyvy_linesize;

                OPTIMIZED_FOR(int x = 0; x < width / 2; ++x)
                {
                                *dst++ = *src_cbcr++;
                                *dst++ = *src_y++;
                                *dst++ = *src_cbcr++;
                                *dst++ = *src_y++;
                }
        }
}

/// Apparently DirectShow uses bottom-to-top line ordering so we want make
/// it top-to-bottom
static void
bgr_flip_lines(int width, int height, const unsigned char *in,
               unsigned char *out)
{
        const size_t linesize = vc_get_linesize(width, BGR);
        for (int i = 0; i < height; ++i) {
                memcpy(out + i * linesize, in + (height - i - 1) * linesize,
                       linesize);
        }
}

/// convert from ABGR and bottom-to-top
static void
abgr_flip_and_swap(int width, int height, const unsigned char *in,
                      unsigned char *out)
{
        const size_t linesize = vc_get_linesize(width, RGBA);
        for (int i = 0; i < height; ++i) {
                vc_copylineRGBA(out + i * linesize,
                                in + (height - i - 1) * linesize, linesize, 16,
                                8, 0);
        }
}

#define GUID_FROM_FOURCC(fourcc) {fourcc, 0x0000, 0x10, {0x80,0x0,0x0,0xAA,0x0,0x38,0x9B,0x71}}
static const GUID GUID_R210 = GUID_FROM_FOURCC(0x30313272);
static const GUID GUID_v210 = GUID_FROM_FOURCC(0x30313276);
static const GUID GUID_V210 = GUID_FROM_FOURCC(0x30313256);
static const GUID GUID_HDYC = GUID_FROM_FOURCC(0x43594448);
static const GUID GUID_I420 = GUID_FROM_FOURCC(0x30323449);
static const GUID GUID_H264 = GUID_FROM_FOURCC(to_fourcc('H', '2', '6', '4')); // 0x34363248

static const struct {
        const GUID *pSubtype;
        const CHAR *pName;
	codec_t ug_codec;
	convert_t convert;
} BitCountMap[] = {
        {&MEDIASUBTYPE_RGB1,     "RGB Monochrome",   VC_NONE, nullptr           },
        { &MEDIASUBTYPE_RGB4,    "RGB VGA",          VC_NONE, nullptr           },
        { &MEDIASUBTYPE_RGB8,    "RGB 8",            VC_NONE, nullptr           },
        { &MEDIASUBTYPE_RGB565,  "RGB 565 (16 bit)", VC_NONE, nullptr           },
        { &MEDIASUBTYPE_RGB555,  "RGB 555 (16 bit)", VC_NONE, nullptr           },
        { &MEDIASUBTYPE_RGB24,   "RGB 24",           BGR,     bgr_flip_lines    },
        { &MEDIASUBTYPE_RGB32,   "RGB 32",           RGBA,    abgr_flip_and_swap},
        { &MEDIASUBTYPE_ARGB32,  "ARGB 32",          VC_NONE, nullptr           },
        { &MEDIASUBTYPE_Overlay, "Overlay",          VC_NONE, nullptr           },
        { &GUID_I420,            "I420",             UYVY,    i420_8_to_uyvy    },
        { &MEDIASUBTYPE_YUY2,    "YUY2",             YUYV,    nullptr           },
        { &GUID_R210,            "r210",             VC_NONE, nullptr           },
        { &GUID_v210,            "v210",             v210,    nullptr           },
        { &GUID_V210,            "V210",             v210,    nullptr           },
        { &MEDIASUBTYPE_UYVY,    "UYVY",             UYVY,    nullptr           },
        { &GUID_HDYC,            "HDYC",             UYVY,    nullptr           },
        { &MEDIASUBTYPE_MJPG,    "MJPG",             MJPG,    nullptr           },
        { &GUID_H264,            "H264",             H264,    nullptr           },
        { &MEDIASUBTYPE_NV12,    "NV12",             UYVY,    nv12_to_uyvy      },
        { &GUID_NULL,            "UNKNOWN",          VC_NONE, nullptr           },
};

static codec_t get_ug_codec(const GUID *pSubtype)
{
        return BitCountMap[LocateSubtype(pSubtype)].ug_codec;
}

static convert_t
get_conversion(const GUID *pSubtype)
{
        return BitCountMap[LocateSubtype(pSubtype)].convert;
}

static codec_t get_ug_from_subtype_name(const char *subtype_name) {
	for (unsigned int i = 0; i < sizeof BitCountMap / sizeof BitCountMap[0]; ++i) {
		if (strcmp(BitCountMap[i].pName, subtype_name) == 0) {
			return BitCountMap[i].ug_codec;
		}
	}
	return VIDEO_CODEC_NONE;
}

static int LocateSubtype(const GUID *pSubtype)
{
        assert(pSubtype);
        const GUID *pMediaSubtype;
        INT iPosition = 0;

/* this code is only to optain type data if UNKNOWN
        fprintf(stderr, "%s ", &pSubtype->Data1);
        fprintf(stderr, "%X ", pSubtype->Data1);
        fprintf(stderr, "%hX ", pSubtype->Data2);
        fprintf(stderr, "%hX ", pSubtype->Data3);
        for (int i = 0;i < 8; ++i)
                fprintf(stderr, "%.2hhX", pSubtype->Data4[i]);
*/

        // Scan the mapping list seeing if the source GUID matches any known
        // bitmap subtypes, the list is terminated by a GUID_NULL entry

        while (TRUE) {
                pMediaSubtype = BitCountMap[iPosition].pSubtype;
                if (IsEqualGUID(*pMediaSubtype,*pSubtype) ||
                                IsEqualGUID(*pMediaSubtype,GUID_NULL)
                   )
                {
                        break;
                }

                iPosition++;
        }

        return iPosition;
}

static const CHAR * GetSubtypeNameA(const GUID *pSubtype)
{
        return BitCountMap[LocateSubtype(pSubtype)].pName;
}

// this is here for people that linked to it directly; most people
// would use the header file that picks the A or W version.
static const CHAR * GetSubtypeName(const GUID *pSubtype)
{
        // the type is unknown to us, so print FourCC
        if (LocateSubtype(pSubtype) == sizeof BitCountMap / sizeof BitCountMap[0] - 1) {
                return pretty_print_fourcc(&pSubtype->Data1);
        }
        return GetSubtypeNameA(pSubtype);
}

extern "C" const struct video_capture_info vidcap_dshow_info = {
        vidcap_dshow_probe,
        vidcap_dshow_init,
        vidcap_dshow_done,
        vidcap_dshow_grab,
        MOD_NAME,
};

REGISTER_MODULE(dshow, &vidcap_dshow_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set noexpandtab: */

