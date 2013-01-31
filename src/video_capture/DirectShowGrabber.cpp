// DirectShowGrabber.cpp : Defines the exported functions for the DLL application.
//

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif


#include "debug.h"
#include "tv.h"
#include "video.h"
#include "video_capture.h"
#include "video_codec.h"

#include "DirectShowGrabber.h"

#include <tchar.h>
#include <DShow.h>
#include <Qedit.h>
#include <OleAuto.h>
//#include <Streams.h>

#include <iostream>
#include <stdio.h>
#include <Windows.h>

using namespace std;

static void DeleteMediaType(AM_MEDIA_TYPE *mediaType);
static const CHAR * GetSubtypeName(const GUID *pSubtype);

void ErrorDescription(HRESULT hr) 
{ 
        if(FACILITY_WINDOWS == HRESULT_FACILITY(hr)) 
                hr = HRESULT_CODE(hr); 
        TCHAR* szErrMsg; 

        if(FormatMessage( 
                                FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM, 
                                NULL, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
                                (LPTSTR)&szErrMsg, 0, NULL) != 0) 
        { 
                _tprintf(TEXT("%s"), szErrMsg);
                LocalFree(szErrMsg); 
        } else 
                _tprintf( TEXT("[Could not find a description for error # %#x.]\n"), hr);
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
	int deviceNumber;
	char *deviceName;
	int modeNumber;
	int width;
	int height;
	int fps;
	codec_t color_spec;
	bool convert_YUYV_RGB;

	struct video_frame *frame;
	struct tile *tile;
	long frameLength;
	long grabBufferLen;
	long returnBufferLen;
	bool haveNewReturnBuffer;
	//BYTE *buffer;
	BYTE *grabBuffer;
	BYTE *returnBuffer;
	BYTE *convert_buffer;

	unsigned long frames;
	struct timeval t0;

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
	s(s),
	frameNumber(frameNumber) {
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
		if (s->grabBufferLen != len) {
			s->grabBuffer = (BYTE *) realloc((void *) s->grabBuffer, len * sizeof(BYTE));
			if (s->grabBuffer == NULL) {
				s->grabBufferLen = 0;
				return S_OK;
			}
			s->grabBufferLen = len;
		}

		// We need to make a copy, DirectShow will do something with the data
                // Apparently DirectShow uses bottom-to-top line ordering so we want make
                // it top-to-bottom
                int linesize = vc_get_linesize(s->width, s->color_spec);
                for(int i = 0; i < s->height; ++i) {
                        memcpy((char *) s->grabBuffer + i * linesize,
                                        (char *) buffer + (s->height - i - 1) * linesize,
                                        linesize);
                }
		bool grabMightWait = false;
		if (!s->haveNewReturnBuffer) {
			grabMightWait = true;
			s->haveNewReturnBuffer = true;
		}
		LeaveCriticalSection(&s->returnBufferCS);
		if (grabMightWait) WakeConditionVariable(&s->grabWaitCV);

		return S_OK;
	}
};

static bool cleanup(struct vidcap_dshow_state *s) {
	if (s->mediaControl != NULL) s->mediaControl->Release();
	fprintf(stderr, "[dshow] Released: mediaControl\n");
	if (s->nullRenderer != NULL) s->nullRenderer->Release();
	fprintf(stderr, "[dshow] Released: nullRenderer\n");
	if (s->captureFilter != NULL) s->captureFilter->Release();
	fprintf(stderr, "[dshow] Released: captureFilter\n");
	if (s->sampleGrabberFilter != NULL) s->sampleGrabberFilter->Release();
	fprintf(stderr, "[dshow] Released: sampleGrabberFilter\n");
	if (s->moniker != NULL) s->moniker->Release();
	fprintf(stderr, "[dshow] Released: moniker\n");
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
	CoUninitialize();

	if (s->frame != NULL) vf_free(s->frame);
	if (s->grabBuffer != NULL) free(s->grabBuffer);
	if (s->returnBuffer != NULL) free(s->returnBuffer);
	if (s->convert_buffer != NULL) free(s->convert_buffer);
	if (yuv_clamp != NULL) free(yuv_clamp);

	DeleteCriticalSection(&s->returnBufferCS);
	return true;
}

static bool common_init(struct vidcap_dshow_state *s) {
	// set defaults
	s->deviceNumber = 1;
	s->modeNumber = 0;
	s->width = DEFAULT_VIDEO_WIDTH;
	s->height = DEFAULT_VIDEO_HEIGHT;
	s->fps = DEFAULT_FPS;

	s->frame = NULL;
	s->tile = NULL;
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
	res = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: COM library initialization failed.\n");
		ErrorDescription(res);
		return false;
	}

	// create device enumerator
	res = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&s->devEnumerator));
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot create System Device Enumerator.\n");
		return false;
	}

	res = s->devEnumerator->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &s->videoInputEnumerator, 0);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot create Video Input Device enumerator.\n");
		return false;
	}

	// Media processing classes (filters) are conected to a graph.
	// Create graph builder -- helper class for connecting of the graph
	res = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER,
			IID_ICaptureGraphBuilder2, (void **) &s->graphBuilder);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot create instance of Capture Graph Builder 2.\n");
		goto error;
	}

	// create the filter graph
	res = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER,
			IID_IGraphBuilder, (void **) &s->filterGraph);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot create instance of Filter Graph.\n");
		goto error;
	}

	// specify the graph builder a graph to be built
	res = s->graphBuilder->SetFiltergraph(s->filterGraph);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot attach Filter Graph to Graph Builder.\n");
		goto error;
	}

	return true;

error:
	cleanup(s);
	return false;
}

void show_help(struct vidcap_dshow_state *s) {
	printf("dshow grabber options:\n");
	printf("\t-t dshow:[Device]<DeviceNumber>:[Mode]<ModeNumber>[:fps]\n");
	printf("\t\tor\n");
	printf("\t-t dshow:[Device]<DeviceNumber>:RGB:width:height:fps\n\n");

	if (!common_init(s)) return;

	HRESULT res;
	int n = 1;
	// Enumerate all capture devices
	while ((res = s->videoInputEnumerator->Next(1, &s->moniker, NULL)) == S_OK) {
		// Attach structure for reading basic device properties
		IPropertyBag *properties;
		res = s->moniker->BindToStorage(0, 0, IID_PPV_ARGS(&properties));
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Failed to read device properties.\n");
			// Ignore the device
			continue;
		}

		// Read name of the device
		VARIANT var;
		VariantInit(&var);
		res = properties->Read(L"FriendlyName", &var, NULL);
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Failed to read device properties.\n");
			VariantClear(&var);
			// Ignore the device
			continue;
		}
		printf("Device %d: %ls\n", n++, var.bstrVal);

		// clean up structures
		VariantClear(&var);
		properties->Release();

		// bind the selected device to the capture filter
		IBaseFilter *captureFilter;
		res = s->moniker->BindToObject(NULL, NULL, IID_IBaseFilter, (void **) &captureFilter);
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot bind capture filter to device.\n");
			ErrorDescription(res);
			goto error;
		}

		// add the capture filter to the filter graph
		res = s->filterGraph->AddFilter(captureFilter, L"Capture filter");
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot add capture filter to filter graph.\n");
			goto error;
		}

		// connect stream config interface to the capture filter
		res = s->graphBuilder->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, captureFilter,
				IID_IAMStreamConfig, (void **) &s->streamConfig);
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot find interface for reading capture capabilites.\n");
			goto error;
		}

		int capCount, capSize;
		// read number of capture device capabilities
		res = s->streamConfig->GetNumberOfCapabilities(&capCount, &capSize);
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot read number of capture capabilites.\n");
			goto error;
		}
		// check if the format of capture capabilities is the right one
		if (capSize != sizeof(VIDEO_STREAM_CONFIG_CAPS)) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Unknown format of capture capabilites.\n");
			goto error;
		}

		// iterate over all capabilities
		for (int i = 0; i < capCount; i++) {
			AM_MEDIA_TYPE *mediaType;
			VIDEO_STREAM_CONFIG_CAPS streamCaps;

			res = s->streamConfig->GetStreamCaps(i, &mediaType, (BYTE*) &streamCaps);
			if (res != S_OK) {
				fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot read stream capabilities #%d.\n", i);
				continue;
			}
			if (mediaType->formattype != FORMAT_VideoInfo || mediaType->cbFormat < sizeof(VIDEOINFOHEADER)) {
				fprintf(stderr, "[dshow] vidcap_dshow_help: Unsupported format type for capability #%d.\n", i);
				continue;
			}
			VIDEOINFOHEADER *infoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mediaType->pbFormat);
			printf("\tMode %d: %s %dx%d\t", i, GetSubtypeName(&mediaType->subtype),
				infoHeader->bmiHeader.biWidth,
				infoHeader->bmiHeader.biHeight);

			DeleteMediaType(mediaType);

			if (i % 2 == 1) putchar('\n');
		}

		s->streamConfig->Release();
		res = s ->filterGraph->RemoveFilter(captureFilter);
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot remove capture filter from filter graph.\n");
			goto error;
		}
		captureFilter->Release();
		s->moniker->Release();

		printf("\n\n");
	}

error:
	return;
}

struct vidcap_type * vidcap_dshow_probe(void)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DSHOW_ID;
		vt->name        = "dshow";
		vt->description = "DirectShow Capture";
	}
	return vt;
}

static bool process_args(struct vidcap_dshow_state *s, char *init_fmt) {
	char *token;
	char *strtok_context;
	int i = 1;

	while ((token = strtok_s(init_fmt, ":", &strtok_context)) != NULL) {
		init_fmt = NULL;
		switch (i) {
			case 1 :
				if (strstr(token, "Device") == token) token = token + strlen("Device");
				if (isdigit(token[0])) { // device specified by number
					s->deviceNumber = atoi(token);
				} else { // device specified by name
					s->deviceName = (char *) malloc(sizeof(char) * (strlen(token) + 100));
					if (s->deviceName == false) return false;
					strcpy_s(s->deviceName, strlen(token), token);
					s->deviceNumber = -1;
				}
				break;
			case 2 :
				if (strstr(token, "Mode") == token) token = token + strlen("Mode");
				if (isdigit(token[0])) {
					s->modeNumber = atoi(token);
				} else {
					s->modeNumber = -1;
					if (strcmp(token, "YUYV") == 0) s->color_spec = YUYV;
					else if (strcmp(token, "RGB") == 0) s->color_spec = RGB;
					else {
						fprintf(stderr, "[dshow] Unsupported video format: %s.\n", token);
						return false;
					}
				}
				break;
			case 3 :
				if (s->modeNumber != -1) {
					s->fps = atoi(token);
					if (s->fps <= 0) {
						fprintf(stderr, "[dshow] Invalid FPS parameter: %s.\n", token);
						return false;
					}
					break;
				}
				s->width = atoi(token);
				if (s->width <= 0) {
					fprintf(stderr, "[dshow] Invalid video width parameter: %s.\n", token);
					return false;
				}
				break;
			case 4 :
				if (s->modeNumber != -1) break;
				s->height = atoi(token);
				if (s->height <= 0) {
					fprintf(stderr, "[dshow] Invalid video height parameter: %s.\n", token);
					return false;
				}
				break;
			case 5 :
				if (s->modeNumber != -1) break;
				s->fps = atoi(token);
				if (s->fps <= 0) {
					fprintf(stderr, "[dshow] Invalid video fps parameter: %s.\n", token);
					return false;
				}
				break;
			default :
				fprintf(stderr, "[dshow] More arguments than expected, ignoring.\n");
				break;
		}
		i++;
	}

	return true;
}

HRESULT PinIsConnected(IPin *pin, bool *result) {
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

HRESULT PinHasDirection(IPin *pin, PIN_DIRECTION direction, bool *result) {
	PIN_DIRECTION pinDir;

	HRESULT res = pin->QueryDirection(&pinDir);
	if (res == S_OK) {
		*result = (pinDir == direction);
	}

	return res;
}

HRESULT FindUnconnectedPin(IBaseFilter *filter, PIN_DIRECTION direction, IPin **pin) {
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

HRESULT ConnectFilters(IGraphBuilder *g, IPin *fromPin, IBaseFilter *toFilter) {
	IPin *toPin = NULL;

	HRESULT res = FindUnconnectedPin(toFilter, PINDIR_INPUT, &toPin);
	if (res == S_OK) {
		res = g->Connect(fromPin, toPin);
		toPin->Release();
	}

	return res;
}

HRESULT ConnectFilters(IGraphBuilder *g, IBaseFilter *fromFilter, IBaseFilter *toFilter) {
	IPin *fromPin = NULL;

	HRESULT res = FindUnconnectedPin(fromFilter, PINDIR_OUTPUT, &fromPin);
	if (res == S_OK) {
		res = ConnectFilters(g, fromPin, toFilter);
		fromPin->Release();
	}

	return res;
}

HRESULT GraphRun(IMediaControl *mc) {
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
		fprintf(stderr, "[dshow] Cannot run filter graph.\n");
		ErrorDescription(res);
	}

	return res;
}

HRESULT GraphPause(IMediaControl *mc) {
	HRESULT res;

	if ((res = mc->Pause()) == S_FALSE) {
		FILTER_STATE fs;

		while ((res = mc->GetState(500, (OAFilterState*) &fs)) != VFW_S_CANT_CUE && !(res == S_OK && fs == State_Paused)) {
			if (res != VFW_S_STATE_INTERMEDIATE && res != S_OK) break;
		}
	}

	if (res != S_OK) {
		fprintf(stderr, "[dshow] Cannot pause filter graph.\n");
		ErrorDescription(res);
	}

	return res;
}

HRESULT GetPinCategory(IPin *pPin, GUID *pPinCategory)
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

void * vidcap_dshow_init(char *init_fmt, unsigned int flags) {
	struct vidcap_dshow_state *s;
	HRESULT res;

	s = (struct vidcap_dshow_state *) malloc(sizeof(struct vidcap_dshow_state));
	if (s == NULL) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: memory allocation error\n");
		return NULL;
	}

	if (init_fmt && strcmp(init_fmt, "help") == 0) {
		show_help(s); 
		cleanup(s);
		return NULL;
	}

	if (!common_init(s)) {
		goto error;
	}

	if (init_fmt != NULL) {
		if (!process_args(s, init_fmt)) goto error;
	}

	InitializeConditionVariable(&s->grabWaitCV);

	// Select video capture device
	if (s->deviceNumber != -1) { // Device was specified by number
		for (int i = 1; i <= s->deviceNumber; i++) {
			// Take one device. We could take more at once, but it would require allocation of more moniker objects
			res = s->videoInputEnumerator->Next(1, &s->moniker, NULL);
			s->moniker->Release();
			if (res != S_OK) break;
		}
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_init: Device number %d\n was not found.\n", s->deviceNumber);
			goto error;
		}
	} else { // device specified by name
		while ((res = s->videoInputEnumerator->Next(1, &s->moniker, NULL)) == S_OK) {
			IPropertyBag *properties;
			res = s->moniker->BindToStorage(0, 0, IID_PPV_ARGS(&properties));
			if (res != S_OK) {
				fprintf(stderr, "[dshow] vidcap_dshow_init: Failed to read device properties.\n");
				// continue to other device; we still may find the correct one
				continue;
			}

			VARIANT var;
			VariantInit(&var);
			res = properties->Read(L"FriendlyName", &var, NULL);
			if (res != S_OK) {
				fprintf(stderr, "[dshow] vidcap_dshow_init: Failed to read device properties.\n");
				VariantClear(&var);
				// continue to other device; we still may find the correct one
				continue;
			}

			char *buf;
			buf = (char *) malloc(sizeof(char) * MAX_STRING_LEN);
			if (buf == NULL) {
				fprintf(stderr, "[dshow] vidcap_dshow_init: memory allocation error\n");
				goto error;
			}
			// convert to standard C string
			_snprintf_s(buf, MAX_STRING_LEN, _TRUNCATE, "%ls", var.bstrVal);

			VariantClear(&var);
			properties->Release();

			if (strcmp(s->deviceName, buf) == 0) {
				break;
			}
			s->moniker->Release();
		}

		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_init: Device named %s\n was not found.\n", s->deviceName);
			goto error;
		}
	}

	res = s->moniker->BindToObject(NULL, NULL, IID_IBaseFilter, (void **) &s->captureFilter);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot bind capture filter to device.\n");
		goto error;
	}

	res = s->filterGraph->AddFilter(s->captureFilter, L"Capture filter");
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot add capture filter to filter graph.\n");
		goto error;
	}

	res = s->graphBuilder->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, s->captureFilter,
			IID_IAMStreamConfig, (void **) &s->streamConfig);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot find interface for reading capture capabilites.\n");
		goto error;
	}

	// create instance of sample grabber
	res = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&s->sampleGrabberFilter));
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot create instance of sample grabber.\n");
		goto error;
	}

	// add the sample grabber to filter graph
	res = s->filterGraph->AddFilter(s->sampleGrabberFilter, L"Sample Grabber");
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot add sample grabber to filter graph.\n");
		goto error;
	}

	// query the sample grabber filter for control interface
	res = s->sampleGrabberFilter->QueryInterface(IID_PPV_ARGS(&s->sampleGrabber));
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot query sample grabber filter for control interface.\n");
		goto error;
	}

	// make the sample grabber buffer frames
	res = s->sampleGrabber->SetBufferSamples(TRUE);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot set sample grabber to buffer samples.\n");
		goto error;
	}

	// set media type for sample grabber; this is not done a very detailed setup, because it would be unneccessarily complicated to do so
	AM_MEDIA_TYPE sampleGrabberMT;
	ZeroMemory(&sampleGrabberMT, sizeof(sampleGrabberMT));
	sampleGrabberMT.majortype = MEDIATYPE_Video;
	sampleGrabberMT.subtype = MEDIASUBTYPE_RGB24;
	res = s->sampleGrabber->SetMediaType(&sampleGrabberMT);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot setup media type of grabber filter.\n");
		goto error;
	}

	int capCount, capSize;
	res = s->streamConfig->GetNumberOfCapabilities(&capCount, &capSize);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot read number of capture capabilites.\n");
		goto error;
	}
	if (capSize != sizeof(VIDEO_STREAM_CONFIG_CAPS)) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Unknown format of capture capabilites.\n");
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
			fprintf(stderr, "[dshow] Invalid mode index!\n");
			goto error;
		}
		// Some other error occured
		if (res != S_OK) {
			fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot read stream capabilities #%d (index is correct).\n", s->modeNumber);
			goto error;
		}

		if (mediaType->subtype == MEDIASUBTYPE_RGB24) s->color_spec = RGB;
		else if (mediaType->subtype == MEDIASUBTYPE_YUY2) s->color_spec = YUYV;
		else {
			fprintf(stderr, "[dshow] Unknown color specifiation of the chosen format, cannot grab.\n");
			goto error;
		}

		VIDEOINFOHEADER *infoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mediaType->pbFormat);
		s->width = infoHeader->bmiHeader.biWidth;
		s->height = infoHeader->bmiHeader.biHeight;

		format_found = true;
	} else {
		for (int i = 0; i < capCount; i++) {
			res = s->streamConfig->GetStreamCaps(i, &mediaType, (BYTE*) &streamCaps);
			if (res != S_OK) {
				fprintf(stderr, "[dshow] vidcap_dshow_help: Cannot read stream capabilities #%d.\n", i);
				continue;
			}
			if (mediaType->formattype != FORMAT_VideoInfo || mediaType->cbFormat < sizeof(VIDEOINFOHEADER)) {
				fprintf(stderr, "[dshow] vidcap_dshow_help: Unsupported format type for capability #%d.\n", i);
				continue;
			}
			if ((s->color_spec == RGB  && mediaType->subtype != MEDIASUBTYPE_RGB24) ||
				(s->color_spec == YUYV && mediaType->subtype != MEDIASUBTYPE_YUY2))
				continue;

			VIDEOINFOHEADER *infoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mediaType->pbFormat);
			if (infoHeader->bmiHeader.biHeight == s->height &&
				infoHeader->bmiHeader.biWidth  == s->width) {
					format_found = true;
					break;
			}

			DeleteMediaType(mediaType);
		}
	}

	if (!format_found) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Requested format not supported by the device. Quitting.\n");
		goto error;
	}

	s->streamConfig->GetFormat(&mediaType);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot get current capture format.\n");
		goto error;
	}
	switch (s->color_spec) {
		case RGB : mediaType->subtype = MEDIASUBTYPE_RGB24;
			   break;
		case YUYV : mediaType->subtype = MEDIASUBTYPE_YUY2;
			    break;
	}
	VIDEOINFOHEADER *infoHeader;
	infoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mediaType->pbFormat);
	infoHeader->rcSource.bottom = s->height;
	infoHeader->rcSource.right = s->width;
	infoHeader->AvgTimePerFrame = (REFERENCE_TIME) (1e7 / s->fps);
	res = s->streamConfig->SetFormat(mediaType);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot set capture format.\n");
		goto error;
	}
	DeleteMediaType(mediaType);

	if (s->convert_YUYV_RGB) {
		s->convert_buffer = (BYTE *) malloc(s->height * s->width * 3);
		if (s->convert_buffer == NULL) {
			fprintf(stderr, "[dshow] vidcap_dshow_init: memory allocation error\n");
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
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot create NullRenderer.\n");
		goto error;
	}

	res = s->filterGraph->AddFilter(s->nullRenderer, L"NullRenderer");
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot add null renderer to filter graph.\n");
		goto error;
	}

	IEnumPins *pinEnum;
	IPin *pin;

	res = s->captureFilter->EnumPins(&pinEnum);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Error enumerating pins of capture filter.\n");
		goto error;
	}

	while (pinEnum->Next(1, &pin, NULL) == S_OK) {
		res = ConnectFilters(s->filterGraph, pin, s->sampleGrabberFilter);
		if (pin != NULL) pin->Release();
		if (res == S_OK) {
			break;
		}
	}
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot connect capture filter to sample grabber.\n");
		goto error;
	}

	res = ConnectFilters(s->filterGraph, s->sampleGrabberFilter, s->nullRenderer);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot connect sample grabber to null renderer.\n");
		goto error;
	}

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
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot find media control interface.\n");
		goto error;
	}

	FILTER_STATE fs;
	res = s->mediaControl->Run();
	while ((res = s->mediaControl->GetState(500, (OAFilterState*) &fs)) != VFW_S_CANT_CUE && !(res == S_OK && fs == State_Running)) {
			if (res != VFW_S_STATE_INTERMEDIATE && res != S_OK) {
				break;
			}
	}
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Cannot run filter graph.\n");
		ErrorDescription(res);
		goto error;
	}
	res = s->sampleGrabberFilter->GetState(INFINITE, &fs);
	if (res != S_OK) {
		fprintf(stderr, "[dshow] filter getstate error\n");
		ErrorDescription(res);
		goto error;
	}

	// Create non-inheritable mutex without a name, owned by this thread
	InitializeCriticalSectionAndSpinCount(&s->returnBufferCS, 0x40);
	s->haveNewReturnBuffer = false;

	s->frame = vf_alloc(1);
	s->tile = vf_get_tile(s->frame, 0);
	s->frame->interlacing = PROGRESSIVE;
	s->frame->color_spec = BGR;
	s->frame->fps = s->fps;
	s->tile->width = s->width;
	s->tile->height = s->height;
	s->frameLength = 0;

	s->frames = 0;
	gettimeofday(&s->t0, NULL);

	return s;

error:
	cleanup(s);
	return NULL;
}

void vidcap_dshow_finish(void *state) {
	UNUSED(state);
}

void vidcap_dshow_done(void *state) {
	struct vidcap_dshow_state *s = (struct vidcap_dshow_state *) state;

	HRESULT res = s->mediaControl->Stop();
	if (res != S_OK) {
		fprintf(stderr, "[dshow] vidcap_dshow_init: Failed to stop filter graph.\n");
	}

	vf_free(s->frame);

	cleanup(s);
	free(s);
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
	BYTE *d = dst;

	for (int i = 0; i < input_len; i += 4) {
		convert_yuv_rgb(s[0], s[1], s[3], dst);
		convert_yuv_rgb(s[2], s[1], s[3], dst + 3);
		s += 4;
		dst += 6;
	}
}

struct video_frame * vidcap_dshow_grab(void *state, struct audio_frame **audio) {
	struct vidcap_dshow_state *s = (struct vidcap_dshow_state *) state;
	*audio = NULL;

	//fprintf(stderr, "[dshow] GRAB: enter: %d\n", s->deviceNumber);
	EnterCriticalSection(&s->returnBufferCS);
	//fprintf(stderr, "[dshow] s: %p\n", s);
	while (!s->haveNewReturnBuffer) {
		//fprintf(stderr, "[dshow] s: %p\n", s);
		SleepConditionVariableCS(&s->grabWaitCV, &s->returnBufferCS, INFINITE);
		//fprintf(stderr, "[dshow] s: %p\n", s);
	}

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

	s->tile->data = (char *) s->returnBuffer;
	//fprintf(stderr, "[dshow] s: %p\n", s);
	//s->tile->data_len = s->width * s->height * 3;
	s->tile->data_len = s->returnBufferLen;

/*
	fprintf(stderr, "[dshow] s5: %p\n", s);
	fprintf(stderr, "[dshow] s6: %p\n", s);
	fprintf(stderr, "[dshow] s7: %p\n", s);
	fprintf(stderr, "[dshow] s: %p\n", s);
	fprintf(stderr, "[dshow] %lf\n", seconds);
	*/

        s->frames++;

	struct timeval t;
	gettimeofday(&t, NULL);
	double seconds = tv_diff(t, s->t0);
	if (seconds >= 5) {
		double fps  = s->frames / seconds;
		fprintf(stderr, "[dshow] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
		s->t0 = t;
		s->frames = 0;
	}

	return s->frame;
}

static const CHAR * GetSubtypeNameA(const GUID *pSubtype);
static const WCHAR * GetSubtypeNameW(const GUID *pSubtype);
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

static const struct {
        const GUID *pSubtype;
        WORD BitCount;
        const CHAR *pName;
        const WCHAR *wszName;
} BitCountMap[] =  { &MEDIASUBTYPE_RGB1,        1,   "RGB Monochrome",     L"RGB Monochrome",   
        &MEDIASUBTYPE_RGB4,        4,   "RGB VGA",            L"RGB VGA",          
        &MEDIASUBTYPE_RGB8,        8,   "RGB 8",              L"RGB 8",            
        &MEDIASUBTYPE_RGB565,      16,  "RGB 565 (16 bit)",   L"RGB 565 (16 bit)", 
        &MEDIASUBTYPE_RGB555,      16,  "RGB 555 (16 bit)",   L"RGB 555 (16 bit)", 
        &MEDIASUBTYPE_RGB24,       24,  "RGB 24",             L"RGB 24",           
        &MEDIASUBTYPE_RGB32,       32,  "RGB 32",             L"RGB 32",
        &MEDIASUBTYPE_ARGB32,    32,  "ARGB 32",             L"ARGB 32",
        &MEDIASUBTYPE_Overlay,     0,   "Overlay",            L"Overlay",          
        &MEDIASUBTYPE_I420,       12,   "I420",               L"I420",          
        &GUID_NULL,                0,   "UNKNOWN",            L"UNKNOWN"           
};

static int LocateSubtype(const GUID *pSubtype)
{
        assert(pSubtype);
        const GUID *pMediaSubtype;
        INT iPosition = 0;

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

static const WCHAR * GetSubtypeNameW(const GUID *pSubtype)
{
        return BitCountMap[LocateSubtype(pSubtype)].wszName;
}

static const CHAR * GetSubtypeNameA(const GUID *pSubtype)
{
        return BitCountMap[LocateSubtype(pSubtype)].pName;
}

// this is here for people that linked to it directly; most people
// would use the header file that picks the A or W version.
static const CHAR * GetSubtypeName(const GUID *pSubtype)
{
        return GetSubtypeNameA(pSubtype);
}

